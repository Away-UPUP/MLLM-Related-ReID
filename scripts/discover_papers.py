"""Find recent MLLM ReID papers and add them to the README."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from html import escape, unescape
import json
from pathlib import Path
import re
import sys
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
START = "<!-- discovered-papers:start -->"
END = "<!-- discovered-papers:end -->"
TABLE_HEADER = "| Publication date | Paper | Source |\n|---|---|---|"
LOOKBACK_DAYS = 60
SEARCH_TERMS = (
    "person re-identification",
    "object re-identification",
    "interactive person retrieval",
    "text-based person retrieval",
)
ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_ID = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/|10\.48550/arxiv\.)(\d{4}\.\d{4,5})", re.I)
DOI = re.compile(r"10\.\d{4,9}/[^\s<>\"|]+", re.I)
TASK = re.compile(
    r"\b(?:person|pedestrian|object|vehicle)\s+re[ -]?(?:id|identification)\b"
    r"|\b(?:interactive|text[- ]based|text[- ]guided|text[- ]to[- ]image)[- ]"
    r"(?:person|pedestrian)[- ](?:retrieval|search)\b",
    re.I,
)
REID = re.compile(r"\bre[ -]?(?:id|identification)\b", re.I)
MODEL = re.compile(
    r"\b(?:mllms?|lvlms?|llms?|mllmreid|large (?:vision[- ]language|language|multimodal) models?"
    r"|general[- ]purpose vision[- ]language models?|multimodal large language models?|multimodal llms?|llava(?:[- ]\w+)?"
    r"|qwen[\w.]*[- ]vl|internvl\w*|gpt[- ]4[ov]|gemini(?:[- ](?:pro|flash|\d[\w.]*))?)\b",
    re.I,
)
METHOD = re.compile(
    r"\b(?:we\s+(?:propose|present|introduce|develop|use|utili[sz]e|leverage|employ|adopt|harness|investigate|explore|adapt|design|show|demonstrate)"
    r"|our\s+(?:method|approach|framework|model|system|pipeline|benchmark|work)"
    r"|(?:this|the proposed)\s+(?:paper|work|method|framework|approach|model))\b",
    re.I,
)


@dataclass(frozen=True)
class Paper:
    title: str
    abstract: str
    published: date
    url: str
    source: str
    aliases: str = ""


def fetch(endpoint, params):
    request = Request(
        f"{endpoint}?{urlencode(params)}",
        headers={"User-Agent": "MLLM-Related-ReID/1.0 (https://github.com/Away-UPUP/MLLM-Related-ReID)"},
    )
    for attempt in range(2):
        try:
            with urlopen(request, timeout=20) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt or isinstance(error, HTTPError) and error.code not in (429, 500, 502, 503, 504):
                raise
            sleep(3)


def clean_text(value):
    return " ".join(re.sub(r"<[^>]+>", " ", unescape(value or "")).split())


def doi_url(value):
    value = (value or "").removeprefix("https://doi.org/")
    return f"https://doi.org/{quote(value, safe='/._-:')}" if DOI.fullmatch(value) else ""


def arxiv_papers(term, since):
    today = datetime.now(timezone.utc).date()
    query = f'ti:"{term}" AND submittedDate:[{since:%Y%m%d}0000 TO {today:%Y%m%d}2359]'
    feed = ET.fromstring(fetch("https://export.arxiv.org/api/query", {
        "search_query": query,
        "max_results": 60,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }))
    for item in feed.findall("atom:entry", ATOM):
        identifier = ARXIV_ID.search(item.findtext("atom:id", default="", namespaces=ATOM))
        published = item.findtext("atom:published", default="", namespaces=ATOM)
        if not identifier or not published:
            continue
        yield Paper(
            clean_text(item.findtext("atom:title", namespaces=ATOM)),
            clean_text(item.findtext("atom:summary", namespaces=ATOM)),
            date.fromisoformat(published[:10]),
            f"https://arxiv.org/abs/{identifier[1]}",
            "arXiv",
            doi_url(item.findtext("arxiv:doi", namespaces=ATOM)),
        )


def openalex_papers(term, since):
    payload = json.loads(fetch("https://api.openalex.org/works", {
        "search": f'"{term}" "language model"',
        "filter": f"from_publication_date:{since},type:article|preprint",
        "per_page": 35,
        "select": "id,display_name,doi,publication_date,abstract_inverted_index",
    }))
    for item in payload["results"]:
        identifier = item.get("id") or ""
        published = item.get("publication_date")
        url = doi_url(item.get("doi"))
        if url:
            arxiv_id = ARXIV_ID.search(url)
            if arxiv_id:
                url = f"https://arxiv.org/abs/{arxiv_id[1]}"
        elif re.fullmatch(r"https://openalex\.org/W[0-9]+", identifier):
            url = identifier
        if not url or not published:
            continue
        indexed_words = item.get("abstract_inverted_index") or {}
        positions = sorted((position, word) for word, locations in indexed_words.items() for position in locations)
        yield Paper(
            clean_text(item.get("display_name")),
            clean_text(" ".join(word for _, word in positions)),
            date.fromisoformat(published),
            url,
            "OpenAlex",
        )


def crossref_papers(term, since):
    payload = json.loads(fetch("https://api.crossref.org/works", {
        "query.title": term,
        "query.bibliographic": "multimodal large language model vision",
        "filter": f"from-pub-date:{since},type:proceedings-article,type:journal-article",
        "rows": 35,
        "select": "DOI,title,published,abstract",
    }))
    for item in payload["message"]["items"]:
        url = doi_url(item.get("DOI"))
        titles = item.get("title") or []
        parts = (item.get("published") or {}).get("date-parts", [[]])[0]
        if url and titles and len(parts) == 3 and all(parts):
            yield Paper(clean_text(titles[0]), clean_text(item.get("abstract")), date(*parts), url, "Crossref")


def collect(source, search, since):
    papers = []
    errors = []
    for term in SEARCH_TERMS:
        try:
            papers.extend(list(search(term, since)))
        except (OSError, ValueError, KeyError, ET.ParseError) as error:
            errors.append(f"{source} / {term}: {error}")
        if source == "arXiv" and term != SEARCH_TERMS[-1]:
            sleep(3)
    return papers, errors


def is_relevant(paper):
    if re.match(r"^(?:review of|comment on|correction|retraction)\b", paper.title, re.I):
        return False
    has_task = TASK.search(paper.title) or (REID.search(paper.title) and TASK.search(paper.abstract))
    if not has_task:
        return False
    if MODEL.search(paper.title):
        return True
    return any(
        MODEL.search(sentence) and METHOD.search(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", paper.abstract)
    )


def title_key(title):
    # Ignore a short method prefix such as "ChatReID:" so conference and arXiv titles can match.
    title = re.sub(r"^[\w-]{2,24}:\s+", "", clean_text(title))
    return " ".join(re.findall(r"\w+", title.lower().replace("re-identification", "reidentification")))


def identifiers(text):
    text = unescape(text)
    arxiv_ids = {f"arxiv:{match[1]}" for match in ARXIV_ID.finditer(text)}
    dois = {
        match[0].lower().rstrip(").,")
        for match in re.finditer(r"10\.\d{4,9}/[^\s<>\")|]+", text, re.I)
    }
    return arxiv_ids | dois


def markdown_title(title):
    return re.sub(r"[\x60\\*_~!\[\]|]", lambda char: f"&#{ord(char[0])};", escape(title))


def update_readme(readme, papers, curated_titles):
    if readme.count(START) != 1 or readme.count(END) != 1:
        raise ValueError("Expected exactly one pair of discovery markers in README.md")
    before, section = readme.split(START, 1)
    body, after = section.split(END, 1)
    rows = [line for line in body.splitlines() if re.match(r"^\| \d{4}-\d{2}-\d{2} \|", line)]
    titles = {title_key(title) for title in curated_titles.splitlines() if title.strip()}
    titles.update(title_key(row.split(" | ")[1]) for row in rows)
    known_ids = identifiers(readme)
    known_urls = set(re.findall(r"""(?:href="|\]\()([^")\s]+)""", readme))
    added = []
    for paper in papers:
        key = title_key(paper.title)
        ids = identifiers(f"{paper.url} {paper.aliases}")
        if not key or paper.url in known_urls or known_ids & ids:
            continue
        if key in titles or (
            len(key.split()) >= 6 and any(
                abs(len(key) - len(title)) <= 15 and SequenceMatcher(None, key, title).ratio() >= 0.94
                for title in titles
            )
        ):
            continue
        titles.add(key)
        known_ids.update(ids)
        known_urls.add(paper.url)
        added.append(paper)
        alias = f" <!-- {paper.aliases} -->" if paper.aliases else ""
        rows.append(f"| {paper.published} | {markdown_title(paper.title)} | [{paper.source}]({paper.url}){alias} |")
    if not added:
        return readme, []
    rows.sort(reverse=True)
    table = "\n".join([TABLE_HEADER, *rows])
    return f"{before}{START}\n{table}\n{END}{after}", added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print matches and leave README.md intact.")
    args = parser.parse_args()
    today = datetime.now(timezone.utc).date()
    since = today - timedelta(days=LOOKBACK_DAYS)
    sources = {"arXiv": arxiv_papers, "OpenAlex": openalex_papers, "Crossref": crossref_papers}
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        jobs = {name: pool.submit(collect, name, search, since) for name, search in sources.items()}
        eligible = []
        completed_queries = 0
        for source, job in jobs.items():
            papers, errors = job.result()
            matches = [paper for paper in papers if since <= paper.published <= today and is_relevant(paper)]
            completed_queries += len(SEARCH_TERMS) - len(errors)
            eligible.extend(matches)
            print(f"{source}: {len(papers)} records, {len(matches)} relevant matches")
            for error in errors:
                print(f"Warning: {error}", file=sys.stderr)
    if not completed_queries:
        raise SystemExit("All paper searches failed. README.md was left intact.")

    readme = ROOT / "README.md"
    curated_titles = (ROOT / "scripts" / "reviewed_titles.txt").read_text(encoding="utf-8")
    updated, added = update_readme(readme.read_text(encoding="utf-8"), eligible, curated_titles)
    if added and not args.dry_run:
        readme.write_text(updated, encoding="utf-8", newline="\n")
    print(f"{'Would add' if args.dry_run else 'Added'} {len(added)} new papers.")
    for paper in added:
        print(f"- {paper.published}: {paper.title} ({paper.url})")


if __name__ == "__main__":
    main()
