#!/usr/bin/env python3
"""Collect Australia-focused AI news from Guardian and GDELT.

The goal is not maximum volume. The goal is a corpus where AI is discussed
from an Australian context rather than as generic US/China technology news.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from lxml import html as lxml_html


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "results" / "collection"

GUARDIAN_ENDPOINT = "https://content.guardianapis.com/search"
GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

USER_AGENT = "student-australia-ai-perspective-news-research/0.1"

AI_PATTERNS = {
    "artificial intelligence": r"\bartificial intelligence\b",
    "generative ai": r"\bgenerative ai\b|\bgenai\b",
    "chatgpt": r"\bchatgpt\b",
    "openai": r"\bopenai\b",
    "machine learning": r"\bmachine learning\b",
    "large language model": r"\blarge language model(s)?\b|\bllm(s)?\b",
    "deepfake": r"\bdeepfake(s)?\b",
    "algorithm": r"\balgorithm(ic)?\b",
    "automation": r"\bautomation\b|\bautomated\b",
    "data centre": r"\bdata[\s-]?cent(?:re|er)(s)?\b",
    "chatbot": r"\bchatbot(s)?\b",
    "ai": r"\bAI\b",
}

AU_PATTERNS = {
    "australia": r"\baustralia\b",
    "australian": r"\baustralian(s)?\b|\baussie(s)?\b",
    "federal government": r"\bfederal government\b|\balbanese government\b|\baustralian government\b",
    "state": r"\bNSW\b|\bNew South Wales\b|\bVictoria\b|\bQueensland\b|\bWestern Australia\b|\bWA\b|\bTasmania\b|\bSouth Australia\b|\bACT\b|\bNorthern Territory\b",
    "city": r"\bSydney\b|\bMelbourne\b|\bCanberra\b|\bBrisbane\b|\bPerth\b|\bAdelaide\b|\bHobart\b|\bDarwin\b",
    "regulator": r"\beSafety\b|\bOAIC\b|\bASIC\b|\bACCC\b|\bCSIRO\b|\bAEMO\b",
}

FOREIGN_PATTERNS = {
    "us": r"\bUS\b|\bUnited States\b|\bAmerica(n)?\b|\bWashington\b|\bTrump\b|\bBiden\b|\bWhite House\b|\bSilicon Valley\b|\bWall Street\b",
    "china": r"\bChina\b|\bChinese\b|\bBeijing\b|\bXi Jinping\b|\bHong Kong\b",
    "global_tech": r"\bNvidia\b|\bMicrosoft\b|\bGoogle\b|\bMeta\b|\bApple\b|\bTesla\b|\bAnthropic\b|\bOpenAI\b|\bDeepSeek\b|\bSam Altman\b|\bElon Musk\b",
    "markets": r"\bstock(s)?\b|\bshares?\b|\bearnings\b|\bIPO\b|\bNasdaq\b|\bS&P\b|\bDow Jones\b|\binvestor(s)?\b",
}

SOURCE_DOMAINS = [
    "abc.net.au",
    "sbs.com.au",
    "theguardian.com",
    "theconversation.com",
    "smartcompany.com.au",
    "itnews.com.au",
    "governmentnews.com.au",
    "innovationaus.com",
    "smh.com.au",
    "theage.com.au",
    "watoday.com.au",
    "afr.com",
    "news.com.au",
    "crikey.com.au",
]

LIVE_OR_LOW_SIGNAL_PATTERNS = [
    r"\bas it happened\b",
    r"\bmorning mail\b",
    r"\bafternoon update\b",
    r"\bfive great reads\b",
    r"\bthe crunch\b",
    r"\bcozzie livs\b",
    r"\blive\b",
    r"\bmarkets today\b",
    r"\bstock explodes\b",
    r"\bstock rockets\b",
    r"\bq[1-4]\b.*\bearnings\b",
    r"\bpress release\b",
]

GUARDIAN_QUERIES = [
    '"artificial intelligence" AND (Australia OR Australian OR Australians)',
    '"generative AI" AND (Australia OR Australian OR Australians)',
    'ChatGPT AND (Australia OR Australian OR Australians)',
    'AI AND Australia AND (regulation OR workplace OR jobs OR privacy OR schools OR healthcare OR copyright)',
    'AI AND Australian AND (government OR workers OR students OR courts OR authors OR artists)',
    'deepfake AND (Australia OR Australian OR Australians)',
    '"AI data centre" AND (Australia OR Australian)',
]

GDELT_QUERIES = [
    '"artificial intelligence" Australia sourcecountry:australia',
    '"artificial intelligence" Australian sourcecountry:australia',
    '"generative AI" Australia sourcecountry:australia',
    'ChatGPT Australia sourcecountry:australia',
    '"AI regulation" Australia sourcecountry:australia',
    '"artificial intelligence" workplace Australia sourcecountry:australia',
    '"artificial intelligence" jobs Australia sourcecountry:australia',
    '"artificial intelligence" privacy Australia sourcecountry:australia',
    '"generative AI" schools Australia sourcecountry:australia',
    '"artificial intelligence" healthcare Australia sourcecountry:australia',
    'deepfake Australia sourcecountry:australia',
    '"AI data centre" Australia sourcecountry:australia',
    '"artificial intelligence" domain:abc.net.au',
    '"generative AI" domain:abc.net.au',
    '"artificial intelligence" domain:sbs.com.au',
    '"artificial intelligence" domain:smartcompany.com.au',
    '"artificial intelligence" domain:itnews.com.au',
    '"artificial intelligence" domain:governmentnews.com.au',
    '"artificial intelligence" domain:theconversation.com',
]


@dataclass
class NewsRecord:
    dataset_id: str
    source_system: str
    collection_method: str
    source_domain: str
    canonical_url: str
    title: str
    published_at: str = ""
    section: str = ""
    author: str = ""
    summary: str = ""
    body_text: str = ""
    query: str = ""
    ai_terms: list[str] = field(default_factory=list)
    australia_terms: list[str] = field(default_factory=list)
    foreign_terms: list[str] = field(default_factory=list)
    australia_score: int = 0
    foreign_score: int = 0
    relevance_score: int = 0
    audit_status: str = "pending"
    audit_reason: str = ""


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def request_json(url: str, params: dict[str, str], timeout: int = 30) -> dict:
    encoded = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{encoded}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    if not body.strip().startswith("{"):
        raise ValueError(body[:240])
    return json.loads(body)


def fetch_html(url: str, timeout: int = 8) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return ""
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def extract_article_text(page_html: str) -> str:
    if not page_html.strip():
        return ""
    try:
        tree = lxml_html.fromstring(page_html)
    except Exception:
        return ""
    for bad in tree.xpath("//script|//style|//noscript|//nav|//footer|//header|//form|//aside"):
        parent = bad.getparent()
        if parent is not None:
            parent.remove(bad)
    candidates = tree.xpath("//article")
    text_parts: list[str] = []
    for node in candidates or [tree]:
        paragraphs = node.xpath(".//p//text()")
        if paragraphs:
            text_parts.extend(paragraphs)
    text = html.unescape(" ".join(t.strip() for t in text_parts if t.strip()))
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urllib.parse.urlsplit(url.strip())
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    clean_query = urllib.parse.urlencode(
        [(k, v) for k, v in query if not k.lower().startswith(("utm_", "fbclid", "gclid", "mc_"))],
        doseq=True,
    )
    return urllib.parse.urlunsplit(
        (parsed.scheme or "https", parsed.netloc.lower(), parsed.path.rstrip("/"), clean_query, "")
    )


def domain_from_url(url: str) -> str:
    try:
        return urllib.parse.urlsplit(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def matched_terms(text: str, patterns: dict[str, str], flags: int = re.I) -> list[str]:
    found = [name for name, pattern in patterns.items() if re.search(pattern, text or "", flags)]
    return sorted(set(found))


def low_signal_title_or_url(title: str, url: str) -> bool:
    text = f"{title} {url}".lower()
    return any(re.search(pattern, text, re.I) for pattern in LIVE_OR_LOW_SIGNAL_PATTERNS)


def metadata_priority(title: str, url: str, domain: str) -> int:
    title_url = f"{title} {url} {domain}"
    score = 0
    score += 12 * len(matched_terms(title_url, AI_PATTERNS))
    score += 10 * len(matched_terms(title_url, AU_PATTERNS))
    score += 4 if domain.endswith(".au") else 0
    score -= 12 * len(matched_terms(title_url, FOREIGN_PATTERNS, flags=0))
    if low_signal_title_or_url(title, url):
        score -= 80
    return score


def normalized_title(title: str) -> str:
    title = re.split(r"\s+\|\s+", title, maxsplit=1)[0]
    text = re.sub(r"[^\w\s]", " ", title.lower())
    return re.sub(r"\s+", " ", text).strip()


def audit_record(record: NewsRecord) -> NewsRecord:
    title_url = f"{record.title} {record.canonical_url} {record.section}"
    full = f"{title_url} {record.summary} {record.body_text[:8000]}"
    lead = f"{title_url} {record.summary} {record.body_text[:1500]}"
    title_ai = matched_terms(record.title, AI_PATTERNS)
    title_summary_ai = matched_terms(f"{record.title} {record.summary} {record.canonical_url}", AI_PATTERNS)
    ai_terms = matched_terms(full, AI_PATTERNS)
    au_title_only_terms = matched_terms(record.title, AU_PATTERNS)
    au_title_terms = matched_terms(title_url, AU_PATTERNS)
    au_lead_terms = matched_terms(lead, AU_PATTERNS)
    au_terms = matched_terms(full, AU_PATTERNS)
    foreign_title_terms = matched_terms(title_url, FOREIGN_PATTERNS, flags=0)
    foreign_terms = matched_terms(full, FOREIGN_PATTERNS, flags=0)

    au_hits_body = sum(len(re.findall(pattern, full, re.I)) for pattern in AU_PATTERNS.values())
    foreign_hits_body = sum(len(re.findall(pattern, full)) for pattern in FOREIGN_PATTERNS.values())
    title_au_bonus = 6 if au_title_terms else 0
    title_ai_bonus = 4 if title_ai else 0
    local_domain_bonus = 2 if record.source_domain.endswith(".au") or record.source_domain == "theguardian.com" else 0
    australia_score = min(40, au_hits_body + title_au_bonus + local_domain_bonus)
    foreign_score = min(40, foreign_hits_body + (5 if foreign_title_terms else 0))
    relevance_score = min(100, 12 * len(ai_terms) + 7 * len(au_terms) + title_ai_bonus + title_au_bonus - foreign_score)

    reasons: list[str] = []
    status = "accepted"
    if low_signal_title_or_url(record.title, record.canonical_url):
        status = "rejected"
        reasons.append("low_signal_live_blog_or_market_wrap")
    if not ai_terms:
        status = "rejected"
        reasons.append("no_ai_terms")
    if not title_summary_ai:
        status = "rejected"
        reasons.append("ai_not_in_title_summary_or_url")
    if not au_terms:
        status = "rejected"
        reasons.append("no_australia_context_terms")
    has_strong_au_context = bool(
        au_title_terms
        or au_lead_terms
        or "australia-news" in record.canonical_url
        or record.section.lower() == "australia news"
    )
    if not has_strong_au_context:
        status = "rejected"
        reasons.append("australia_context_not_in_title_lead_or_section")
    if record.section.lower() in {"world news", "uk news", "us news"} and not au_title_terms:
        status = "rejected"
        reasons.append("foreign_section_without_australia_title")
    if not au_title_only_terms and foreign_title_terms:
        status = "rejected"
        reasons.append("foreign_title_without_australia_context")
    if foreign_score > australia_score:
        status = "rejected"
        reasons.append("foreign_score_exceeds_australia_score")
    if record.body_text and len(record.body_text) < 300:
        status = "rejected"
        reasons.append("body_too_short")
    if record.source_system == "gdelt" and not record.body_text:
        status = "rejected"
        reasons.append("gdelt_body_fetch_failed")
    if status == "accepted" and not (au_title_terms or australia_score >= 6):
        status = "rejected"
        reasons.append("weak_australian_perspective")

    record.ai_terms = ai_terms
    record.australia_terms = au_terms
    record.foreign_terms = foreign_terms
    record.australia_score = australia_score
    record.foreign_score = foreign_score
    record.relevance_score = relevance_score
    record.audit_status = status
    record.audit_reason = ";".join(reasons) if reasons else "passes_australia_perspective_filter"
    return record


def collect_guardian(api_key: str, start_date: str, end_date: str, max_pages: int, run_raw_dir: Path) -> list[NewsRecord]:
    if not api_key:
        print("[Guardian] skipped: GUARDIAN_API_KEY is not set")
        return []
    rows_by_url: dict[str, NewsRecord] = {}
    for query_index, query in enumerate(GUARDIAN_QUERIES, start=1):
        print(f"[Guardian] query {query_index}/{len(GUARDIAN_QUERIES)}: {query}", flush=True)
        for page in range(1, max_pages + 1):
            params = {
                "api-key": api_key,
                "q": query,
                "from-date": start_date,
                "to-date": end_date,
                "page-size": "50",
                "page": str(page),
                "order-by": "newest",
                "show-fields": "headline,trailText,bodyText,byline,wordcount,lastModified",
                "show-tags": "keyword",
            }
            try:
                payload = request_json(GUARDIAN_ENDPOINT, params)
            except Exception as exc:
                print(f"[Guardian] query {query_index} page {page} failed: {exc}")
                break
            raw_path = run_raw_dir / "guardian" / f"q{query_index:02d}_p{page:03d}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            response = payload.get("response", {})
            results = response.get("results", [])
            print(
                f"[Guardian] query {query_index} page {page}: "
                f"{len(results)} results; unique={len(rows_by_url)}",
                flush=True,
            )
            if not results:
                break
            for item in results:
                fields = item.get("fields", {}) or {}
                url = canonicalize_url(item.get("webUrl", ""))
                title = fields.get("headline") or item.get("webTitle", "")
                body = fields.get("bodyText", "") or ""
                record = NewsRecord(
                    dataset_id=sha16(url + "|" + title),
                    source_system="guardian",
                    collection_method="guardian_open_platform",
                    source_domain=domain_from_url(url),
                    canonical_url=url,
                    title=title.strip(),
                    published_at=item.get("webPublicationDate", ""),
                    section=item.get("sectionName", ""),
                    author=fields.get("byline", ""),
                    summary=re.sub(r"<[^>]+>", " ", fields.get("trailText", "") or "").strip(),
                    body_text=re.sub(r"\s+", " ", body).strip(),
                    query=query,
                )
                rows_by_url[url] = audit_record(record)
            pages = int(response.get("pages", 1) or 1)
            if page >= pages:
                break
            time.sleep(0.2)
        time.sleep(0.4)
    print(f"[Guardian] collected {len(rows_by_url)} unique records", flush=True)
    return list(rows_by_url.values())


def collect_gdelt(start_date: str, end_date: str, max_records_per_query: int, fetch_limit: int, run_raw_dir: Path) -> list[NewsRecord]:
    metadata_by_url: dict[str, dict] = {}
    startdt = start_date.replace("-", "") + "000000"
    enddt = end_date.replace("-", "") + "235959"
    for query_index, query in enumerate(GDELT_QUERIES, start=1):
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": str(max_records_per_query),
            "sort": "datedesc",
            "startdatetime": startdt,
            "enddatetime": enddt,
            "format": "json",
        }
        payload = None
        for attempt in range(1, 3):
            try:
                payload = request_json(GDELT_ENDPOINT, params)
                break
            except HTTPError as exc:
                if exc.code == 429 and attempt == 1:
                    print(f"[GDELT] query {query_index} rate-limited; retrying after 25s", flush=True)
                    time.sleep(25)
                    continue
                print(f"[GDELT] query {query_index} failed: {exc}", flush=True)
            except Exception as exc:
                print(f"[GDELT] query {query_index} failed: {exc}", flush=True)
                break
        if payload is None:
            time.sleep(3.0)
            continue
        raw_path = run_raw_dir / "gdelt" / f"q{query_index:02d}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        for article in payload.get("articles", []):
            url = canonicalize_url(article.get("url", ""))
            title = (article.get("title") or "").strip()
            if not url or not title:
                continue
            title_url = f"{title} {url}"
            domain = (article.get("domain") or domain_from_url(url)).lower()
            if not matched_terms(title_url, AI_PATTERNS):
                continue
            if metadata_priority(title, url, domain) <= 0:
                continue
            metadata_by_url[url] = {
                "url": url,
                "title": title,
                "date": article.get("seendate", ""),
                "domain": domain,
                "query": query,
            }
        print(f"[GDELT] query {query_index}/{len(GDELT_QUERIES)} metadata pool={len(metadata_by_url)}", flush=True)
        time.sleep(5.0)

    records: list[NewsRecord] = []
    candidates = sorted(
        metadata_by_url.values(),
        key=lambda row: metadata_priority(row["title"], row["url"], row["domain"]),
        reverse=True,
    )[:fetch_limit]
    for index, item in enumerate(candidates, start=1):
        body = ""
        try:
            page_html = fetch_html(item["url"])
            body = extract_article_text(page_html)
            if page_html:
                html_path = run_raw_dir / "gdelt_html" / f"{sha16(item['url'])}.html"
                html_path.parent.mkdir(parents=True, exist_ok=True)
                html_path.write_text(page_html, encoding="utf-8", errors="ignore")
        except Exception:
            body = ""
        record = NewsRecord(
            dataset_id=sha16(item["url"] + "|" + item["title"]),
            source_system="gdelt",
            collection_method="gdelt_doc_api_plus_html_parse",
            source_domain=item["domain"],
            canonical_url=item["url"],
            title=item["title"],
            published_at=item["date"],
            body_text=body,
            query=item["query"],
        )
        records.append(audit_record(record))
        if index % 20 == 0:
            print(f"[GDELT] fetched/filtered {index}/{len(candidates)}", flush=True)
        time.sleep(0.4)
    print(f"[GDELT] collected {len(records)} fetched records from {len(metadata_by_url)} metadata URLs", flush=True)
    return records


def dedupe_records(records: Iterable[NewsRecord]) -> list[NewsRecord]:
    best: dict[str, NewsRecord] = {}
    for record in records:
        existing = best.get(record.canonical_url)
        if not existing or record.relevance_score > existing.relevance_score:
            best[record.canonical_url] = record
    best_by_title: dict[str, NewsRecord] = {}
    for record in best.values():
        title_key = normalized_title(record.title)
        existing = best_by_title.get(title_key)
        if not existing or record.relevance_score > existing.relevance_score:
            best_by_title[title_key] = record
    return sorted(best_by_title.values(), key=lambda r: (r.audit_status == "accepted", r.relevance_score, r.published_at), reverse=True)


def write_csv(path: Path, records: list[NewsRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else list(asdict(NewsRecord("", "", "", "", "", "")).keys())
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            for key in ("ai_terms", "australia_terms", "foreign_terms"):
                row[key] = "; ".join(row[key])
            writer.writerow(row)


def write_jsonl(path: Path, records: list[NewsRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_audit_report(path: Path, accepted: list[NewsRecord], rejected: list[NewsRecord], all_records: list[NewsRecord]) -> None:
    reasons = Counter()
    for record in rejected:
        for reason in record.audit_reason.split(";"):
            if reason:
                reasons[reason] += 1
    accepted_sources = Counter(record.source_system for record in accepted)
    accepted_domains = Counter(record.source_domain for record in accepted)
    foreign_warnings = [
        record for record in accepted
        if record.foreign_score > record.australia_score or "foreign_title_without_australia_context" in record.audit_reason
    ]
    lines = [
        "# Australia AI News Collection Summary",
        "",
        f"- Created UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Total unique records filtered: {len(all_records)}",
        f"- Accepted records: {len(accepted)}",
        f"- Rejected records: {len(rejected)}",
        f"- Foreign-dominance warnings among accepted: {len(foreign_warnings)}",
        "",
        "## Accepted By Source",
    ]
    for source, count in accepted_sources.most_common():
        lines.append(f"- {source}: {count}")
    lines += ["", "## Accepted Top Domains"]
    for domain, count in accepted_domains.most_common(20):
        lines.append(f"- {domain}: {count}")
    lines += ["", "## Rejection Reasons"]
    for reason, count in reasons.most_common():
        lines.append(f"- {reason}: {count}")
    lines += ["", "## Accepted Sample"]
    for record in accepted[:25]:
        lines.append(f"- [{record.source_system}] {record.published_at} | {record.source_domain} | {record.title}")
    if foreign_warnings:
        lines += ["", "## Foreign-Dominance Warnings"]
        for record in foreign_warnings[:25]:
            lines.append(
                f"- au_score={record.australia_score} foreign_score={record.foreign_score} | "
                f"{record.source_domain} | {record.title}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2026-06-28")
    parser.add_argument("--guardian-max-pages", type=int, default=8)
    parser.add_argument("--gdelt-max-records-per-query", type=int, default=80)
    parser.add_argument("--gdelt-fetch-limit", type=int, default=160)
    parser.add_argument("--skip-guardian", action="store_true")
    parser.add_argument("--skip-gdelt", action="store_true")
    args = parser.parse_args()

    run_id = now_stamp()
    run_raw_dir = RAW_DIR / run_id
    api_key = os.environ.get("GUARDIAN_API_KEY", "").strip()

    records: list[NewsRecord] = []
    if not args.skip_guardian:
        records.extend(collect_guardian(api_key, args.start_date, args.end_date, args.guardian_max_pages, run_raw_dir))
    if not args.skip_gdelt:
        records.extend(collect_gdelt(args.start_date, args.end_date, args.gdelt_max_records_per_query, args.gdelt_fetch_limit, run_raw_dir))

    records = dedupe_records(records)
    accepted = [record for record in records if record.audit_status == "accepted"]
    rejected = [record for record in records if record.audit_status != "accepted"]

    clean_csv = PROCESSED_DIR / f"au_ai_guardian_gdelt_{run_id}_clean.csv"
    clean_jsonl = PROCESSED_DIR / f"au_ai_guardian_gdelt_{run_id}_clean.jsonl"
    rejected_csv = INTERIM_DIR / f"au_ai_guardian_gdelt_{run_id}_rejected.csv"
    audit_md = REPORTS_DIR / f"collection_summary_{run_id}.md"

    write_csv(clean_csv, accepted)
    write_jsonl(clean_jsonl, accepted)
    write_csv(rejected_csv, rejected)
    write_audit_report(audit_md, accepted, rejected, records)

    latest_csv = PROCESSED_DIR / "au_ai_guardian_gdelt_latest_clean.csv"
    latest_jsonl = PROCESSED_DIR / "au_ai_guardian_gdelt_latest_clean.jsonl"
    latest_audit = REPORTS_DIR / "collection_summary_latest.md"
    shutil.copyfile(clean_csv, latest_csv)
    shutil.copyfile(clean_jsonl, latest_jsonl)
    shutil.copyfile(audit_md, latest_audit)

    print(json.dumps({
        "run_id": run_id,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "clean_csv": str(clean_csv),
        "collection_report": str(audit_md),
        "foreign_warnings": sum(1 for r in accepted if r.foreign_score > r.australia_score),
    }, indent=2))


if __name__ == "__main__":
    main()
