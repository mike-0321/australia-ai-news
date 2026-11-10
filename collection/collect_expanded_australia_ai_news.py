#!/usr/bin/env python3
"""Build a larger Australia-focused AI news corpus.

This script keeps the strict dataset as a core tier, then expands with broader
Guardian records. It relaxes the strict rule that AI/Australia must both appear
in the title or lead, but still requires both in the article text and removes
roundups/live blogs/foreign-dominant records.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().with_name("collect_australia_ai_news.py")
spec = importlib.util.spec_from_file_location("strict_collector", SRC)
strict_collector = importlib.util.module_from_spec(spec)
sys.modules["strict_collector"] = strict_collector
assert spec.loader is not None
spec.loader.exec_module(strict_collector)


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
INTERIM_DIR = ROOT / "data" / "interim"
REPORTS_DIR = ROOT / "results" / "collection"

GUARDIAN_QUERIES_EXPANDED = [
    'AI AND (Australia OR Australian)',
    '"artificial intelligence" AND (Australia OR Australian)',
    '"generative AI" AND (Australia OR Australian)',
    'ChatGPT AND (Australia OR Australian)',
    'deepfake AND (Australia OR Australian)',
    'algorithm AND (Australia OR Australian)',
    'automation AND (Australia OR Australian)',
    '"data centre" AND AI AND Australia',
    'machine learning AND (Australia OR Australian)',
]

GUARDIAN_SECTION_QUERIES_EXPANDED = [
    ("AI", "australia-news"),
    ('"artificial intelligence"', "australia-news"),
    ('"generative AI"', "australia-news"),
    ("ChatGPT", "australia-news"),
    ("deepfake", "australia-news"),
    ("algorithm", "australia-news"),
    ("automation", "australia-news"),
    ('"machine learning"', "australia-news"),
    ("chatbot", "australia-news"),
    ('"data centre"', "australia-news"),
    ("datacentre", "australia-news"),
]


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def load_core(path: Path) -> list[strict_collector.NewsRecord]:
    rows: list[strict_collector.NewsRecord] = []
    if not path.exists():
        return rows
    for row in csv.DictReader(path.open(encoding="utf-8-sig")):
        rec = strict_collector.NewsRecord(
            dataset_id=row["dataset_id"],
            source_system=row["source_system"],
            collection_method=row["collection_method"],
            source_domain=row["source_domain"],
            canonical_url=row["canonical_url"],
            title=row["title"],
            published_at=row.get("published_at", ""),
            section=row.get("section", ""),
            author=row.get("author", ""),
            summary=row.get("summary", ""),
            body_text=row.get("body_text", ""),
            query=row.get("query", ""),
            ai_terms=parse_list(row.get("ai_terms", "")),
            australia_terms=parse_list(row.get("australia_terms", "")),
            foreign_terms=parse_list(row.get("foreign_terms", "")),
            australia_score=int(row.get("australia_score", 0) or 0),
            foreign_score=int(row.get("foreign_score", 0) or 0),
            relevance_score=int(row.get("relevance_score", 0) or 0),
            audit_status="accepted",
            audit_reason="strict_core_preserved",
        )
        rec.quality_tier = "core"  # dynamic field handled during write
        rows.append(rec)
    return rows


def guardian_record_from_item(item: dict, query: str) -> strict_collector.NewsRecord:
    fields = item.get("fields", {}) or {}
    url = strict_collector.canonicalize_url(item.get("webUrl", ""))
    title = fields.get("headline") or item.get("webTitle", "")
    body = fields.get("bodyText", "") or ""
    summary = re.sub(r"<[^>]+>", " ", fields.get("trailText", "") or "").strip()
    return strict_collector.NewsRecord(
        dataset_id=strict_collector.sha16(url + "|" + title),
        source_system="guardian",
        collection_method="guardian_open_platform_expanded",
        source_domain=strict_collector.domain_from_url(url),
        canonical_url=url,
        title=title.strip(),
        published_at=item.get("webPublicationDate", ""),
        section=item.get("sectionName", ""),
        author=fields.get("byline", ""),
        summary=re.sub(r"\s+", " ", summary).strip(),
        body_text=re.sub(r"\s+", " ", body).strip(),
        query=query,
    )


def expanded_audit(record: strict_collector.NewsRecord) -> strict_collector.NewsRecord:
    record = strict_collector.audit_record(record)
    if record.audit_status == "accepted":
        record.quality_tier = "high"
        return record

    title_url = f"{record.title} {record.canonical_url} {record.section}"
    full = f"{title_url} {record.summary} {record.body_text[:12000]}"
    ai_terms = strict_collector.matched_terms(full, strict_collector.AI_PATTERNS)
    au_terms = strict_collector.matched_terms(full, strict_collector.AU_PATTERNS)
    foreign_terms = strict_collector.matched_terms(full, strict_collector.FOREIGN_PATTERNS, flags=0)
    title_ai = strict_collector.matched_terms(record.title, strict_collector.AI_PATTERNS)
    title_au = strict_collector.matched_terms(record.title, strict_collector.AU_PATTERNS)
    lead = f"{record.title} {record.summary} {record.body_text[:2000]} {record.section} {record.canonical_url}"
    lead_ai = strict_collector.matched_terms(lead, strict_collector.AI_PATTERNS)
    lead_au = strict_collector.matched_terms(lead, strict_collector.AU_PATTERNS)

    au_hits = sum(len(re.findall(pattern, full, re.I)) for pattern in strict_collector.AU_PATTERNS.values())
    ai_hits = sum(len(re.findall(pattern, full, re.I)) for pattern in strict_collector.AI_PATTERNS.values())
    foreign_hits = sum(len(re.findall(pattern, full)) for pattern in strict_collector.FOREIGN_PATTERNS.values())
    australia_score = min(40, au_hits + (6 if title_au else 0) + (4 if lead_au else 0))
    foreign_score = min(40, foreign_hits + (5 if strict_collector.matched_terms(title_url, strict_collector.FOREIGN_PATTERNS, flags=0) else 0))
    relevance_score = min(100, 10 * len(ai_terms) + 6 * len(au_terms) + (5 if title_ai else 0) + (3 if lead_ai else 0) - int(foreign_score * 0.45))
    has_ai_focus = bool(lead_ai or title_ai or ai_hits >= 2)
    has_au_context = bool(
        lead_au
        or record.section.lower() == "australia news"
        or "australia-news" in record.canonical_url
        or au_hits >= 4
    )
    has_foreign_title_without_au = bool(
        strict_collector.matched_terms(title_url, strict_collector.FOREIGN_PATTERNS, flags=0)
        and not strict_collector.matched_terms(title_url, strict_collector.AU_PATTERNS)
    )

    reasons: list[str] = []
    status = "accepted"
    if strict_collector.low_signal_title_or_url(record.title, record.canonical_url):
        status = "rejected"
        reasons.append("low_signal_live_blog_or_roundup")
    if not ai_terms:
        status = "rejected"
        reasons.append("no_ai_terms_in_text")
    if not au_terms:
        status = "rejected"
        reasons.append("no_hard_australia_terms_in_text")
    if not has_ai_focus:
        status = "rejected"
        reasons.append("weak_ai_focus")
    if not has_au_context:
        status = "rejected"
        reasons.append("weak_australia_context")
    if has_foreign_title_without_au and australia_score < 10:
        status = "rejected"
        reasons.append("foreign_title_without_enough_australia_context")
    if foreign_score > max(24, australia_score * 2.4):
        status = "rejected"
        reasons.append("foreign_strongly_dominant")
    if record.body_text and len(record.body_text) < 250:
        status = "rejected"
        reasons.append("body_too_short")

    record.ai_terms = ai_terms
    record.australia_terms = au_terms
    record.foreign_terms = foreign_terms
    record.australia_score = australia_score
    record.foreign_score = foreign_score
    record.relevance_score = relevance_score
    record.audit_status = status
    record.audit_reason = ";".join(reasons) if reasons else "expanded_text_filter"
    if status == "accepted" and not (lead_ai or title_ai):
        record.quality_tier = "contextual"
    else:
        record.quality_tier = "expanded" if status == "accepted" else "rejected"
    return record


def request_guardian_with_retries(params: dict[str, str], attempts: int = 3) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return strict_collector.request_json(strict_collector.GUARDIAN_ENDPOINT, params, timeout=60)
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 * attempt)
    assert last_error is not None
    raise last_error


def collect_guardian_expanded(api_key: str, start_date: str, end_date: str, pages_per_query: int, page_size: int, run_raw_dir: Path) -> list[strict_collector.NewsRecord]:
    records: list[strict_collector.NewsRecord] = []
    seen_urls: set[str] = set()
    query_specs = [(query, None) for query in GUARDIAN_QUERIES_EXPANDED] + GUARDIAN_SECTION_QUERIES_EXPANDED
    for query_index, (query, section) in enumerate(query_specs, start=1):
        query_label = f"{query} section:{section}" if section else query
        print(f"[Guardian expanded] query {query_index}/{len(query_specs)} {query_label}", flush=True)
        for page in range(1, pages_per_query + 1):
            params = {
                "api-key": api_key,
                "q": query,
                "from-date": start_date,
                "to-date": end_date,
                "page-size": str(page_size),
                "page": str(page),
                "order-by": "newest",
                "show-fields": "headline,trailText,bodyText,byline,wordcount,lastModified",
                "show-tags": "keyword",
            }
            if section:
                params["section"] = section
            try:
                payload = request_guardian_with_retries(params)
            except Exception as exc:
                print(f"[Guardian expanded] query {query_index} page {page} failed: {exc}", flush=True)
                break
            raw_path = run_raw_dir / "guardian_expanded" / f"q{query_index:02d}_p{page:03d}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            response = payload.get("response", {})
            results = response.get("results", [])
            print(f"[Guardian expanded] page {page}: {len(results)} results; collected={len(records)}", flush=True)
            if not results:
                break
            for item in results:
                rec = guardian_record_from_item(item, query_label)
                if not rec.canonical_url or rec.canonical_url in seen_urls:
                    continue
                seen_urls.add(rec.canonical_url)
                records.append(expanded_audit(rec))
            if page >= int(response.get("pages", 1) or 1):
                break
            time.sleep(0.25)
        time.sleep(0.5)
    return records


def dedupe_with_tiers(records: list[strict_collector.NewsRecord]) -> list[strict_collector.NewsRecord]:
    tier_rank = {"core": 4, "high": 3, "expanded": 2, "contextual": 1, "rejected": 0}
    best_by_url: dict[str, strict_collector.NewsRecord] = {}
    for record in records:
        existing = best_by_url.get(record.canonical_url)
        if (
            existing is None
            or tier_rank.get(getattr(record, "quality_tier", "expanded"), 0) > tier_rank.get(getattr(existing, "quality_tier", "expanded"), 0)
            or record.relevance_score > existing.relevance_score
        ):
            best_by_url[record.canonical_url] = record

    best_by_title: dict[str, strict_collector.NewsRecord] = {}
    for record in best_by_url.values():
        key = strict_collector.normalized_title(record.title)
        existing = best_by_title.get(key)
        if (
            existing is None
            or tier_rank.get(getattr(record, "quality_tier", "expanded"), 0) > tier_rank.get(getattr(existing, "quality_tier", "expanded"), 0)
            or record.relevance_score > existing.relevance_score
        ):
            best_by_title[key] = record
    return sorted(
        best_by_title.values(),
        key=lambda r: (tier_rank.get(getattr(r, "quality_tier", "expanded"), 0), r.relevance_score, r.published_at),
        reverse=True,
    )


def write_records_csv(path: Path, records: list[strict_collector.NewsRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = list(asdict(records[0]).keys()) if records else list(asdict(strict_collector.NewsRecord("", "", "", "", "", "")).keys())
    fieldnames = base_fields + ["quality_tier"]
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            for key in ("ai_terms", "australia_terms", "foreign_terms"):
                row[key] = "; ".join(row[key])
            row["quality_tier"] = getattr(record, "quality_tier", "expanded")
            writer.writerow(row)


def write_records_jsonl(path: Path, records: list[strict_collector.NewsRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = asdict(record)
            row["quality_tier"] = getattr(record, "quality_tier", "expanded")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, accepted: list[strict_collector.NewsRecord], rejected: list[strict_collector.NewsRecord], target: int) -> None:
    tiers = Counter(getattr(r, "quality_tier", "expanded") for r in accepted)
    domains = Counter(r.source_domain for r in accepted)
    sources = Counter(r.source_system for r in accepted)
    reasons = Counter()
    for record in rejected:
        for reason in record.audit_reason.split(";"):
            if reason:
                reasons[reason] += 1
    warnings = [r for r in accepted if r.foreign_score > r.australia_score]
    lines = [
        "# Expanded Australia AI News Collection Summary",
        "",
        f"- Created UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Target records: {target}",
        f"- Accepted records: {len(accepted)}",
        f"- Rejected records: {len(rejected)}",
        f"- Foreign-score > Australia-score warnings: {len(warnings)}",
        "",
        "## Accepted By Tier",
    ]
    for tier, count in tiers.most_common():
        lines.append(f"- {tier}: {count}")
    lines += ["", "## Accepted By Source"]
    for source, count in sources.most_common():
        lines.append(f"- {source}: {count}")
    lines += ["", "## Top Domains"]
    for domain, count in domains.most_common(30):
        lines.append(f"- {domain}: {count}")
    lines += ["", "## Rejection Reasons"]
    for reason, count in reasons.most_common(20):
        lines.append(f"- {reason}: {count}")
    lines += ["", "## Accepted Sample"]
    for record in accepted[:40]:
        lines.append(f"- [{getattr(record, 'quality_tier', 'expanded')}] {record.published_at} | {record.source_domain} | {record.title}")
    if warnings:
        lines += ["", "## Foreign Warnings Sample"]
        for record in warnings[:30]:
            lines.append(f"- au={record.australia_score} foreign={record.foreign_score} | {record.title}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2026-06-28")
    parser.add_argument("--pages-per-query", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--core-csv", default=str(PROCESSED_DIR / "au_ai_guardian_gdelt_latest_clean.csv"))
    args = parser.parse_args()

    api_key = os.environ.get("GUARDIAN_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("GUARDIAN_API_KEY is required")

    run_id = strict_collector.now_stamp()
    raw_dir = RAW_DIR / run_id
    core = load_core(Path(args.core_csv))
    print(f"[core] loaded {len(core)} strict records", flush=True)
    expanded = collect_guardian_expanded(api_key, args.start_date, args.end_date, args.pages_per_query, args.page_size, raw_dir)
    all_records = dedupe_with_tiers(core + expanded)
    accepted = [r for r in all_records if getattr(r, "quality_tier", "expanded") != "rejected" and r.audit_status == "accepted"]
    rejected = [r for r in all_records if r.audit_status != "accepted" or getattr(r, "quality_tier", "expanded") == "rejected"]

    # Keep the best target records; always core records sort first through tier rank.
    accepted = accepted[: args.target]
    out_csv = PROCESSED_DIR / f"au_ai_guardian_expanded_{run_id}_clean.csv"
    out_jsonl = PROCESSED_DIR / f"au_ai_guardian_expanded_{run_id}_clean.jsonl"
    reject_csv = INTERIM_DIR / f"au_ai_guardian_expanded_{run_id}_rejected.csv"
    report = REPORTS_DIR / f"expanded_collection_summary_{run_id}.md"
    write_records_csv(out_csv, accepted)
    write_records_jsonl(out_jsonl, accepted)
    write_records_csv(reject_csv, rejected)
    write_report(report, accepted, rejected, args.target)

    shutil.copyfile(out_csv, PROCESSED_DIR / "au_ai_guardian_expanded_latest_clean.csv")
    shutil.copyfile(out_jsonl, PROCESSED_DIR / "au_ai_guardian_expanded_latest_clean.jsonl")
    shutil.copyfile(report, REPORTS_DIR / "expanded_collection_summary_latest.md")

    print(json.dumps({
        "run_id": run_id,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "output_csv": str(out_csv),
        "collection_report": str(report),
        "target_met": len(accepted) >= args.target,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
