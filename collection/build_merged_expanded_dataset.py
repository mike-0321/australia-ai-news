#!/usr/bin/env python3
"""Merge prior Guardian/GDELT candidate pools into a 2,000-record corpus.

This is a controlled relaxation step. It preserves the strict Guardian+GDELT
records as the core tier, then adds Guardian-expanded records in lower tiers.
The weakest tier is labelled background so downstream analysis can include or
exclude it explicitly.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().with_name("collect_australia_ai_news.py")
spec = importlib.util.spec_from_file_location("strict_collector", SRC)
strict_collector = importlib.util.module_from_spec(spec)
sys.modules["strict_collector"] = strict_collector
assert spec.loader is not None
spec.loader.exec_module(strict_collector)

PROCESSED_DIR = ROOT / "data" / "processed"
INTERIM_DIR = ROOT / "data" / "interim"
REPORTS_DIR = ROOT / "results" / "collection"
csv.field_size_limit(100_000_000)


DEFAULT_INPUTS = sorted(PROCESSED_DIR.glob("au_ai_guardian_expanded_[0-9]*_clean.csv")) + sorted(
    INTERIM_DIR.glob("au_ai_guardian_expanded_[0-9]*_rejected.csv")
)



def parse_list(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(";") if item.strip()]


def count_terms(text: str, patterns: dict[str, str], flags: int = re.I) -> int:
    return sum(len(re.findall(pattern, text or "", flags)) for pattern in patterns.values())


def normalized_title(title: str) -> str:
    return strict_collector.normalized_title(title or "")


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    by_url: dict[str, dict[str, str]] = {}
    by_title: dict[str, dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if "canonical_url" not in row and "url" in row:
                    row = {
                        "dataset_id": strict_collector.sha16(row.get("url", "") + "|" + row.get("title", "")),
                        "source_system": "gdelt",
                        "collection_method": row.get("collection_method") or "gdelt_doc_api_local_cache",
                        "source_domain": row.get("source_domain", ""),
                        "canonical_url": strict_collector.canonicalize_url(row.get("url", "")),
                        "title": row.get("title", ""),
                        "published_at": row.get("date", ""),
                        "section": "Australia source" if row.get("country", "").lower() == "australia" else "",
                        "author": "",
                        "summary": row.get("source_guess", ""),
                        "body_text": "",
                        "query": row.get("gdelt_query", ""),
                        "ai_terms": row.get("matched_terms", ""),
                        "australia_terms": "australia" if row.get("country", "").lower() == "australia" else "",
                        "foreign_terms": "",
                        "australia_score": "0",
                        "foreign_score": "0",
                        "relevance_score": row.get("relevance_score", "0"),
                        "audit_status": "pending",
                        "audit_reason": "",
                        "quality_tier": "",
                    }
                url = row.get("canonical_url", "").strip()
                if not url:
                    continue
                current = by_url.get(url)
                row_score = (
                    row.get("audit_status") == "accepted",
                    len(row.get("body_text", "")),
                    row.get("published_at", ""),
                )
                current_score = (
                    current.get("audit_status") == "accepted",
                    len(current.get("body_text", "")),
                    current.get("published_at", ""),
                ) if current else None
                if current is None or row_score > current_score:
                    by_url[url] = row
    for row in by_url.values():
        title_key = normalized_title(row.get("title", ""))
        current = by_title.get(title_key)
        row_score = (row.get("audit_status") == "accepted", len(row.get("body_text", "")))
        current_score = (current.get("audit_status") == "accepted", len(current.get("body_text", ""))) if current else None
        if current is None or row_score > current_score:
            by_title[title_key] = row
    return list(by_title.values())


def classify(row: dict[str, str]) -> dict[str, object]:
    title = row.get("title", "")
    section = row.get("section", "")
    url = row.get("canonical_url", "")
    summary = row.get("summary", "")
    body = row.get("body_text", "")
    full = f"{title} {url} {section} {summary} {body[:12000]}"
    lead = f"{title} {summary} {body[:2000]} {section} {url}"
    title_url = f"{title} {url} {section}"

    ai_terms = strict_collector.matched_terms(full, strict_collector.AI_PATTERNS)
    australia_terms = strict_collector.matched_terms(full, strict_collector.AU_PATTERNS)
    foreign_terms = strict_collector.matched_terms(full, strict_collector.FOREIGN_PATTERNS, flags=0)
    title_ai = strict_collector.matched_terms(title, strict_collector.AI_PATTERNS)
    lead_ai = strict_collector.matched_terms(lead, strict_collector.AI_PATTERNS)
    title_au = strict_collector.matched_terms(title_url, strict_collector.AU_PATTERNS)
    lead_au = strict_collector.matched_terms(lead, strict_collector.AU_PATTERNS)
    foreign_title = strict_collector.matched_terms(title_url, strict_collector.FOREIGN_PATTERNS, flags=0)
    guardian_au_section = section.lower() == "australia news" or "australia-news" in url

    ai_hits = count_terms(full, strict_collector.AI_PATTERNS)
    australia_hits = count_terms(full, strict_collector.AU_PATTERNS)
    foreign_hits = count_terms(full, strict_collector.FOREIGN_PATTERNS, flags=0)
    is_low_signal = strict_collector.low_signal_title_or_url(title, url)
    is_core = row.get("quality_tier") == "core" or row.get("audit_reason") == "strict_core_preserved"

    tier = "rejected"
    reasons: list[str] = []
    if is_low_signal:
        reasons.append("low_signal_live_blog_or_roundup")
    if not ai_terms:
        reasons.append("no_ai_terms")
    if not australia_terms:
        reasons.append("no_australia_terms")
    if foreign_hits > max(45, australia_hits * 6):
        reasons.append("foreign_strongly_dominant")
    if foreign_title and not title_au and australia_hits < 2:
        reasons.append("foreign_title_without_australia_context")

    if not reasons:
        if is_core:
            tier = "core"
        elif row.get("audit_status") == "accepted" and row.get("quality_tier") in {"high", "expanded", "contextual"}:
            tier = row.get("quality_tier", "expanded")
        elif (title_ai or lead_ai) and (title_au or lead_au or australia_hits >= 1):
            if row.get("source_system") == "guardian" and not guardian_au_section and australia_hits < 2:
                reasons.append("weak_guardian_background_australia_context")
            else:
                tier = "background"
        elif ai_hits >= 1 and australia_hits >= 2:
            tier = "background"
        else:
            reasons.append("insufficient_ai_australia_signal")

    score = (
        18 * min(ai_hits, 8)
        + 12 * min(australia_hits, 8)
        + (25 if title_ai else 0)
        + (12 if lead_ai else 0)
        + (25 if title_au else 0)
        + (12 if lead_au else 0)
        + (15 if section.lower() == "australia news" else 0)
        - 5 * min(foreign_hits, 12)
        - (24 if foreign_title and not title_au else 0)
    )

    return {
        "quality_tier": tier,
        "audit_status": "accepted" if tier != "rejected" else "rejected",
        "audit_reason": "merged_relaxed_filter" if tier != "rejected" else ";".join(reasons),
        "ai_terms": ai_terms,
        "australia_terms": australia_terms,
        "foreign_terms": foreign_terms,
        "ai_hit_count": ai_hits,
        "australia_hit_count": australia_hits,
        "foreign_hit_count": foreign_hits,
        "selection_score": score,
    }


def enrich_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    enriched: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for row in rows:
        meta = classify(row)
        out: dict[str, object] = dict(row)
        out.update(meta)
        if meta["audit_status"] == "accepted":
            enriched.append(out)
        else:
            rejected.append(out)
    tier_rank = {"core": 5, "high": 4, "expanded": 3, "contextual": 2, "background": 1}
    enriched.sort(
        key=lambda r: (
            tier_rank.get(str(r["quality_tier"]), 0),
            int(r["selection_score"]),
            str(r.get("published_at", "")),
        ),
        reverse=True,
    )
    return enriched, rejected


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    base_fields = [
        "dataset_id", "source_system", "collection_method", "source_domain",
        "canonical_url", "title", "published_at", "section", "author",
        "summary", "body_text", "query", "ai_terms", "australia_terms",
        "foreign_terms", "australia_score", "foreign_score",
        "relevance_score", "audit_status", "audit_reason", "quality_tier",
        "ai_hit_count", "australia_hit_count", "foreign_hit_count",
        "selection_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = dict(row)
            for key in ("ai_terms", "australia_terms", "foreign_terms"):
                value = cleaned.get(key, "")
                if isinstance(value, list):
                    cleaned[key] = "; ".join(value)
                elif not isinstance(value, str):
                    cleaned[key] = str(value)
            writer.writerow(cleaned)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, accepted: list[dict[str, object]], rejected: list[dict[str, object]], target: int) -> None:
    tiers = Counter(str(row["quality_tier"]) for row in accepted)
    sources = Counter(str(row.get("source_system", "")) for row in accepted)
    domains = Counter(str(row.get("source_domain", "")) for row in accepted)
    reasons = Counter()
    for row in rejected:
        for reason in str(row.get("audit_reason", "")).split(";"):
            if reason:
                reasons[reason] += 1
    foreign_warnings = [
        row for row in accepted
        if int(row.get("foreign_hit_count", 0) or 0) > int(row.get("australia_hit_count", 0) or 0) * 3
    ]
    lines = [
        "# Merged 2000-Record Australia AI News Summary",
        "",
        f"- Created UTC: {datetime.now(timezone.utc).isoformat()}",
        f"- Target records: {target}",
        f"- Accepted records written: {len(accepted)}",
        f"- Rejected candidates retained in collection logs: {len(rejected)}",
        f"- Strong foreign-dominance warnings in accepted: {len(foreign_warnings)}",
        "",
        "## Accepted By Tier",
    ]
    for tier, count in tiers.most_common():
        lines.append(f"- {tier}: {count}")
    lines += ["", "## Accepted By Source"]
    for source, count in sources.most_common():
        lines.append(f"- {source}: {count}")
    lines += ["", "## Top Domains"]
    for domain, count in domains.most_common(25):
        lines.append(f"- {domain}: {count}")
    lines += ["", "## Rejection Reasons"]
    for reason, count in reasons.most_common(20):
        lines.append(f"- {reason}: {count}")
    lines += ["", "## Accepted Sample"]
    for row in accepted[:45]:
        lines.append(
            f"- [{row['quality_tier']}] {row.get('published_at', '')} | "
            f"{row.get('source_domain', '')} | {row.get('title', '')}"
        )
    if foreign_warnings:
        lines += ["", "## Foreign Warning Sample"]
        for row in foreign_warnings[:30]:
            lines.append(
                f"- au_hits={row.get('australia_hit_count')} foreign_hits={row.get('foreign_hit_count')} | "
                f"{row.get('title', '')}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--input", action="append", default=[],
                        help="Candidate CSV; repeat to replace the default expanded pools")
    parser.add_argument("--gdelt-input", action="append", type=Path, default=[],
                        help="Optional historical GDELT CSV cache; may be repeated")
    args = parser.parse_args()

    if args.target <= 0:
        parser.error("--target must be greater than zero")
    input_paths = ([Path(item) for item in args.input] if args.input else DEFAULT_INPUTS) + args.gdelt_input
    for path in input_paths:
        if not path.is_file():
            parser.error(f"Input CSV does not exist: {path}")
    if not input_paths:
        parser.error("No candidate CSVs found; run collection or supply --input/--gdelt-input")
    rows = load_rows(input_paths)
    accepted, rejected = enrich_rows(rows)
    selected = accepted[: args.target]
    run_id = strict_collector.now_stamp()

    out_csv = PROCESSED_DIR / f"au_ai_guardian_gdelt_expanded_2000_{run_id}_clean.csv"
    out_jsonl = PROCESSED_DIR / f"au_ai_guardian_gdelt_expanded_2000_{run_id}_clean.jsonl"
    rejected_csv = INTERIM_DIR / f"au_ai_guardian_gdelt_expanded_2000_{run_id}_rejected.csv"
    report = REPORTS_DIR / f"expanded_2000_collection_summary_{run_id}.md"

    write_csv(out_csv, selected)
    write_jsonl(out_jsonl, selected)
    write_csv(rejected_csv, rejected)
    write_report(report, selected, rejected, args.target)

    shutil.copyfile(out_csv, PROCESSED_DIR / "au_ai_guardian_gdelt_expanded_2000_latest_clean.csv")
    shutil.copyfile(out_jsonl, PROCESSED_DIR / "au_ai_guardian_gdelt_expanded_2000_latest_clean.jsonl")
    shutil.copyfile(report, REPORTS_DIR / "expanded_2000_collection_summary_latest.md")

    print(json.dumps({
        "candidate_records": len(rows),
        "accepted_before_cut": len(accepted),
        "written": len(selected),
        "target_met": len(selected) >= args.target,
        "output_csv": str(out_csv),
        "collection_report": str(report),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
