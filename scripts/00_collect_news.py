#!/usr/bin/env python3
"""Collect, merge and export the Australian AI news corpus.

The three collection stages preserve the original query, exact-deduplication,
quality-tier and ranking rules. Set GUARDIAN_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLLECTION_DIR = ROOT / "collection"
FINAL_CSV = ROOT / "data/processed/au_ai_guardian_gdelt_expanded_2000_latest_clean.csv"
ARTICLE_CSV = ROOT / "data/input/articles.csv"
ARTICLE_FIELDS = [
    "dataset_id", "source_system", "source_domain", "canonical_url", "title",
    "published_at", "summary", "body_text", "quality_tier",
]


def run_step(name: str, command: list[str], dry_run: bool) -> None:
    print(f"\n[{name}]\n{shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def export_articles(source: Path, destination: Path, target: int) -> int:
    """Export only the fields used by the analysis stages."""
    csv.field_size_limit(100_000_000)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(ARTICLE_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Merged corpus is missing columns: {sorted(missing)}")
        rows = list(reader)
    if len(rows) < target:
        raise ValueError(
            f"Only {len(rows)} records were collected; target is {target}. "
            "Increase query coverage or supply historical GDELT candidate CSVs."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ARTICLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2026-06-28")
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--guardian-max-pages", type=int, default=8)
    parser.add_argument("--gdelt-max-records-per-query", type=int, default=80)
    parser.add_argument("--gdelt-fetch-limit", type=int, default=160)
    parser.add_argument("--pages-per-query", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--gdelt-input", action="append", type=Path, default=[],
                        help="Optional historical GDELT CSV cache; may be repeated")
    parser.add_argument("--skip-strict", action="store_true",
                        help="Reuse an existing strict core corpus")
    parser.add_argument("--skip-expanded", action="store_true",
                        help="Reuse existing expanded candidate pools")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without API calls or file writes")
    args = parser.parse_args()
    if args.target <= 0:
        parser.error("--target must be greater than zero")
    try:
        if date.fromisoformat(args.start_date) > date.fromisoformat(args.end_date):
            parser.error("--start-date must be on or before --end-date")
    except ValueError:
        parser.error("Dates must use YYYY-MM-DD")
    args.gdelt_input = [path.expanduser().resolve() for path in args.gdelt_input]
    if not args.dry_run:
        for path in args.gdelt_input:
            if not path.is_file():
                parser.error(f"GDELT candidate CSV does not exist: {path}")
    return args


def main() -> None:
    args = parse_args()
    needs_guardian = not args.skip_strict or not args.skip_expanded
    if not args.dry_run and needs_guardian and not os.environ.get("GUARDIAN_API_KEY", "").strip():
        raise SystemExit("Set the GUARDIAN_API_KEY environment variable before collection.")

    python = sys.executable
    if not args.skip_strict:
        run_step("1. Strict Guardian and GDELT collection", [
            python, str(COLLECTION_DIR / "collect_australia_ai_news.py"),
            "--start-date", args.start_date, "--end-date", args.end_date,
            "--guardian-max-pages", str(args.guardian_max_pages),
            "--gdelt-max-records-per-query", str(args.gdelt_max_records_per_query),
            "--gdelt-fetch-limit", str(args.gdelt_fetch_limit),
        ], args.dry_run)

    if not args.skip_expanded:
        run_step("2. Expanded Guardian candidate collection", [
            python, str(COLLECTION_DIR / "collect_expanded_australia_ai_news.py"),
            "--target", str(args.target),
            "--start-date", args.start_date, "--end-date", args.end_date,
            "--pages-per-query", str(args.pages_per_query),
            "--page-size", str(args.page_size),
        ], args.dry_run)

    merge = [python, str(COLLECTION_DIR / "build_merged_expanded_dataset.py"),
             "--target", str(args.target)]
    for path in args.gdelt_input:
        merge.extend(["--gdelt-input", str(path)])
    run_step("3. Merge, deduplicate and select the corpus", merge, args.dry_run)

    if args.dry_run:
        print(f"\n4. Export analysis columns to {ARTICLE_CSV}")
        print("Dry run complete; no API requests or file writes were made.")
        return

    if not FINAL_CSV.is_file():
        raise SystemExit(f"Merged corpus was not created: {FINAL_CSV}")
    try:
        count = export_articles(FINAL_CSV, ARTICLE_CSV, args.target)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"\nAnalysis input: {ARTICLE_CSV}\nRecords: {count}")


if __name__ == "__main__":
    main()
