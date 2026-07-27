#!/usr/bin/env python3
"""Consolidate archived final_comparison jobs into clean MHD/Euler CSV files.

The archived timing rows contain ISO-8601 nanosecond timestamps whose decimal
separator is a comma.  Those commas were not CSV-quoted, so this script repairs
the two timestamp fields before writing standards-compliant output.
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "timing" / "final_comparison"
OUTPUTS = {
    "mhd": INPUT_ROOT / "mhd_timing_31_all.csv",
    "euler": INPUT_ROOT / "euler_timing_32_all.csv",
}

ORIGINAL_FIELDS = [
    "backend",
    "case",
    "solver",
    "n",
    "repeat",
    "repeats_requested",
    "build_variant",
    "threads",
    "ranks",
    "nx",
    "ny",
    "total_cells",
    "steps",
    "app_elapsed_s",
    "steps_per_s",
    "Mcell_updates_s",
    "wall_seconds",
    "user_seconds",
    "sys_seconds",
    "cpu_percent",
    "max_rss_kb",
    "major_page_faults",
    "minor_page_faults",
    "voluntary_context_switches",
    "involuntary_context_switches",
    "start_utc",
    "end_utc",
    "exit_status",
    "hostname",
    "git_branch",
    "git_commit",
]
PROVENANCE_FIELDS = [
    "experiment",
    "source_job_task",
    "source_job_id",
    "array_task_id",
    "source_file",
    "is_canonical",
]

JOB_TASK_RE = re.compile(r"^(?P<job>\d+)_(?P<task>\d+)$")
TIMESTAMP_START_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T")


def repair_timestamps(tokens: List[str], path: Path, line_number: int) -> List[str]:
    """Return the two timestamp fields from their unquoted CSV tokens."""
    starts = [i for i, token in enumerate(tokens) if TIMESTAMP_START_RE.match(token)]
    if len(starts) != 2 or starts[0] != 0:
        raise RuntimeError(
            f"Cannot identify two timestamps in {path}:{line_number}: {tokens}"
        )
    second = starts[1]
    start_utc = ".".join(tokens[:second])
    end_utc = ".".join(tokens[second:])
    return [start_utc, end_utc]


def parse_timing_row(raw_line: str, path: Path, line_number: int) -> Dict[str, str]:
    tokens = next(csv.reader([raw_line]))
    if len(tokens) < len(ORIGINAL_FIELDS):
        raise RuntimeError(
            f"Short timing row in {path}:{line_number}: "
            f"{len(tokens)} fields, expected at least {len(ORIGINAL_FIELDS)}"
        )

    # The first 25 and last 4 fields never contain commas.  Everything between
    # them is the pair of malformed timestamps.
    prefix = tokens[:25]
    timestamps = repair_timestamps(tokens[25:-4], path, line_number)
    suffix = tokens[-4:]
    repaired = prefix + timestamps + suffix
    if len(repaired) != len(ORIGINAL_FIELDS):
        raise RuntimeError(f"Repair produced the wrong field count in {path}:{line_number}")
    return dict(zip(ORIGINAL_FIELDS, repaired))


def read_family(experiment: str, prefix: str) -> List[Dict[str, str]]:
    rows = []  # type: List[Dict[str, str]]
    for job_dir in sorted(INPUT_ROOT.iterdir()):
        if not job_dir.is_dir() or not job_dir.name.startswith(prefix):
            continue
        match = JOB_TASK_RE.match(job_dir.name)
        if not match:
            continue
        for path in sorted(job_dir.rglob("*.csv")):
            with path.open(encoding="utf-8", newline="") as handle:
                lines = handle.read().splitlines()
            if not lines:
                continue
            header = next(csv.reader([lines[0]]))
            if header != ORIGINAL_FIELDS:
                raise RuntimeError(f"Unexpected header in {path}: {header}")
            for line_number, raw_line in enumerate(lines[1:], start=2):
                if not raw_line.strip():
                    continue
                row = parse_timing_row(raw_line, path, line_number)
                row.update(
                    {
                        "experiment": experiment,
                        "source_job_task": job_dir.name,
                        "source_job_id": match.group("job"),
                        "array_task_id": match.group("task"),
                        "source_file": str(path.relative_to(INPUT_ROOT)).replace("\\", "/"),
                    }
                )
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No {prefix}-prefixed timing rows found under {INPUT_ROOT}")
    return rows


def mark_canonical(rows: List[Dict[str, str]]) -> None:
    """Mark the newest job for each backend/solver as the plotting dataset."""
    newest = {}  # type: Dict[Tuple[str, str], int]
    for row in rows:
        key = (row["backend"], row["solver"])
        newest[key] = max(newest.get(key, -1), int(row["source_job_id"]))
    for row in rows:
        key = (row["backend"], row["solver"])
        row["is_canonical"] = (
            "1" if int(row["source_job_id"]) == newest[key] else "0"
        )


def write_family(experiment: str, rows: List[Dict[str, str]]) -> None:
    mark_canonical(rows)
    backend_order = {"cpu_openmp": 0, "mpi": 1, "gpu": 2}
    rows.sort(
        key=lambda row: (
            backend_order.get(row["backend"], 99),
            row["case"],
            row["solver"],
            int(row["n"]),
            int(row["source_job_id"]),
            int(row["repeat"]),
        )
    )
    output = OUTPUTS[experiment]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PROVENANCE_FIELDS + ORIGINAL_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    successful = sum(row["exit_status"] == "0" for row in rows)
    canonical = sum(row["is_canonical"] == "1" for row in rows)
    print(
        f"{experiment}: wrote {len(rows)} rows "
        f"({successful} successful, {canonical} canonical) -> {output}"
    )


def main() -> None:
    write_family("mhd", read_family("mhd", "31"))
    write_family("euler", read_family("euler", "32"))


if __name__ == "__main__":
    main()
