"""Turn ``results.csv`` into the Markdown tables of ``REPORT.md``.

The runner measures; this script only aggregates (stdlib only, no pandas or
matplotlib: the exercise asks for curves as tables). The two reading
paragraphs of the report are written by hand on top of these numbers.

Usage::

    uv run python -m evals.stress.analyze evals/stress/results.csv > /tmp/tables.md
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

Row = dict[str, str]

CHECKPOINTS = (1, 2, 3, 5, 6, 7, 8, 10, 12, 15, 20)
TOKEN_BUCKETS = ((0, 2999), (3000, 4999), (5000, 6999), (7000, 9999), (10000, 10**9))


def load(path: Path) -> list[Row]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ok(rows: Iterable[Row]) -> list[Row]:
    return [r for r in rows if not r["error"]]


def num(row: Row, key: str) -> float:
    return float(row[key])


def pct(values: list[float], p: float) -> float:
    """Nearest-rank percentile (no interpolation): honest with small n."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(p / 100 * len(ordered) + 0.5) - 1))]


def rate(rows: list[Row], key: str) -> str:
    values = [r[key] for r in rows if r[key] != ""]
    if not values:
        return "–"
    return f"{100 * sum(v == 'True' for v in values) / len(values):.0f}%"


def mean(rows: list[Row], key: str, fmt: str = ".2f") -> str:
    values = [num(r, key) for r in rows if r[key] != ""]
    return format(statistics.mean(values), fmt) if values else "–"


def table(header: list[str], body: list[list[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return "\n".join(lines)


def group(rows: list[Row], key: Callable[[Row], object]) -> dict[object, list[Row]]:
    out: dict[object, list[Row]] = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def session_cost(rows: list[Row]) -> float:
    """Average total bill (estimation + extractor) of one session in ``rows``."""
    sessions = group(rows, lambda r: r["session_id"])
    return statistics.mean(sum(num(r, "turn_cost_usd") for r in s) for s in sessions.values())


def summary(rows: list[Row]) -> str:
    body = []
    for (scenario, kb), g in sorted(
        group(rows, lambda r: (r["scenario"], int(r["attachment_size_kb"]))).items()
    ):
        lat = [num(r, "latency_ms") for r in g]
        turn = [num(r, "turn_latency_ms") for r in g]
        body.append(
            [
                scenario,
                kb,
                len(g),
                f"{pct(lat, 50):.0f}",
                f"{pct(lat, 95):.0f}",
                f"{pct(turn, 95):.0f}",
                f"{session_cost(g):.4f}",
                f"{sum(num(r, 'turn_cost_usd') for r in g):.4f}",
                f"{100 * sum(r['cache_hit_kind'] == 'exact' for r in g) / len(g):.0f}%",
                "n/a",
                rate(g, "latency_budget_passed"),
                rate(g, "memory_drift_passed"),
                mean(g, "facts_recall"),
                mean(g, "facts_recall_metadata"),
            ]
        )
    return table(
        [
            "Scenario",
            "KB",
            "n",
            "P50 LLM ms",
            "P95 LLM ms",
            "P95 turn ms",
            "$ / session",
            "$ total",
            "Exact hit",
            "Semantic hit",
            "Latency ≤ budget",
            "Drift pass (turn-1 name)",
            "Fact recall (window+memory)",
            "Fact recall (memory only)",
        ],
        body,
    )


def latency_vs_tokens(rows: list[Row]) -> str:
    body = []
    for low, high in TOKEN_BUCKETS:
        g = [r for r in rows if low <= num(r, "tokens_in") <= high]
        if not g:
            continue
        lat = [num(r, "latency_ms") for r in g]
        out = [num(r, "tokens_out") for r in g]
        per_token = [num(r, "latency_ms") / max(1, num(r, "tokens_out")) for r in g]
        label = f"{low}+" if high >= 10**9 else f"{low}–{high}"
        body.append(
            [
                label,
                len(g),
                f"{pct(lat, 50):.0f}",
                f"{pct(lat, 95):.0f}",
                f"{statistics.median(out):.0f}",
                f"{statistics.median(per_token):.1f}",
            ]
        )
    return table(
        ["tokens_in", "n", "Median ms", "P95 ms", "Median tokens_out", "Median ms / token_out"],
        body,
    )


def correlation(rows: list[Row], a: str, b: str) -> float:
    xs, ys = [num(r, a) for r in rows], [num(r, b) for r in rows]
    return statistics.correlation(xs, ys)


def cumulative_cost(rows: list[Row], size_kb: int = 0) -> str:
    rows = [r for r in rows if int(r["attachment_size_kb"]) == size_kb]
    by_scenario = group(rows, lambda r: r["scenario"])
    scenarios = sorted(by_scenario)
    body = []
    for n in CHECKPOINTS:
        line: list[object] = [n]
        for s in scenarios:
            sessions = group(by_scenario[s], lambda r: r["session_id"]).values()
            totals = [
                sum(num(r, "turn_cost_usd") for r in sess if int(r["turn_index"]) <= n)
                for sess in sessions
                if any(int(r["turn_index"]) >= n for r in sess)
            ]
            turn_n = [num(r, "turn_cost_usd") for r in by_scenario[s] if int(r["turn_index"]) == n]
            line.append(f"{statistics.mean(totals):.4f}" if totals else "–")
            line.append(f"{statistics.mean(turn_n):.5f}" if turn_n else "–")
        body.append(line)
    header: list[str] = ["Turn"]
    for s in scenarios:
        header += [f"{s} cumulative $", f"{s} $ of turn"]
    return table(header, body)


def tokens_by_turn(rows: list[Row]) -> str:
    sizes = sorted({int(r["attachment_size_kb"]) for r in rows})
    body = []
    for n in CHECKPOINTS:
        line: list[object] = [n]
        for kb in sizes:
            g = [
                num(r, "tokens_in")
                for r in rows
                if int(r["turn_index"]) == n and int(r["attachment_size_kb"]) == kb
            ]
            line.append(f"{statistics.mean(g):.0f}" if g else "–")
        body.append(line)
    return table(["Turn"] + [f"tokens_in @ {kb} KB" for kb in sizes], body)


def recall_vs_turn(rows: list[Row]) -> str:
    body = []
    for n in CHECKPOINTS[1:]:
        g = [r for r in rows if int(r["turn_index"]) == n]
        if not g:
            continue
        body.append(
            [
                n,
                len(g),
                mean(g, "facts_tracked", ".0f"),
                rate(g, "memory_drift_passed"),
                mean(g, "facts_recall"),
                mean(g, "facts_recall_history"),
                mean(g, "facts_recall_metadata"),
            ]
        )
    return table(
        [
            "Turn N",
            "n",
            "Facts stated before N",
            "Turn-1 name present",
            "Recall window+memory",
            "Recall window only",
            "Recall memory only",
        ],
        body,
    )


def recall_by_scenario(rows: list[Row]) -> str:
    scenarios = sorted({r["scenario"] for r in rows})
    body = []
    for n in (6, 7, 10, 15, 20):
        line: list[object] = [n]
        for s in scenarios:
            g = [r for r in rows if r["scenario"] == s and int(r["turn_index"]) == n]
            line.append(mean(g, "facts_recall"))
        body.append(line)
    return table(["Turn N"] + [f"{s} recall" for s in scenarios], body)


def attachments(rows: list[Row]) -> str:
    body = []
    for kb, g in sorted(group(rows, lambda r: int(r["attachment_size_kb"])).items()):
        if kb == 0:
            continue
        first = [r for r in g if r["turn_index"] == "1"]

        def at(n: int, rows_kb: list[Row] = g) -> list[Row]:
            return [r for r in rows_kb if int(r["turn_index"]) == n]

        body.append(
            [
                kb,
                mean(first, "attachments_total_chars", ".0f"),
                mean(first, "tokens_in", ".0f"),
                f"{pct([num(r, 'latency_ms') for r in first], 50):.0f}",
                rate(first, "attach_head_in_answer"),
                rate(first, "attach_tail_in_answer"),
                rate(at(6), "attach_head_in_session"),
                rate(at(7), "attach_head_in_session"),
                rate(at(20), "attach_head_in_session"),
                rate(at(20), "attach_head_in_answer"),
            ]
        )
    return table(
        [
            "KB",
            "Extracted chars (turn 1)",
            "tokens_in turn 1",
            "P50 ms turn 1",
            "Head marker in answer t1",
            "Tail marker in answer t1",
            "Head in session t6",
            "Head in session t7",
            "Head in session t20",
            "Head in answer t20",
        ],
        body,
    )


def supersession(rows: list[Row]) -> str:
    body = []
    for scenario in ("pivot", "contradiction"):
        for n in (5, 6, 8, 9, 10, 12, 14, 15, 20):
            g = [
                r
                for r in rows
                if r["scenario"] == scenario
                and int(r["turn_index"]) == n
                and r["superseded_in_metadata"] != ""
            ]
            if g:
                body.append(
                    [
                        scenario,
                        n,
                        len(g),
                        rate(g, "superseded_in_metadata"),
                        rate(g, "updated_in_session"),
                        rate(g, "updated_in_answer"),
                    ]
                )
    return table(
        [
            "Scenario",
            "Turn",
            "n",
            "Old fact still in memory",
            "New fact in session",
            "New fact in answer",
        ],
        body,
    )


def main(argv: list[str] | None = None) -> int:
    path = Path((argv or sys.argv[1:] or ["evals/stress/results.csv"])[0])
    all_rows = load(path)
    rows = ok(all_rows)
    size0 = [r for r in rows if r["attachment_size_kb"] == "0"]
    errors = len(all_rows) - len(rows)
    print(f"<!-- generated from {path}: {len(all_rows)} rows, {errors} errors -->\n")
    print("## Summary\n\n" + summary(rows) + "\n")
    print("## Latency vs tokens_in (all rows)\n\n" + latency_vs_tokens(rows) + "\n")
    print(
        f"corr(latency, tokens_in) = {correlation(rows, 'latency_ms', 'tokens_in'):.2f}; "
        f"corr(latency, tokens_out) = {correlation(rows, 'latency_ms', 'tokens_out'):.2f}\n"
    )
    print("## tokens_in vs turn\n\n" + tokens_by_turn(rows) + "\n")
    print("## Cumulative cost vs turn (0 KB)\n\n" + cumulative_cost(rows, 0) + "\n")
    print("## Cumulative cost vs turn (100 KB)\n\n" + cumulative_cost(rows, 100) + "\n")
    print("## Recall vs N (0 KB, pooled)\n\n" + recall_vs_turn(size0) + "\n")
    print("## Recall vs N by scenario (0 KB)\n\n" + recall_by_scenario(size0) + "\n")
    print("## Attachments\n\n" + attachments(rows) + "\n")
    print("## Supersession\n\n" + supersession(rows) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
