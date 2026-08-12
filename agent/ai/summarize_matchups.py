"""Aggregate per-chunk matchup rows (from ``run_chunked``) into a summary.

    python -m agent.ai.summarize_matchups output/matchups.csv

Each CSV row is one chunk: mode, kind, opponent, days, iters, seed_start,
seed_end, win_rate0, mean_reward0, median_reward0, std_reward0, mean_reward1,
mean_steps, latency_ms, transitions. Aggregation weights by the number of games
in each chunk; median/std/latency are the (weighted) means of the chunk values
since raw per-game rewards are not persisted per chunk.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _Acc:
    games: int = 0
    wins: int = 0
    mean0: list[tuple[float, int]] = field(default_factory=list)
    median0: list[tuple[float, int]] = field(default_factory=list)
    std0: list[tuple[float, int]] = field(default_factory=list)
    mean1: list[tuple[float, int]] = field(default_factory=list)
    steps: list[tuple[float, int]] = field(default_factory=list)
    latency: list[tuple[float, int]] = field(default_factory=list)
    transitions: int = 0


def _weighted_mean(pairs: list[tuple[float, int]]) -> float:
    total = sum(g for _, g in pairs)
    if total <= 0:
        return 0.0
    return sum(v * g for v, g in pairs) / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate chunked matchup results")
    parser.add_argument("csv_path")
    args = parser.parse_args()

    acc: dict[tuple[str, str, str], _Acc] = defaultdict(_Acc)
    with open(args.csv_path, newline="") as handle:
        for row in csv.reader(handle):
            if not row or len(row) < 15 or not row[0].strip():
                continue
            mode, kind, opponent = row[0], row[1], row[2]
            chunk_games = int(row[6]) - int(row[5])
            a = acc[(mode, kind, opponent)]
            a.games += chunk_games
            a.wins += round(float(row[7]) * chunk_games)
            a.mean0.append((float(row[8]), chunk_games))
            a.median0.append((float(row[9]), chunk_games))
            a.std0.append((float(row[10]), chunk_games))
            a.mean1.append((float(row[11]), chunk_games))
            a.steps.append((float(row[12]), chunk_games))
            a.latency.append((float(row[13]), chunk_games))
            a.transitions += int(row[14])

    print(
        f"{'matchup':<26}{'games':>6}{'win%':>6}{'mean_r0':>9}{'med_r0':>8}{'sd_r0':>7}"
        f"{'lat_ms':>8}{'trans':>11}"
    )
    for (mode, kind, opponent), a in sorted(acc.items()):
        label = f"m{mode}{'_' + kind if kind != 'new' else ''}_vs_{opponent}"
        print(
            f"{label:<26}{a.games:>6}{a.wins / a.games * 100:>6.0f}"
            f"{_weighted_mean(a.mean0):>9.1f}{_weighted_mean(a.median0):>8.1f}"
            f"{_weighted_mean(a.std0):>7.1f}{_weighted_mean(a.latency):>8.1f}"
            f"{a.transitions:>11}"
        )


if __name__ == "__main__":
    main()
