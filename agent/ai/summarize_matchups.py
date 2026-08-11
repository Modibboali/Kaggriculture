"""Aggregate per-chunk matchup rows (from ``run_chunked``) into a summary.

    python -m agent.ai.summarize_matchups output/matchups.csv

Each CSV row is one chunk: kind, opponent, days, iters, seed_start, seed_end,
win_rate0, mean_reward0, median_reward0, std_reward0, mean_reward1, mean_steps.
Aggregation weights by the number of games in each chunk; the reported median
and std are the (weighted) means of the chunk medians/stds since the raw
per-game rewards are not persisted per chunk.
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
    mean0: list[tuple[float, int]] = field(default_factory=list)  # (value, games)
    median0: list[tuple[float, int]] = field(default_factory=list)
    std0: list[tuple[float, int]] = field(default_factory=list)
    mean1: list[tuple[float, int]] = field(default_factory=list)
    steps: list[tuple[float, int]] = field(default_factory=list)


def _weighted_mean(pairs: list[tuple[float, int]]) -> float:
    total = sum(g for _, g in pairs)
    if total <= 0:
        return 0.0
    return sum(v * g for v, g in pairs) / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate chunked matchup results")
    parser.add_argument("csv_path")
    args = parser.parse_args()

    acc: dict[tuple[str, str], _Acc] = defaultdict(_Acc)
    with open(args.csv_path, newline="") as handle:
        for row in csv.reader(handle):
            if not row or len(row) < 12 or not row[0].strip():
                continue
            kind, opponent = row[0], row[1]
            chunk_games = int(row[5]) - int(row[4])
            a = acc[(kind, opponent)]
            a.games += chunk_games
            a.wins += round(float(row[6]) * chunk_games)
            a.mean0.append((float(row[7]), chunk_games))
            a.median0.append((float(row[8]), chunk_games))
            a.std0.append((float(row[9]), chunk_games))
            a.mean1.append((float(row[10]), chunk_games))
            a.steps.append((float(row[11]), chunk_games))

    print(f"{'matchup':<30}{'games':>6}{'win%':>6}{'mean_r0':>9}{'med_r0':>9}{'sd_r0':>8}{'mean_r1':>9}")
    for (kind, opponent), a in sorted(acc.items()):
        print(
            f"{kind + '_vs_' + opponent:<30}{a.games:>6}{a.wins / a.games * 100:>6.0f}"
            f"{_weighted_mean(a.mean0):>9.1f}{_weighted_mean(a.median0):>9.1f}"
            f"{_weighted_mean(a.std0):>8.1f}{_weighted_mean(a.mean1):>9.1f}"
        )


if __name__ == "__main__":
    main()
