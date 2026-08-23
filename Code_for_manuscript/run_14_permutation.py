"""Permutation null analysis for the four PMC text representations.

The analysis is deliberately standalone: it consumes the already generated
``outputs/tables/overlap_diagnostics.csv`` from ``run_03_overlap.py`` and does
not rebuild policy or text distances. City labels on the rule-distance side
are permuted while each pair's text distance remains fixed.

The output reports the observed Spearman rho, the empirical 95% permutation
null band, the observed discordant-pair count and its null band, and the
absolute-rho resolution supported by this design. Defaults are B=1,000 and
seed=42, matching Code 10.4's reproducibility contract.
"""
from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config


B = int(os.environ.get("JHBE_P1_PERMUTATIONS", "1000"))
SEED = int(os.environ.get("JHBE_P1_PERMUTATION_SEED", str(config.SEED)))
REPS = ("T", "I", "F", "BERT")
SRC = config.TABLE_DIR / "overlap_diagnostics.csv"
OUT = config.TABLE_DIR / "p1_permutation_null.csv"
REQUIRED_COLUMNS = {
    "city_a",
    "city_b",
    "rule_distance_gower",
    *(f"text_dist_{rep}" for rep in REPS),
}


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranked = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranked[order[k]] = average_rank
        i = j + 1
    return ranked


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("Spearman correlation requires two equal vectors with at least 3 values")
    a, b = ranks(x), ranks(y)
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((p - mean_a) * (q - mean_b) for p, q in zip(a, b))
    denominator = (
        sum((p - mean_a) ** 2 for p in a) * sum((q - mean_b) ** 2 for q in b)
    ) ** 0.5
    if denominator == 0:
        raise ValueError("Spearman correlation is undefined for a constant vector")
    return numerator / denominator


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty vector")
    ordered = sorted(values)
    h = (len(ordered) - 1) * q
    lo = int(h)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def load_pair_data() -> tuple[list[dict[str, str]], list[str], dict[tuple[int, int], float], list[tuple[int, int]]]:
    if not SRC.exists():
        raise FileNotFoundError(
            f"Missing {SRC}. Run run_03_overlap.py first so the fixed pair distances exist."
        )

    with SRC.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{SRC} contains no pair rows")

    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"{SRC} is missing required columns: {sorted(missing)}")

    cities = sorted({row["city_a"] for row in rows} | {row["city_b"] for row in rows})
    if len(cities) != config.N_CITIES:
        raise ValueError(f"Expected {config.N_CITIES} cities, found {len(cities)}")
    expected_pairs = config.N_CITIES * (config.N_CITIES - 1) // 2
    if len(rows) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} city pairs, found {len(rows)}")

    city_index = {city: i for i, city in enumerate(cities)}
    rule: dict[tuple[int, int], float] = {}
    pairs: list[tuple[int, int]] = []
    for row in rows:
        if row["city_a"] == row["city_b"]:
            raise ValueError("Self-pairs are not valid for a city-label permutation")
        a, b = city_index[row["city_a"]], city_index[row["city_b"]]
        key = (min(a, b), max(a, b))
        if key in rule:
            raise ValueError(f"Duplicate city pair in {SRC}: {row['city_a']} / {row['city_b']}")
        rule[key] = float(row["rule_distance_gower"])
        pairs.append(key)

    if len(rule) != expected_pairs:
        raise ValueError(f"Rule-distance matrix is incomplete: {len(rule)} unique pairs")
    return rows, cities, rule, pairs


def permutation_maps(n_cities: int) -> list[list[int]]:
    rng = random.Random(SEED)
    maps: list[list[int]] = []
    for _ in range(B):
        mapping = list(range(n_cities))
        rng.shuffle(mapping)
        maps.append(mapping)
    return maps


def endpoint(sorted_values: list[float], fraction: float) -> float:
    """Return the same empirical order-statistic convention as the draft."""
    index = int(fraction * B) - 1
    return sorted_values[index]


def main() -> list[dict[str, object]]:
    if B < 40:
        raise ValueError("JHBE_P1_PERMUTATIONS must be at least 40 for 2.5% tail order statistics")
    config.ensure_directories()
    rows, cities, rule, pairs = load_pair_data()
    maps = permutation_maps(len(cities))
    observed_rule = [rule[pair] for pair in pairs]

    output_rows: list[dict[str, object]] = []
    for rep in REPS:
        text = [float(row[f"text_dist_{rep}"]) for row in rows]
        q3_rule = quantile(observed_rule, 0.75)
        q1_text = quantile(text, 0.25)
        observed_rho = spearman(observed_rule, text)
        observed_count = sum(
            rule_distance >= q3_rule and text_distance <= q1_text
            for rule_distance, text_distance in zip(observed_rule, text)
        )

        null_rhos: list[float] = []
        null_counts: list[int] = []
        for mapping in maps:
            permuted_rule = [
                rule[(min(mapping[a], mapping[b]), max(mapping[a], mapping[b]))]
                for a, b in pairs
            ]
            null_rhos.append(spearman(permuted_rule, text))
            q3_permuted = quantile(permuted_rule, 0.75)
            null_counts.append(
                sum(
                    rule_distance >= q3_permuted and text_distance <= q1_text
                    for rule_distance, text_distance in zip(permuted_rule, text)
                )
            )

        null_rhos.sort()
        null_counts.sort()
        low_rho, high_rho = endpoint(null_rhos, 0.025), endpoint(null_rhos, 0.975)
        low_count, high_count = endpoint(null_counts, 0.025), endpoint(null_counts, 0.975)
        output_rows.append(
            {
                "representation": f"PMC_{rep}",
                "n_pairs": len(pairs),
                "B": B,
                "seed": SEED,
                "rule_cut_q3": round(q3_rule, 6),
                "text_cut_q1": round(q1_text, 6),
                "observed_rho": round(observed_rho, 6),
                "null_rho_p2_5": round(low_rho, 6),
                "null_rho_p97_5": round(high_rho, 6),
                "observed_discordant": observed_count,
                "null_count_p2_5": low_count,
                "null_count_p97_5": high_count,
                "min_detectable_abs_rho": round(max(abs(low_rho), abs(high_rho)), 6),
                "inside_band": low_rho <= observed_rho <= high_rho,
                "scope": "descriptive P1 measurement-validity evidence; not causal",
            }
        )

    with OUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    for row in output_rows:
        print(
            row["representation"],
            row["observed_rho"],
            [row["null_rho_p2_5"], row["null_rho_p97_5"]],
            row["observed_discordant"],
            [row["null_count_p2_5"], row["null_count_p97_5"]],
            row["min_detectable_abs_rho"],
        )
    return output_rows


if __name__ == "__main__":
    main()
