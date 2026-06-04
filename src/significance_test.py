"""
significance_test.py — Bootstrap and cross-seed significance testing for subchar-mt.

Two modes:
  1. Cross-seed test (uses finetuned_results_ALL.csv — available now)
     Wilcoxon signed-rank on per-seed scores; tests each representation vs. baseline.

  2. Bootstrap test (uses per-prediction CSVs produced by finetune_experiment.py v2)
     Paired bootstrap resampling (Koehn 2004) at the sentence level; more powerful.

Usage:
    python src/significance_test.py                           # cross-seed mode
    python src/significance_test.py --mode bootstrap \\
        --preds_dir results/predictions/                      # bootstrap mode
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ── Cross-seed significance test ──────────────────────────────────────────────

def cross_seed_test(
    all_csv: str = "finetuned_results_ALL.csv",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For each (model, train_size) group, compare every representation vs. baseline
    using a Wilcoxon signed-rank test on per-seed metric values.

    With only 3 seeds the test has very low power; use this as a sanity check
    and note the limitation in the paper. Expand to 5+ seeds before final submission.
    """
    df = pd.read_csv(all_csv)
    ft = df[df["condition"] == "finetuned"].copy()
    ft["comet"] = pd.to_numeric(ft["comet"], errors="coerce")
    ft["seed"] = pd.to_numeric(ft["seed"], errors="coerce")

    records = []
    representations = [r for r in ft["representation"].unique() if r != "baseline"]

    for model in ft["model"].unique():
        for train_size in sorted(ft["train_size"].unique()):
            base = ft[
                (ft["model"] == model)
                & (ft["train_size"] == train_size)
                & (ft["representation"] == "baseline")
            ].sort_values("seed")

            if len(base) < 2:
                continue

            for rep in representations:
                rep_rows = ft[
                    (ft["model"] == model)
                    & (ft["train_size"] == train_size)
                    & (ft["representation"] == rep)
                ].sort_values("seed")

                if len(rep_rows) < 2:
                    continue

                # Align on seed
                merged = base[["seed", "bleu", "chrf", "comet"]].merge(
                    rep_rows[["seed", "bleu", "chrf", "comet"]],
                    on="seed",
                    suffixes=("_base", "_rep"),
                )
                if len(merged) < 2:
                    continue

                row: dict = {
                    "model": model,
                    "representation": rep,
                    "train_size": train_size,
                    "n_seeds": len(merged),
                }

                for metric in ["bleu", "chrf", "comet"]:
                    base_vals = merged[f"{metric}_base"].dropna()
                    rep_vals = merged[f"{metric}_rep"].dropna()

                    if len(base_vals) < 2 or base_vals.std() == 0 and rep_vals.std() == 0:
                        # Identical values — note the zero-variance anomaly
                        row[f"{metric}_delta"] = float(rep_vals.mean() - base_vals.mean())
                        row[f"{metric}_p"] = float("nan")
                        row[f"{metric}_sig"] = "zero_var"
                        row[f"{metric}_cohens_d"] = float("nan")
                    else:
                        delta = float(rep_vals.mean() - base_vals.mean())
                        try:
                            _, p = stats.wilcoxon(rep_vals, base_vals)
                        except Exception:
                            _, p = stats.ttest_rel(rep_vals, base_vals)
                        pooled_sd = np.sqrt(
                            (rep_vals.std() ** 2 + base_vals.std() ** 2) / 2
                        )
                        d = delta / pooled_sd if pooled_sd > 0 else float("nan")
                        row[f"{metric}_delta"] = delta
                        row[f"{metric}_p"] = float(p)
                        row[f"{metric}_sig"] = "yes" if p < alpha else "no"
                        row[f"{metric}_cohens_d"] = float(d)

                records.append(row)

    results = pd.DataFrame(records)
    return results


def print_cross_seed_summary(results: pd.DataFrame) -> None:
    print("=" * 100)
    print("SIGNIFICANCE TESTS (cross-seed Wilcoxon/t-test, each rep vs. baseline)")
    print("NOTE: 3 seeds has very low statistical power — treat p-values as indicative only.")
    print("=" * 100)

    for model in results["model"].unique():
        print(f"\n[{model}]")
        sub = results[results["model"] == model]
        for rep in sub["representation"].unique():
            rep_rows = sub[sub["representation"] == rep].sort_values("train_size")
            print(f"  {rep}:")
            for _, r in rep_rows.iterrows():
                bleu_str  = f"ΔBLEU={r['bleu_delta']:+.2f}  (p={r['bleu_p']:.3f}  {r['bleu_sig']}  d={r['bleu_cohens_d']:.2f})"
                chrf_str  = f"ΔchrF={r['chrf_delta']:+.2f}  (p={r['chrf_p']:.3f}  {r['chrf_sig']})"
                comet_str = f"ΔCOMET={r['comet_delta']:+.4f}  (p={r['comet_p']:.3f}  {r['comet_sig']})"
                print(f"    n={int(r['train_size']):4d}: {bleu_str}   {chrf_str}   {comet_str}")


def zero_variance_report(all_csv: str = "finetuned_results_ALL.csv") -> None:
    """
    Report conditions where all seeds produced identical scores.
    This is suspicious and likely indicates a seeding or data-sampling bug.
    """
    df = pd.read_csv(all_csv)
    ft = df[df["condition"] == "finetuned"].copy()
    ft["comet"] = pd.to_numeric(ft["comet"], errors="coerce")

    summary = (
        ft.groupby(["model", "representation", "train_size"])
        .agg(bleu_sd=("bleu", "std"), chrf_sd=("chrf", "std"), n=("seed", "count"))
        .reset_index()
    )
    zero_var = summary[(summary["bleu_sd"] == 0) & (summary["chrf_sd"] == 0) & (summary["n"] > 1)]

    print("\n" + "=" * 80)
    print("ZERO-VARIANCE CONDITIONS (all seeds identical — investigate before reporting)")
    print("=" * 80)
    if zero_var.empty:
        print("  None found.")
    else:
        print(f"  {len(zero_var)} conditions with bleu_sd=0 and chrf_sd=0 across >1 seed:")
        print(zero_var.to_string(index=False))
    print()


# ── Bootstrap significance test (requires saved per-prediction files) ─────────

def bootstrap_test_fast(
    base_sent_scores: np.ndarray,
    rep_sent_scores: np.ndarray,
    n_iterations: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> dict:
    """
    Fast paired bootstrap over precomputed sentence-level scores (numpy only).

    Bootstraps over mean sentence BLEU / mean COMET rather than recomputing
    corpus BLEU from scratch — 100× faster than the sacrebleu-based version
    with nearly identical results for large test sets (n≥100).

    base_sent_scores / rep_sent_scores: 1-D arrays of per-sentence scores.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n = len(base_sent_scores)
    base_mean = float(base_sent_scores.mean())
    rep_mean  = float(rep_sent_scores.mean())
    observed_delta = rep_mean - base_mean

    # Paired difference array — resample this
    diffs = rep_sent_scores - base_sent_scores

    # Bootstrap: resample differences, test if resampled mean > observed delta
    # (equivalent to Koehn's formulation for the mean metric)
    samples = rng.choice(diffs, size=(n_iterations, n), replace=True)
    resampled_deltas = samples.mean(axis=1)

    better_count = int((resampled_deltas > observed_delta).sum())
    p_value = better_count / n_iterations
    ci_lo, ci_hi = float(np.percentile(resampled_deltas, 2.5)), \
                   float(np.percentile(resampled_deltas, 97.5))

    return {
        "base_mean": base_mean,
        "rep_mean":  rep_mean,
        "delta":     observed_delta,
        "p_value":   p_value,
        "significant": p_value < alpha,
        "ci_95_lo":  ci_lo,
        "ci_95_hi":  ci_hi,
    }


def bootstrap_test(
    base_preds: list,
    rep_preds: list,
    references: list,
    n_iterations: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """
    Paired bootstrap resampling (Koehn 2004) using sacrebleu corpus_bleu.
    Prefer bootstrap_test_fast() when sentence-level scores are available.
    """
    from sacrebleu import corpus_bleu

    n = len(references)
    base_score = corpus_bleu(base_preds, [[r] for r in references]).score
    rep_score  = corpus_bleu(rep_preds,  [[r] for r in references]).score
    observed_delta = rep_score - base_score

    rng = np.random.RandomState(42)
    better_count = 0
    deltas = []

    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        b = corpus_bleu([base_preds[i] for i in idx],
                        [[references[i]] for i in idx]).score
        r = corpus_bleu([rep_preds[i]  for i in idx],
                        [[references[i]] for i in idx]).score
        delta = r - b
        deltas.append(delta)
        if delta > observed_delta:
            better_count += 1

    p_value = better_count / n_iterations
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "base_bleu": base_score, "rep_bleu": rep_score,
        "delta": observed_delta, "p_value": p_value,
        "significant": p_value < alpha,
        "ci_95_lo": ci_lo, "ci_95_hi": ci_hi,
    }


def _parse_pred_filename(stem: str):
    """Parse '{model}_{rep}_{train_size}_s{seed}' from a prediction file stem."""
    parts = stem.split("_")
    seed_part = [p for p in parts if p.startswith("s") and p[1:].isdigit()]
    if not seed_part:
        return None
    idx_seed   = parts.index(seed_part[0])
    seed       = int(seed_part[0][1:])
    train_size = int(parts[idx_seed - 1])
    rep        = parts[idx_seed - 2]
    model      = "_".join(parts[:idx_seed - 2])
    return model, rep, train_size, seed


def run_bootstrap_from_dir(preds_dir: str, alpha: float = 0.05,
                           n_iterations: int = 1000,
                           out_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Load prediction JSON files and run fast bootstrap tests.
    Uses precomputed sent_bleus / comet_scores — finishes in seconds, not hours.
    """
    import glob
    from collections import defaultdict

    pred_files = glob.glob(os.path.join(preds_dir, "*_preds.json"))
    if not pred_files:
        print(f"No prediction files found in {preds_dir}")
        return pd.DataFrame()

    # Load all files and group by (model, train_size, seed) → {rep: data}
    groups: dict = defaultdict(dict)
    for fpath in sorted(pred_files):
        stem = Path(fpath).stem.replace("_preds", "")
        parsed = _parse_pred_filename(stem)
        if parsed is None:
            continue
        model, rep, train_size, seed = parsed
        with open(fpath) as f:
            data = json.load(f)
        groups[(model, train_size, seed)][rep] = data

    print("=" * 100)
    print(f"BOOTSTRAP SIGNIFICANCE TESTS (fast numpy, {n_iterations} iterations, each rep vs. baseline)")
    print("=" * 100)

    records = []
    rng = np.random.RandomState(42)

    for (model, train_size, seed), reps in sorted(groups.items()):
        if "baseline" not in reps:
            continue
        base_data = reps["baseline"]

        for rep, rep_data in sorted(reps.items()):
            if rep == "baseline":
                continue

            row: dict = {"model": model, "representation": rep,
                         "train_size": train_size, "seed": seed}

            for metric, key in [("bleu", "sent_bleus"), ("comet", "comet_scores")]:
                base_scores = base_data.get(key)
                rep_scores  = rep_data.get(key)

                if not base_scores or not rep_scores:
                    row[f"{metric}_delta"] = row[f"{metric}_p"] = row[f"{metric}_sig"] = None
                    row[f"{metric}_ci_lo"] = row[f"{metric}_ci_hi"] = None
                    continue

                result = bootstrap_test_fast(
                    np.array(base_scores), np.array(rep_scores),
                    n_iterations=n_iterations, alpha=alpha, rng=rng,
                )
                row[f"{metric}_base"]  = round(result["base_mean"], 4)
                row[f"{metric}_rep"]   = round(result["rep_mean"],  4)
                row[f"{metric}_delta"] = round(result["delta"],     4)
                row[f"{metric}_p"]     = round(result["p_value"],   4)
                row[f"{metric}_sig"]   = result["significant"]
                row[f"{metric}_ci_lo"] = round(result["ci_95_lo"],  4)
                row[f"{metric}_ci_hi"] = round(result["ci_95_hi"],  4)

            sig_bleu  = "***" if row.get("bleu_sig")  else "   "
            sig_comet = "***" if row.get("comet_sig") else "   "
            comet_str = (f"  ΔCOMET={row['comet_delta']:+.4f}{sig_comet}"
                         f"  p={row['comet_p']:.3f}"
                         if row.get("comet_delta") is not None else "")
            print(
                f"  {model:9s} | {rep:14s} | n={train_size:4d} | s={seed} | "
                f"ΔBLEU={row['bleu_delta']:+.2f}{sig_bleu}  p={row['bleu_p']:.3f}"
                f"  CI=[{row['bleu_ci_lo']:+.2f},{row['bleu_ci_hi']:+.2f}]"
                f"{comet_str}"
            )
            records.append(row)

    df = pd.DataFrame(records)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\n✓ Saved to {out_csv}")
    return df


# ── Metric-divergence analysis ────────────────────────────────────────────────

def metric_divergence_report(all_csv: str = "finetuned_results_ALL.csv") -> None:
    """
    Identify conditions where BLEU and COMET disagree on ranking vs. baseline.
    This is the core analytical finding: morphemes win BLEU but lose COMET.
    """
    df = pd.read_csv(all_csv)
    ft = df[df["condition"] == "finetuned"].copy()
    ft["comet"] = pd.to_numeric(ft["comet"], errors="coerce")

    summary = (
        ft.groupby(["model", "representation", "train_size"])
        .agg(
            bleu_mean=("bleu", "mean"),
            chrf_mean=("chrf", "mean"),
            comet_mean=("comet", "mean"),
        )
        .reset_index()
    )

    baseline = summary[summary["representation"] == "baseline"][
        ["model", "train_size", "bleu_mean", "chrf_mean", "comet_mean"]
    ].rename(columns={
        "bleu_mean": "base_bleu", "chrf_mean": "base_chrf", "comet_mean": "base_comet"
    })

    merged = summary[summary["representation"] != "baseline"].merge(
        baseline, on=["model", "train_size"]
    )
    merged["bleu_delta"]  = merged["bleu_mean"]  - merged["base_bleu"]
    merged["chrf_delta"]  = merged["chrf_mean"]  - merged["base_chrf"]
    merged["comet_delta"] = merged["comet_mean"] - merged["base_comet"]

    # Disagreement: BLEU improves but COMET degrades (or vice versa)
    merged["bleu_comet_disagree"] = (
        ((merged["bleu_delta"] > 0) & (merged["comet_delta"] < 0)) |
        ((merged["bleu_delta"] < 0) & (merged["comet_delta"] > 0))
    )

    print("=" * 100)
    print("METRIC DIVERGENCE: conditions where BLEU and COMET disagree on direction vs. baseline")
    print("=" * 100)
    disagree = merged[merged["bleu_comet_disagree"]].sort_values(
        "bleu_delta", ascending=False
    )
    if disagree.empty:
        print("  No disagreements found.")
    else:
        cols = ["model", "representation", "train_size",
                "bleu_delta", "chrf_delta", "comet_delta"]
        print(disagree[cols].round(3).to_string(index=False))

    print("\nAll deltas (rep vs. baseline):")
    cols = ["model", "representation", "train_size",
            "bleu_delta", "chrf_delta", "comet_delta"]
    print(merged[cols].round(3).sort_values(
        ["model", "representation", "train_size"]
    ).to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cross_seed", "bootstrap", "divergence", "all"],
                        default="all")
    parser.add_argument("--all_csv", default="finetuned_results_ALL.csv")
    parser.add_argument("--preds_dir", default="results/predictions/")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default=None, help="Save results CSV to this path")
    args = parser.parse_args()

    if args.mode in ("cross_seed", "all"):
        zero_variance_report(args.all_csv)
        results = cross_seed_test(args.all_csv, alpha=args.alpha)
        print_cross_seed_summary(results)
        if args.out:
            results.to_csv(args.out, index=False)
            print(f"\n✓ Saved to {args.out}")

    if args.mode in ("divergence", "all"):
        metric_divergence_report(args.all_csv)

    if args.mode in ("bootstrap", "all"):
        if os.path.isdir(args.preds_dir):
            run_bootstrap_from_dir(args.preds_dir, alpha=args.alpha, out_csv=args.out)
        else:
            print(f"\nBootstrap mode skipped: {args.preds_dir} not found.")
            print("Run finetune_experiment.py v2 with --save_predictions to generate them.")


if __name__ == "__main__":
    main()
