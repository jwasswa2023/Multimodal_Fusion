# ==========================================
# CODE B (with plots)
# Uncertainty & calibration:
#   Four-modality fusion:
#   Mol2Vec + RDKit + AttentiveFP STRICT embeddings + SMILES BiGRU embeddings
#   → LGBM ensemble
#
# Requires from Code A:
#   - y_test
#   - ensemble_preds_fourmod    (list of length = n_seeds, each (n_test,))
#   - baseline_pred_fourmod     (first-seed predictions, (n_test,))
# ==========================================
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Uncertainty & Calibration helpers
# ============================================================
def compute_uncertainty_from_ensemble(ensemble_preds):
    """
    Return ensemble mean prediction and epistemic uncertainty
    (std across ensemble members).
    """
    preds = np.vstack([np.asarray(p).reshape(-1) for p in ensemble_preds])
    mean_pred = preds.mean(axis=0)
    std_pred  = preds.std(axis=0)
    return mean_pred, std_pred


def regression_uncertainty_ece(abs_errors, uncertainties, n_bins=10):
    """
    Regression-style Expected Calibration Error (ECE):
    Compare mean |error| vs mean uncertainty per bin.
    """
    abs_errors = np.asarray(abs_errors)
    uncertainties = np.asarray(uncertainties)

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(uncertainties, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.nan, []

    bin_idx = np.digitize(uncertainties, edges[1:-1], right=True)
    ece = 0.0
    bin_stats = []

    for b in range(len(edges) - 1):
        mask = (bin_idx == b)
        if not np.any(mask):
            continue
        w = mask.mean()
        mean_err = abs_errors[mask].mean()
        mean_unc = uncertainties[mask].mean()
        ece += w * abs(mean_err - mean_unc)

        bin_stats.append(dict(
            bin=b,
            frac=w,
            n_samples=int(mask.sum()),
            mean_error=float(mean_err),
            mean_uncertainty=float(mean_unc),
            unc_min=float(edges[b]),
            unc_max=float(edges[b+1]),
        ))

    return float(ece), bin_stats


def miscalibration_area(abs_errors, uncertainties, n_points=100):
    """
    Continuous Miscalibration Area between the marginal distributions
    of |error| and uncertainty (L1 area between their quantile curves).
    Smaller is better.
    """
    abs_errors = np.asarray(abs_errors)
    uncertainties = np.asarray(uncertainties)

    if abs_errors.size == 0 or uncertainties.size == 0:
        return np.nan

    qs = np.linspace(0.0, 1.0, n_points)
    e_q = np.quantile(abs_errors, qs)
    u_q = np.quantile(uncertainties, qs)
    area = np.trapz(np.abs(e_q - u_q), qs)
    return float(area)


def summarize_outliers(abs_errors, threshold=1.0):
    """Tail error metrics."""
    abs_errors = np.asarray(abs_errors)
    return {
        "max_error": float(abs_errors.max()),
        "p90": float(np.quantile(abs_errors, 0.90)),
        "p95": float(np.quantile(abs_errors, 0.95)),
        "p99": float(np.quantile(abs_errors, 0.99)),
        "frac_above_threshold": float((abs_errors > threshold).mean()),
        "n_above_threshold": int((abs_errors > threshold).sum()),
    }


def coverage_rates(abs_errors, uncertainties, ks=(1.0, 2.0, 3.0)):
    """
    Coverage of k·σ intervals:
    Fraction of samples with |error| <= k * uncertainty.
    """
    abs_errors = np.asarray(abs_errors)
    uncertainties = np.asarray(uncertainties)
    cov = {}
    for k in ks:
        cov_k = float((abs_errors <= k * uncertainties).mean())
        cov[f"cov_{k:.0f}sigma"] = cov_k
    return cov


def calibration_slope_intercept(abs_errors, uncertainties):
    """
    Fit a simple linear model:
        |error| ≈ a + b * uncertainty
    and return (a, b). A slope b ≈ 1 and small intercept
    is a rough sign of good calibration (for magnitudes).
    """
    abs_errors = np.asarray(abs_errors)
    uncertainties = np.asarray(uncertainties)
    if uncertainties.std() == 0 or abs_errors.std() == 0:
        return np.nan, np.nan
    b, a = np.polyfit(uncertainties, abs_errors, 1)  # y = a + b x
    return float(a), float(b)


def kappa_slope_intercept(y_true, y_pred):
    """
    Regression calibration line: y_true ≈ alpha + kappa * y_pred

    κ-slope style calibration:
      - kappa ~ 1 and alpha ~ 0 → well calibrated in mean.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mu_y = y_true.mean()
    mu_p = y_pred.mean()
    var_p = ((y_pred - mu_p) ** 2).mean()
    if var_p <= 0:
        return np.nan, np.nan

    cov_yp = ((y_pred - mu_p) * (y_true - mu_y)).mean()
    kappa = cov_yp / var_p
    alpha = mu_y - kappa * mu_p
    return float(alpha), float(kappa)


def reliability_curve(abs_errors, uncertainties, n_bins=8):
    """
    Return (bin_centers, mean_uncertainty, mean_error, counts)
    for plotting a reliability-style curve in regression.
    """
    abs_errors = np.asarray(abs_errors)
    uncertainties = np.asarray(uncertainties)

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(uncertainties, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return None

    bin_idx = np.digitize(uncertainties, edges[1:-1], right=True)
    centers, mean_uncs, mean_errs, counts = [], [], [], []

    for b in range(len(edges) - 1):
        mask = (bin_idx == b)
        if not np.any(mask):
            continue
        centers.append(0.5 * (edges[b] + edges[b+1]))
        mean_uncs.append(float(uncertainties[mask].mean()))
        mean_errs.append(float(abs_errors[mask].mean()))
        counts.append(int(mask.sum()))

    return (
        np.array(centers),
        np.array(mean_uncs),
        np.array(mean_errs),
        np.array(counts),
    )


# ============================================================
# Main UQ + calibration analysis
# ============================================================
def analyze_uncertainty(name, y_true, ensemble_preds,
                        baseline_pred=None,
                        error_threshold=1.0, ece_bins=8,
                        make_plots=True):
    """
    Run full uncertainty pipeline and optionally make plots.

    Epistemic uncertainty: std across LGBM seeds.
    Calibration metrics:
      - |error| vs σ slope/intercept
      - κ-slope on y_true vs y_pred
      - ECE and miscalibration area
      - Coverage and tail stats
    """
    y_true = np.asarray(y_true).reshape(-1)

    # 1) Ensemble mean + std (epistemic uncertainty)
    mean_pred, std_pred = compute_uncertainty_from_ensemble(ensemble_preds)
    abs_errors = np.abs(y_true - mean_pred)

    # 2) Accuracy metrics
    mse = ((y_true - mean_pred)**2).mean()
    rmse = float(np.sqrt(mse))
    mae  = float(abs_errors.mean())
    ss_res = ((y_true - mean_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    # 3) Uncertainty–error correlation
    if std_pred.std() > 0 and abs_errors.std() > 0:
        corr = float(np.corrcoef(std_pred, abs_errors)[0, 1])
    else:
        corr = np.nan

    # 4) ECE + Miscalibration Area
    ece, bin_stats = regression_uncertainty_ece(abs_errors, std_pred, n_bins=ece_bins)
    miscal_area = miscalibration_area(abs_errors, std_pred, n_points=100)

    # 5) Tail analysis
    tail = summarize_outliers(abs_errors, threshold=error_threshold)

    # 6) Coverage
    cov = coverage_rates(abs_errors, std_pred, ks=(1.0, 2.0, 3.0))

    # 7) Calibration slope/intercept for |err| vs σ
    intercept_mag, slope_mag = calibration_slope_intercept(abs_errors, std_pred)

    # 8) κ-slope for y_true vs mean_pred
    alpha, kappa = kappa_slope_intercept(y_true, mean_pred)

    # 9) Reliability curve points
    rel_curve = reliability_curve(abs_errors, std_pred, n_bins=ece_bins)

    # ---- Print summary ----
    print("\n" + "="*80)
    print(f"UNCERTAINTY & CALIBRATION — {name}")
    print("="*80)
    print(f"Samples                      : {len(y_true)}")
    print(f"R2 (ensemble mean)          : {r2:.4f}")
    print(f"RMSE                        : {rmse:.4f}")
    print(f"MAE                         : {mae:.4f}")
    print(f"Mean epistemic std          : {std_pred.mean():.4f}")
    print(f"Corr(|error|, uncertainty)  : {corr:.4f}")
    print(f"ECE (regression-style)      : {ece:.4f}")
    print(f"Miscalibration Area (|err| vs σ): {miscal_area:.4f}")
    print("Calibration |err| ≈ a + b·σ:")
    print(f"  a = {intercept_mag:.4f}, b = {slope_mag:.4f}")
    print("κ-slope calibration y ≈ alpha + kappa·ŷ:")
    print(f"  alpha = {alpha:.4f}, kappa = {kappa:.4f}")
    print(f"Tail errors (threshold={error_threshold}): {tail}")
    print("Coverage rates (|err| <= k·σ):")
    for k, v in cov.items():
        print(f"  {k}: {v:.3f}")

    baseline_tail_stats = None
    if baseline_pred is not None:
        baseline_pred = np.asarray(baseline_pred).reshape(-1)
        baseline_abs = np.abs(y_true - baseline_pred)
        baseline_tail_stats = summarize_outliers(
            baseline_abs, threshold=error_threshold
        )

        print("\nBaseline vs Ensemble — Outlier Reduction")
        print("----------------------------------------")
        print(f"Baseline p95:        {baseline_tail_stats['p95']:.4f}")
        print(f"Ensemble  p95:       {tail['p95']:.4f}")
        print(f"Baseline max error:  {baseline_tail_stats['max_error']:.4f}")
        print(f"Ensemble  max error: {tail['max_error']:.4f}")
        print(f"Baseline frac >{error_threshold:.2f}:  "
              f"{baseline_tail_stats['frac_above_threshold']:.3f}")
        print(f"Ensemble  frac >{error_threshold:.2f}: "
              f"{tail['frac_above_threshold']:.3f}")

    # ---- Optional plots ----
    if make_plots:
        # (1) |error| vs epistemic std
        plt.figure(figsize=(5, 4))
        plt.scatter(std_pred, abs_errors, alpha=0.6)
        plt.xlabel("Epistemic std (four-mod LGBM ensemble)")
        plt.ylabel("|Error|")
        plt.title(f"|Error| vs Epistemic std — {name}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        # (2) Histogram of epistemic std
        plt.figure(figsize=(5, 4))
        plt.hist(std_pred, bins=30, edgecolor="black", alpha=0.8)
        plt.xlabel("Epistemic std (four-mod LGBM ensemble)")
        plt.ylabel("Count")
        plt.title(f"Distribution of Epistemic std — {name}")
        plt.tight_layout()
        plt.show()

        # (3) Reliability-style curve: mean |err| vs mean σ
        if rel_curve is not None:
            centers, mean_uncs, mean_errs, counts = rel_curve
            plt.figure(figsize=(5, 4))
            plt.plot(mean_uncs, mean_errs, marker="o")
            max_val = max(mean_uncs.max(), mean_errs.max())
            plt.plot([0, max_val], [0, max_val], linestyle="--")  # ideal line
            plt.xlabel("Mean epistemic std per bin")
            plt.ylabel("Mean |error| per bin")
            plt.title(f"Reliability-style curve — {name}")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    results = dict(
        mean_pred=mean_pred,
        std_pred=std_pred,
        abs_errors=abs_errors,
        r2=r2,
        rmse=rmse,
        mae=mae,
        corr_error_unc=corr,
        ece=ece,
        miscal_area=miscal_area,
        coverage=cov,
        bin_stats=bin_stats,
        tail_stats=tail,
        baseline_tail_stats=baseline_tail_stats,
        calibration_intercept_mag=intercept_mag,
        calibration_slope_mag=slope_mag,
        alpha=alpha,
        kappa=kappa,
        reliability_curve=rel_curve,
    )
    return results


# ------------------------------------------------------------
# Run analysis for FOUR-modality fusion ensemble
# ------------------------------------------------------------
MAKE_PLOTS = True

unc_results_fourmod = analyze_uncertainty(
    name="Four-modality fusion (Mol2Vec + RDKit + AttentiveFP-emb + SMILES-emb)",
    y_true=y_test,
    ensemble_preds=ensemble_preds_fourmod,
    baseline_pred=baseline_pred_fourmod,
    error_threshold=1.0,
    ece_bins=8,
    make_plots=MAKE_PLOTS,
)

