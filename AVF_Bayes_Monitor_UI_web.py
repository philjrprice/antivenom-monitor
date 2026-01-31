"""
AVF_Bayes_Monitor_UI_web (24).py
--------------------------------
Single-arm Bayesian trial monitor (AVF). This Streamlit application supports:
  • Real-time Bayesian updating for efficacy and safety (Beta–Binomial model)
  • Sequential decision rules (success, futility via PPoS, safety) with look schedules
  • Forecasting (PPoS to final), sensitivity analyses, and regulatory boundary tables
  • Design integrity checks (sequential Type I error), power analysis, and power curves
  • On-demand Operating Characteristics (OC) scenario suite with charts and CSV export

NEW IN THIS VERSION (24):
  • Scenario Suite stop-reasons stacked bar uses fixed colors:
      Safety stop = red, Futility stop = orange, Interim success = green,
      Final success = blue, No decision = grey
  • A compact “stop reason legend” is added to the Power section (not just the Scenario Suite)
  • Extensive comments have been added throughout the code to aid inspection and review

All prior functionality from (23) is preserved.
"""

# ============ Imports ============
import streamlit as st
import numpy as np
from scipy.stats import beta
from scipy.special import betaln
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# ============ Streamlit page config ============
# Wide layout improves horizontal space for multi-column metrics and charts
st.set_page_config(page_title="Bayes Trial Monitor", layout="wide")

# ==============================================================
# 1. SIDEBAR INPUTS — Trial data, Priors, Schedules, and Controls
#    These widgets define the trial design and the current observed data.
# ==============================================================
st.sidebar.header("📋 Current Trial Data")

# --- Trial size and observed data ---
max_n_val = st.sidebar.number_input(
    "Maximum Sample Size (N)", 10, 500, 70,
    help="The total planned sample size for the trial design."
)
# Current evaluable sample size
total_n = st.sidebar.number_input(
    "Total Patients Enrolled", 0, max_n_val, 20,
    help="Current number of patients who have generated evaluable data."
)
# Primary efficacy endpoint: binary success count
successes = st.sidebar.number_input(
    "Total Successes", 0, total_n, value=min(14, total_n),
    help="Number of patients achieving the primary efficacy endpoint."
)
# Safety endpoint: binary SAE per patient (≥1 SAE => SAE=1)
saes = st.sidebar.number_input(
    "Serious Adverse Events (SAEs)", 0, total_n, value=min(1, total_n),
    help="Binary safety endpoint: ≥1 SAE per patient counts as 1 SAE event."
)

# --- Data Integrity Validation ---
if successes > total_n:
    st.error("⚠️ Data Integrity Error: Successes cannot exceed total patients enrolled.")
    st.stop()
if saes > total_n:
    st.error("⚠️ Data Integrity Error: SAEs cannot exceed total patients enrolled.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Study Parameters")

# --- Efficacy prior (Beta prior for true success rate) ---
with st.sidebar.expander("Base Study Priors (Efficacy)", expanded=True):
    prior_alpha = st.slider(
        "Prior Successes (Alpha_eff)", 0.1, 10.0, 1.0, step=0.1,
        help="Initial belief strength for efficacy: equivalent 'phantom' successes."
    )
    prior_beta = st.slider(
        "Prior Failures (Beta_eff)", 0.1, 10.0, 1.0, step=0.1,
        help="Initial belief strength for efficacy: equivalent 'phantom' failures."
    )

# --- Efficacy look schedule (interim analysis cadence) ---
with st.sidebar.expander("Adaptive Timing & Look Points (Efficacy/Futility)", expanded=True):
    min_interim = st.number_input(
        "Min N before first check", 1, max_n_val, 14,
        help="Burn-in period: No stop decisions will be made before this sample size."
    )
    eff_schedule_mode = st.selectbox(
        "Efficacy look schedule",
        ["Every N", "Number of looks (equal spacing)", "Custom % of remaining"], index=0,
        help="Choose schedule type. Run-in is always included as the first look."
    )
    # Single 'value' control depending on selected mode
    eff_value = None
    if eff_schedule_mode == "Every N":
        eff_value = st.number_input(
 "Every N patients", 1, max(20, max_n_val), 5, key='eff_every_n',
 help="Check after every N patients following the run-in."
        )
    elif eff_schedule_mode == "Number of looks (equal spacing)":
        eff_value = st.number_input(
        "Total number of looks (incl. final)", 1, 100, 8, 1,
            help="Equally spaced looks from run-in to max N (includes run-in and final, key='eff_nlooks')."
        )
    else:  # Custom % of remaining
        eff_value = st.text_input(
    "Custom % of remaining (comma-separated)", "20,20,20,40", key='eff_pctseq',
    help="Enter percentages like 20,20,20,40. Each value schedules the next look after that % of remaining to max N."
)

with st.sidebar.expander("Success & Futility Rules"):

    null_eff = st.slider(
        "Null Efficacy (p0) (%)", 0.1, 1.0, 0.50,
        help="The historical or placebo rate we must beat."
    )
    target_eff = st.slider(
        "Target Efficacy (p1) (%)", 0.1, 1.0, 0.60,
        help="The desired efficacy rate we hope to achieve."
    )
    dream_eff = st.slider(
        "Goal/Dream Efficacy (%)", 0.1, 1.0, 0.70,
        help="An aspirational target used for secondary probability metrics."
    )
    success_conf_req_interim = st.slider(
        "Interim Success Confidence", 0.5, 0.99, 0.74,
        help="Posterior probability threshold to declare efficacy at interim looks."
    )
    success_conf_req_final = st.slider(
        "Final Success Confidence", 0.5, 0.999, 0.74,
        help="Posterior probability threshold to declare efficacy at the final look."
    )
    bpp_futility_limit = st.slider(
        "BPP Futility Limit", 0.01, 0.20, 0.05,
        help="Stop if the chance of future success (PPoS) drops below this level."
    )

# --- Safety rules: independent safety prior, safety schedule, and gating toggle ---
with st.sidebar.expander("Safety Rules, Priors & Timing", expanded=True):
    safe_limit = st.slider(
        "SAE Upper Limit (% )", 0.05, 0.50, 0.15,
        help="Maximum acceptable toxicity rate."
    )
    safe_conf_req = st.slider(
        "Safety Stop Confidence", 0.5, 0.99, 0.90,
        help="Stop if we are this confident the true toxicity rate exceeds the limit."
    )
    st.markdown("**Safety Priors (independent from efficacy):**")
    prior_alpha_saf = st.slider(
        "Safety prior α (toxic events)", 0.1, 10.0, 1.0, step=0.1,
        help="Prior pseudo-count for toxic events (α)."
    )
    prior_beta_saf = st.slider(
        "Safety prior β (non-toxic)", 0.1, 10.0, 1.0, step=0.1,
        help="Prior pseudo-count for non-toxic outcomes (β)."
    )
    st.markdown("**Safety Look Schedule (optional; defaults preserve original behavior):**")
    safety_min_interim = st.number_input(
        "Safety: Min N before first check", 1, max_n_val, min_interim,
        help="First safety look (often earlier than efficacy)."
    )
    safety_schedule_mode = st.selectbox(
        "Safety look schedule",
        ["Every N", "Number of looks (equal spacing)", "Custom % of remaining"], index=0,
        help="Choose schedule type. Run-in is always included as the first look."
    )
    safety_value = None
    if safety_schedule_mode == "Every N":
        safety_value = st.number_input(
 "Every N patients", 1, max(20, max_n_val), 5, key='saf_every_n',
 help="Check after every N patients following the run-in."
        )
    elif safety_schedule_mode == "Number of looks (equal spacing)":
         safety_value = st.number_input(
        "Total number of looks (incl. final)", 1, 100, 8, 1, key='saf_nlooks',
        help="Equally spaced looks from run-in to max N (includes run-in and final)."
    )
else:
        safety_value = st.text_input(
    "Custom % of remaining (comma-separated)", "20,20,20,40", key='saf_pctseq',
    help="Enter percentages like 20,20,20,40. Each value schedules the next look after that % of remaining to max N."
)
    safety_gate_to_schedule = st.checkbox(
        "Apply safety decision only at scheduled safety looks",
        value=False,
        help="If unchecked, safety is monitored continuously (original behavior)."
    )

with st.sidebar.expander("Sensitivity Priors (Adjustable) — Efficacy", expanded=True):

    st.caption("Define three efficacy priors (α, β) for sensitivity overlays.")
    eff1_a = st.number_input("Efficacy S1 α", 0.1, 20.0, 2.0, 0.1)
    eff1_b = st.number_input("Efficacy S1 β", 0.1, 20.0, 1.0, 0.1)
    eff2_a = st.number_input("Efficacy S2 α", 0.1, 20.0, 1.0, 0.1)
    eff2_b = st.number_input("Efficacy S2 β", 0.1, 20.0, 1.0, 0.1)
    eff3_a = st.number_input("Efficacy S3 α", 0.1, 20.0, 1.0, 0.1)
    eff3_b = st.number_input("Efficacy S3 β", 0.1, 20.0, 2.0, 0.1)

# --- Sensitivity priors (safety) for robustness overlays ---
with st.sidebar.expander("Sensitivity Priors (Adjustable) — Safety", expanded=True):
    st.caption("Define three safety priors (α, β) for sensitivity overlays.")
    saf1_a = st.number_input("Safety S1 α", 0.1, 20.0, 0.5, 0.1)
    saf1_b = st.number_input("Safety S1 β", 0.1, 20.0, 2.0, 0.1)
    saf2_a = st.number_input("Safety S2 α", 0.1, 20.0, 1.0, 0.1)
    saf2_b = st.number_input("Safety S2 β", 0.1, 20.0, 1.0, 0.1)
    saf3_a = st.number_input("Safety S3 α", 0.1, 20.0, 2.0, 0.1)
    saf3_b = st.number_input("Safety S3 β", 0.1, 20.0, 0.5, 0.1)

# --- Equivalence band and optional heatmap ---
with st.sidebar.expander("Equivalence & Heatmap Settings"):
    equiv_bound = st.slider("Equivalence Bound (+/-)", 0.01, 0.10, 0.05,
                            help="Zone around Null Efficacy considered 'Practical Equivalence'.")
    include_heatmap = st.checkbox("Generate Risk-Benefit Heatmap", value=True)
    st.caption("Note: Heatmap utility is illustrative (score = efficacy − w×toxicity).")

# --- Forecasting (PPoS) simulation controls ---
with st.sidebar.expander("Forecasting Controls (PPoS)", expanded=True):
    mc_draws = st.number_input(
        "Monte Carlo draws for PPoS", 5000, 100000, 20000, step=5000,
        help="Higher draws reduce Monte Carlo noise at boundaries."
    )
    mc_seed = st.number_input(
        "Random seed (PPoS)", 0, 10_000_000, 42, step=1,
        help="Controls reproducibility of PPoS simulations."
    )

# --- Power analysis controls & toggles ---
with st.sidebar.expander("Power Analysis Settings", expanded=False):
    power_true_eff = st.slider(
        "Assumed TRUE efficacy rate for power", 0.05, 0.95, float(target_eff), step=0.01,
        help="Used to simulate trials when estimating power (probability of declaring success)."
    )
    power_true_saf = st.slider(
        "Assumed TRUE SAE rate for power", 0.0, 0.9, float(safe_limit), step=0.01,
        help="Used to simulate SAEs during power estimation."
    )
    power_sims = st.number_input(
        "Number of simulations (power point)", 1000, 100000, 10000, step=1000,
        help="Trials simulated at the chosen true rates to estimate power and expected sample size."
    )
    power_seed = st.number_input("Random seed (power)", 0, 10_000_000, 17, step=1)
    st.markdown("**Power curve options**")
    power_curve_points = st.slider(
        "Power curve: points across efficacy range", 5, 30, 12, step=1,
        help="Number of grid points (e.g., from p0 to 0.9) for the power curve."
    )
    power_curve_sims = st.number_input(
        "Simulations per curve point", 500, 50000, 5000, step=500,
        help="Trials simulated at each efficacy grid value to draw the power curve."
    )
    power_use_eff_looks = st.checkbox(
        "Simulate using user-defined efficacy look schedule",
        value=True,
        help="If unchecked, evaluate efficacy at every N from 1..max N (continuous)."
    )
    power_use_saf_looks = st.checkbox(
        "Simulate using user-defined safety look schedule",
        value=True,
        help="If unchecked, evaluate safety at every N from 1..max N (continuous; safety gating ignored)."
    )

# ==============================================================
# 2. BAYESIAN UPDATES — Posteriors, Means, Credible Intervals & Key Probs
#    Compute Beta posterior parameters for efficacy and safety.
# ==============================================================
# Posterior parameters = prior + data
a_eff, b_eff = prior_alpha + successes, prior_beta + (total_n - successes)
a_saf, b_saf = prior_alpha_saf + saes, prior_beta_saf + (total_n - saes)

# Key posterior tail probabilities used across the app
p_null = 1 - beta.cdf(null_eff, a_eff, b_eff)      # Pr(p_eff > p0)
p_target = 1 - beta.cdf(target_eff, a_eff, b_eff)  # Pr(p_eff > p1)
p_goal = 1 - beta.cdf(dream_eff, a_eff, b_eff)     # Pr(p_eff > goal)
p_toxic = 1 - beta.cdf(safe_limit, a_saf, b_saf)   # Pr(p_saf > safety limit)

# Equivalence (practical equivalence band around p0)
lb, ub = max(0.0, null_eff - equiv_bound), min(1.0, null_eff + equiv_bound)
p_equiv = beta.cdf(ub, a_eff, b_eff) - beta.cdf(lb, a_eff, b_eff)

# Means & 95% credible intervals
eff_mean, eff_ci = a_eff / (a_eff + b_eff), beta.ppf([0.025, 0.975], a_eff, b_eff)
saf_mean, saf_ci = a_saf / (a_saf + b_saf), beta.ppf([0.025, 0.975], a_saf, b_saf)

# ==============================================================
# 3. FORECASTING ENGINE — Predictive Probability of Success (PPoS)
#    Simulates remaining patients to max N using posterior predictive draws.
# ==============================================================
@st.cache_data
def get_enhanced_forecasts(curr_s, curr_n, m_n, t_eff, s_conf_final, p_a, p_b, draws, seed):
    """Compute PPoS at final N and an 95% MC CI around it.
    - Draw a predictive efficacy rate from Beta(p_a+curr_s, p_b+curr_n-curr_s)
    - Simulate remaining successes and evaluate final success rule against s_conf_final
    """
    rng = np.random.default_rng(seed)
    rem_n = m_n - curr_n
    if rem_n <= 0:
        # At final N: PPoS is degenerate (0 or 1) depending on meeting final threshold
        is_success = (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf_final
        ppos = 1.0 if is_success else 0.0
        se = np.sqrt(ppos * (1 - ppos) / max(1, draws))
        ci_low = max(0.0, ppos - 1.96 * se)
        ci_high = min(1.0, ppos + 1.96 * se)
        return ppos, [curr_s, curr_s], se, ci_low, ci_high
    # Predictive draws of rate and future successes
    future_rates = rng.beta(p_a + curr_s, p_b + curr_n - curr_s, draws)
    future_successes = rng.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    # Posterior tail at final
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    ppos = float(np.mean(final_confs > s_conf_final))
    s_range = [int(np.percentile(total_proj_s, 5)), int(np.percentile(total_proj_s, 95))]
    se = float(np.sqrt(ppos * (1 - ppos) / max(1, draws)))
    ci_low = max(0.0, ppos - 1.96 * se)
    ci_high = min(1.0, ppos + 1.96 * se)
    return ppos, s_range, se, ci_low, ci_high

# ==============================================================
# 4. BOUNDARY HELPERS — Success, Futility (PPoS), Safety Stops
#    Used by corridors, governing rules, Type I / Power / OC sims.
# ==============================================================
@st.cache_data
def success_boundary(lp, prior_alpha, prior_beta, target_eff, conf_req):
    """Smallest S at look size lp such that Pr(p_eff > target_eff) > conf_req; else None."""
    return next((s for s in range(lp + 1)
                 if (1 - beta.cdf(target_eff, prior_alpha + s, prior_beta + (lp - s))) > conf_req),
                None)

def safety_stop_threshold(lp: int, prior_alpha_saf: float, prior_beta_saf: float,
                          safe_limit: float, safe_conf_req: float):
    """Smallest SAE count s at look size lp with Pr(p_saf > safe_limit) > safe_conf_req; else None."""
    for s in range(lp + 1):
        post_tail = 1 - beta.cdf(safe_limit, prior_alpha_saf + s, prior_beta_saf + (lp - s))
        if post_tail > safe_conf_req:
            return s
    return None

@st.cache_data
def futility_boundary_ppos(lp, max_n_val, target_eff, success_conf_final,
                           prior_alpha, prior_beta, draws, seed, futility_floor):
    """Largest S at look size lp where PPoS(final) <= futility_floor. Monotone-corrected."""
    ppos_by_S = np.empty(lp + 1, dtype=float)
    for s in range(lp + 1):
        ppos_by_S[s] = get_enhanced_forecasts(s, lp, max_n_val, target_eff, success_conf_final,
                                              prior_alpha, prior_beta, draws, seed)[0]
    # Enforce monotonicity (non-decreasing in S)
    ppos_monotone = np.maximum.accumulate(ppos_by_S)
    idx = np.where(ppos_monotone <= futility_floor)[0]
    return int(idx[-1]) if idx.size > 0 else -1

# --- Wilson score CI for MC proportions (robust near 0 and 1) ---
def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (0.0, 1.0)
    from math import sqrt
    z = 1.959963984540054  # 97.5% quantile for two-sided 95% CI
    p = k / n
    denom = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    halfwidth = (z * sqrt((p*(1-p)/n) + ((z*z)/(4*n*n)))) / denom
    return (max(0.0, center - halfwidth), min(1.0, center + halfwidth))

# --- Run forecasting for current state (uses FINAL success threshold) ---
bpp, ps_range, bpp_se, bpp_ci_low, bpp_ci_high = get_enhanced_forecasts(
    successes, total_n, max_n_val, target_eff, success_conf_req_final, prior_alpha, prior_beta, mc_draws, mc_seed
)

# ==============================================================
# 5. MAIN DASHBOARD — Title, Design Summary, KPIs, and Governing Rules
# ==============================================================
st.title("🛡️ AVF-Single Arm Bayesian Trial Monitor")

# --- Design summary panel (compact) ---
st.subheader("🧭 Design Summary")
ds1, ds2, ds3 = st.columns(3)
with ds1:
    st.markdown("**Efficacy (primary)**")
    st.write(f"Null p0: **{null_eff:.0%}** | Target p1: **{target_eff:.0%}** | Goal: **{dream_eff:.0%}**")
    st.write(f"Prior (α, β): **({prior_alpha:.1f}, {prior_beta:.1f})** → ESS **{prior_alpha+prior_beta:.1f}**")
    sch_text_eff = (
    f'run-in N={min_interim}, then every {eff_value}' if eff_schedule_mode=='Every N' else
    f'run-in N={min_interim}, then {int(eff_value)} equally spaced looks to N={max_n_val}' if eff_schedule_mode=='Number of looks (equal spacing)' else
    f'run-in N={min_interim}, then % remaining sequence [{eff_value}] to N={max_n_val}'
)
st.write(f'Schedule: {sch_text_eff}')
with ds2:
    st.markdown("**Safety (binary SAE)**")
    st.write(f"Safety limit: **{safe_limit:.0%}** | Stop confidence: **{safe_conf_req:.0%}**")
    st.write(f"Prior (α, β): **({prior_alpha_saf:.1f}, {prior_beta_saf:.1f})** → ESS **{prior_alpha_saf+prior_beta_saf:.1f}**")
    sch_text_saf = (
    f'run-in N={safety_min_interim}, then every {safety_value}' if safety_schedule_mode=='Every N' else
    f'run-in N={safety_min_interim}, then {int(safety_value)} equally spaced looks to N={max_n_val}' if safety_schedule_mode=='Number of looks (equal spacing)' else
    f'run-in N={safety_min_interim}, then % remaining sequence [{safety_value}] to N={max_n_val}'
)
st.write(f'Schedule: {sch_text_saf}')
    st.write(f"Gating to schedule: **{'ON' if safety_gate_to_schedule else 'OFF'}**")
with ds3:
    st.markdown("**Decision & Simulation settings**")
    st.write(f"Interim success threshold: **{success_conf_req_interim:.0%}**")
    st.write(f"Final success threshold: **{success_conf_req_final:.0%}**")
    st.write(f"Futility floor (PPoS): **{bpp_futility_limit:.0%}**")
    st.write(f"PPoS draws: **{mc_draws:,}** | seed: **{mc_seed}**")
    st.write(f"Power toggles → Eff: **{'Scheduled' if power_use_eff_looks else 'Continuous'}**, "
             f"Safety: **{'Scheduled' if power_use_saf_looks else 'Continuous'}**")

# --- Top KPIs ---
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample N", f"{total_n}/{max_n_val}")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}")
m3.metric(f"P(>{target_eff:.0%})", f"{p_target:.1%}")
m4.metric("Safety Risk", f"{p_toxic:.1%}")
m5.metric("PPoS (Final)", f"{bpp:.1%}", help=f"95% CI [{bpp_ci_low:.1%}, {bpp_ci_high:.1%}]")
m6.metric("Prior ESS (Eff.)", f"{prior_alpha + prior_beta:.1f}")
st.caption(
    f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}**  \n"
    f"Prob Equivalence: **{p_equiv:.1%}** (band applied: [{lb:.0%}, {ub:.0%}])"
)

st.markdown("---")

# --- Governing Rules (safety → futility → success; final uses final threshold) ---
is_efficacy_look = (total_n >= min_interim) and (((total_n - min_interim) % check_cohort) == 0)
is_safety_look = (total_n >= safety_min_interim) and (((total_n - safety_min_interim) % safety_check_cohort) == 0)
apply_safety_now = (not safety_gate_to_schedule) or is_safety_look
is_final_look = (total_n == max_n_val)
eff_success_threshold_now = success_conf_req_final if is_final_look else success_conf_req_interim

if apply_safety_now and (p_toxic > safe_conf_req):
    msg_schedule = "" if not safety_gate_to_schedule else " (scheduled safety look)"
    st.error(f"🛑 **GOVERNING RULE: SAFETY STOP{msg_schedule}.** Risk ({p_toxic:.1%}) exceeds {safe_conf_req:.0%} threshold.")
elif is_efficacy_look or is_final_look:
    if not is_final_look and (bpp < bpp_futility_limit):
        st.warning(f"⚠️ **GOVERNING RULE: FUTILITY STOP.** PPoS ({bpp:.1%}) below floor. (SE={bpp_se:.3f})")
    elif p_target > eff_success_threshold_now:
        label = "FINAL EFFICACY SUCCESS" if is_final_look else "INTERIM EFFICACY SUCCESS"
        st.success(f"✅ **GOVERNING RULE: {label}.** Evidence achieved at {p_target:.1%} "
                   f"(threshold={eff_success_threshold_now:.0%}).")
    else:
        phase = "final look" if is_final_look else f"interim N={total_n}"
        st.info(f"🛡️ **GOVERNING RULE: CONTINUE.** {phase} is indeterminate.")
elif total_n < min_interim:
    st.info(f"⏳ **STATUS: LEAD-IN.** Enrollment phase; first efficacy check at N={min_interim}.")
else:
    # Determine upcoming scheduled looks from the constructed schedules
    next_check = next((n for n in look_points if n > total_n), None)
    next_safety = next((n for n in safety_look_points if n > total_n), None)
    if next_check and next_safety:
        st.info(f'🧬 **STATUS: MONITORING.** Next efficacy check at N={next_check}; next safety check at N={next_safety}.')
    elif next_check and not next_safety:
        st.info(f'🧬 **STATUS: MONITORING.** Next efficacy check at N={next_check}; no further safety looks scheduled.')
    elif next_safety and not next_check:
        st.info(f'🧬 **STATUS: MONITORING.** Next safety check at N={next_safety}; no further efficacy looks scheduled.')
    else:
        st.info('🧬 **STATUS: MONITORING.** No further scheduled looks; decisions occur only at final.')
    # Show next-look thresholds explicitly for review clarity
    try:
        succ_req_next = success_boundary(
            next_check, prior_alpha, prior_beta, target_eff,
            success_conf_req_final if (next_check == max_n_val) else success_conf_req_interim
        )
        futi_req_next = futility_boundary_ppos(
            next_check, max_n_val, target_eff, success_conf_req_final,
            prior_alpha, prior_beta, mc_draws, mc_seed, bpp_futility_limit
        )
        succ_txt = (f"Success Stop if S ≥ {succ_req_next}" if succ_req_next is not None else "No success stop at that look")
        futi_txt = (f"Futility Stop if S ≤ {futi_req_next}" if futi_req_next >= 0 else "No futility stop at that look")
        st.caption(f"**Next efficacy look (N={next_check}):** {succ_txt}; {futi_txt}.")
    except Exception:
        pass
    # Predictive safety stop probability to next safety look
    try:
        if next_safety > total_n:
            saf_thr_next = safety_stop_threshold(next_safety, prior_alpha_saf, prior_beta_saf, safe_limit, safe_conf_req)
            if saf_thr_next is not None:
                rem_to_next = next_safety - total_n
                draws_pred = min(5000, mc_draws)
                rng_pred = np.random.default_rng(mc_seed + 101)
                p_saf_draws = rng_pred.beta(a_saf, b_saf, draws_pred)
                new_saes = rng_pred.binomial(rem_to_next, p_saf_draws)
                pr_stop_next = float(np.mean((saes + new_saes) >= saf_thr_next))
                st.caption(f"**Predictive Pr[safety stop at next safety look (N={next_safety})]: {pr_stop_next:.1%}** "
                           f"(threshold: SAEs ≥ {saf_thr_next})")
    except Exception:
        pass

# ==============================================================
# 6. DECISION CORRIDORS — Efficacy (success/futility) and Safety boundaries
# ==============================================================
st.subheader("📈 Trial Decision Corridors")
# Build canonical look points from the schedule definitions
look_points = build_looks(
    max_n_val, min_interim, eff_schedule_mode, eff_value,
    is_pct_list=(eff_schedule_mode=='Custom % of remaining')
)
viz_n = np.array(look_points)
safety_look_points = build_looks(
    max_n_val, safety_min_interim, safety_schedule_mode, safety_value,
    is_pct_list=(safety_schedule_mode=='Custom % of remaining')
)
viz_n_safety = np.array(safety_look_points)

# Compute success and futility boundaries across efficacy looks
succ_line, fut_line = [], []
for lp in viz_n:
    use_conf = success_conf_req_final if (lp == max_n_val) else success_conf_req_interim
    s_req = success_boundary(lp, prior_alpha, prior_beta, target_eff, use_conf)
    f_req = futility_boundary_ppos(lp, max_n_val, target_eff, success_conf_req_final,
                                   prior_alpha, prior_beta, mc_draws, mc_seed, bpp_futility_limit)
    succ_line.append(s_req)
    fut_line.append(max(0, f_req) if f_req >= 0 else -1)

# Compute safety stop thresholds across safety looks
saf_line = []
for lp in viz_n_safety:
    saf_req = safety_stop_threshold(lp, prior_alpha_saf, prior_beta_saf, safe_limit, safe_conf_req)
    saf_line.append(saf_req)

# Maps for simulators and tables
succ_req_by_n = {int(n): (None if s is None else int(s)) for n, s in zip(viz_n, succ_line)}
futi_max_by_n = {int(n): int(f) if isinstance(f, (int, np.integer)) and f >= 0 else -1 for n, f in zip(viz_n, fut_line)}
safety_req_by_n = {int(n): (None if s is None else int(s)) for n, s in zip(viz_n_safety, saf_line)}

# Efficacy corridor figure
fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=viz_n, y=[np.nan if s is None else s for s in succ_line],
                              name="Success Boundary", line=dict(color='green', dash='dash')))
fig_corr.add_trace(go.Scatter(x=viz_n, y=[np.nan if f < 0 else f for f in fut_line],
                              name="Futility Boundary", line=dict(color='red', dash='dash')))
fig_corr.add_trace(go.Scatter(x=[total_n], y=[successes], mode='markers+text', text=["Current"],
                              name="Current Efficacy", marker=dict(size=12, color='blue')))
fig_corr.update_layout(xaxis_title="Sample Size (N)", yaxis_title="Successes (S)", height=400, margin=dict(t=20, b=0))
st.plotly_chart(fig_corr, use_container_width=True)

# Safety corridor figure
st.subheader("🧯 Safety Decision Corridor")
fig_safety_corr = go.Figure()
fig_safety_corr.add_trace(go.Scatter(
    x=viz_n_safety,
    y=[np.nan if s is None else s for s in saf_line],
    name="Safety Stop Boundary (SAEs ≥)",
    line=dict(color='black', dash='dot')
))
fig_safety_corr.add_trace(go.Scatter(
    x=[total_n], y=[saes],
    mode='markers+text', text=["Current"],
    name="Current Safety", marker=dict(size=12, color='black')
))
fig_safety_corr.update_layout(
    xaxis_title="Sample Size (N)", yaxis_title="SAE count (to trigger stop)",
    height=400, margin=dict(t=20, b=0)
)
st.plotly_chart(fig_safety_corr, use_container_width=True)

# ==============================================================
# 7. POSTERIOR VISUALS — PDFs with 95% CI shading and optional heatmap
# ==============================================================
st.subheader("Statistical Distributions (95% CI Shaded)")
x = np.linspace(0, 1, 500)
fig = go.Figure()
# Efficacy PDF + shaded CI
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_eff, b_eff), name="Efficacy Belief",
                         line=dict(color='#2980b9', width=3)))
x_ci_e = np.linspace(eff_ci[0], eff_ci[1], 100)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_ci_e, x_ci_e[::-1]]),
    y=np.concatenate([beta.pdf(x_ci_e, a_eff, b_eff), np.zeros(100)]),
    fill='toself', fillcolor='rgba(41, 128, 185, 0.2)',
    line=dict(color='rgba(255,255,255,0)'), showlegend=False
))
# Safety PDF + shaded CI
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_saf, b_saf), name="Safety Belief",
                         line=dict(color='#c0392b', width=3)))
x_ci_s = np.linspace(saf_ci[0], saf_ci[1], 100)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_ci_s, x_ci_s[::-1]]),
    y=np.concatenate([beta.pdf(x_ci_s, a_saf, b_saf), np.zeros(100)]),
    fill='toself', fillcolor='rgba(192, 57, 43, 0.2)',
    line=dict(color='rgba(255,255,255,0)'), showlegend=False
))
fig.add_vline(x=target_eff, line_dash="dash", line_color="green", annotation_text="Target")
fig.add_vline(x=safe_limit, line_dash="dash", line_color="black", annotation_text="Safety Limit")
fig.update_layout(xaxis=dict(range=[0, 1]), height=400, margin=dict(l=0, r=0, t=50, b=0))
st.plotly_chart(fig, use_container_width=True)

# Optional risk–benefit heatmap (illustrative utility)
if include_heatmap:
    st.subheader("⚖️ Risk-Benefit Trade-off Heatmap (Illustrative)")
    eff_grid, saf_grid = np.linspace(0.2, 0.9, 50), np.linspace(0.0, 0.4, 50)
    w_tox = st.slider("Heatmap toxicity weight (w)", 0.5, 5.0, 2.0, step=0.5,
                      help="Illustrative utility: efficacy − w×toxicity")
    E, S = np.meshgrid(eff_grid, saf_grid)
    score = E - (w_tox * S)
    fig_heat = px.imshow(score, x=eff_grid, y=saf_grid,
                         labels=dict(x="Efficacy Rate", y="SAE Rate", color="Benefit Score"),
                         color_continuous_scale="RdYlGn", origin="lower")
    fig_heat.add_trace(go.Scatter(x=[eff_mean], y=[saf_mean], mode='markers+text',
                                  text=["Current"], marker=dict(color='white', size=12, symbol='x')))
    st.plotly_chart(fig_heat, use_container_width=True)

# ==============================================================
# 8. FULL BREAKDOWN — Efficacy & Safety summaries and Sensitivity analysis
# ==============================================================
with st.expander("📊 Full Statistical Breakdown", expanded=True):
    c1, c2, c3 = st.columns(3)
    # Efficacy summary
    with c1:
        st.markdown("**Efficacy Summary**")
        st.write(f"Mean Efficacy: **{eff_mean:.1%}**")
        st.write(f"95% CI: **[{eff_ci[0]:.1%} - {eff_ci[1]:.1%}]**")
        st.write(f"95% CI width (precision): **{(eff_ci[1]-eff_ci[0]):.1%}**")
        st.write(f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}**")
        st.write(f"Prob > Target ({target_eff:.0%}): **{p_target:.1%}**")
        st.write(f"Prob > Goal ({dream_eff:.0%}): **{p_goal:.1%}**")
        st.write(f"Prob Equivalence: **{p_equiv:.1%}** (band: [{lb:.0%}, {ub:.0%}])")
        st.write(f"Projected Success Range: **{ps_range[0]} - {ps_range[1]} successes**")
    # Safety summary
    with c2:
        st.markdown("**Safety Summary (Binary SAE)**")
        st.write(f"Mean Toxicity: **{saf_mean:.1%}**")
        st.write(f"95% CI: **[{saf_ci[0]:.1%} - {saf_ci[1]:.1%}]**")
        st.write(f"95% CI width (precision): **{(saf_ci[1]-saf_ci[0]):.1%}**")
        st.write(f"Prob > Limit ({safe_limit:.0%}): **{p_toxic:.1%}**")
        st.caption("Safety is modeled as binary per patient: ≥1 SAE ⇒ SAE=1.")
    # Operational info
    with c3:
        st.markdown("**Operational Info**")
        st.write(f"BPP Success Forecast: **{bpp:.1%}**  (SE={bpp_se:.3f}, 95% CI [{bpp_ci_low:.1%}, {bpp_ci_high:.1%}])")
        st.write(f"PPoS (Predicted Prob): **{bpp:.1%}**")
        st.write(f"Posterior pseudo-count (efficacy): **{a_eff + b_eff:.1f}**")
        st.write(f"Posterior pseudo-count (safety): **{a_saf + b_saf:.1f}**")
        st.write(f"Efficacy Look Points: **N = {', '.join(map(str, look_points)) or 'None'}**")
        st.write(f"Safety Look Points: **N = {', '.join(map(str, safety_look_points)) or 'None'}**")

# --- Efficacy sensitivity with Bayes factors (point null at p0) ---
st.subheader("🧪 Efficacy Sensitivity & Robustness (with Bayes Factors)")

def bayes_factor_point_null(s, n, a, b, p0, eps=1e-12):
    """BF10 = marginal_likelihood(H1=Beta(a,b))/likelihood(H0=point p0)."""
    p0 = float(np.clip(p0, eps, 1.0 - eps))
    successes_local = s
    failures_local = n - s
    log_m1 = betaln(a + successes_local, b + failures_local) - betaln(a, b)
    log_m0 = successes_local * np.log(p0) + failures_local * np.log(1 - p0)
    return float(np.exp(log_m1 - log_m0))

eff_sens_list = [
    (f"Efficacy S1 (α={eff1_a:.1f}, β={eff1_b:.1f})", eff1_a, eff1_b, "#27ae60"),
    (f"Efficacy S2 (α={eff2_a:.1f}, β={eff2_b:.1f})", eff2_a, eff2_b, "#34495e"),
    (f"Efficacy S3 (α={eff3_a:.1f}, β={eff3_b:.1f})", eff3_a, eff3_b, "#8e44ad"),
]
cols, target_probs = st.columns(3), []
sens_rows = []
for i, (name, ap, bp, color) in enumerate(eff_sens_list):
    ae_s, be_s = ap + successes, bp + (total_n - successes)
    m_eff_s = ae_s / (ae_s + be_s)
    p_n_s = 1 - beta.cdf(null_eff, ae_s, be_s)
    p_t_s = 1 - beta.cdf(target_eff, ae_s, be_s)
    p_g_s = 1 - beta.cdf(dream_eff, ae_s, be_s)
    bf10 = bayes_factor_point_null(successes, total_n, ap, bp, null_eff)
    target_probs.append(p_t_s)
    sens_rows.append({
        "Prior": name, "color": color,
        "a": ae_s, "b": be_s,
        "Mean": m_eff_s,
        "P>Null": p_n_s, "P>Target": p_t_s, "P>Goal": p_g_s,
        "BF10": bf10
    })
    with cols[i]:
        st.info(f"**{name}**")
        st.write(f"Mean Efficacy: **{m_eff_s:.1%}**")
        st.write(f"Prob > Null: **{p_n_s:.1%}**")
        st.write(f"Prob > Target: **{p_t_s:.1%}**")
        st.write(f"Prob > Goal: **{p_g_s:.1%}**")
        bf_str = f"{bf10:.2e}" if (bf10 >= 1e4 or bf10 <= 1e-4) else f"{bf10:.2f}"
        st.write(f"Bayes Factor BF₁₀ (H1 vs H0 p0): **{bf_str}**")

st.caption("Interpretation: BF₁₀>1 favors H1; >10 is strong evidence. BF₁₀<1/10 is strong evidence for H0.")
spread = max(target_probs) - min(target_probs) if target_probs else 0.0
st.markdown(f"**Interpretation (efficacy):** Results are **{'ROBUST' if spread < 0.15 else 'SENSITIVE'}** "
            f"({spread:.1%} variance in P(>target) across efficacy priors).")

# PDFs and bar summaries for sensitivity priors
st.subheader("🎛️ Efficacy Sensitivity: Posterior Distributions")
x_grid = np.linspace(0, 1, 600)
fig_sens_pdf = go.Figure()
for row in sens_rows:
    fig_sens_pdf.add_trace(go.Scatter(
        x=x_grid, y=beta.pdf(x_grid, row["a"], row["b"]),
        name=row["Prior"], line=dict(width=3, color=row["color"])
    ))
fig_sens_pdf.add_vline(x=null_eff, line_dash="dot", line_color="#7f8c8d", annotation_text="Null (p0)")
fig_sens_pdf.add_vline(x=target_eff, line_dash="dash", line_color="#2ecc71", annotation_text="Target")
fig_sens_pdf.add_vline(x=dream_eff, line_dash="dash", line_color="#f39c12", annotation_text="Goal")
fig_sens_pdf.update_layout(xaxis_title="Efficacy rate", yaxis_title="Posterior density",
                           height=420, margin=dict(t=20, b=0))
st.plotly_chart(fig_sens_pdf, use_container_width=True)

st.subheader("🎛️ Efficacy Sensitivity: Key Probabilities by Prior")
prob_df = pd.DataFrame([{
    "Prior": r["Prior"], "P>Null": r["P>Null"], "P>Target": r["P>Target"], "P>Goal": r["P>Goal"]
} for r in sens_rows])
prob_df_long = prob_df.melt(id_vars="Prior", var_name="Metric", value_name="Probability")
fig_sens_bars = px.bar(
    prob_df_long, x="Prior", y="Probability", color="Metric",
    barmode="group", text="Probability",
    color_discrete_map={"P>Null": "#1abc9c", "P>Target": "#2980b9", "P>Goal": "#e67e22"}
)
fig_sens_bars.update_traces(texttemplate="%{text:.1%}", textposition="outside")
fig_sens_bars.update_yaxes(tickformat=".0%")
fig_sens_bars.update_layout(height=420, margin=dict(t=20, b=0))
st.plotly_chart(fig_sens_bars, use_container_width=True)

# ==============================================================

st.subheader("🧪 Safety Sensitivity & Robustness (with Bayes Factors)")
# Build safety sensitivity prior list
saf_sens_list = [
    (f"Safety S1 (α={saf1_a:.1f}, β={saf1_b:.1f})", saf1_a, saf1_b, "#e74c3c"),
    (f"Safety S2 (α={saf2_a:.1f}, β={saf2_b:.1f})", saf2_a, saf2_b, "#2c3e50"),
    (f"Safety S3 (α={saf3_a:.1f}, β={saf3_b:.1f})", saf3_a, saf3_b, "#8e44ad"),
]
s_cols, saf_target_probs = st.columns(3), []
saf_rows = []
for i, (name, ap, bp, color) in enumerate(saf_sens_list):
    as_ = ap + saes
    bs_ = bp + (total_n - saes)
    m_saf = as_ / (as_ + bs_)
    p_gt_lim = 1 - beta.cdf(safe_limit, as_, bs_)  # Pr(toxicity > limit)
    p_lt_lim = beta.cdf(safe_limit, as_, bs_)     # Pr(toxicity <= limit)
    bf10_s = bayes_factor_point_null(saes, total_n, ap, bp, safe_limit)
    saf_target_probs.append(p_gt_lim)
    saf_rows.append({
        'Prior': name, 'color': color, 'a': as_, 'b': bs_,
        'MeanTox': m_saf, 'P>ToxLimit': p_gt_lim, 'P≤ToxLimit': p_lt_lim, 'BF10_vs_point_limit': bf10_s
    })
    with s_cols[i]:
        st.info(f"**{name}**")
        st.write(f"Mean Toxicity: **{m_saf:.1%}**")
        st.write(f"Prob > Limit: **{p_gt_lim:.1%}**")
        st.write(f"Prob ≤ Limit: **{p_lt_lim:.1%}**")
        bf_s = f"{bf10_s:.2e}" if (bf10_s >= 1e4 or bf10_s <= 1e-4) else f"{bf10_s:.2f}"
        st.write(f"Bayes Factor BF₁₀ (Alt vs point at limit): **{bf_s}**")
st.caption("Interpretation: larger P>limit suggests higher safety risk; BF₁₀>1 favors the alternative vs a point null at the safety limit.")
# PDFs overlay
st.subheader("🏛️ Safety Sensitivity: Posterior Distributions")
x_grid_s = np.linspace(0, 1, 600)
fig_sens_s_pdf = go.Figure()
for row in saf_rows:
    fig_sens_s_pdf.add_trace(go.Scatter(
        x=x_grid_s, y=beta.pdf(x_grid_s, row['a'], row['b']),
        name=row['Prior'], line=dict(width=3, color=row['color'])
    ))
fig_sens_s_pdf.add_vline(x=safe_limit, line_dash='dash', line_color='black', annotation_text='Safety limit')
fig_sens_s_pdf.update_layout(xaxis_title='Toxicity rate', yaxis_title='Posterior density', height=420, margin=dict(t=20,b=0))
st.plotly_chart(fig_sens_s_pdf, use_container_width=True)
# Bars for P>limit and P<=limit
st.subheader("🏛️ Safety Sensitivity: Key Probabilities by Prior")
sprob_df = pd.DataFrame([{
    'Prior': r['Prior'], 'P>ToxLimit': r['P>ToxLimit'], 'P≤ToxLimit': r['P≤ToxLimit']
} for r in saf_rows])
sprob_long = sprob_df.melt(id_vars='Prior', var_name='Metric', value_name='Probability')
fig_sens_s_bars = px.bar(sprob_long, x='Prior', y='Probability', color='Metric', barmode='group', text='Probability',
    color_discrete_map={'P>ToxLimit':'#c0392b','P≤ToxLimit':'#27ae60'})
fig_sens_s_bars.update_traces(texttemplate='%{text:.1%}', textposition='outside')
fig_sens_s_bars.update_yaxes(tickformat='.0%')
fig_sens_s_bars.update_layout(height=420, margin=dict(t=20,b=0))
st.plotly_chart(fig_sens_s_bars, use_container_width=True)

# 9. DESIGN INTEGRITY CHECK — Sequential Type I Error (with toggles)
#    Simulates trials under H0 (p_eff=p0) and user-defined safety rate.
# ==============================================================
st.markdown("---")
st.subheader("🧪 Design Integrity Check")

num_sims = 10000  # default shown; now user-configurable below
with st.expander("Simulation Options", expanded=False):
    sim_safety_rate = st.slider(
        "Assumed TRUE SAE rate for Type I simulation", 0.0, 0.9, float(safe_limit), step=0.01,
        help="Used to simulate SAEs while testing Type I (efficacy false positive) under p=p0."
    )
    sim_seed = st.number_input("Random seed (Type I sim)", 0, 10_000_000, 7, step=1)
    typei_sims = st.number_input(
        "Number of Type I simulations", 1000, 200000, num_sims, step=1000,
        help="Monte Carlo trials for Type I error estimate."
    )
    # Separate toggles for look schedules during Type I calculation
    typei_use_eff_looks = st.checkbox(
        "Use user-defined efficacy look schedule",
        value=True,
        help="If unchecked, evaluate efficacy at every N from 1..max N (continuous)."
    )
    typei_use_saf_looks = st.checkbox(
        "Use user-defined safety look schedule",
        value=True,
        help="If unchecked, evaluate safety at every N from 1..max N (continuous; safety gating ignored)."
    )

if st.button(f"Calculate Sequential Type I Error ({typei_sims:,} sims)"):
    if (typei_use_eff_looks and typei_use_saf_looks and len(look_points) == 0 and len(safety_look_points) == 0):
        st.warning("No scheduled interim looks based on current settings; Type I error is not computed.")
    else:
        with st.spinner(f"Simulating {typei_sims:,} trials..."):
            rng = np.random.default_rng(sim_seed)
            # Build efficacy look set
            eff_looks = set(look_points) if typei_use_eff_looks else set(range(1, max_n_val + 1))
            eff_looks.add(max_n_val)  # always include final look
            # Build safety look set
            saf_looks = set(safety_look_points) if typei_use_saf_looks else set(range(1, max_n_val + 1))
            all_looks = sorted(eff_looks.union(saf_looks))
            # Safety check set (gating logic respected when scheduled)
            if typei_use_saf_looks:
                safety_check_set = saf_looks if safety_gate_to_schedule else set(all_looks)
            else:
                safety_check_set = set(range(1, max_n_val + 1))
            # Precompute thresholds
            succ_req_sim = dict(succ_req_by_n)
            futi_max_sim = dict(futi_max_by_n)
            safety_req_sim = dict(safety_req_by_n)
            for lp in eff_looks:
                use_conf = success_conf_req_final if (lp == max_n_val) else success_conf_req_interim
                if lp not in succ_req_sim:
                    succ_req_sim[lp] = success_boundary(lp, prior_alpha, prior_beta, target_eff, use_conf)
                if lp != max_n_val and (lp not in futi_max_sim):
                    futi_max_sim[lp] = futility_boundary_ppos(lp, max_n_val, target_eff, success_conf_req_final,
                                                             prior_alpha, prior_beta, mc_draws, mc_seed, bpp_futility_limit)
            for lp in saf_looks:
                if lp not in safety_req_sim:
                    safety_req_sim[lp] = safety_stop_threshold(lp, prior_alpha_saf, prior_beta_saf, safe_limit, safe_conf_req)
            # Simulate trials under H0 (p_eff = p0)
            fp_count = 0
            for _ in range(typei_sims):
                trial_eff = rng.binomial(1, null_eff, max_n_val)
                trial_saf = rng.binomial(1, sim_safety_rate, max_n_val)
                for lp in all_looks:
                    s = int(trial_eff[:lp].sum())
                    t = int(trial_saf[:lp].sum())
                    # Safety stop preempts any efficacy decisions
                    if lp in safety_check_set:
                        saf_thr = safety_req_sim.get(lp, None)
                        if saf_thr is not None and t >= saf_thr:
                            break
                    # Futility at interim preempts success
                    if lp in eff_looks:
                        if lp != max_n_val:
                            fut_thr = futi_max_sim.get(lp, -1)
                            if fut_thr >= 0 and s <= fut_thr:
                                break
                        # Success
                        suc_thr = succ_req_sim.get(lp, None)
                        if suc_thr is not None and s >= suc_thr:
                            fp_count += 1
                            break
            type_i_estimate = fp_count / typei_sims
            ci_lo, ci_hi = wilson_ci(fp_count, typei_sims)
            st.warning(
                f"Estimated Sequential Type I Error (with safety & futility): **{type_i_estimate:.2%}** "
                f"(95% CI {ci_lo:.2%}–{ci_hi:.2%})"
            )
            st.caption("Alpha respects the toggles: using user look schedules or continuous checks at every N.")

# ==============================================================
# 10. POWER ANALYSIS — Operating characteristics (single point + curve)
#     Uses same decision logic and schedules (or continuous) per toggles.
# ==============================================================
st.markdown("---")
st.subheader("📐 Power Analysis (Operating Characteristics)")

@st.cache_data
def simulate_power_once(p_true_eff, p_true_saf, sims, seed,
                        max_n_val,
                        look_points, safety_look_points, safety_gate_to_schedule,
                        prior_alpha, prior_beta, target_eff,
                        success_conf_req_interim, success_conf_req_final,
                        prior_alpha_saf, prior_beta_saf, safe_limit, safe_conf_req,
                        succ_req_by_n, futi_max_by_n, safety_req_by_n,
                        mc_draws, mc_seed, bpp_futility_limit,
                        power_use_eff_looks, power_use_saf_looks):
    """Simulate trials at given true rates to estimate power and expected N."""
    rng = np.random.default_rng(seed)
    # Build look sets according to toggles
    eff_looks = set(look_points) if power_use_eff_looks else set(range(1, max_n_val + 1))
    eff_looks.add(max_n_val)
    saf_looks = set(safety_look_points) if power_use_saf_looks else set(range(1, max_n_val + 1))
    all_looks = sorted(eff_looks.union(saf_looks))
    # Safety check locus (respect gating only for scheduled safety)
    if power_use_saf_looks:
        safety_check_set = saf_looks if safety_gate_to_schedule else set(all_looks)
    else:
        safety_check_set = set(range(1, max_n_val + 1))
    # Precompute thresholds if needed
    succ_req_sim = dict(succ_req_by_n)
    futi_max_sim = dict(futi_max_by_n)
    safety_req_sim = dict(safety_req_by_n)
    for lp in eff_looks:
        use_conf = success_conf_req_final if (lp == max_n_val) else success_conf_req_interim
        if lp not in succ_req_sim:
            succ_req_sim[lp] = success_boundary(lp, prior_alpha, prior_beta, target_eff, use_conf)
        if lp != max_n_val and (lp not in futi_max_sim):
            futi_max_sim[lp] = futility_boundary_ppos(lp, max_n_val, target_eff, success_conf_req_final,
                                                      prior_alpha, prior_beta, mc_draws, mc_seed, bpp_futility_limit)
    for lp in saf_looks:
        if lp not in safety_req_sim:
            safety_req_sim[lp] = safety_stop_threshold(lp, prior_alpha_saf, prior_beta_saf, safe_limit, safe_conf_req)
    # Run sims
    success_count = 0
    stop_n_list = []
    stop_reason_counts = {"safety": 0, "futility": 0, "interim_success": 0, "final_success": 0, "no_decision": 0}
    for _ in range(sims):
        trial_eff = rng.binomial(1, p_true_eff, max_n_val)
        trial_saf = rng.binomial(1, p_true_saf, max_n_val)
        decided = False
        for lp in all_looks:
            s = int(trial_eff[:lp].sum())
            t = int(trial_saf[:lp].sum())
            # Safety stop
            if lp in safety_check_set:
                saf_thr = safety_req_sim.get(lp, None)
                if saf_thr is not None and t >= saf_thr:
                    stop_reason_counts["safety"] += 1
                    stop_n_list.append(lp)
                    decided = True
                    break
            # Efficacy/futility
            if lp in eff_looks:
                if lp != max_n_val:
                    fut_thr = futi_max_sim.get(lp, -1)
                    if fut_thr >= 0 and s <= fut_thr:
                        stop_reason_counts["futility"] += 1
                        stop_n_list.append(lp)
                        decided = True
                        break
                suc_thr = succ_req_sim.get(lp, None)
                if suc_thr is not None and s >= suc_thr:
                    success_count += 1
                    stop_reason_counts["final_success" if lp == max_n_val else "interim_success"] += 1
                    stop_n_list.append(lp)
                    decided = True
                    break
        if not decided:
            stop_reason_counts["no_decision"] += 1
            stop_n_list.append(max_n_val)
    power_est = success_count / sims
    exp_n = float(np.mean(stop_n_list)) if stop_n_list else float(max_n_val)
    return power_est, exp_n, stop_reason_counts

# --- Single-point power estimate at chosen true rates ---
power_est, power_exp_n, power_reasons = simulate_power_once(
    power_true_eff, power_true_saf, power_sims, power_seed,
    max_n_val,
    look_points, safety_look_points, safety_gate_to_schedule,
    prior_alpha, prior_beta, target_eff,
    success_conf_req_interim, success_conf_req_final,
    prior_alpha_saf, prior_beta_saf, safe_limit, safe_conf_req,
    succ_req_by_n, futi_max_by_n, safety_req_by_n,
    mc_draws, mc_seed, bpp_futility_limit,
    power_use_eff_looks, power_use_saf_looks
)

# Wilson 95% CI around Monte Carlo power proportion
c_pow_lo, c_pow_hi = wilson_ci(power_reasons['interim_success'] + power_reasons['final_success'], power_sims)

# Present the main OC metrics
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Power (Pr[declare success])", f"{power_est:.1%}",
              help=f"Monte Carlo 95% CI {c_pow_lo:.1%}–{c_pow_hi:.1%}")
with c2:
    st.metric("Expected sample size at stop", f"{power_exp_n:.1f}")
with c3:
    st.write("Stop reasons:")
    st.write(f"- Safety stop: **{power_reasons['safety'] / power_sims:.1%}**")
    st.write(f"- Futility stop: **{power_reasons['futility'] / power_sims:.1%}**")
    st.write(f"- Interim success: **{power_reasons['interim_success'] / power_sims:.1%}**")
    st.write(f"- Final success: **{power_reasons['final_success'] / power_sims:.1%}**")
    st.write(f"- No decision at looks (ended at max N): **{power_reasons['no_decision'] / power_sims:.1%}**")

# Add stop-reason legend to the Power section (as requested)
st.caption(
    "**Stop reason legend:**\n"
    "\n- **Safety stop**: at a safety look (or continuously if selected), cumulative SAEs reach or exceed the safety stop threshold for that N."
    "\n- **Futility stop**: at an interim efficacy look, observed successes are at/below the futility boundary derived from the PPoS floor."
    "\n- **Interim success**: observed successes meet/exceed the success boundary at an interim look."
    "\n- **Final success**: observed successes meet/exceed the success boundary at the final look (N = max N)."
    "\n- **No decision**: no stop criteria were met; the trial reached max N without declaring success or triggering safety/futility."
)

# --- Power curve vs true efficacy (using same toggles for looks) ---
st.subheader("📐 Power Curve vs True Efficacy")
@st.cache_data
def power_curve(p0, p_high, points, sims_per_point, seed_base,
                max_n_val, look_points, safety_look_points, safety_gate_to_schedule,
                prior_alpha, prior_beta, target_eff,
                success_conf_req_interim, success_conf_req_final,
                prior_alpha_saf, prior_beta_saf, safe_limit, safe_conf_req,
                succ_req_by_n, futi_max_by_n, safety_req_by_n,
                true_saf_rate, mc_draws, mc_seed, bpp_futility_limit,
                power_use_eff_looks, power_use_saf_looks):
    grid = np.linspace(p0, p_high, points)
    curve = []
    for i, p_eff in enumerate(grid):
        seed = int(seed_base + i * 9973)
        pw, expN, _ = simulate_power_once(
            p_eff, true_saf_rate, sims_per_point, seed,
            max_n_val, look_points, safety_look_points, safety_gate_to_schedule,
            prior_alpha, prior_beta, target_eff,
            success_conf_req_interim, success_conf_req_final,
            prior_alpha_saf, prior_beta_saf, safe_limit, safe_conf_req,
            succ_req_by_n, futi_max_by_n, safety_req_by_n,
            mc_draws, mc_seed, bpp_futility_limit,
            power_use_eff_looks, power_use_saf_looks
        )
        curve.append({"True efficacy": p_eff, "Power": pw, "Expected N": expN})
    return pd.DataFrame(curve)

power_df = power_curve(
    null_eff, 0.90, power_curve_points, power_curve_sims, power_seed,
    max_n_val, look_points, safety_look_points, safety_gate_to_schedule,
    prior_alpha, prior_beta, target_eff,
    success_conf_req_interim, success_conf_req_final,
    prior_alpha_saf, prior_beta_saf, safe_limit, safe_conf_req,
    succ_req_by_n, futi_max_by_n, safety_req_by_n,
    power_true_saf, mc_draws, mc_seed, bpp_futility_limit,
    power_use_eff_looks, power_use_saf_looks
)

fig_power = go.Figure()
fig_power.add_trace(go.Scatter(
    x=power_df["True efficacy"], y=power_df["Power"],
    name="Power", mode="lines+markers", line=dict(color="#2c3e50", width=3)
))
fig_power.add_vline(x=null_eff, line_dash="dot", line_color="#7f8c8d", annotation_text="Null (p0)")
fig_power.add_vline(x=target_eff, line_dash="dash", line_color="#2ecc71", annotation_text="Target (p1)")
fig_power.update_layout(xaxis_title="True efficacy rate", yaxis_title="Power",
                        yaxis=dict(range=[0,1]), height=420, margin=dict(t=20, b=0))
st.plotly_chart(fig_power, use_container_width=True)

fig_expN = go.Figure()
fig_expN.add_trace(go.Scatter(
    x=power_df["True efficacy"], y=power_df["Expected N"],
    name="Expected N at stop", mode="lines+markers", line=dict(color="#8e44ad", width=3)
))
fig_expN.update_layout(xaxis_title="True efficacy rate", yaxis_title="Expected sample size at stop",
                       height=380, margin=dict(t=20, b=0))
st.plotly_chart(fig_expN, use_container_width=True)

# ==============================================================
# 10B. ADAPTIVE SCENARIO SUITE — On-demand OC across multiple scenarios
#      (Now with fixed colors for stop-reason stacked bars)
# ==============================================================
st.markdown("---")
st.subheader("📊 Adaptive Scenario Suite — Operating Characteristics (OC)")

with st.expander("Scenario Suite Options", expanded=False):
    oc_sims = st.number_input("Simulations per scenario", 1000, 200000, max(20000, power_sims), step=1000,
                              help="Monte Carlo trials per scenario.")
    oc_seed = st.number_input("Random seed (OC)", 0, 10_000_000, max(123, power_seed), step=1)
    oc_use_eff_looks = st.checkbox("Use user-defined efficacy look schedule (OC)", value=power_use_eff_looks)
    oc_use_saf_looks = st.checkbox("Use user-defined safety look schedule (OC)", value=power_use_saf_looks)
    include_borderline = st.checkbox("Include borderline scenarios (near thresholds)", value=True)

# Small helpers to construct scenarios adaptively from p0/p1 and safety limit

def _clip(x, lo, hi):
    return float(max(lo, min(hi, x)))

def build_adaptive_scenarios(null_eff, target_eff, dream_eff, safe_limit, include_borderline=True):
    scenarios = []
    eff_null = _clip(null_eff, 0.01, 0.99)
    eff_target = _clip(target_eff, 0.01, 0.99)
    eff_dream = _clip(dream_eff, 0.01, 0.99)
    eff_low = _clip(null_eff - 0.10, 0.01, 0.99)         # clearly futile
    eff_high = _clip(max(target_eff + 0.10, eff_dream), 0.01, 0.99)  # strong efficacy
    eff_mid = _clip((null_eff + target_eff)/2.0, 0.01, 0.99)        # marginal efficacy

    saf_border = _clip(safe_limit, 0.0, 0.9)             # borderline safety
    saf_low = _clip(safe_limit/2.0, 0.0, 0.9)            # reassuring safety
    saf_high = _clip(safe_limit*1.5, 0.0, 0.9)           # high toxicity
    saf_very_high = _clip(safe_limit + 0.10, 0.0, 0.9)   # very high toxicity

    scenarios += [
        {"name": "Null efficacy, acceptable safety", "p_eff": eff_null,   "p_saf": saf_low},
        {"name": "Target efficacy, acceptable safety", "p_eff": eff_target, "p_saf": saf_low},
        {"name": "High efficacy, acceptable safety",  "p_eff": eff_high,  "p_saf": saf_low},
        {"name": "Futile efficacy, acceptable safety","p_eff": eff_low,   "p_saf": saf_low},
        {"name": "Null efficacy, borderline safety",  "p_eff": eff_null,   "p_saf": saf_border},
        {"name": "Target efficacy, borderline safety","p_eff": eff_target, "p_saf": saf_border},
        {"name": "High efficacy, high toxicity",      "p_eff": eff_high,  "p_saf": saf_high},
        {"name": "Futile efficacy, high toxicity",    "p_eff": eff_low,   "p_saf": saf_high},
        {"name": "Marginal efficacy, acceptable safety","p_eff": eff_mid,  "p_saf": saf_low},
        {"name": "Marginal efficacy, borderline safety","p_eff": eff_mid,  "p_saf": saf_border},
        {"name": "Marginal efficacy, high toxicity",   "p_eff": eff_mid,  "p_saf": saf_high},
    ]
    if include_borderline:
        scenarios += [
            {"name": "Target efficacy, very high toxicity", "p_eff": eff_target, "p_saf": saf_very_high},
            {"name": "Null efficacy, very high toxicity",   "p_eff": eff_null,   "p_saf": saf_very_high},
        ]
    # Deduplicate and clip
    out, seen = [], set()
    for sc in scenarios:
        key = (sc["name"], round(sc["p_eff"], 4), round(sc["p_saf"], 4))
        if key in seen:
            continue
        seen.add(key)
        sc["p_eff"] = _clip(sc["p_eff"], 0.01, 0.99)
        sc["p_saf"] = _clip(sc["p_saf"], 0.0, 0.9)
        out.append(sc)
    return out

# Execute scenario suite when button is pressed
if st.button("▶️ Run Adaptive Scenario Suite (OC)"):
    scenarios = build_adaptive_scenarios(null_eff, target_eff, dream_eff, safe_limit, include_borderline)
    st.info(f"Running {len(scenarios)} scenarios @ {oc_sims:,} sims each…")

    # Prepare look sets once per run
    eff_looks = set(look_points) if oc_use_eff_looks else set(range(1, max_n_val + 1))
    eff_looks.add(max_n_val)
    saf_looks = set(safety_look_points) if oc_use_saf_looks else set(range(1, max_n_val + 1))
    all_looks = sorted(eff_looks.union(saf_looks))
    safety_check_set_sched = saf_looks if safety_gate_to_schedule else set(all_looks)
    safety_check_set_cont = set(range(1, max_n_val + 1))
    safety_check_set = safety_check_set_sched if oc_use_saf_looks else safety_check_set_cont

    # Threshold caches
    succ_req_sim = dict(succ_req_by_n)
    futi_max_sim = dict(futi_max_by_n)
    safety_req_sim = dict(safety_req_by_n)
    for lp in eff_looks:
        use_conf = success_conf_req_final if (lp == max_n_val) else success_conf_req_interim
        if lp not in succ_req_sim:
            succ_req_sim[lp] = success_boundary(lp, prior_alpha, prior_beta, target_eff, use_conf)
        if lp != max_n_val and (lp not in futi_max_sim):
            futi_max_sim[lp] = futility_boundary_ppos(lp, max_n_val, target_eff, success_conf_req_final,
                                                      prior_alpha, prior_beta, mc_draws, mc_seed, bpp_futility_limit)
    for lp in saf_looks:
        if lp not in safety_req_sim:
            safety_req_sim[lp] = safety_stop_threshold(lp, prior_alpha_saf, prior_beta_saf, safe_limit, safe_conf_req)

    rows = []
    # Run each scenario with distinct RNG seed offset
    for i, sc in enumerate(scenarios):
        rng = np.random.default_rng(int(oc_seed + i*9973))
        success_count = 0
        stop_n_list = []
        stop_reason_counts = {"safety": 0, "futility": 0, "interim_success": 0, "final_success": 0, "no_decision": 0}
        p_true_eff = sc["p_eff"]
        p_true_saf = sc["p_saf"]
        for _ in range(oc_sims):
            trial_eff = rng.binomial(1, p_true_eff, max_n_val)
            trial_saf = rng.binomial(1, p_true_saf, max_n_val)
            decided = False
            for lp in all_looks:
                s = int(trial_eff[:lp].sum())
                t = int(trial_saf[:lp].sum())
                if lp in safety_check_set:
                    saf_thr = safety_req_sim.get(lp, None)
                    if saf_thr is not None and t >= saf_thr:
                        stop_reason_counts["safety"] += 1
                        stop_n_list.append(lp)
                        decided = True
                        break
                if lp in eff_looks:
                    if lp != max_n_val:
                        fut_thr = futi_max_sim.get(lp, -1)
                        if fut_thr >= 0 and s <= fut_thr:
                            stop_reason_counts["futility"] += 1
                            stop_n_list.append(lp)
                            decided = True
                            break
                    suc_thr = succ_req_sim.get(lp, None)
                    if suc_thr is not None and s >= suc_thr:
                        success_count += 1
                        stop_reason_counts["final_success" if lp == max_n_val else "interim_success"] += 1
                        stop_n_list.append(lp)
                        decided = True
                        break
            if not decided:
                stop_reason_counts["no_decision"] += 1
                stop_n_list.append(max_n_val)
        oc_power = success_count / oc_sims
        oc_pow_lo, oc_pow_hi = wilson_ci(success_count, oc_sims)
        oc_exp_n = float(np.mean(stop_n_list)) if stop_n_list else float(max_n_val)
        rows.append({
            "Scenario": sc["name"],
            "TRUE efficacy": p_true_eff,
            "TRUE SAE": p_true_saf,
            "Power": oc_power,
            "Power_CI_low": oc_pow_lo,
            "Power_CI_high": oc_pow_hi,
            "Expected N": oc_exp_n,
            "Safety stop %": stop_reason_counts['safety']/oc_sims,
            "Futility stop %": stop_reason_counts['futility']/oc_sims,
            "Interim success %": stop_reason_counts['interim_success']/oc_sims,
            "Final success %": stop_reason_counts['final_success']/oc_sims,
            "No decision %": stop_reason_counts['no_decision']/oc_sims,
        })

    suite_df = pd.DataFrame(rows)

    # Pretty table for display (percentages and rounding)
    fmt_df = suite_df.copy()
    for col in ["Power", "Power_CI_low", "Power_CI_high", "Safety stop %", "Futility stop %", "Interim success %", "Final success %", "No decision %"]:
        fmt_df[col] = (fmt_df[col]*100.0).round(1)
    fmt_df["Expected N"] = fmt_df["Expected N"].round(1)

    st.write("**Scenario Suite Summary**")
    st.dataframe(fmt_df)

    # --- Power by scenario (unchanged colors) ---
    fig_suite_power = px.bar(suite_df, x="Scenario", y="Power", text="Power")
    fig_suite_power.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig_suite_power.update_yaxes(tickformat=".0%")
    fig_suite_power.update_layout(height=420, margin=dict(t=20, b=80), xaxis_tickangle=30)
    st.plotly_chart(fig_suite_power, use_container_width=True)

    # --- Stacked stop-reason rates with fixed color mapping as requested ---
    reasons_long = suite_df.melt(id_vars=["Scenario"],
                                 value_vars=["Safety stop %", "Futility stop %", "Interim success %", "Final success %", "No decision %"],
                                 var_name="Reason", value_name="Rate")
    fig_suite_reasons = px.bar(
        reasons_long, x="Scenario", y="Rate", color="Reason", barmode="stack",
        color_discrete_map={
            "Safety stop %": "#e74c3c",   # red
            "Futility stop %": "#f39c12",  # orange
            "Interim success %": "#27ae60",# green
            "Final success %": "#2980b9",  # blue
            "No decision %": "#95a5a6"     # grey
        }
    )
    fig_suite_reasons.update_yaxes(tickformat=".0%")
    fig_suite_reasons.update_layout(height=460, margin=dict(t=20, b=80), xaxis_tickangle=30,
                                    legend_title_text="Stop reason")
    st.plotly_chart(fig_suite_reasons, use_container_width=True)

    # Compact legend (also appears in Power section, for consistency)
    st.caption(
        "**Stop reason legend:**\n"
        "\n- **Safety stop**: at a safety look (or continuously if selected), cumulative SAEs reach or exceed the safety stop threshold for that N."
        "\n- **Futility stop**: at an interim efficacy look, observed successes are at/below the futility boundary derived from the PPoS floor."
        "\n- **Interim success**: observed successes meet/exceed the success boundary at an interim look."
        "\n- **Final success**: observed successes meet/exceed the success boundary at the final look (N = max N)."
        "\n- **No decision**: no stop criteria were met; the trial reached max N without declaring success or triggering safety/futility."
    )

    # CSV export for audit or offline review
    st.download_button(
        label="⬇️ Download Scenario Suite (CSV)",
        data=suite_df.to_csv(index=False).encode("utf-8"),
        file_name=f"OC_scenario_suite_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# ==============================================================
# 11. REGULATORY TABLES — Efficacy/Futility and Safety boundaries by look
# ==============================================================
with st.expander("📋 Regulatory Decision Boundary Tables", expanded=True):
    st.markdown("**Efficacy/Futility Boundaries (by efficacy look schedule)**")
    boundary_data_eff = []
    for lp in look_points:
        if lp <= total_n:
            continue
        use_conf = success_conf_req_final if (lp == max_n_val) else success_conf_req_interim
        s_req = success_boundary(lp, prior_alpha, prior_beta, target_eff, use_conf)
        f_req = futility_boundary_ppos(lp, max_n_val, target_eff, success_conf_req_final,
                                       prior_alpha, prior_beta, mc_draws, mc_seed, bpp_futility_limit)
        boundary_data_eff.append({
            "N (eff)": lp,
            "Success Stop S ≥": s_req if s_req is not None else "No success at this look",
            "Futility Stop S ≤": (f_req if f_req >= 0 else "No stop"),
            "Success threshold used": "Final" if (lp == max_n_val) else "Interim"
        })
    if boundary_data_eff:
        st.table(pd.DataFrame(boundary_data_eff))
    else:
        st.write("No future efficacy/futility looks (or trial is at/final analysis).")

    st.markdown("---")
    st.markdown("**Safety Boundaries (by safety look schedule)**")
    boundary_data_saf = []
    for lp in safety_look_points:
        if lp <= total_n:
            continue
        safe_req = safety_stop_threshold(lp, prior_alpha_saf, prior_beta_saf, safe_limit, safe_conf_req)
        boundary_data_saf.append({
            "N (safety)": lp,
            "Safety Stop SAEs ≥": safe_req if safe_req is not None else "No safety stop at this look"
        })
    if boundary_data_saf:
        st.table(pd.DataFrame(boundary_data_saf))
    else:
        st.write("No future safety looks (or trial is at/final analysis).")

st.markdown("---")

# ==============================================================
# 12. EXPORT / AUDIT SNAPSHOT — One-click CSV of current state & settings
# ==============================================================
if st.button("📥 Prepare Audit-Ready Snapshot"):
    report_data = {
        "Metric": [
            "Timestamp", "N", "Successes", "SAEs",
            "Interim Success Threshold (%)", "Final Success Threshold (%)",
            "Futility Threshold (PPoS floor)", "PPoS", "PPoS SE", "PPoS 95% CI",
            "Prior ESS (eff)", "Prior ESS (saf)",
            "Posterior pseudo-count (eff)", "Posterior pseudo-count (saf)",
            "Efficacy Looks", "Safety Looks", "Safety gated to schedule?",
            "Power (at chosen true rates)", "Power Expected N"
        ],
        "Value": [
            datetime.now().isoformat(), total_n, successes, saes,
            f"{success_conf_req_interim:.1%}", f"{success_conf_req_final:.1%}",
            f"{bpp_futility_limit:.2%}", f"{bpp:.2%}", f"{bpp_se:.4f}", f"[{bpp_ci_low:.1%}, {bpp_ci_high:.1%}]",
            f"{prior_alpha+prior_beta:.1f}", f"{prior_alpha_saf+prior_beta_saf:.1f}",
            f"{a_eff+b_eff:.1f}", f"{a_saf+b_saf:.1f}",
            ", ".join(map(str, look_points)) or "None",
            ", ".join(map(str, safety_look_points)) or "None",
            "Yes" if safety_gate_to_schedule else "No",
            f"{power_est:.2%}", f"{power_exp_n:.1f}"
        ]
    }
    df_report = pd.DataFrame(report_data)
    st.download_button(
        label="Click here to Download CSV",
        data=df_report.to_csv(index=False).encode('utf-8'),
        file_name=f"Trial_Snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime='text/csv'
    )
    st.table(df_report)

