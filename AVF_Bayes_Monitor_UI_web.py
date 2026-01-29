import streamlit as st
import numpy as np
from scipy.stats import beta
from scipy.special import betaln
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Universal Trial Monitor: Airtight Pro", layout="wide")

# ==========================================
# 1. SIDEBAR INPUTS (With Explanatory Tooltips)
# ==========================================
st.sidebar.header("📋 Current Trial Data")

# Trial size and observed data
max_n_val = st.sidebar.number_input(
    "Maximum Sample Size (N)", 10, 500, 70,
    help="The total planned sample size for the trial design."
)
total_n = st.sidebar.number_input(
    "Total Patients Enrolled", 0, max_n_val, 20,
    help="Current number of patients who have generated evaluable data."
)
successes = st.sidebar.number_input(
    "Total Successes", 0, total_n, value=min(14, total_n),
    help="Number of patients achieving the primary efficacy endpoint."
)
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

# Priors (Efficacy)
with st.sidebar.expander("Base Study Priors (Efficacy)", expanded=True):
    prior_alpha = st.slider(
        "Prior Successes (Alpha_eff)", 0.1, 10.0, 1.0, step=0.1,
        help="Initial belief strength for efficacy: equivalent 'phantom' successes."
    )
    prior_beta = st.slider(
        "Prior Failures (Beta_eff)", 0.1, 10.0, 1.0, step=0.1,
        help="Initial belief strength for efficacy: equivalent 'phantom' failures."
    )

# Timing: efficacy/futility schedule
with st.sidebar.expander("Adaptive Timing & Look Points (Efficacy/Futility)", expanded=True):
    min_interim = st.number_input(
        "Min N before first check", 1, max_n_val, 14,
        help="Burn-in period: No stop decisions will be made before this sample size."
    )
    check_cohort = st.number_input(
        "Check every X patients (Cohort)", 1, 20, 5,
        help="Frequency of interim analysis (e.g., check data every 5 patients)."
    )

# Thresholds (Efficacy) — separate interim vs final
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

# Safety rules + Separate Safety Priors + Safety schedule + gating toggle
with st.sidebar.expander("Safety Rules, Priors & Timing", expanded=True):
    safe_limit = st.slider(
        "SAE Upper Limit (%)", 0.05, 0.50, 0.15,
        help="Maximum acceptable toxicity rate."
    )
    safe_conf_req = st.slider(
        "Safety Stop Confidence", 0.5, 0.99, 0.90,
        help="Stop if we are this confident the true toxicity rate exceeds the limit."
    )
    st.markdown("**Safety Priors (independent from efficacy):**")
    prior_alpha_saf = st.slider(
        "Safety Prior Successes (Alpha_safety)", 0.1, 10.0, 1.0, step=0.1,
        help="Initial belief strength for toxicity: equivalent 'phantom' toxic events."
    )
    prior_beta_saf = st.slider(
        "Safety Prior Failures (Beta_safety)", 0.1, 10.0, 1.0, step=0.1,
        help="Initial belief strength for toxicity: equivalent 'phantom' non-toxic outcomes."
    )
    st.markdown("**Safety Look Schedule (optional; defaults preserve original behavior):**")
    safety_min_interim = st.number_input(
        "Safety: Min N before first check", 1, max_n_val, min_interim,
        help="First safety look (often earlier than efficacy)."
    )
    safety_check_cohort = st.number_input(
        "Safety: Check every X patients", 1, 20, check_cohort,
        help="Frequency of safety checks."
    )
    safety_gate_to_schedule = st.checkbox(
        "Apply safety decision only at scheduled safety looks",
        value=False,
        help="If unchecked, safety is monitored continuously (original behavior)."
    )

# Sensitivity & Equivalence
with st.sidebar.expander("Sensitivity Prior Settings (Legacy weights)"):
    opt_p = st.slider("Optimistic Prior Weight (efficacy α)", 1, 10, 4,
                      help="Legacy alpha weight for the Optimistic efficacy sensitivity.")
    skp_p = st.slider("Skeptical Prior Weight (efficacy β)", 1, 10, 4,
                      help="Legacy beta weight for the Skeptical efficacy sensitivity.")

with st.sidebar.expander("Sensitivity Priors (Adjustable) — Efficacy", expanded=True):
    st.caption("Define three efficacy priors (α, β) for sensitivity overlays.")
    eff1_a = st.number_input("Efficacy S1 α", 0.1, 20.0, float(opt_p), 0.1)
    eff1_b = st.number_input("Efficacy S1 β", 0.1, 20.0, 1.0, 0.1)
    eff2_a = st.number_input("Efficacy S2 α", 0.1, 20.0, 1.0, 0.1)
    eff2_b = st.number_input("Efficacy S2 β", 0.1, 20.0, 1.0, 0.1)
    eff3_a = st.number_input("Efficacy S3 α", 0.1, 20.0, 1.0, 0.1)
    eff3_b = st.number_input("Efficacy S3 β", 0.1, 20.0, float(skp_p), 0.1)

with st.sidebar.expander("Sensitivity Priors (Adjustable) — Safety", expanded=True):
    st.caption("Define three safety priors (α, β) for sensitivity overlays.")
    saf1_a = st.number_input("Safety S1 α", 0.1, 20.0, 0.5, 0.1)
    saf1_b = st.number_input("Safety S1 β", 0.1, 20.0, 2.0, 0.1)
    saf2_a = st.number_input("Safety S2 α", 0.1, 20.0, 1.0, 0.1)
    saf2_b = st.number_input("Safety S2 β", 0.1, 20.0, 1.0, 0.1)
    saf3_a = st.number_input("Safety S3 α", 0.1, 20.0, 2.0, 0.1)
    saf3_b = st.number_input("Safety S3 β", 0.1, 20.0, 0.5, 0.1)

with st.sidebar.expander("Equivalence & Heatmap Settings"):
    equiv_bound = st.slider("Equivalence Bound (+/-)", 0.01, 0.10, 0.05,
                            help="Zone around Null Efficacy considered 'Practical Equivalence'.")
    include_heatmap = st.checkbox("Generate Risk-Benefit Heatmap", value=True)
    st.caption("Note: Heatmap utility is illustrative (score = efficacy − w×toxicity).")

# Monte Carlo controls
with st.sidebar.expander("Forecasting Controls (PPoS)", expanded=True):
    mc_draws = st.number_input(
        "Monte Carlo draws for PPoS", 5000, 100000, 20000, step=5000,
        help="Higher draws reduce Monte Carlo noise at boundaries."
    )
    mc_seed = st.number_input(
        "Random seed (PPoS)", 0, 10_000_000, 42, step=1,
        help="Controls reproducibility of PPoS simulations."
    )

# --- Power analysis settings
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
    # NEW: Toggle for using look schedules in POWER sim
    sim_use_look_settings_power = st.checkbox(
        "Simulate using user-defined look schedules (efficacy & safety)",
        value=True,
        help="If off, evaluate safety & efficacy at every N from 1..max N (continuous monitoring)."
    )

# ==========================================
# 2. MATH ENGINE (Bayesian Updates)
# ==========================================
# Posterior = Prior + Data
a_eff, b_eff = prior_alpha + successes, prior_beta + (total_n - successes)
a_saf, b_saf = prior_alpha_saf + saes, prior_beta_saf + (total_n - saes)

# Probability Calculations (CDF = Cumulative Distribution Function)
p_null = 1 - beta.cdf(null_eff, a_eff, b_eff)      # Prob > Null
p_target = 1 - beta.cdf(target_eff, a_eff, b_eff)  # Prob > Target
p_goal = 1 - beta.cdf(dream_eff, a_eff, b_eff)     # Prob > Dream
p_toxic = 1 - beta.cdf(safe_limit, a_saf, b_saf)   # Prob > Safety Limit

# Equivalence around Null (clip to [0,1])
lb, ub = max(0.0, null_eff - equiv_bound), min(1.0, null_eff + equiv_bound)
p_equiv = beta.cdf(ub, a_eff, b_eff) - beta.cdf(lb, a_eff, b_eff)

# Credible Intervals & Means
eff_mean, eff_ci = a_eff / (a_eff + b_eff), beta.ppf([0.025, 0.975], a_eff, b_eff)
saf_mean, saf_ci = a_saf / (a_saf + b_saf), beta.ppf([0.025, 0.975], a_saf, b_saf)

# ==========================================
# 3. FORECASTING ENGINE (BPP) + CI
# ==========================================
@st.cache_data
def get_enhanced_forecasts(curr_s, curr_n, m_n, t_eff, s_conf_final, p_a, p_b, draws, seed):
    """
    Simulates the remaining trial (Monte Carlo) to calculate BPP (Bayesian Predictive Probability).
    Uses the FINAL success confidence threshold because PPoS refers to success at max N.
    Returns: (PPoS, [success range low, high], SE, CI_low, CI_high)
    """
    rng = np.random.default_rng(seed)
    rem_n = m_n - curr_n
    if rem_n <= 0:
        is_success = (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf_final
        ppos = 1.0 if is_success else 0.0
        se = np.sqrt(ppos * (1 - ppos) / max(1, draws))
        ci_low = max(0.0, ppos - 1.96 * se)
        ci_high = min(1.0, ppos + 1.96 * se)
        return ppos, [curr_s, curr_s], se, ci_low, ci_high

    # 1. Sample potential true rates from current posterior
    future_rates = rng.beta(p_a + curr_s, p_b + curr_n - curr_s, draws)
    # 2. Simulate outcomes for remaining patients based on those rates
    future_successes = rng.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    # 3. Check how many simulations end in success (against FINAL threshold)
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    ppos = float(np.mean(final_confs > s_conf_final))
    s_range = [int(np.percentile(total_proj_s, 5)), int(np.percentile(total_proj_s, 95))]

    # MC uncertainty (normal approx on Bernoulli)
    se = float(np.sqrt(ppos * (1 - ppos) / max(1, draws)))
    ci_low = max(0.0, ppos - 1.96 * se)
    ci_high = min(1.0, ppos + 1.96 * se)
    return ppos, s_range, se, ci_low, ci_high

# Run the forecast for the current state (use FINAL threshold)
bpp, ps_range, bpp_se, bpp_ci_low, bpp_ci_high = get_enhanced_forecasts(
    successes, total_n, max_n_val, target_eff, success_conf_req_final, prior_alpha, prior_beta, mc_draws, mc_seed
)

# ==========================================
# 4. MAIN DASHBOARD LAYOUT
# ==========================================
st.title("🛡️ Hybrid Antivenom Trial Monitor: Airtight Pro")

# TOP ROW METRICS
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample N", f"{total_n}/{max_n_val}", help="Current Enrollment / Max Planned N")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}", help="The average expected efficacy based on current data.")
m3.metric(f"P(>{target_eff:.0%})", f"{p_target:.1%}", help=f"Probability that True Efficacy is greater than {target_eff:.0%}.")
m4.metric("Safety Risk", f"{p_toxic:.1%}", help=f"Probability that SAE Rate is greater than {safe_limit:.0%}.")
m5.metric("PPoS (Final)", f"{bpp:.1%}", help=f"Predicted Probability of Success at max N; 95% CI [{bpp_ci_low:.1%}, {bpp_ci_high:.1%}]")
m6.metric("Prior ESS (Eff.)", f"{prior_alpha + prior_beta:.1f}", help="Prior effective sample size for efficacy only.")
st.caption(
    f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}**  |  Prob Equivalence: **{p_equiv:.1%}** "
    f"(band applied: [{lb:.0%}, {ub:.0%}])"
)

st.markdown("---")

# ==========================================
# 5. GOVERNING RULES (Stop Logic)
# ==========================================
# Look-point checks
is_efficacy_look = (total_n >= min_interim) and (((total_n - min_interim) % check_cohort) == 0)
is_safety_look = (total_n >= safety_min_interim) and (((total_n - safety_min_interim) % safety_check_cohort) == 0)

# Safety rule application: gated or continuous
apply_safety_now = (not safety_gate_to_schedule) or is_safety_look

# Determine which efficacy confidence to use (interim vs final)
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
    next_check = total_n + (check_cohort - (total_n - min_interim) % check_cohort)
    next_safety = total_n + (safety_check_cohort - (total_n - safety_min_interim) % safety_check_cohort)
    st.info(f"🧬 **STATUS: MONITORING.** Next efficacy check at N={next_check}; "
            f"next safety check at N={next_safety}.")

# ==========================================
# 6. SEQUENTIAL DECISION CORRIDORS
# ==========================================
st.subheader("📈 Trial Decision Corridors")

# Efficacy/futility look points
look_points = [min_interim + (i * check_cohort) for i in range(100) if (min_interim + (i * check_cohort)) <= max_n_val]
viz_n = np.array(look_points)

# Safety look points (independent schedule)
safety_look_points = [safety_min_interim + (i * safety_check_cohort)
                      for i in range(100) if (safety_min_interim + (i * safety_check_cohort)) <= max_n_val]
viz_n_safety = np.array(safety_look_points)

@st.cache_data
def futility_boundary_ppos(lp, max_n_val, target_eff, success_conf_final, prior_alpha, prior_beta, draws, seed, futility_floor):
    # PPoS computed against FINAL success threshold
    ppos_by_S = np.empty(lp + 1, dtype=float)
    for s in range(lp + 1):
        ppos_by_S[s] = get_enhanced_forecasts(s, lp, max_n_val, target_eff, success_conf_final,
                                              prior_alpha, prior_beta, draws, seed)[0]
    ppos_monotone = np.maximum.accumulate(ppos_by_S)
    idx = np.where(ppos_monotone <= futility_floor)[0]
    return int(idx[-1]) if idx.size > 0 else -1

@st.cache_data
def success_boundary(lp, prior_alpha, prior_beta, target_eff, conf_req):
    return next((s for s in range(lp + 1)
                 if (1 - beta.cdf(target_eff, prior_alpha + s, prior_beta + (lp - s))) > conf_req),
                None)

succ_line, fut_line = [], []
for lp in viz_n:
    use_conf = success_conf_req_final if (lp == max_n_val) else success_conf_req_interim
    s_req = success_boundary(lp, prior_alpha, prior_beta, target_eff, use_conf)
    f_req = futility_boundary_ppos(lp, max_n_val, target_eff, success_conf_req_final,
                                   prior_alpha, prior_beta, mc_draws, mc_seed, bpp_futility_limit)
    succ_line.append(s_req)
    fut_line.append(max(0, f_req) if f_req >= 0 else -1)

# Safety boundaries computed on the safety schedule
saf_line = []
for lp in viz_n_safety:
    saf_req = next((s for s in range(lp + 1)
                    if (1 - beta.cdf(safe_limit, prior_alpha_saf + s, prior_beta_saf + (lp - s))) > safe_conf_req),
                   None)
    saf_line.append(saf_req)

# For quick lookup during simulations
succ_req_by_n = {int(n): (None if s is None else int(s)) for n, s in zip(viz_n, succ_line)}
futi_max_by_n = {int(n): int(f) if isinstance(f, (int, np.integer)) and f >= 0 else -1 for n, f in zip(viz_n, fut_line)}
safety_req_by_n = {int(n): (None if s is None else int(s)) for n, s in zip(viz_n_safety, saf_line)}

# Plot efficacy corridors
fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=viz_n, y=[np.nan if s is None else s for s in succ_line],
                              name="Success Boundary", line=dict(color='green', dash='dash')))
fig_corr.add_trace(go.Scatter(x=viz_n, y=[np.nan if f < 0 else f for f in fut_line],
                              name="Futility Boundary", line=dict(color='red', dash='dash')))
fig_corr.add_trace(go.Scatter(x=[total_n], y=[successes], mode='markers+text', text=["Current"],
                              name="Current Efficacy", marker=dict(size=12, color='blue')))
fig_corr.update_layout(xaxis_title="Sample Size (N)", yaxis_title="Successes (S)", height=400, margin=dict(t=20, b=0))
st.plotly_chart(fig_corr, use_container_width=True)

# Safety Decision Corridor
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

# ==========================================
# 7. VISUALIZATIONS & HEATMAP
# ==========================================
st.subheader("Statistical Distributions (95% CI Shaded)")
x = np.linspace(0, 1, 500)
fig = go.Figure()

# Efficacy Curve
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_eff, b_eff), name="Efficacy Belief",
                         line=dict(color='#2980b9', width=3)))
x_ci_e = np.linspace(eff_ci[0], eff_ci[1], 100)
fig.add_trace(go.Scatter(
    x=np.concatenate([x_ci_e, x_ci_e[::-1]]),
    y=np.concatenate([beta.pdf(x_ci_e, a_eff, b_eff), np.zeros(100)]),
    fill='toself', fillcolor='rgba(41, 128, 185, 0.2)',
    line=dict(color='rgba(255,255,255,0)'), showlegend=False
))

# Safety Curve
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

if include_heatmap:
    st.subheader("⚖️ Risk-Benefit Trade-off Heatmap (Illustrative)")
    eff_grid, saf_grid = np.linspace(0.2, 0.9, 50), np.linspace(0.0, 0.4, 50)
    w_tox = st.slider("Heatmap toxicity weight (w)", 0.5, 5.0, 2.0, step=0.5,
                      help="Illustrative utility: efficacy − w×toxicity")
    E, S = np.meshgrid(eff_grid, saf_grid)
    score = E - (w_tox * S)  # Illustrative linear index
    fig_heat = px.imshow(score, x=eff_grid, y=saf_grid,
                         labels=dict(x="Efficacy Rate", y="SAE Rate", color="Benefit Score"),
                         color_continuous_scale="RdYlGn", origin="lower")
    fig_heat.add_trace(go.Scatter(x=[eff_mean], y=[saf_mean], mode='markers+text',
                                  text=["Current"], marker=dict(color='white', size=12, symbol='x')))
    st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# 8. TEXTUAL BREAKDOWN & SENSITIVITY + Proper BF10 (hardened)
# ==========================================
with st.expander("📊 Full Statistical Breakdown", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Efficacy Summary**")
        st.write(f"Mean Efficacy: **{eff_mean:.1%}**")
        st.write(f"95% CI: **[{eff_ci[0]:.1%} - {eff_ci[1]:.1%}]**")
        st.write(f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}**")
        st.write(f"Prob > Target ({target_eff:.0%}): **{p_target:.1%}**")
        st.write(f"Prob > Goal ({dream_eff:.0%}): **{p_goal:.1%}**")
        st.write(f"Prob Equivalence: **{p_equiv:.1%}** (band: [{lb:.0%}, {ub:.0%}])")
        st.write(f"Projected Success Range: **{ps_range[0]} - {ps_range[1]} successes**")
    with c2:
        st.markdown("**Safety Summary (Binary SAE)**")
        st.write(f"Mean Toxicity: **{saf_mean:.1%}**")
        st.write(f"95% CI: **[{saf_ci[0]:.1%} - {saf_ci[1]:.1%}]**")
        st.write(f"Prob > Limit ({safe_limit:.0%}): **{p_toxic:.1%}**")
        st.caption("Safety is modeled as binary per patient: ≥1 SAE ⇒ SAE=1.")
    with c3:
        st.markdown("**Operational Info**")
        st.write(f"BPP Success Forecast: **{bpp:.1%}**  (SE={bpp_se:.3f}, 95% CI [{bpp_ci_low:.1%}, {bpp_ci_high:.1%}])")
        st.write(f"PPoS (Predicted Prob): **{bpp:.1%}**")
        st.write(f"Posterior pseudo-count (efficacy): **{a_eff + b_eff:.1f}**")
        st.write(f"Posterior pseudo-count (safety): **{a_saf + b_saf:.1f}**")
        st.write(f"Efficacy Look Points: **N = {', '.join(map(str, look_points)) or 'None'}**")
        st.write(f"Safety Look Points: **N = {', '.join(map(str, safety_look_points)) or 'None'}**")

st.subheader("🧪 Efficacy Sensitivity & Robustness (with Bayes Factors)")
def bayes_factor_point_null(s, n, a, b, p0, eps=1e-12):
    """
    Proper Bayes Factor BF10 comparing:
      H1: p ~ Beta(a, b)  vs  H0: p = p0
    Clips p0 to (eps, 1-eps) to avoid log(0) at boundaries.
    BF10 = [B(a+s, b+n-s) / B(a, b)] / [p0^s * (1-p0)^(n-s)]
    """
    p0 = float(np.clip(p0, eps, 1.0 - eps))
    successes_local = s
    failures_local = n - s
    log_m1 = betaln(a + successes_local, b + failures_local) - betaln(a, b)
    log_m0 = successes_local * np.log(p0) + failures_local * np.log(1 - p0)
    return float(np.exp(log_m1 - log_m0))

# ---- Adjustable efficacy sensitivity priors ----
eff_sens_list = [
    (f"Efficacy S1 (α={eff1_a:.1f}, β={eff1_b:.1f})", eff1_a, eff1_b, "#27ae60"),
    (f"Efficacy S2 (α={eff2_a:.1f}, β={eff2_b:.1f})", eff2_a, eff2_b, "#34495e"),
    (f"Efficacy S3 (α={eff3_a:.1f}, β={eff3_b:.1f})", eff3_a, eff3_b, "#8e44ad"),
]
cols, target_probs = st.columns(3), []

# Collect efficacy sensitivity results for plotting
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

st.subheader("🎛️ Efficacy Sensitivity: Posterior Distributions")
x_grid = np.linspace(0, 1, 600)
fig_sens_pdf = go.Figure()
for row in sens_rows:
    fig_sens_pdf.add_trace(go.Scatter(
        x=x_grid,
        y=beta.pdf(x_grid, row["a"], row["b"]),
        name=row["Prior"],
        line=dict(width=3, color=row["color"])
    ))
fig_sens_pdf.add_vline(x=null_eff, line_dash="dot", line_color="#7f8c8d", annotation_text="Null (p0)")
fig_sens_pdf.add_vline(x=target_eff, line_dash="dash", line_color="#2ecc71", annotation_text="Target")
fig_sens_pdf.add_vline(x=dream_eff, line_dash="dash", line_color="#f39c12", annotation_text="Goal")
fig_sens_pdf.update_layout(
    xaxis_title="Efficacy rate", yaxis_title="Posterior density",
    height=420, margin=dict(t=20, b=0)
)
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

# ==========================================
# 8B. SAFETY SENSITIVITY (adjustable priors + richer stats + UCL badge)
# ==========================================
st.subheader("🧯 Safety Sensitivity Analysis (Adjustable Priors)")

# Build safety sensitivity specs from sidebar
safety_sens_specs = [
    (f"Safety S1 (α={saf1_a:.1f}, β={saf1_b:.1f})", saf1_a, saf1_b, "#16a085"),
    (f"Safety S2 (α={saf2_a:.1f}, β={saf2_b:.1f})", saf2_a, saf2_b, "#2c3e50"),
    (f"Safety S3 (α={saf3_a:.1f}, β={saf3_b:.1f})", saf3_a, saf3_b, "#c0392b"),
]
safety_rows = []
cols_saf = st.columns(3)
for i, (name, a0, b0, color) in enumerate(safety_sens_specs):
    a_post = a0 + saes
    b_post = b0 + (total_n - saes)
    mean_tox = a_post / (a_post + b_post)
    ci_lo, ci_hi = beta.ppf([0.025, 0.975], a_post, b_post)
    # One-sided 95% upper credible limit (UCL95)
    ucl95 = beta.ppf(0.95, a_post, b_post)
    p_tox_gt = 1 - beta.cdf(safe_limit, a_post, b_post)
    p_tox_le = 1 - p_tox_gt
    # Safety Bayes factor vs point-null at the safety limit (hardened BF with clipping)
    bf10_safety = bayes_factor_point_null(saes, total_n, a0, b0, safe_limit)

    safety_rows.append({
        "Prior": name, "color": color,
        "a": a_post, "b": b_post,
        "Mean tox": mean_tox,
        "CI_lo": ci_lo, "CI_hi": ci_hi,
        "UCL95": ucl95,
        "P(tox>limit)": p_tox_gt,
        "P(tox≤limit)": p_tox_le,
        "BF10_safety": bf10_safety
    })
    with cols_saf[i]:
        st.info(f"**{name}**")
        st.write(f"Mean toxicity: **{mean_tox:.1%}**")
        st.write(f"95% CI: **[{ci_lo:.1%} – {ci_hi:.1%}]**")
        st.write(f"UCL95 (one-sided): **{ucl95:.1%}**")
        # Micro-badge comparing UCL95 to safety limit (quick reassurance cue)
        if ucl95 < safe_limit:
            st.caption("✅ UCL95 < safety limit — reassuring")
        else:
            st.caption("⚠️ UCL95 ≥ safety limit — monitor closely")
        st.write(f"P(tox > {safe_limit:.0%}): **{p_tox_gt:.1%}**")
        st.write(f"P(tox ≤ {safe_limit:.0%}): **{p_tox_le:.1%}**")
        bf_str = f"{bf10_safety:.2e}" if (bf10_safety >= 1e4 or bf10_safety <= 1e-4) else f"{bf10_safety:.2f}"
        st.write(f"Safety BF₁₀ (H1 vs H0 p={safe_limit:.0%}): **{bf_str}**")

# Safety PDFs overlay
x_grid = np.linspace(0, 1, 600)
fig_safety_pdf = go.Figure()
for row in safety_rows:
    fig_safety_pdf.add_trace(go.Scatter(
        x=x_grid, y=beta.pdf(x_grid, row["a"], row["b"]),
        name=row["Prior"], line=dict(width=3, color=row["color"])
    ))
fig_safety_pdf.add_vline(x=safe_limit, line_dash="dash", line_color="#000000", annotation_text="Safety limit")
fig_safety_pdf.update_layout(xaxis_title="SAE rate", yaxis_title="Posterior density",
                             height=380, margin=dict(t=20, b=0))
st.plotly_chart(fig_safety_pdf, use_container_width=True)

# Grouped bars for P(tox>limit)
sdf = pd.DataFrame([{"Prior": r["Prior"], "P(tox>limit)": r["P(tox>limit)"]} for r in safety_rows])
fig_safety_bar = px.bar(sdf, x="Prior", y="P(tox>limit)", color="Prior",
                        color_discrete_sequence=[r["color"] for r in safety_rows],
                        text="P(tox>limit)")
fig_safety_bar.update_traces(texttemplate="%{text:.1%}", textposition="outside", showlegend=False)
fig_safety_bar.update_yaxes(tickformat=".0%")
fig_safety_bar.update_layout(height=360, margin=dict(t=20, b=0))
st.plotly_chart(fig_safety_bar, use_container_width=True)

# Helper interpretation for safety robustness vs the stop threshold
if safety_rows:
    probs = [r["P(tox>limit)"] for r in safety_rows]
    min_p, max_p = float(np.min(probs)), float(np.max(probs))
    if max_p < safe_conf_req:
        interp = (f"Across safety priors, **P(tox > {safe_limit:.0%}) ranges {min_p:.1%}–{max_p:.1%}**, "
                  f"which is **below** your Safety Stop Confidence (**{safe_conf_req:.0%}**). "
                  "Interpretation: safety is **robustly acceptable** under current data and priors.")
        st.success(interp)
    elif min_p > safe_conf_req:
        interp = (f"Across safety priors, **P(tox > {safe_limit:.0%}) ranges {min_p:.1%}–{max_p:.1%}**, "
                  f"which is **above** your Safety Stop Confidence (**{safe_conf_req:.0%}**). "
                  "Interpretation: safety is **consistently concerning** across priors—consider stopping/mitigation.")
        st.error(interp)
    else:
        interp = (f"Across safety priors, **P(tox > {safe_limit:.0%}) ranges {min_p:.1%}–{max_p:.1%}**, "
                  f"straddling the Safety Stop Confidence (**{safe_conf_req:.0%}**). "
                  "Interpretation: safety judgment is **sensitive to prior assumptions**—"
                  "DSMB should review exposure, severity, and trend before deciding.")
        st.warning(interp)

# ==========================================
# 9. TYPE I ERROR SIMULATION (Monte Carlo) -- Safety & Futility + Guardrails, mixed thresholds
# ==========================================
st.markdown("---")
st.subheader("🧪 Design Integrity Check")

num_sims = 10000
with st.expander("Simulation Options", expanded=False):
    sim_safety_rate = st.slider(
        "Assumed TRUE SAE rate for Type I simulation", 0.0, 0.9, float(safe_limit), step=0.01,
        help="Used to simulate SAEs while testing Type I (efficacy false positive) under p=p0."
    )
    sim_seed = st.number_input("Random seed (Type I sim)", 0, 10_000_000, 7, step=1)
    # NEW: Toggle for using look schedules in TYPE I sim
    sim_use_look_settings_typei = st.checkbox(
        "Simulate using user-defined look schedules (efficacy & safety)",
        value=True,
        help="If off, evaluate safety & efficacy at every N from 1..max N (continuous monitoring)."
    )

if st.button(f"Calculate Sequential Type I Error ({num_sims:,} sims)"):
    if len(look_points) == 0 and len(safety_look_points) == 0:
        st.warning("No scheduled interim looks based on current settings; Type I error is not computed.")
    else:
        with st.spinner(f"Simulating {num_sims:,} trials..."):
            rng = np.random.default_rng(sim_seed)

            # Build look sets based on the toggle
            if sim_use_look_settings_typei:
                eff_looks = set(look_points)
                eff_looks.add(max_n_val)  # always include final
                saf_looks = set(safety_look_points)
                all_looks = sorted(eff_looks.union(saf_looks))
                safety_check_set = saf_looks if safety_gate_to_schedule else set(all_looks)
            else:
                eff_looks = set(range(1, max_n_val + 1))
                saf_looks = set(range(1, max_n_val + 1))
                all_looks = list(range(1, max_n_val + 1))
                safety_check_set = set(all_looks)

            # Precompute any missing thresholds for expanded looks (cached)
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
                    safety_req_sim[lp] = next((s for s in range(lp + 1)
                                               if (1 - beta.cdf(safe_limit, prior_alpha_saf + s, prior_beta_saf + (lp - s))) > safe_conf_req),
                                              None)

            fp_count = 0  # False positives for efficacy

            for _ in range(num_sims):
                trial_eff = rng.binomial(1, null_eff, max_n_val)
                trial_saf = rng.binomial(1, sim_safety_rate, max_n_val)

                for lp in all_looks:
                    s = int(trial_eff[:lp].sum())
                    t = int(trial_saf[:lp].sum())

                    # Safety per toggle/gating
                    if lp in safety_check_set:
                        saf_thr = safety_req_sim.get(lp, None)
                        if saf_thr is not None and t >= saf_thr:
                            # safety stop (not FP)
                            break

                    # Efficacy/Futility
                    if lp in eff_looks:
                        if lp != max_n_val:
                            fut_thr = futi_max_sim.get(lp, -1)
                            if fut_thr >= 0 and s <= fut_thr:
                                break  # futility stop (not FP)

                        suc_thr = succ_req_sim.get(lp, None)
                        if suc_thr is not None and s >= suc_thr:
                            fp_count += 1  # efficacy success under H0 => false positive
                            break

            type_i_estimate = fp_count / num_sims
            st.warning(f"Estimated Sequential Type I Error (with safety & futility): **{type_i_estimate:.2%}**")
            st.caption("Alpha respects the toggle: using user look schedules (default) or continuous checks at every N.")

# ==========================================
# 10. POWER ANALYSIS (Operating Characteristics)
# ==========================================
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
                        sim_use_look_settings_power):
    """
    Simulate full trials to estimate:
      - Power (probability of declaring efficacy success)
      - Expected sample size at stop
      - Stop reason proportions
    Applies rules: safety -> futility -> efficacy, with separate safety schedule and
    interim vs final thresholds. Final efficacy look at N = max_n_val is ensured in schedule mode.
    Toggle supports schedule mode vs continuous checks at every N.
    """
    rng = np.random.default_rng(seed)

    # Build look sets per toggle
    if sim_use_look_settings_power:
        eff_looks = set(look_points)
        eff_looks.add(max_n_val)
        saf_looks = set(safety_look_points)
        all_looks = sorted(eff_looks.union(saf_looks))
        safety_check_set = saf_looks if safety_gate_to_schedule else set(all_looks)
    else:
        eff_looks = set(range(1, max_n_val + 1))
        saf_looks = set(range(1, max_n_val + 1))
        all_looks = list(range(1, max_n_val + 1))
        safety_check_set = set(all_looks)

    # Precompute any missing thresholds for expanded looks (cached)
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
            safety_req_sim[lp] = next((s for s in range(lp + 1)
                                       if (1 - beta.cdf(safe_limit, prior_alpha_saf + s, prior_beta_saf + (lp - s))) > safe_conf_req),
                                      None)

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

            # Safety per toggle/gating
            if lp in safety_check_set:
                saf_thr = safety_req_sim.get(lp, None)
                if saf_thr is not None and t >= saf_thr:
                    stop_reason_counts["safety"] += 1
                    stop_n_list.append(lp)
                    decided = True
                    break

            # Efficacy/Futility
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

# Run power simulation at the chosen true rates (respect toggle)
power_est, power_exp_n, power_reasons = simulate_power_once(
    power_true_eff, power_true_saf, power_sims, power_seed,
    max_n_val,
    look_points, safety_look_points, safety_gate_to_schedule,
    prior_alpha, prior_beta, target_eff,
    success_conf_req_interim, success_conf_req_final,
    prior_alpha_saf, prior_beta_saf, safe_limit, safe_conf_req,
    succ_req_by_n, futi_max_by_n, safety_req_by_n,
    mc_draws, mc_seed, bpp_futility_limit,
    sim_use_look_settings_power
)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Power (Pr[declare success])", f"{power_est:.1%}")
with c2:
    st.metric("Expected sample size at stop", f"{power_exp_n:.1f}")
with c3:
    st.write("Stop reasons:")
    st.write(f"- Safety stop: **{power_reasons['safety'] / power_sims:.1%}**")
    st.write(f"- Futility stop: **{power_reasons['futility'] / power_sims:.1%}**")
    st.write(f"- Interim success: **{power_reasons['interim_success'] / power_sims:.1%}**")
    st.write(f"- Final success: **{power_reasons['final_success'] / power_sims:.1%}**")
    st.write(f"- No decision at looks (ended at max N): **{power_reasons['no_decision'] / power_sims:.1%}**")

# Power curve vs true efficacy
st.subheader("📐 Power Curve vs True Efficacy")
@st.cache_data
def power_curve(p0, p_high, points, sims_per_point, seed_base,
                max_n_val, look_points, safety_look_points, safety_gate_to_schedule,
                prior_alpha, prior_beta, target_eff,
                success_conf_req_interim, success_conf_req_final,
                prior_alpha_saf, prior_beta_saf, safe_limit, safe_conf_req,
                succ_req_by_n, futi_max_by_n, safety_req_by_n,
                true_saf_rate, mc_draws, mc_seed, bpp_futility_limit,
                sim_use_look_settings_power):
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
            sim_use_look_settings_power
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
    sim_use_look_settings_power
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

# ==========================================
# 11. REGULATORY TABLES (efficacy/futility) & (safety)
# ==========================================
with st.expander("📋 Regulatory Decision Boundary Tables", expanded=True):
    st.markdown("**Efficacy/Futility Boundaries (by efficacy look schedule)**")
    boundary_data_eff = []
    for lp in look_points:
        if lp <= total_n:
            continue

        use_conf = success_conf_req_final if (lp == max_n_val) else success_conf_req_interim

        # Success threshold
        s_req = success_boundary(lp, prior_alpha, prior_beta, target_eff, use_conf)

        # Futility threshold (highest S that triggers a stop)
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

        # Safety threshold (using safety priors)
        safe_req = next((s for s in range(lp + 1)
                         if (1 - beta.cdf(safe_limit, prior_alpha_saf + s, prior_beta_saf + (lp - s))) > safe_conf_req),
                        None)

        boundary_data_saf.append({
            "N (safety)": lp,
            "Safety Stop SAEs ≥": safe_req if safe_req is not None else "No safety stop at this look"
        })
    if boundary_data_saf:
        st.table(pd.DataFrame(boundary_data_saf))
    else:
        st.write("No future safety looks (or trial is at/final analysis).")

st.markdown("---")

# ==========================================
# 12. EXPORT / AUDIT SNAPSHOT
# ==========================================
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


