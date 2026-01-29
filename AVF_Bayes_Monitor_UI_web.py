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

# Thresholds (Efficacy)
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
    success_conf_req = st.slider(
        "Success Confidence Req.", 0.5, 0.99, 0.74,
        help="Posterior Probability threshold required to declare early success (e.g., 0.975)."
    )
    bpp_futility_limit = st.slider(
        "BPP Futility Limit", 0.01, 0.20, 0.05,
        help="Stop if the chance of future success (BPP/PPoS) drops below this level."
    )

# Safety rules + Separate Safety Priors + Safety schedule
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

# Sensitivity & Equivalence
with st.sidebar.expander("Sensitivity Prior Settings"):
    opt_p = st.slider("Optimistic Prior Weight", 1, 10, 4,
                      help="Alpha weight for the Optimistic sensitivity analysis.")
    skp_p = st.slider("Skeptical Prior Weight", 1, 10, 4,
                      help="Beta weight for the Skeptical sensitivity analysis.")
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

# ==========================================
# 2. MATH ENGINE (Bayesian Updates)
# ==========================================
# Posterior = Prior + Data
a_eff, b_eff = prior_alpha + successes, prior_beta + (total_n - successes)
a_saf, b_saf = prior_alpha_saf + saes, prior_beta_saf + (total_n - saes)

# Probability Calculations (CDF = Cumulative Distribution Function)
p_null = 1 - beta.cdf(null_eff, a_eff, b_eff)    # Prob > Null
p_target = 1 - beta.cdf(target_eff, a_eff, b_eff)  # Prob > Target
p_goal = 1 - beta.cdf(dream_eff, a_eff, b_eff)     # Prob > Dream
p_toxic = 1 - beta.cdf(safe_limit, a_saf, b_saf)   # Prob > Safety Limit

# Equivalence around Null (clip to [0,1] for clarity + display)
lb, ub = max(0.0, null_eff - equiv_bound), min(1.0, null_eff + equiv_bound)
p_equiv = beta.cdf(ub, a_eff, b_eff) - beta.cdf(lb, a_eff, b_eff)

# Credible Intervals & Means
eff_mean, eff_ci = a_eff / (a_eff + b_eff), beta.ppf([0.025, 0.975], a_eff, b_eff)
saf_mean, saf_ci = a_saf / (a_saf + b_saf), beta.ppf([0.025, 0.975], a_saf, b_saf)

# ==========================================
# 3. FORECASTING ENGINE (BPP) + CI
# ==========================================
@st.cache_data
def get_enhanced_forecasts(curr_s, curr_n, m_n, t_eff, s_conf, p_a, p_b, draws, seed):
    """
    Simulates the remaining trial (Monte Carlo) to calculate BPP (Bayesian Predictive Probability).
    Returns: (PPoS, [success range low, high], SE, CI_low, CI_high)
    """
    rng = np.random.default_rng(seed)
    rem_n = m_n - curr_n
    if rem_n <= 0:
        is_success = (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf
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
    # 3. Check how many simulations end in success
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    ppos = float(np.mean(final_confs > s_conf))
    s_range = [int(np.percentile(total_proj_s, 5)), int(np.percentile(total_proj_s, 95))]

    # MC uncertainty (normal approx on Bernoulli)
    se = float(np.sqrt(ppos * (1 - ppos) / max(1, draws)))
    ci_low = max(0.0, ppos - 1.96 * se)
    ci_high = min(1.0, ppos + 1.96 * se)
    return ppos, s_range, se, ci_low, ci_high

# Run the forecast for the current state
bpp, ps_range, bpp_se, bpp_ci_low, bpp_ci_high = get_enhanced_forecasts(
    successes, total_n, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta, mc_draws, mc_seed
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
m5.metric("PPoS (Final)", f"{bpp:.1%}", help=f"Predicted Probability of Success; 95% CI [{bpp_ci_low:.1%}, {bpp_ci_high:.1%}]")
m6.metric("Prior ESS (Eff.)", f"{prior_alpha + prior_beta:.1f}", help="Prior effective sample size for efficacy only.")
st.caption(
    f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}**  |  Prob Equivalence: **{p_equiv:.1%}** "
    f" (band applied: [{lb:.0%}, {ub:.0%}])"
)

st.markdown("---")

# ==========================================
# 5. GOVERNING RULES (Stop Logic)
# ==========================================
is_look_point = (total_n >= min_interim) and (((total_n - min_interim) % check_cohort) == 0)
if p_toxic > safe_conf_req:
    st.error(f"🛑 **GOVERNING RULE: SAFETY STOP.** Risk ({p_toxic:.1%}) exceeds {safe_conf_req:.0%} threshold.")
elif is_look_point:
    if bpp < bpp_futility_limit:
        st.warning(f"⚠️ **GOVERNING RULE: FUTILITY STOP.** PPoS ({bpp:.1%}) below floor. (SE={bpp_se:.3f})")
    elif p_target > success_conf_req:
        st.success(f"✅ **GOVERNING RULE: EFFICACY SUCCESS.** Evidence achieved at {p_target:.1%}.")
    else:
        st.info(f"🛡️ **GOVERNING RULE: CONTINUE.** Interim check at N={total_n} is indeterminate.")
elif total_n < min_interim:
    st.info(f"⏳ **STATUS: LEAD-IN.** Enrollment phase; first check at N={min_interim}.")
else:
    next_check = total_n + (check_cohort - (total_n - min_interim) % check_cohort)
    st.info(f"🧬 **STATUS: MONITORING.** Trial between cohorts. Next check at N={next_check}.")

# ==========================================
# 6. SEQUENTIAL DECISION CORRIDORS
# ==========================================
st.subheader("📈 Trial Decision Corridors")

# Efficacy/futility look points
look_points = [min_interim + (i * check_cohort) for i in range(100) if (min_interim + (i * check_cohort)) <= max_n_val]
viz_n = np.array(look_points)

# Safety look points (independent schedule; defaults match efficacy schedule)
safety_look_points = [safety_min_interim + (i * safety_check_cohort)
                      for i in range(100) if (safety_min_interim + (i * safety_check_cohort)) <= max_n_val]
viz_n_safety = np.array(safety_look_points)

# Helper: monotone futility boundary via isotonic enforcement
@st.cache_data
def futility_boundary_ppos(lp, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta, draws, seed):
    ppos_by_S = np.empty(lp + 1, dtype=float)
    for s in range(lp + 1):
        ppos_by_S[s] = get_enhanced_forecasts(s, lp, max_n_val, target_eff, success_conf_req,
                                              prior_alpha, prior_beta, draws, seed)[0]
    # Enforce monotonicity (PPoS should be non-decreasing in S)
    ppos_monotone = np.maximum.accumulate(ppos_by_S)
    idx = np.where(ppos_monotone <= bpp_futility_limit)[0]
    return int(idx[-1]) if idx.size > 0 else -1

succ_line, fut_line = [], []
for lp in viz_n:
    # Success boundary: smallest S where posterior P(>target) > requirement
    s_req = next((s for s in range(lp + 1)
                  if (1 - beta.cdf(target_eff, prior_alpha + s, prior_beta + (lp - s))) > success_conf_req),
                 None)
    # Futility boundary: highest S where PPoS is still BELOW the limit (monotone-safe)
    f_req = futility_boundary_ppos(lp, max_n_val, target_eff, success_conf_req,
                                   prior_alpha, prior_beta, mc_draws, mc_seed)
    succ_line.append(s_req)  # may be None (no success at this look)
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

# Plot efficacy success/futility corridors (unchanged visual; safety shown in table)
fig_corr = go.Figure()
# Success boundary: draw with gaps where None
fig_corr.add_trace(go.Scatter(x=viz_n, y=[np.nan if s is None else s for s in succ_line],
                              name="Success Boundary", line=dict(color='green', dash='dash')))
fig_corr.add_trace(go.Scatter(x=viz_n, y=[np.nan if f < 0 else f for f in fut_line],
                              name="Futility Boundary", line=dict(color='red', dash='dash')))
fig_corr.add_trace(go.Scatter(x=[total_n], y=[successes], mode='markers+text', text=["Current"],
                              name="Current Data", marker=dict(size=12, color='blue')))
fig_corr.update_layout(xaxis_title="Sample Size (N)", yaxis_title="Successes (S)", height=400, margin=dict(t=20, b=0))
st.plotly_chart(fig_corr, use_container_width=True)

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
    # Editable toxicity weight (optional)
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
# 8. TEXTUAL BREAKDOWN & SENSITIVITY + Proper BF10
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

st.subheader("🧪 Sensitivity Analysis & Robustness (with Bayes Factors)")
def bayes_factor_point_null(s, n, a, b, p0):
    """
    Proper Bayes Factor BF10 comparing:
      H1: p ~ Beta(a, b)  vs  H0: p = p0
    Using marginal likelihoods (combinatorial factors cancel).
    BF10 = [B(a+s, b+n-s) / B(a, b)] / [p0^s * (1-p0)^(n-s)]
    """
    successes = s
    failures = n - s
    log_m1 = betaln(a + successes, b + failures) - betaln(a, b)
    log_m0 = successes * np.log(p0) + failures * np.log(1 - p0)
    return float(np.exp(log_m1 - log_m0))

priors_list = [
    (f"Optimistic ({opt_p}:1)", opt_p, 1),
    ("Neutral (1:1)", 1, 1),
    (f"Skeptical (1:{skp_p})", 1, skp_p)
]
cols, target_probs = st.columns(3), []

for i, (name, ap, bp) in enumerate(priors_list):
    ae_s, be_s = ap + successes, bp + (total_n - successes)
    m_eff_s = ae_s / (ae_s + be_s)
    p_n_s = 1 - beta.cdf(null_eff, ae_s, be_s)
    p_t_s = 1 - beta.cdf(target_eff, ae_s, be_s)
    p_g_s = 1 - beta.cdf(dream_eff, ae_s, be_s)
    target_probs.append(p_t_s)

    with cols[i]:
        st.info(f"**{name}**")
        st.write(f"Mean Efficacy: **{m_eff_s:.1%}**")
        st.write(f"Prob > Null: **{p_n_s:.1%}**")
        st.write(f"Prob > Target: **{p_t_s:.1%}**")
        st.write(f"Prob > Goal: **{p_g_s:.1%}**")

        bf10 = bayes_factor_point_null(successes, total_n, ap, bp, null_eff)
        bf_str = f"{bf10:.2e}" if (bf10 >= 1e4 or bf10 <= 1e-4) else f"{bf10:.2f}"
        st.write(f"Bayes Factor BF₁₀ (H1: Beta({ap},{bp}) vs H0: p={null_eff:.0%}): **{bf_str}**")

st.caption("Interpretation: BF₁₀>1 favors H1; >10 is strong evidence. BF₁₀<1/10 is strong evidence for H0.")
spread = max(target_probs) - min(target_probs)
st.markdown(f"**Interpretation:** Results are **{'ROBUST' if spread < 0.15 else 'SENSITIVE'}** "
            f"({spread:.1%} variance between prior mindsets).")

# ==========================================
# 9. TYPE I ERROR SIMULATION (Monte Carlo) -- Extended with Safety & Futility + Guardrails
# ==========================================
