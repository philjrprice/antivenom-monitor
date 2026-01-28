import streamlit as st
import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Universal Trial Monitor: Airtight Pro", layout="wide")

# --- SIDEBAR: INPUT SECTIONS ---
# The sidebar captures real-time data from the clinical trial site.
st.sidebar.header("📋 Current Trial Data")
max_n_val = st.sidebar.number_input("Maximum Sample Size (N)", 10, 500, 70, 
                                     help="The total planned enrollment for the trial (N_max).")
total_n = st.sidebar.number_input("Total Patients Enrolled", 0, max_n_val, 20, 
                                   help="The number of patients who have completed follow-up and have evaluable data.")
successes = st.sidebar.number_input("Total Successes", 0, total_n, value=min(14, total_n), 
                                     help="The number of patients reaching the primary efficacy endpoint.")
saes = st.sidebar.number_input("Serious Adverse Events (SAEs)", 0, total_n, value=min(1, total_n), 
                                help="The count of Serious Adverse Events used for the Bayesian safety monitoring.")

# --- DATA INTEGRITY VALIDATION ---
# Ensures that the entered data is mathematically possible before proceeding.
if successes > total_n:
    st.error("⚠️ Data Integrity Error: Successes cannot exceed total patients enrolled.")
    st.stop()
if saes > total_n:
    st.error("⚠️ Data Integrity Error: SAEs cannot exceed total patients enrolled.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Study Parameters")

# --- BAYESIAN PRIORS ---
# Priors represent our 'starting belief' before we see any data. 
# Alpha=1, Beta=1 is a 'Flat/Non-informative' prior (all rates 0-100% are equally likely).
with st.sidebar.expander("Base Study Priors", expanded=True):
    prior_alpha = st.slider("Prior Successes (Alpha)", 0.1, 10.0, 1.0, step=0.1, 
                            help="Initial 'phantom' successes added to the model. Higher values represent stronger pre-existing belief in efficacy.")
    prior_beta = st.slider("Prior Failures (Beta)", 0.1, 10.0, 1.0, step=0.1, 
                           help="Initial 'phantom' failures. Higher values represent a more skeptical starting belief.")

# --- ADAPTIVE TIMING ---
with st.sidebar.expander("Adaptive Timing & Look Points", expanded=True):
    min_interim = st.number_input("Min N before first check", 1, max_n_val, 14, 
                                  help="The 'lead-in' phase. No stopping is allowed before this many patients are enrolled.")
    check_cohort = st.number_input("Check every X patients (Cohort)", 1, 20, 5, 
                                   help="The frequency of interim analyses (e.g., check every 5 patients).")

# --- SUCCESS & FUTILITY RULES ---
with st.sidebar.expander("Success & Futility Rules"):
    null_eff = st.slider("Null Efficacy (p0) (%)", 0.1, 1.0, 0.50, 
                         help="The 'Standard of Care' or placebo rate we must beat.")
    target_eff = st.slider("Target Efficacy (p1) (%)", 0.1, 1.0, 0.60, 
                           help="The minimum efficacy required to consider the treatment clinically meaningful.")
    dream_eff = st.slider("Goal/Dream Efficacy (%)", 0.1, 1.0, 0.70, 
                          help="High-bar efficacy goal (used for secondary probability metrics).")
    success_conf_req = st.slider("Success Confidence Req.", 0.5, 0.99, 0.74, 
                                 help="The required posterior probability that Efficacy > Target to declare success early.")
    bpp_futility_limit = st.slider("BPP Futility Limit", 0.01, 0.20, 0.05, 
                                   help="Bayesian Predictive Probability floor. If the chance of future success drops below this, the trial stops for futility.")

# --- SAFETY RULES ---
with st.sidebar.expander("Safety Rules", expanded=True):
    safe_limit = st.slider("SAE Upper Limit (%)", 0.05, 0.50, 0.15, 
                           help="The maximum acceptable Serious Adverse Event rate.")
    safe_conf_req = st.slider("Safety Stop Confidence", 0.5, 0.99, 0.90, 
                              help="The threshold for stopping due to safety. If P(SAE Rate > Limit) exceeds this, stop the trial.")

# --- SENSITIVITY & ANALYSIS SETTINGS ---
with st.sidebar.expander("Sensitivity Prior Settings"):
    opt_p = st.slider("Optimistic Prior Weight", 1, 10, 4, help="Prior successes used for the 'Best Case' sensitivity model.")
    skp_p = st.slider("Skeptical Prior Weight", 1, 10, 4, help="Prior failures used for the 'Worst Case' sensitivity model.")

with st.sidebar.expander("Equivalence & Heatmap Settings"):
    equiv_bound = st.slider("Equivalence Bound (+/-)", 0.01, 0.10, 0.05, help="The margin for calculating if the treatment is 'equivalent' to the Null rate.")
    include_heatmap = st.checkbox("Generate Risk-Benefit Heatmap", value=True)

# --- BAYESIAN MATH ENGINE ---
# Posterior = Prior + Data. We update our Beta distribution here.
a_eff, b_eff = prior_alpha + successes, prior_beta + (total_n - successes)
a_safe, b_safe = prior_alpha + saes, prior_beta + (total_n - saes)

# Probability Calculations (Integrals of the Beta Distribution)
p_null = 1 - beta.cdf(null_eff, a_eff, b_eff)
p_target = 1 - beta.cdf(target_eff, a_eff, b_eff)
p_goal = 1 - beta.cdf(dream_eff, a_eff, b_eff)
p_toxic = 1 - beta.cdf(safe_limit, a_safe, b_safe)
p_equiv = beta.cdf(null_eff + equiv_bound, a_eff, b_eff) - beta.cdf(null_eff - equiv_bound, a_eff, b_eff)

# Posterior Statistics
eff_mean, eff_ci = a_eff / (a_eff + b_eff), beta.ppf([0.025, 0.975], a_eff, b_eff)
safe_mean, safe_ci = a_safe / (a_safe + b_safe), beta.ppf([0.025, 0.975], a_safe, b_safe)

# --- FORECASTING ENGINE (BPP) ---
# Calculates the Predicted Probability of Success (PPoS) by simulating the remainder of the trial.
@st.cache_data
def get_enhanced_forecasts(curr_s, curr_n, m_n, t_eff, s_conf, p_a, p_b):
    np.random.seed(42) 
    rem_n = m_n - curr_n
    if rem_n <= 0:
        is_success = (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf
        return 1.0 if is_success else 0.0, [curr_s, curr_s]
    
    # Monte Carlo simulation of future patients
    future_rates = np.random.beta(p_a + curr_s, p_b + curr_n - curr_s, 5000)
    future_successes = np.random.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    
    ppos = np.mean(final_confs > s_conf)
    s_range = [int(np.percentile(total_proj_s, 5)), int(np.percentile(total_proj_s, 95))]
    return ppos, s_range

bpp, ps_range = get_enhanced_forecasts(successes, total_n, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)

# Robustness/Spread calculation (difference between optimistic and skeptical models)
skep_a, skep_b = 1 + successes, skp_p + (total_n - successes)
opt_a, opt_b = opt_p + successes, 1 + (total_n - successes)
spread = abs((1 - beta.cdf(target_eff, opt_a, opt_b)) - (1 - beta.cdf(target_eff, skep_a, skep_b)))

# Bayes Factor calculation (Ratio of Evidence)
skep_prob = 1 - beta.cdf(target_eff, skep_a, skep_b)
evidence_shift = p_target / skep_prob if skep_prob > 0 else 1.0

# --- DASHBOARD UI ---
st.title("🛡️ Hybrid Antivenom Trial Monitor: Airtight Pro")

# Metrics Row
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample N", f"{total_n}/{max_n_val}", 
          help="Enrollment progress toward Maximum Sample Size.")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}", 
          help="The posterior mean: our best single estimate of the true efficacy rate.")
m3.metric(f"P(>{target_eff:.0%})", f"{p_target:.1%}", 
          help="The Bayesian confidence level that the drug efficacy is above your target threshold.")
m4.metric("Safety Risk", f"{p_toxic:.1%}", 
          help="The probability that the true SAE rate is higher than your safety limit.")
m5.metric("PPoS (Final)", f"{bpp:.1%}", 
          help="Predicted Probability of Success: The likelihood this trial will pass the final analysis if it continues to Max N.")
m6.metric("Prior Weight", f"{prior_alpha + prior_beta:.1f}", 
          help="The strength of your starting priors expressed as 'equivalent number of patients' (ESS_prior).")

st.caption(f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}** | Prob Equivalence: **{p_equiv:.1%}** | Robustness Spread: **{spread:.1%}**")
st.markdown("---")

# --- GOVERNING LOGIC (INTERIM RULES) ---
# This section automates the decision-making process based on the rules defined in the sidebar.
is_look_point = (total_n >= min_interim) and ((total_n - min_interim) % check_cohort == 0)

if p_toxic > safe_conf_req:
    st.error(f"🛑 **GOVERNING RULE: SAFETY STOP.** SAE risk ({p_toxic:.1%}) exceeds the {safe_conf_req:.0%} threshold. Recommend immediate cessation.")
elif is_look_point:
    if bpp < bpp_futility_limit:
        st.warning(f"⚠️ **GOVERNING RULE: FUTILITY STOP.** The PPoS ({bpp:.1%}) has fallen below the {bpp_futility_limit:.0%} limit. High risk of trial failure.")
    elif p_target > success_conf_req:
        st.success(f"✅ **GOVERNING RULE: EFFICACY SUCCESS.** Current confidence ({p_target:.1%}) meets the requirement for early success declaration.")
    else:
        st.info(f"🛡️ **GOVERNING RULE: CONTINUE.** Enrollment cohort N={total_n} check complete. Evidence is indeterminate; continue enrollment.")
elif total_n < min_interim:
    st.info(f"⏳ **STATUS: LEAD-IN PHASE.** Enrollment is below the minimum interim threshold (N={min_interim}). Monitoring only.")
else:
    next_check = total_n + (check_cohort - (total_n - min_interim) % check_cohort)
    st.info(f"🧬 **STATUS: MONITORING.** Trial is between analysis points. Next cohort check at N={next_check}.")

# --- SEQUENTIAL DECISION CORRIDORS (VISUALIZATION) ---
# This calculates and plots the 'Go/No-Go' lines for the entire trial.
st.subheader("📈 Trial Decision Corridors")
look_points = [min_interim + (i * check_cohort) for i in range(100) if (min_interim + (i * check_cohort)) <= max_n_val]
viz_n = np.array(look_points)
succ_line, fut_line = [], []

for lp in viz_n:
    # Smallest S where confidence exceeds threshold
    s_req = next((s for s in range(lp+1) if (1 - beta.cdf(target_eff, prior_alpha+s, prior_beta+(lp-s))) > success_conf_req), lp+1)
    # Highest S where BPP is still below futility limit
    f_req = next((s for s in reversed(range(lp+1)) if get_enhanced_forecasts(s, lp, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)[0] <= bpp_futility_limit), -1)
    succ_line.append(s_req)
    fut_line.append(max(0, f_req))

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=viz_n, y=succ_line, name="Success Boundary (S ≥)", line=dict(color='green', dash='dash')))
fig_corr.add_trace(go.Scatter(x=viz_n, y=fut_line, name="Futility Boundary (S ≤)", line=dict(color='red', dash='dash')))
fig_corr.add_trace(go.Scatter(x=[total_n], y=[successes], mode='markers+text', name="Current Position", text=["Current"], textposition="top center", marker=dict(size=12, color='blue')))
fig_corr.update_layout(title="Sequential Decision Boundaries", xaxis_title="Sample Size (N)", yaxis_title="Successes (S)", hovermode="x")
st.plotly_chart(fig_corr, use_container_width=True)

# --- DIAGNOSTICS & SENSITIVITY ---
col_op1, col_op2 = st.columns(2)
with col_op1:
    st.write("### Operational Info")
    st.metric("Effective Sample Size (ESS)", f"{a_eff + b_eff:.1f}", help="Total information (Current N + Prior Weights).")
    st.write(f"95% Efficacy Credible Interval: **[{eff_ci[0]:.1%}, {eff_ci[1]:.1%}]**")
    st.write(f"95% Safety Credible Interval: **[{safe_ci[0]:.1%}, {safe_ci[1]:.1%}]**")

with col_op2:
    st.write("### Sensitivity Analysis")
    priors = [("Neutral (Current)", prior_alpha, prior_beta), ("Optimistic", opt_p, 1), ("Skeptical", 1, skp_p)]
    for name, pa, pb in priors:
        p_s = 1 - beta.cdf(target_eff, pa + successes, pb + (total_n - successes))
        st.write(f"{name} Prior Confidence: **{p_s:.1%}**")
        if "Neutral" in name:
            st.metric("Bayes Factor (BF₁₀)", f"{evidence_shift:.2f}x", help="Evidence ratio: Factors > 1 indicate the data supports the efficacy hypothesis.")

# --- REGULATORY DECISION TABLE ---
with st.expander("📋 Regulatory Decision Boundary Table", expanded=True):
    boundary_data = []
    for lp in look_points:
        if lp <= total_n: continue
        
        s_req = next((s for s in range(lp+1) if (1 - beta.cdf(target_eff, prior_alpha+s, prior_beta+(lp-s))) > success_conf_req), "N/A")
        f_req = next((s for s in reversed(range(lp+1)) if get_enhanced_forecasts(s, lp, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)[0] <= bpp_futility_limit), -1)
        safe_req = next((s for s in range(lp+1) if (1 - beta.cdf(safe_limit, prior_alpha+s, prior_beta+(lp-s))) > safe_conf_req), "N/A")
        
        boundary_data.append({
            "N": lp, 
            "Success S ≥": s_req, 
            "Futility S ≤": f_req if f_req != -1 else "No Stop", 
            "Safety SAEs ≥": safe_req
        })
    
    if boundary_data: 
        st.table(pd.DataFrame(boundary_data))
    else: 
        st.write("Trial is at the final analysis point.")

# --- SNAPSHOT & AUDIT LOG ---
st.markdown("---")
if st.button("📥 Prepare Audit-Ready Snapshot"):
    report_data = {
        "Metric": ["Timestamp", "N", "Successes", "SAEs", "Success Threshold (%)", "Futility Threshold (%)", "PPoS", "ESS"],
        "Value": [datetime.now().isoformat(), total_n, successes, saes, f"{success_conf_req:.1%}", f"{bpp_futility_limit:.1%}", f"{bpp:.2%}", f"{a_eff+b_eff:.1f}"]
    }
    df_report = pd.DataFrame(report_data)
    st.download_button(label="Download CSV", data=df_report.to_csv(index=False).encode('utf-8'), 
                       file_name=f"Trial_Snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime='text/csv')
    st.table(df_report)
