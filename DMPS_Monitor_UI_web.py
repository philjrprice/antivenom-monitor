import streamlit as st
import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Universal Trial Monitor: Airtight Pro", layout="wide")

# --- SIDEBAR: INPUT SECTIONS ---
st.sidebar.header("📋 Current Trial Data")
max_n_val = st.sidebar.number_input("Maximum Sample Size (N)", 10, 500, 70)
total_n = st.sidebar.number_input("Total Patients Enrolled", 0, max_n_val, 20)
successes = st.sidebar.number_input("Total Successes", 0, total_n, value=min(14, total_n))
saes = st.sidebar.number_input("Serious Adverse Events (SAEs)", 0, total_n, value=min(1, total_n))

# --- DATA INTEGRITY ---
if successes > total_n:
    st.error("⚠️ Data Integrity Error: Successes cannot exceed total patients enrolled.")
    st.stop()
if saes > total_n:
    st.error("⚠️ Data Integrity Error: SAEs cannot exceed total patients enrolled.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Study Parameters")

with st.sidebar.expander("Base Study Priors", expanded=True):
    prior_alpha = st.slider("Prior Successes (Alpha)", 0.1, 10.0, 1.0, step=0.1)
    prior_beta = st.slider("Prior Failures (Beta)", 0.1, 10.0, 1.0, step=0.1)

with st.sidebar.expander("Adaptive Timing & Look Points", expanded=True):
    min_interim = st.number_input("Min N before first check", 1, max_n_val, 14)
    check_cohort = st.number_input("Check every X patients (Cohort)", 1, 20, 5)

with st.sidebar.expander("Success & Futility Rules"):
    null_eff = st.slider("Null Efficacy (p0) (%)", 0.1, 1.0, 0.50)
    target_eff = st.slider("Target Efficacy (p1) (%)", 0.1, 1.0, 0.60)
    dream_eff = st.slider("Goal/Dream Efficacy (%)", 0.1, 1.0, 0.70)
    success_conf_req = st.slider("Success Confidence Req.", 0.5, 0.99, 0.74)
    bpp_futility_limit = st.slider("BPP Futility Limit", 0.01, 0.20, 0.05)

with st.sidebar.expander("Safety Rules", expanded=True):
    safe_limit = st.slider("SAE Upper Limit (%)", 0.05, 0.50, 0.15)
    safe_conf_req = st.slider("Safety Stop Confidence", 0.5, 0.99, 0.90)

with st.sidebar.expander("Sensitivity Prior Settings"):
    opt_p = st.slider("Optimistic Prior Weight", 1, 10, 4)
    skp_p = st.slider("Skeptical Prior Weight", 1, 10, 4)

with st.sidebar.expander("Equivalence & Heatmap Settings"):
    equiv_bound = st.slider("Equivalence Bound (+/-)", 0.01, 0.10, 0.05)
    include_heatmap = st.checkbox("Generate Risk-Benefit Heatmap", value=True)

# --- MATH ENGINE ---
a_eff, b_eff = prior_alpha + successes, prior_beta + (total_n - successes)
a_safe, b_safe = prior_alpha + saes, prior_beta + (total_n - saes)

p_null = 1 - beta.cdf(null_eff, a_eff, b_eff)
p_target = 1 - beta.cdf(target_eff, a_eff, b_eff)
p_goal = 1 - beta.cdf(dream_eff, a_eff, b_eff)
p_toxic = 1 - beta.cdf(safe_limit, a_safe, b_safe)
p_equiv = beta.cdf(null_eff + equiv_bound, a_eff, b_eff) - beta.cdf(null_eff - equiv_bound, a_eff, b_eff)

eff_mean, eff_ci = a_eff / (a_eff + b_eff), beta.ppf([0.025, 0.975], a_eff, b_eff)
safe_mean, safe_ci = a_safe / (a_safe + b_safe), beta.ppf([0.025, 0.975], a_safe, b_safe)

@st.cache_data
def get_enhanced_forecasts(curr_s, curr_n, m_n, t_eff, s_conf, p_a, p_b):
    np.random.seed(42) 
    rem_n = m_n - curr_n
    if rem_n <= 0:
        is_success = (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf
        return 1.0 if is_success else 0.0, [curr_s, curr_s]
    
    future_rates = np.random.beta(p_a + curr_s, p_b + curr_n - curr_s, 5000)
    future_successes = np.random.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    
    ppos = np.mean(final_confs > s_conf)
    s_range = [int(np.percentile(total_proj_s, 5)), int(np.percentile(total_proj_s, 95))]
    return ppos, s_range

bpp, ps_range = get_enhanced_forecasts(successes, total_n, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)

skep_a, skep_b = 1 + successes, skp_p + (total_n - successes)
skep_prob = 1 - beta.cdf(target_eff, skep_a, skep_b)
evidence_shift = p_target / skep_prob if skep_prob > 0 else 1.0

# --- MAIN DASHBOARD ---
st.title("🛡️ Hybrid Antivenom Trial Monitor: Airtight Pro")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample N", f"{total_n}/{max_n_val}")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}", help="Posterior mean efficacy based on current data.")
m3.metric(f"P(>{target_eff:.0%})", f"{p_target:.1%}", help="Confidence efficacy exceeds target.")
m4.metric("Safety Risk", f"{p_toxic:.1%}", help="Probability SAE rate exceeds limit.")
m5.metric("PPoS (Final)", f"{bpp:.1%}", help="Predicted Probability of Success at Max N.")
m6.metric("Prior Weight", f"{prior_alpha + prior_beta:.1f}", help="Information weight of priors.")

st.caption(f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}** | Prob Equivalence: **{p_equiv:.1%}**")
st.markdown("---")

# Governing Rules
is_look_point = (total_n >= min_interim) and ((total_n - min_interim) % check_cohort == 0)
if p_toxic > safe_conf_req:
    st.error(f"🛑 **GOVERNING RULE: SAFETY STOP.** Risk ({p_toxic:.1%}) exceeds {safe_conf_req:.0%} threshold.")
elif is_look_point:
    if bpp < bpp_futility_limit: st.warning(f"⚠️ **GOVERNING RULE: FUTILITY STOP.** PPoS ({bpp:.1%}) below floor.")
    elif p_target > success_conf_req: st.success(f"✅ **GOVERNING RULE: EFFICACY SUCCESS.** Evidence achieved at {p_target:.1%}.")
    else: st.info(f"🛡️ **GOVERNING RULE: CONTINUE.** Interim check at N={total_n} is indeterminate.")
elif total_n < min_interim:
    st.info(f"⏳ **STATUS: LEAD-IN.** Enrollment phase; first check at N={min_interim}.")
else:
    next_check = total_n + (check_cohort - (total_n - min_interim) % check_cohort)
    st.info(f"🧬 **STATUS: MONITORING.** Trial between cohorts. Next check at N={next_check}.")

# Decision Corridors
st.subheader("📈 Trial Decision Corridors")
look_points = [min_interim + (i * check_cohort) for i in range(100) if (min_interim + (i * check_cohort)) <= max_n_val]
viz_n = np.array(look_points)
succ_line, fut_line = [], []

for lp in viz_n:
    s_req = next((s for s in range(lp+1) if (1 - beta.cdf(target_eff, prior_alpha+s, prior_beta+(lp-s))) > success_conf_req), lp+1)
    f_req = next((s for s in reversed(range(lp+1)) if get_enhanced_forecasts(s, lp, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)[0] <= bpp_futility_limit), -1)
    succ_line.append(s_req)
    fut_line.append(max(0, f_req))

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=viz_n, y=succ_line, name="Success Boundary", line=dict(color='green', dash='dash')))
fig_corr.add_trace(go.Scatter(x=viz_n, y=fut_line, name="Futility Boundary", line=dict(color='red', dash='dash')))
fig_corr.add_trace(go.Scatter(x=[total_n], y=[successes], mode='markers+text', name="Current Position", text=["Current"], textposition="top center", marker=dict(size=12, color='blue')))
fig_corr.update_layout(title="Sequential Stop Boundaries", xaxis_title="Sample Size (N)", yaxis_title="Successes (S)")
st.plotly_chart(fig_corr, use_container_width=True)

# Diagnostics & Sensitivity
col_op1, col_op2 = st.columns(2)
with col_op1:
    st.write("### Operational Info")
    st.metric("Effective Sample Size (ESS)", f"{a_eff + b_eff:.1f}", 
              help="Total information content (Current N + Prior Weight).")
    st.write(f"95% Efficacy CI: **[{eff_ci[0]:.1%}, {eff_ci[1]:.1%}]**")

with col_op2:
    st.write("### Sensitivity Analysis")
    priors = [("Neutral (Current)", prior_alpha, prior_beta), ("Optimistic", opt_p, 1), ("Skeptical", 1, skp_p)]
    for name, pa, pb in priors:
        p_s = 1 - beta.cdf(target_eff, pa + successes, pb + (total_n - successes))
        st.write(f"{name}: **{p_s:.1%}**")
        if "Neutral" in name:
            st.metric("Bayes Factor (BF₁₀)", f"{evidence_shift:.2f}x", 
                      help="Likelihood ratio supporting the efficacy hypothesis.")

# Regulatory Table
with st.expander("📋 Regulatory Decision Boundary Table", expanded=True):
    boundary_data = []
    for lp in look_points:
        if lp <= total_n: continue
        s_req = next((s for s in range(lp+1) if (1 - beta.cdf(target_eff, prior_alpha+s, prior_beta+(lp-s))) > success_conf_req), "N/A")
        f_req = next((s for s in reversed(range(lp+1)) if get_enhanced_forecasts(s, lp, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)[0] <= bpp_futility_limit), -1)
        safe_req = next((s for s in range(lp+1) if (1 - beta.cdf(safe_limit, prior_alpha+s, prior_beta+(lp-s))) > safe_conf_req), "N/A")
        boundary_data.append({
            "N": lp, "Success S ≥": s_req, 
            "Futility S ≤": f_req if f_req != -1 else "No Stop", 
            "Safety SAEs ≥": safe_req
        })
    if boundary_data:
        st.table(pd.DataFrame(boundary_data))
    else:
        st.write("Trial is at the final analysis point.")

# Export
st.markdown("---")
if st.button("📥 Export Audit-Ready Snapshot"):
    report_data = {
        "Metric": ["Timestamp", "N", "Successes", "SAEs", "Success Conf Req", "Futility Limit", "PPoS", "ESS"],
        "Value": [datetime.now().isoformat(), total_n, successes, saes, f"{success_conf_req:.1%}", f"{bpp_futility_limit:.1%}", f"{bpp:.1%}", f"{a_eff+b_eff:.1f}"]
    }
    st.download_button("Download CSV", pd.DataFrame(report_data).to_csv(index=False).encode('utf-8'), f"Trial_Snapshot_{datetime.now().strftime('%Y%m%d')}.csv")
