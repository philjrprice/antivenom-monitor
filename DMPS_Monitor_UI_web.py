import streamlit as st
import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Universal Trial Monitor: Hybrid", layout="wide")

# --- SIDEBAR: INPUT SECTIONS ---
st.sidebar.header("📋 Current Trial Data")
max_n_val = st.sidebar.number_input("Maximum Sample Size (N)", 10, 500, 70)
total_n = st.sidebar.number_input("Total Patients Enrolled", 0, max_n_val, 20)
successes = st.sidebar.number_input("Total Successes", 0, total_n, value=min(14, total_n))
saes = st.sidebar.number_input("Serious Adverse Events (SAEs)", 0, total_n, value=min(1, total_n))

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

# --- MATH ENGINE ---
a_eff, b_eff = prior_alpha + successes, prior_beta + (total_n - successes)
a_safe, b_safe = prior_alpha + saes, prior_beta + (total_n - saes)

p_null = 1 - beta.cdf(null_eff, a_eff, b_eff)
p_target = 1 - beta.cdf(target_eff, a_eff, b_eff)
p_goal = 1 - beta.cdf(dream_eff, a_eff, b_eff)
p_toxic = 1 - beta.cdf(safe_limit, a_safe, b_safe)

eff_mean, eff_ci = a_eff / (a_eff + b_eff), beta.ppf([0.025, 0.975], a_eff, b_eff)
safe_mean, safe_ci = a_safe / (a_safe + b_safe), beta.ppf([0.025, 0.975], a_safe, b_safe)

# Bayesian Predictive Probability of Success (PPoS)
def get_enhanced_forecasts(curr_s, curr_n, m_n, t_eff, s_conf, p_a, p_b):
    np.random.seed(42) # Ensure audit reproducibility
    rem_n = m_n - curr_n
    if rem_n <= 0:
        is_success = (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf
        return 1.0 if is_success else 0.0, [curr_s, curr_s]
    
    future_rates = np.random.beta(p_a + curr_s, p_b + curr_n - curr_s, 2000)
    future_successes = np.random.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    
    ppos = np.mean(final_confs > s_conf)
    s_range = [int(np.percentile(total_proj_s, 5)), int(np.percentile(total_proj_s, 95))]
    return ppos, s_range

bpp, ps_range = get_enhanced_forecasts(successes, total_n, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)

# Bayes Factor calculation for evidence strength
skep_a, skep_b = 1 + successes, skp_p + (total_n - successes)
skep_prob = 1 - beta.cdf(target_eff, skep_a, skep_b)
evidence_shift = p_target / skep_prob if skep_prob > 0 else 1.0

# --- UI LAYOUT ---
st.title("🛡️ Universal Trial Monitor: Robust Version")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Progress", f"{total_n}/{max_n_val} N")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}")
m3.metric(f"P(>{target_eff:.0%})", f"{p_target:.1%}")
m4.metric("Safety Risk", f"{p_toxic:.1%}")
m5.metric("PPoS", f"{bpp:.1%}")
m6.metric("Prior Weight", f"{prior_alpha + prior_beta:.1f}")

st.markdown("---")

# Decision Logic
is_look_point = (total_n >= min_interim) and ((total_n - min_interim) % check_cohort == 0)

if p_toxic > safe_conf_req:
    st.error(f"🛑 **ACTION: STOP FOR SAFETY.** Risk of SAEs ({p_toxic:.1%}) exceeds threshold.")
elif is_look_point:
    if bpp < bpp_futility_limit: st.warning(f"⚠️ **ACTION: STOP FOR FUTILITY.** Forecasted success ({bpp:.1%}) too low.")
    elif p_target > success_conf_req: st.success(f"✅ **ACTION: STOP FOR SUCCESS.** Efficacy evidence achieved.")
    else: st.info("🧬 **ACTION: CONTINUE ENROLLMENT.** Data is currently indeterminate.")
else:
    st.info("⌛ **ACTION: MONITORING.** Trial is between scheduled look points.")

# Visualizations
c_left, c_right = st.columns([2, 1])

with c_left:
    st.subheader("Statistical Distributions")
    x = np.linspace(0, 1, 500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_eff, b_eff), name="Efficacy", line=dict(color='#2980b9')))
    fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_safe, b_safe), name="Safety (SAE)", line=dict(color='#c0392b')))
    fig.update_layout(height=400, xaxis_title="Rate", yaxis_title="Density", margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig, use_container_width=True)

with c_right:
    st.subheader("Sensitivity Analysis")
    p_opt = 1 - beta.cdf(target_eff, opt_p + successes, 1 + (total_n - successes))
    p_skp = 1 - beta.cdf(target_eff, 1 + successes, skp_p + (total_n - successes))
    st.write(f"**Optimistic Prior:** {p_opt:.1%}")
    st.write(f"**Neutral Prior:** {p_target:.1%}")
    st.write(f"**Skeptical Prior:** {p_skp:.1%}")
    st.write(f"**Bayes Factor (BF₁₀):** {evidence_shift:.2f}")

# The Decision Table
with st.expander("📋 Regulatory Decision Boundary Table", expanded=True):
    st.markdown("Calculated success/futility boundaries for all remaining look points:")
    look_points = [min_interim + (i * check_cohort) for i in range(100) if (min_interim + (i * check_cohort)) <= max_n_val]
    
    boundary_data = []
    for lp in look_points:
        if lp <= total_n: continue
        # Boundary for success stop
        s_stop = next((s for s in range(lp+1) if (1 - beta.cdf(target_eff, prior_alpha+s, prior_beta+(lp-s))) > success_conf_req), "N/A")
        # Boundary for futility stop
        f_stop = next((s for s in reversed(range(lp+1)) if get_enhanced_forecasts(s, lp, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)[0] > bpp_futility_limit), -1)
        boundary_data.append({"N": lp, "Stop for Success if S ≥": s_stop, "Stop for Futility if S ≤": f_stop})
    
    if boundary_data:
        st.table(pd.DataFrame(boundary_data))
    else:
        st.write("Final analysis stage reached.")

# Operating Characteristics (Robustness Check)
with st.expander("🔬 Operating Characteristics & Simulation"):
    if st.button("Run Simulation (1,000 Iterations)"):
        null_sims = np.random.binomial(max_n_val, null_eff, 1000)
        type_1 = sum([(1 - beta.cdf(target_eff, prior_alpha+s, prior_beta+(max_n_val-s))) > success_conf_req for s in null_sims])
        st.info(f"Estimated Final Type I Error: **{type_1/1000:.3f}** (Target usually < 0.05)")

# Export Results
st.markdown("---")
if st.button("📥 Export Audit-Ready Snapshot"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    report = pd.DataFrame({
        "Parameter": ["Total N", "Successes", "SAEs", "PPoS", "Mean Efficacy", "Safety Risk"],
        "Value": [total_n, successes, saes, f"{bpp:.2%}", f"{eff_mean:.2%}", f"{p_toxic:.2%}"]
    })
    st.download_button("Download CSV", report.to_csv(index=False).encode('utf-8'), f"Trial_Audit_{timestamp}.csv")
