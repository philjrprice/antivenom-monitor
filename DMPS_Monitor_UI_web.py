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

# Data Integrity Validation
if successes > total_n or saes > total_n:
    st.error("⚠️ Data Integrity Error: Successes/SAEs cannot exceed total enrolled.")
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
    return np.mean(final_confs > s_conf), [int(np.percentile(total_proj_s, 5)), int(np.percentile(total_proj_s, 95))]

bpp, ps_range = get_enhanced_forecasts(successes, total_n, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)
skep_a, skep_b = 1 + successes, skp_p + (total_n - successes)
evidence_shift = p_target / (1 - beta.cdf(target_eff, skep_a, skep_b)) if (1 - beta.cdf(target_eff, skep_a, skep_b)) > 0 else 1.0

# --- MAIN DASHBOARD ---
st.title("🛡️ Hybrid Antivenom Trial Monitor: Airtight Pro")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample N", f"{total_n}/{max_n_val}")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}")
m3.metric(f"P(>{target_eff:.0%})", f"{p_target:.1%}")
m4.metric("Safety Risk", f"{p_toxic:.1%}")
m5.metric("PPoS (Final)", f"{bpp:.1%}")
m6.metric("Prior ESS", f"{prior_alpha + prior_beta:.1f}")

st.markdown("---")

# Governing Hierarchy logic
is_look_point = (total_n >= min_interim) and ((total_n - min_interim) % check_cohort == 0)
if p_toxic > safe_conf_req:
    st.error(f"🛑 **GOVERNING RULE: SAFETY STOP.** Risk ({p_toxic:.1%}) exceeds {safe_conf_req:.0%}.")
elif is_look_point:
    if bpp < bpp_futility_limit: st.warning(f"⚠️ **FUTILITY STOP.** PPoS ({bpp:.1%}) below floor.")
    elif p_target > success_conf_req: st.success(f"✅ **EFFICACY SUCCESS.** Evidence achieved at {p_target:.1%}.")
    else: st.info(f"🛡️ **CONTINUE.** Interim check at N={total_n} is indeterminate.")
else: st.info(f"🧬 **STATUS: MONITORING.** Trial active. Lead-in or between cohorts.")

# Graph Row
st.subheader("Statistical Distributions")
x = np.linspace(0, 1, 500)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_eff, b_eff), name="Efficacy Belief", line=dict(color='#2980b9', width=3)))
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_safe, b_safe), name="Safety Belief", line=dict(color='#c0392b', width=3)))
fig.add_vline(x=target_eff, line_dash="dash", line_color="green", annotation_text="Target")
fig.add_vline(x=safe_limit, line_dash="dash", line_color="black", annotation_text="Safety Limit")
fig.update_layout(xaxis=dict(range=[0, 1]), height=400, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# Sensitivity Analysis
st.subheader("🧪 Sensitivity Analysis & Robustness")
priors_list = [(f"Optimistic ({opt_p}:1)", opt_p, 1), ("Neutral (1:1)", 1, 1), (f"Skeptical (1:{skp_p})", 1, skp_p)]
cols, target_probs = st.columns(3), []
for i, (name, ap, bp) in enumerate(priors_list):
    ae_s, be_s = ap + successes, bp + (total_n - successes)
    p_t_s = 1 - beta.cdf(target_eff, ae_s, be_s)
    target_probs.append(p_t_s)
    with cols[i]:
        st.info(f"**{name}**")
        st.write(f"Prob > Target: **{p_t_s:.1%}**")
        if "Neutral" in name: st.write(f"Bayes Factor: **{evidence_shift:.2f}x**")

# Robustness Interpretation
spread = max(target_probs) - min(target_probs)
st.markdown(f"**Interpretation:** Results are **{'ROBUST' if spread < 0.15 else 'SENSITIVE'}** ({spread:.1%} variance between prior mindsets).")

# Boundary Table with Safety Added
with st.expander("📋 Regulatory Decision Boundary Table", expanded=True):
    look_points = [min_interim + (i * check_cohort) for i in range(100) if (min_interim + (i * check_cohort)) <= max_n_val]
    boundary_data = []
    for lp in look_points:
        if lp <= total_n: continue
        s_req = next((s for s in range(lp+1) if (1 - beta.cdf(target_eff, prior_alpha+s, prior_beta+(lp-s))) > success_conf_req), "N/A")
        f_req = next((s for s in reversed(range(lp+1)) if get_enhanced_forecasts(s, lp, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)[0] > bpp_futility_limit), -1)
        # Added Robustness: Safety Stop boundary
        safe_stop_req = next((s for s in range(lp+1) if (1 - beta.cdf(safe_limit, prior_alpha+s, prior_beta+(lp-s))) > safe_conf_req), "N/A")
        boundary_data.append({"N": lp, "Efficacy Stop (S ≥)": s_req, "Futility Stop (S ≤)": f_req if f_req != -1 else "Stop", "Safety Stop (SAEs ≥)": safe_stop_req})
    if boundary_data: st.table(pd.DataFrame(boundary_data))
    else: st.write("Trial at final analysis.")

# Export Snapshot
st.markdown("---")
if st.button("📥 Export Audit-Ready Snapshot"):
    report_data = {"Metric": ["Timestamp", "N", "Successes", "SAEs", "Post Mean Eff", "Prob > Target", "Safety Risk", "PPoS", "Robustness Spread"],
                   "Value": [datetime.now().isoformat(), total_n, successes, saes, f"{eff_mean:.2%}", f"{p_target:.2%}", f"{p_toxic:.2%}", f"{bpp:.2%}", f"{spread:.2%}%"]}
    st.download_button("Download CSV", pd.DataFrame(report_data).to_csv(index=False).encode('utf-8'), f"Trial_Audit_{datetime.now().strftime('%Y%m%d')}.csv")
