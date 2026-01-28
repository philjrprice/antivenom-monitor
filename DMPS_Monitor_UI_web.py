import streamlit as st
import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Universal Trial Monitor: Airtight v5", layout="wide")

# --- SIDEBAR: INPUT SECTIONS ---
st.sidebar.header("📋 Current Trial Data")
max_n_val = st.sidebar.number_input("Maximum Sample Size (N)", 10, 500, 70)
total_n = st.sidebar.number_input("Total Patients Enrolled", 0, max_n_val, 20)
successes = st.sidebar.number_input("Total Successes", 0, total_n, value=min(14, total_n))
saes = st.sidebar.number_input("Serious Adverse Events (SAEs)", 0, total_n, value=min(1, total_n))

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Regulatory Parameters")

with st.sidebar.expander("Base Study Priors", expanded=True):
    prior_alpha = st.slider("Prior Successes (Alpha)", 0.1, 10.0, 1.0, step=0.1)
    prior_beta = st.slider("Prior Failures (Beta)", 0.1, 10.0, 1.0, step=0.1)

with st.sidebar.expander("Adaptive Timing & Look Points"):
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

@st.cache_data
def get_cached_forecasts(curr_s, curr_n, m_n, t_eff, s_conf, p_a, p_b):
    np.random.seed(42) # Reproducibility Lock
    rem_n = m_n - curr_n
    if rem_n <= 0:
        return (1.0 if (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf else 0.0), [curr_s, curr_s]
    
    future_rates = np.random.beta(p_a + curr_s, p_b + curr_n - curr_s, 5000)
    future_successes = np.random.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    
    return np.mean(final_confs > s_conf), [int(np.percentile(total_proj_s, 5)), int(np.percentile(total_proj_s, 95))]

bpp, ps_range = get_cached_forecasts(successes, total_n, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)

# --- UI LAYOUT ---
st.title("🛡️ Airtight Bayesian Trial Monitor")

# Status Messaging (Governing Rules)
is_look_point = (total_n >= min_interim) and ((total_n - min_interim) % check_cohort == 0)

if p_toxic > safe_conf_req:
    st.error(f"🚨 **CRITICAL: SAFETY STOP TRIGGERED.** Prob. Toxicity ({p_toxic:.1%}) ≥ {safe_conf_req:.0%}. Discontinue enrollment immediately.")
    status_color = "red"
elif is_look_point:
    if bpp < bpp_futility_limit:
        st.warning(f"🛑 **STOP: FUTILITY TRIGGERED.** Predictive Success ({bpp:.1%}) is below threshold.")
        status_color = "orange"
    elif p_target > success_conf_req:
        st.success(f"✅ **STOP: EFFICACY ACHIEVED.** Prob. Efficacy ({p_target:.1%}) ≥ {success_conf_req:.1%}.")
        status_color = "green"
    else:
        st.info("🧬 **CONTINUE ENROLLMENT.** Data is indeterminate at this interim check.")
        status_color = "blue"
else:
    st.info(f"⌛ **MONITORING.** Next interim check at N={total_n + (check_cohort - (total_n - min_interim) % check_cohort) if total_n >= min_interim else min_interim}.")
    status_color = "grey"

# Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Enrolled N", f"{total_n} / {max_n_val}")
col2.metric("Mean Efficacy", f"{eff_mean:.1%}")
col3.metric(f"P(>{target_eff:.0%})", f"{p_target:.1%}")
col4.metric("SAE Risk", f"{p_toxic:.1%}", delta_color="inverse")
col5.metric("PPoS", f"{bpp:.1%}")

# Visuals
st.subheader("Statistical Distributions & Shaded 95% CI")
x = np.linspace(0, 1, 500)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_eff, b_eff), name="Efficacy", line=dict(color='#2980b9', width=3)))
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_safe, b_safe), name="Safety", line=dict(color='#c0392b', width=3)))
fig.add_vline(x=target_eff, line_dash="dash", line_color="green", annotation_text="Target")
fig.add_vline(x=safe_limit, line_dash="dash", line_color="red", annotation_text="Safety Limit")
fig.update_layout(height=400, margin=dict(l=0,r=0,t=20,b=0), xaxis_title="Efficacy/SAE Rate")
st.plotly_chart(fig, use_container_width=True)



# Decision Boundary Table
with st.expander("📋 Regulatory Decision Boundary Table", expanded=True):
    look_points = [min_interim + (i * check_cohort) for i in range(100) if (min_interim + (i * check_cohort)) <= max_n_val]
    boundary_data = []
    for lp in look_points:
        if lp <= total_n: continue
        s_req = next((s for s in range(lp+1) if (1 - beta.cdf(target_eff, prior_alpha+s, prior_beta+(lp-s))) > success_conf_req), "Unreachable")
        f_req = next((s for s in reversed(range(lp+1)) if get_cached_forecasts(s, lp, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)[0] > bpp_futility_limit), "Immediate Stop")
        boundary_data.append({"Look Point (N)": lp, "Success if S ≥": s_req, "Futility if S ≤": f_req})
    if boundary_data:
        st.table(pd.DataFrame(boundary_data))
    else:
        st.write("Trial is at final stage.")

# Sensitivity Section
st.subheader("🧪 Sensitivity: Prior Robustness")
c1, c2, c3 = st.columns(3)
priors = [("Optimistic", opt_p, 1), ("Neutral", 1, 1), ("Skeptical", 1, skp_p)]
for i, (label, ap, bp) in enumerate(priors):
    p_t = 1 - beta.cdf(target_eff, ap + successes, bp + (total_n - successes))
    with [c1, c2, c3][i]:
        st.metric(f"{label} P(>Target)", f"{p_t:.1%}")
        st.caption(f"Based on {ap}:{bp} prior weight")

# Export
st.markdown("---")
if st.button("📥 Export Audit-Ready Snapshot"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    report = pd.DataFrame({
        "Parameter": ["N", "Successes", "SAEs", "Target Eff", "PPoS", "Mean Eff", "Prior Weight", "Timestamp"],
        "Value": [total_n, successes, saes, target_eff, bpp, eff_mean, prior_alpha+prior_beta, timestamp]
    })
    st.download_button("Confirm Download", report.to_csv(index=False).encode('utf-8'), f"Trial_Audit_{datetime.now().strftime('%Y%m%d')}.csv")
