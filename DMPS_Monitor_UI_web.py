import streamlit as st
import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Universal Trial Monitor: Airtight v16", layout="wide")

# --- DATA INTEGRITY GATEWAY ---
st.sidebar.header("📋 Current Trial Data")
max_n_val = st.sidebar.number_input("Maximum Sample Size (N)", 10, 500, 70)
total_n = st.sidebar.number_input("Total Patients Enrolled", 0, max_n_val, 20)
successes = st.sidebar.number_input("Total Successes", 0, total_n, value=min(14, total_n))
saes = st.sidebar.number_input("Serious Adverse Events (SAEs)", 0, total_n, value=min(1, total_n))

# Airtight Validation logic
if total_n < (successes + saes):
    st.error("⚠️ Data Entry Error: Total N cannot be less than Successes + SAEs.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Regulatory Parameters")

with st.sidebar.expander("Base Study Priors", expanded=True):
    prior_alpha = st.slider("Prior Successes (Alpha)", 0.1, 10.0, 1.0, step=0.1)
    prior_beta = st.slider("Prior Failures (Beta)", 0.1, 10.0, 1.0, step=0.1)

with st.sidebar.expander("Adaptive Timing & Look Points", expanded=True):
    min_interim = st.number_input("Min N before first check", 1, max_n_val, 14)
    check_cohort = st.number_input("Check every X patients (Cohort)", 1, 20, 5)

with st.sidebar.expander("Success & Futility Rules"):
    null_eff = st.slider("Null Efficacy (p0) (%)", 0.1, 1.0, 0.50)
    target_eff = st.slider("Target Efficacy (p1) (%)", 0.1, 1.0, 0.60)
    success_conf_req = st.slider("Success Confidence Req.", 0.5, 0.99, 0.74)
    bpp_futility_limit = st.slider("BPP Futility Limit", 0.01, 0.20, 0.05)

with st.sidebar.expander("Safety Rules", expanded=True):
    safe_limit = st.slider("SAE Upper Limit (%)", 0.05, 0.50, 0.15)
    safe_conf_req = st.slider("Safety Stop Confidence", 0.5, 0.99, 0.90)

# --- MATH ENGINE ---
a_eff, b_eff = prior_alpha + successes, prior_beta + (total_n - successes)
a_safe, b_safe = prior_alpha + saes, prior_beta + (total_n - saes)

p_target = 1 - beta.cdf(target_eff, a_eff, b_eff)
p_toxic = 1 - beta.cdf(safe_limit, a_safe, b_safe)
eff_mean = a_eff / (a_eff + b_eff)

@st.cache_data
def get_ppos(curr_s, curr_n, m_n, t_eff, s_conf, p_a, p_b):
    np.random.seed(42) 
    rem_n = m_n - curr_n
    if rem_n <= 0: return (1.0 if (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf else 0.0)
    future_rates = np.random.beta(p_a + curr_s, p_b + curr_n - curr_s, 5000)
    future_successes = np.random.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    return np.mean(final_confs > s_conf)

ppos = get_ppos(successes, total_n, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)

# --- MAIN UI ---
st.title("🛡️ Airtight Bayesian Trial Monitor")

# Action Logic
is_look_point = (total_n >= min_interim) and ((total_n - min_interim) % check_cohort == 0)

if p_toxic > safe_conf_req:
    st.error(f"🛑 **GOVERNING RULE: SAFETY STOP.** SAE Risk ({p_toxic:.1%}) ≥ {safe_conf_req:.0%}.")
elif is_look_point:
    if ppos < bpp_futility_limit: st.warning(f"⚠️ **GOVERNING RULE: FUTILITY STOP.** PPoS ({ppos:.1%}) < {bpp_futility_limit:.0%}.")
    elif p_target > success_conf_req: st.success(f"✅ **GOVERNING RULE: EFFICACY SUCCESS.** Prob. Efficacy ({p_target:.1%}) ≥ {success_conf_req:.0%}.")
    else: st.info("🧬 **STATUS: CONTINUE.** Current data does not meet stop criteria.")
else:
    st.info("⌛ **STATUS: MONITORING.** Trial between scheduled interim looks.")

# Boundary Visualization (New)
st.subheader("📈 Decision Trajectory")
look_points = [min_interim + (i * check_cohort) for i in range(100) if (min_interim + (i * check_cohort)) <= max_n_val]
bounds = []
for lp in look_points:
    s_req = next((s for s in range(lp+1) if (1 - beta.cdf(target_eff, prior_alpha+s, prior_beta+(lp-s))) > success_conf_req), lp)
    bounds.append({"N": lp, "Success Threshold": s_req})

df_bounds = pd.DataFrame(bounds)
fig_traj = px.line(df_bounds, x="N", y="Success Threshold", title="Success Boundary (S required at N)")
fig_traj.add_scatter(x=[total_n], y=[successes], mode='markers', name='Current Trial', marker=dict(size=12, color='red'))
st.plotly_chart(fig_traj, use_container_width=True)

# Final Audit Export
if st.button("📥 Generate Regulatory Audit Log"):
    log = pd.DataFrame({
        "Variable": ["Timestamp", "Enrolled N", "Successes", "SAEs", "PPoS", "Safety Risk"],
        "Value": [datetime.now(), total_n, successes, saes, f"{ppos:.4f}", f"{p_toxic:.4f}"]
    })
    st.download_button("Download Audit Trail", log.to_csv(index=False).encode('utf-8'), "Trial_Audit_v16.csv")
