import streamlit as st
import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go

st.set_page_config(page_title="Hybrid Antivenom Trial Monitor", layout="wide")

# --- SIDEBAR: INPUT SECTIONS ---
st.sidebar.header("📋 Current Trial Data")
max_n_val = st.sidebar.number_input("Maximum Sample Size (N)", 10, 500, 70)
total_n = st.sidebar.number_input("Total Patients Enrolled", 0, max_n_val, 20)
successes = st.sidebar.number_input("Total Successes", 0, total_n, 14)
saes = st.sidebar.number_input("Serious Adverse Events (SAEs)", 0, total_n, 1)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Timing Controls")
min_interim = st.sidebar.number_input("Min N before first check", 1, max_n_val, 14)
check_cohort = st.sidebar.number_input("Check every X patients (Cohort)", 1, 20, 5)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Study Rules")

with st.sidebar.expander("Efficacy & Futility Rules"):
    target_eff = st.slider("Target Efficacy (%)", 0.1, 1.0, 0.60)
    success_conf_req = st.slider("Success Confidence Req.", 0.5, 0.99, 0.74)
    bpp_futility_limit = st.slider("BPP Futility Limit", 0.01, 0.20, 0.05)

with st.sidebar.expander("Safety Rules"):
    safe_limit = st.slider("SAE Upper Limit (%)", 0.05, 0.50, 0.15)
    safe_conf_req = st.sidebar.slider("Safety Stop Confidence", 0.5, 0.99, 0.90)

with st.sidebar.expander("Sensitivity Prior Settings"):
    opt_p = st.slider("Optimistic Prior Weight", 1, 10, 4)
    skp_p = st.slider("Skeptical Prior Weight", 1, 10, 4)

# --- MATH ENGINE ---
def get_bpp(curr_s, curr_n, m_n, t_eff, s_conf, p_a=1.0, p_b=1.0):
    rem_n = m_n - curr_n
    if rem_n <= 0:
        return 1.0 if (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf else 0.0
    future_rates = np.random.beta(p_a + curr_s, p_b + curr_n - curr_s, 1000)
    future_successes = np.random.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    return np.mean(final_confs > s_conf)

# Base Calculations (using neutral 1,1 prior)
a_eff, b_eff = 1 + successes, 1 + (total_n - successes)
a_safe, b_safe = 1 + saes, 1 + (total_n - saes)

eff_mean = a_eff / (a_eff + b_eff)
p_target = 1 - beta.cdf(target_eff, a_eff, b_eff)
p_toxic = 1 - beta.cdf(safe_limit, a_safe, b_safe)
bpp = get_bpp(successes, total_n, max_n_val, target_eff, success_conf_req)

# --- HYBRID DECISION LOGIC ---
is_at_min_n = total_n >= min_interim
is_on_cohort_beat = (total_n - min_interim) % check_cohort == 0
is_look_point = is_at_min_n and is_on_cohort_beat

# --- MAIN DASHBOARD ---
st.title("🐍 Hybrid Antivenom Trial Monitor")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Current N", f"{total_n}/{max_n_val}")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}")
m3.metric("Efficacy Prob.", f"{p_target:.1%}")
m4.metric("Safety Risk", f"{p_toxic:.1%}")
m5.metric("Success Forecast (BPP)", f"{bpp:.1%}")

st.markdown("---")
if p_toxic > safe_conf_req:
    st.error(f"🚨 **CRITICAL SAFETY STOP:** Probability of Toxicity ({p_toxic:.1%}) exceeds limit.")
elif is_look_point:
    if bpp < bpp_futility_limit:
        st.warning(f"🛑 **STOP: FUTILITY TRIGGERED:** Forecast ({bpp:.1%}) below limit.")
    elif p_target > success_conf_req:
        st.success(f"✅ **EFFICACY MET:** Evidence is significant for >{target_eff:.0%} efficacy.")
    else:
        st.info(f"🛡️ **INTERIM CHECK AT N={total_n}:** No triggers hit. Continue to next cohort.")
elif not is_at_min_n:
    st.info(f"⏳ **LEAD-IN PHASE:** Waiting for Min N ({min_interim}) to begin adaptive monitoring.")
else:
    next_check = total_n + (check_cohort - (total_n - min_interim) % check_cohort)
    st.info(f"🧬 **MONITORING:** Enrolling next cohort. Next review at N={next_check}.")

# --- SENSITIVITY ANALYSIS ---
st.subheader("🧪 Sensitivity Analysis (Prior Robustness Check)")
priors_list = [
    (f"Optimistic ({opt_p}:1)", opt_p, 1), 
    ("Neutral (1:1)", 1, 1), 
    (f"Skeptical (1:{skp_p})", 1, skp_p)
]

cols = st.columns(3)
target_probs = []
for i, (name, ap, bp) in enumerate(priors_list):
    ae_s, be_s = ap + successes, bp + (total_n - successes)
    as_s, bs_s = bp + saes, ap + (total_n - saes) # Safety sensitivity usually flips the prior weight
    
    p_t_s = 1 - beta.cdf(target_eff, ae_s, be_s)
    target_probs.append(p_t_s)
    
    with cols[i]:
        st.info(f"**{name}**")
        st.write(f"Mean Eff: {ae_s/(ae_s+be_s):.1%}")
        st.write(f"P(>{target_eff:.0%}): {p_t_s:.1%}")
        st.write(f"Mean Tox: {as_s/(as_s+bs_s):.1%}")
        st.write(f"P(>{safe_limit:.0%}): {1-beta.cdf(safe_limit, as_s, bs_s):.1%}")

spread = max(target_probs) - min(target_probs)
status = "ROBUST" if spread < 0.15 else "FRAGILE"
st.markdown(f"**Interpretation:** Results are **{status}** ({spread:.1%} variance across priors).")


# --- PLOTS ---
st.subheader("Statistical Distributions")
x = np.linspace(0, 1, 500)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_eff, b_eff), name="Efficacy PDF", line=dict(color='#2980b9', width=3)))
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_safe, b_safe), name="Safety PDF", line=dict(color='#c0392b', width=3)))
fig.update_layout(xaxis=dict(range=[0, 1]), height=400)
st.plotly_chart(fig, use_container_width=True)
