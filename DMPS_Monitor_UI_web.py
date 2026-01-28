import streamlit as st
import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go

st.set_page_config(page_title="Universal Trial Monitor: Hybrid Edition", layout="wide")

# --- SIDEBAR: INPUT SECTIONS ---
st.sidebar.header("📋 Current Trial Data")
max_n_val = st.sidebar.number_input("Maximum Sample Size (N)", 10, 500, 70, 
    help="The total number of patients planned for the trial.")
total_n = st.sidebar.number_input("Total Patients Enrolled", 0, max_n_val, 20,
    help="Number of patients whose outcomes are currently known.")
successes = st.sidebar.number_input("Total Successes", 0, total_n, 14,
    help="Number of patients who met the primary efficacy endpoint.")
saes = st.sidebar.number_input("Serious Adverse Events (SAEs)", 0, total_n, 1,
    help="Number of patients who experienced a Serious Adverse Event.")

st.sidebar.markdown("---")
st.sidebar.header("⏱️ Adaptive Timing Controls")
min_interim = st.sidebar.number_input("Min N before first check", 1, max_n_val, 14, 
    help="The 'Waiting Room'. No efficacy or futility stops will occur before this number of patients.")
check_cohort = st.sidebar.number_input("Check every X patients (Cohort)", 1, 20, 5, 
    help="The 'Heartbeat'. After Min N is reached, decisions only trigger at these intervals.")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Study Rules")

with st.sidebar.expander("Efficacy & Futility Rules"):
    target_eff = st.slider("Target Efficacy (%)", 0.1, 1.0, 0.60, help="The success rate you want to prove.")
    null_eff = st.slider("Null Efficacy (%)", 0.1, 1.0, 0.50, help="The baseline/standard of care rate.")
    success_conf_req = st.slider("Success Confidence Req.", 0.5, 0.99, 0.74, help="Bayesian probability needed to declare a win.")
    bpp_futility_limit = st.slider("BPP Futility Limit", 0.01, 0.20, 0.05, help="Stop if the chance of final success drops below this.")

with st.sidebar.expander("Safety Rules"):
    safe_limit = st.slider("SAE Upper Limit (%)", 0.05, 0.50, 0.15, help="Max allowable toxicity rate.")
    safe_conf_req = st.slider("Safety Stop Confidence", 0.5, 0.99, 0.90, help="Confidence needed to pull the plug for safety.")

# --- MATH ENGINE ---
def get_bpp(curr_s, curr_n, m_n, t_eff, s_conf):
    """Bayesian Predictive Probability of Success at Max N."""
    rem_n = m_n - curr_n
    if rem_n <= 0:
        return 1.0 if (1 - beta.cdf(t_eff, 1 + curr_s, 1 + curr_n - curr_s)) > s_conf else 0.0
    # Simulating future outcomes based on current posterior
    future_rates = np.random.beta(1 + curr_s, 1 + curr_n - curr_s, 1000)
    future_successes = np.random.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    final_confs = 1 - beta.cdf(t_eff, 1 + total_proj_s, 1 + (m_n - total_proj_s))
    return np.mean(final_confs > s_conf)

# Calculations
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

# Metrics Summary Row
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Current N", f"{total_n}/{max_n_val}")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}")
m3.metric("Efficacy Prob.", f"{p_target:.1%}")
m4.metric("Safety Risk", f"{p_toxic:.1%}")
m5.metric("Success Forecast (BPP)", f"{bpp:.1%}")

# Decision Status Boxes
st.markdown("---")
if p_toxic > safe_conf_req:
    st.error(f"🚨 **CRITICAL SAFETY STOP:** Probability of Toxicity ({p_toxic:.1%}) exceeds the {safe_conf_req:.0%} confidence limit. Recommend immediate suspension.")
elif is_look_point:
    if bpp < bpp_futility_limit:
        st.warning(f"🛑 **STOP: FUTILITY TRIGGERED:** At N={total_n}, the chance of eventual success ({bpp:.1%}) has dropped below the {bpp_futility_limit:.0%} threshold.")
    elif p_target > success_conf_req:
        st.success(f"✅ **EFFICACY MET:** At N={total_n}, there is {p_target:.1%} confidence that efficacy is >{target_eff:.0%}. Success criteria satisfied.")
    else:
        st.info(f"🛡️ **INTERIM CHECK AT N={total_n}:** No stop triggers hit. Continue to next cohort.")
elif not is_at_min_n:
    st.info(f"⏳ **PHASE 1 (Lead-in):** Trial is currently in the initial enrollment phase. Adaptive rules will activate at N={min_interim}.")
else:
    next_check = total_n + (check_cohort - (total_n - min_interim) % check_cohort)
    st.info(f"🧬 **PHASE 2 (Adaptive):** Monitoring ongoing. Next efficacy/futility review scheduled for patient N={next_check}.")

# --- PLOTS ---
st.subheader("Current Statistical Posteriors")
x = np.linspace(0, 1, 500)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_eff, b_eff), name="Efficacy Belief", line=dict(color='#2980b9', width=3)))
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_safe, b_safe), name="Safety Belief", line=dict(color='#c0392b', width=3)))
fig.add_vline(x=target_eff, line_dash="dash", line_color="green", annotation_text="Eff. Target")
fig.add_vline(x=safe_limit, line_dash="dash", line_color="black", annotation_text="Safe Limit")
fig.update_layout(xaxis=dict(range=[0, 1], title="Rate"), yaxis=dict(title="Density"), height=400)
st.plotly_chart(fig, use_container_width=True)
