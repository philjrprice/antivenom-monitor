import streamlit as st
import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go

st.set_page_config(page_title="Universal Trial Monitor: Hybrid", layout="wide")

# --- SIDEBAR: INPUT SECTIONS ---
st.sidebar.header("📋 Current Trial Data")
max_n_val = st.sidebar.number_input("Maximum Sample Size (N)", 10, 500, 70)
total_n = st.sidebar.number_input("Total Patients Enrolled", 0, max_n_val, 20)
successes = st.sidebar.number_input("Total Successes", 0, total_n, value=min(14, total_n))
saes = st.sidebar.number_input("Serious Adverse Events (SAEs)", 0, total_n, value=min(1, total_n))

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Study Parameters")

with st.sidebar.expander("Base Study Priors", expanded=False):
    prior_alpha = st.number_input("Prior Successes (Alpha)", 0.1, 10.0, 1.0)
    prior_beta = st.number_input("Prior Failures (Beta)", 0.1, 10.0, 1.0)

with st.sidebar.expander("Success Thresholds", expanded=True):
    null_eff = st.slider("Null Efficacy (p0)", 0.1, 0.9, 0.5)
    target_eff = st.slider("Target Efficacy (p1)", 0.1, 0.9, 0.7)
    dream_eff = st.slider("Goal Efficacy", 0.1, 0.9, 0.8)

with st.sidebar.expander("Safety Parameters", expanded=True):
    safe_limit = st.slider("SAE Upper Limit", 0.05, 0.4, 0.15)
    s_alpha = st.number_input("Saf Prior Events", 0.1, 5.0, 1.0)
    s_beta = st.number_input("Saf Prior Non-Events", 1.0, 20.0, 9.0)

# --- MATH ENGINE ---
post_a, post_b = prior_alpha + successes, prior_beta + (total_n - successes)
s_post_a, s_post_b = s_alpha + saes, s_beta + (total_n - saes)

p_null = 1 - beta.cdf(null_eff, post_a, post_b)
p_target = 1 - beta.cdf(target_eff, post_a, post_b)
p_goal = 1 - beta.cdf(dream_eff, post_a, post_b)
p_toxic = 1 - beta.cdf(safe_limit, s_post_a, s_post_b)

eff_mean = post_a / (post_a + post_b)
eff_ci = beta.interval(0.95, post_a, post_b)
safe_mean = s_post_a / (s_post_a + s_post_b)

# BPP Success Forecast
rem_n = max_n_val - total_n
if rem_n > 0:
    future_s = np.random.binomial(rem_n, eff_mean, 5000)
    final_post_a = post_a + future_s
    final_post_b = post_b + (rem_n - future_s)
    bpp = np.mean((1 - beta.cdf(null_eff, final_post_a, final_post_b)) > 0.95)
else:
    bpp = 1.0 if p_null > 0.95 else 0.0

# --- UI LAYOUT ---
st.title("📊 Bayesian Adaptive Trial Monitor")

c1, c2 = st.columns(2)

with c1:
    fig_eff = go.Figure(go.Indicator(
        mode = "gauge+number", value = p_null * 100,
        title = {'text': f"Efficacy Confidence (> {null_eff:.0%})"},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"},
                 'steps': [{'range': [0, 70], 'color': "lightgray"}, {'range': [70, 95], 'color': "royalblue"}, {'range': [95, 100], 'color': "green"}]}))
    st.plotly_chart(fig_eff, use_container_width=True)

with c2:
    fig_saf = go.Figure(go.Indicator(
        mode = "gauge+number", value = p_toxic * 100,
        title = {'text': f"Toxicity Risk (> {safe_limit:.0%})"},
        gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "darkred"},
                 'steps': [{'range': [0, 10], 'color': "green"}, {'range': [10, 50], 'color': "orange"}, {'range': [50, 100], 'color': "red"}]}))
    st.plotly_chart(fig_saf, use_container_width=True)



with st.expander("📋 Current Performance Metrics", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Efficacy Summary**")
        st.write(f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}**")
        st.write(f"Prob > Target ({target_eff:.0%}): **{p_target:.1%}**")
        st.write(f"Prob > Goal ({dream_eff:.0%}): **{p_goal:.1%}**")
        st.write(f"95% CI: [{eff_ci[0]:.1%} - {eff_ci[1]:.1%}]")
    with col2:
        st.markdown("**Safety Summary**")
        st.write(f"Mean Toxicity: {safe_mean:.1%}")
        st.write(f"Prob > Limit ({safe_limit:.0%}): **{p_toxic:.1%}**")
    with col3:
        st.markdown("**Operational Info**")
        st.write(f"BPP Success Forecast: {bpp:.1%}")
        st.write(f"Prior Strength: {prior_alpha + prior_beta} pts")

# --- SENSITIVITY ANALYSIS (ENHANCED) ---
st.subheader("🧪 Sensitivity Analysis: Prior Robustness")
st.markdown("This section shows how the trial conclusion changes under different starting assumptions.")

opt_p = 4.0
skp_p = 4.0
priors_list = [
    (f"Optimistic ({opt_p}:1)", opt_p, 1.0), 
    ("Neutral (1:1)", 1.0, 1.0), 
    (f"Skeptical (1:{skp_p})", 1.0, skp_p)
]

cols = st.columns(3)
for i, (name, a_pri, b_pri) in enumerate(priors_list):
    # Re-calculate posterior for specific prior
    curr_a, curr_b = a_pri + successes, b_pri + (total_n - successes)
    s_p_null = 1 - beta.cdf(null_eff, curr_a, curr_b)
    s_p_target = 1 - beta.cdf(target_eff, curr_a, curr_b)
    s_p_goal = 1 - beta.cdf(dream_eff, curr_a, curr_b)
    s_mean = curr_a / (curr_a + curr_b)
    
    with cols[i]:
        st.info(f"**{name}**")
        st.metric("Mean Estimate", f"{s_mean:.1%}")
        # RESTORED: Full stats for each prior mindset
        st.write(f"Prob > Null: **{s_p_null:.1%}**")
        st.write(f"Prob > Target: **{s_p_target:.1%}**")
        st.write(f"Prob > Goal: **{s_p_goal:.1%}**")



st.markdown("---")
st.caption("Universal Trial Monitor v2.1 | Bayesian decision support for adaptive clinical designs.")
