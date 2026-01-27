import streamlit as st
import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go

st.set_page_config(page_title="Universal Trial Monitor", layout="wide")

# --- SIDEBAR: INPUT SECTIONS ---
st.sidebar.header("📋 Current Trial Data")
max_n_val = st.sidebar.number_input("Maximum Sample Size (N)", 10, 500, 70)
total_n = st.sidebar.number_input("Total Patients Enrolled", 0, max_n_val, 20)
successes = st.sidebar.number_input("Total Successes", 0, total_n, 14)
saes = st.sidebar.number_input("Serious Adverse Events (SAEs)", 0, total_n, 1)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Study Parameters")

with st.sidebar.expander("Base Study Priors", expanded=False):
    st.write("Set the 'Starting Belief' for this trial:")
    prior_alpha = st.number_input("Prior Successes (Alpha)", 0.1, 10.0, 1.0)
    prior_beta = st.number_input("Prior Failures (Beta)", 0.1, 10.0, 1.0)

with st.sidebar.expander("Success & Futility Rules"):
    target_eff = st.slider("Target Efficacy (%)", 0.1, 1.0, 0.60)
    null_eff = st.slider("Null Efficacy (%)", 0.1, 1.0, 0.50)
    dream_eff = st.slider("Goal/Dream Efficacy (%)", 0.1, 1.0, 0.70)
    success_conf_req = st.slider("Success Confidence Req.", 0.5, 0.99, 0.74)
    bpp_futility_limit = st.slider("BPP Futility Limit", 0.01, 0.20, 0.05)
    min_interim = st.number_input("Minimum N for Interim", 1, max_n_val, 14)

with st.sidebar.expander("Safety Rules"):
    safe_limit = st.slider("SAE Upper Limit (%)", 0.05, 0.50, 0.15)
    safe_conf_req = st.slider("Safety Stop Confidence", 0.5, 0.99, 0.90)

with st.sidebar.expander("Sensitivity Scenario Settings"):
    st.write("Customize the three priors used for robustness checks:")
    opt_p = st.slider("Optimistic Prior (Succ:Fail)", 1, 10, 4)
    flat_p = st.slider("Neutral Prior (Succ:Fail)", 1, 10, 1)
    skp_p = st.slider("Skeptical Prior (Succ:Fail)", 1, 10, 4)

# --- MATH ENGINE ---
def get_bpp(curr_s, curr_n, m_n, t_eff, s_conf, p_a, p_b):
    rem_n = m_n - curr_n
    if rem_n <= 0:
        return 1.0 if (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf else 0.0
    future_rates = np.random.beta(p_a + curr_s, p_b + curr_n - curr_s, 1000)
    future_successes = np.random.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    return np.mean(final_confs > s_conf)

# Calculations
a_eff, b_eff = prior_alpha + successes, prior_beta + (total_n - successes)
a_safe, b_safe = prior_alpha + saes, prior_beta + (total_n - saes)

eff_mean, eff_ci = a_eff / (a_eff + b_eff), beta.ppf([0.025, 0.975], a_eff, b_eff)
safe_mean, safe_ci = a_safe / (a_safe + b_safe), beta.ppf([0.025, 0.975], a_safe, b_safe)

p_target = 1 - beta.cdf(target_eff, a_eff, b_eff)
p_toxic = 1 - beta.cdf(safe_limit, a_safe, b_safe)
bpp = get_bpp(successes, total_n, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)

# --- MAIN DASHBOARD ---
st.title("🐍 Antivenom Foundation Bayesian Trial Monitor")

# Metrics Summary Row
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample N", f"{total_n}/{max_n_val}")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}")
m3.metric("Eff. Confidence", f"{p_target:.1%}")
m4.metric("Mean Safety", f"{safe_mean:.1%}")
m5.metric("Safety Risk", f"{p_toxic:.1%}")
m6.metric("Chance for Trial Success (BPP)", f"{bpp:.1%}")

# Decision Logic
if p_toxic > safe_conf_req:
    st.error(f"🚨 **CRITICAL SAFETY STOP:** Prob. Toxicity ({p_toxic:.1%}) exceeds limit.")
elif total_n >= min_interim and bpp < bpp_futility_limit:
    st.warning(f"🛑 **STOP: FUTILITY TRIGGERED:** Forecast ({bpp:.1%}) below limit.")
elif p_target > success_conf_req:
    st.success(f"✅ **EFFICACY MET:** Significant evidence for >{target_eff:.0%} efficacy.")
else:
    st.info("🛡️ **TRIAL ONGOING:** Monitoring data according to protocol.")

# --- PLOTS ---
st.subheader("Statistical Distributions (Posterior vs. Target)")
x = np.linspace(0, 1, 500)
fig = go.Figure()

# Efficacy with Shaded CI
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_eff, b_eff), name="Efficacy PDF", line=dict(color='#2980b9', width=3)))
x_ci_e = np.linspace(eff_ci[0], eff_ci[1], 100)
fig.add_trace(go.Scatter(x=np.concatenate([x_ci_e, x_ci_e[::-1]]), 
             y=np.concatenate([beta.pdf(x_ci_e, a_eff, b_eff), np.zeros(len(x_ci_e))]),
             fill='toself', fillcolor='rgba(41, 128, 185, 0.2)', line=dict(color='rgba(255,255,255,0)'), name="95% CI (Eff)"))

# Safety with Shaded CI
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_safe, b_safe), name="Safety PDF", line=dict(color='#c0392b', width=3)))
x_ci_s = np.linspace(safe_ci[0], safe_ci[1], 100)
fig.add_trace(go.Scatter(x=np.concatenate([x_ci_s, x_ci_s[::-1]]), 
             y=np.concatenate([beta.pdf(x_ci_s, a_safe, b_safe), np.zeros(len(x_ci_s))]),
             fill='toself', fillcolor='rgba(192, 57, 43, 0.2)', line=dict(color='rgba(255,255,255,0)'), name="95% CI (Safe)"))

fig.add_vline(x=target_eff, line_dash="dash", line_color="green", annotation_text="Target")
fig.add_vline(x=safe_limit, line_dash="dash", line_color="black", annotation_text="Limit")
fig.update_layout(xaxis=dict(range=[0, 1]), legend=dict(orientation="h", yanchor="bottom", y=1.02), height=450)
st.plotly_chart(fig, use_container_width=True)

# --- EXPANDED STATISTICAL BREAKDOWN ---
with st.expander("📊 Full Statistical Breakdown", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Efficacy Summary**")
    c1.write(f"**Mean Efficacy Rate: {eff_mean:.1%}**")
    c1.write(f"95% Credible Interval: [{eff_ci[0]:.1%} - {eff_ci[1]:.1%}]")
    c1.write(f"Prob > Null ({null_eff:.0%}): {1 - beta.cdf(null_eff, a_eff, b_eff):.1%}")
    c1.write(f"Prob > Target ({target_eff:.0%}): {p_target:.1%}")
    c1.write(f"Prob > Goal ({dream_eff:.0%}): {1 - beta.cdf(dream_eff, a_eff, b_eff):.1%}")
    
    c2.markdown(f"**Safety Summary**")
    c2.write(f"**Mean Toxicity Rate: {safe_mean:.1%}**")
    c2.write(f"95% Credible Interval: [{safe_ci[0]:.1%} - {safe_ci[1]:.1%}]")
    c2.write(f"Prob > Limit ({safe_limit:.0%}): {p_toxic:.1%}")
    
    c3.markdown(f"**Prior Configuration**")
    c3.write(f"Base Prior Alpha: {prior_alpha}")
    c3.write(f"Base Prior Beta: {prior_beta}")
    c3.write(f"Forecasted Success (BPP): {bpp:.1%}")
    st.caption("Priors represent pre-trial evidence strength.")

# --- SENSITIVITY ANALYSIS ---
st.subheader("🧪 Comprehensive Sensitivity Analysis")
priors_list = [
    (f"Optimistic ({opt_p}:1)", opt_p, 1), 
    (f"Flat ({flat_p}:{flat_p})", flat_p, flat_p), 
    (f"Skeptical (1:{skp_p})", 1, skp_p)
]

cols = st.columns(3)
target_probs = []
for i, (name, ap, bp) in enumerate(priors_list):
    ae_s, be_s = ap + successes, bp + (total_n - successes)
    as_s, bs_s = (bp if "Optimistic" in name else ap) + saes, (ap if "Optimistic" in name else bp) + (total_n - saes)
    
    p_t_s = 1 - beta.cdf(target_eff, ae_s, be_s)
    target_probs.append(p_t_s)
    
    with cols[i]:
        st.info(f"**{name}**")
        st.write(f"Mean Eff: {ae_s/(ae_s+be_s):.1%}")
        st.write(f"P(>{target_eff:.0%}): {p_t_s:.1%}")
        st.write(f"Mean Tox: {as_s/(as_s+bs_s):.1%}")
        st.write(f"P(>{safe_limit:.0%}): {1-beta.cdf(safe_limit, as_s, bs_s):.1%}")

spread = max(target_probs) - min(target_probs)
st.markdown(f"**Interpretation:** Results are **{'ROBUST' if spread < 0.15 else 'FRAGILE'}** ({spread:.1%} variance across priors).")
