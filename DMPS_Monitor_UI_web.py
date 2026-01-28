import streamlit as st
import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go

st.set_page_config(page_title="Universal Trial Monitor: Hybrid", layout="wide")

# --- SIDEBAR: INPUT SECTIONS ---
st.sidebar.header("📋 Current Trial Data")
max_n_val = st.sidebar.number_input("Maximum Sample Size (N)", 10, 500, 70)
total_n = st.sidebar.number_input("Total Patients Enrolled", 0, max_n_val, 20)
successes = st.sidebar.number_input("Total Successes", 0, total_n, 14)
saes = st.sidebar.number_input("Serious Adverse Events (SAEs)", 0, total_n, 1)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Study Parameters")

# Base Study Priors
with st.sidebar.expander("Base Study Priors", expanded=False):
    st.write("Set the 'Starting Belief' for this trial:")
    prior_alpha = st.number_input("Prior Successes (Alpha)", 0.1, 10.0, 1.0, help="Higher values mean a stronger initial belief in efficacy.")
    prior_beta = st.number_input("Prior Failures (Beta)", 0.1, 10.0, 1.0, help="Higher values mean a more skeptical starting point.")

# Adaptive Timing & Look Points
with st.sidebar.expander("Adaptive Timing & Look Points", expanded=True):
    min_interim = st.number_input("Min N before first check", 1, max_n_val, 14, help="Trial must reach this N before efficacy/futility rules activate.")
    check_cohort = st.number_input("Check every X patients (Cohort)", 1, 20, 5, help="Interval for interim analysis after Min N is met.")

# Success & Futility Rules
with st.sidebar.expander("Success & Futility Rules"):
    target_eff = st.slider("Target Efficacy (%)", 0.1, 1.0, 0.60)
    null_eff = st.slider("Null Efficacy (%)", 0.1, 1.0, 0.50)
    dream_eff = st.slider("Goal/Dream Efficacy (%)", 0.1, 1.0, 0.70)
    success_conf_req = st.slider("Success Confidence Req.", 0.5, 0.99, 0.74)
    bpp_futility_limit = st.slider("BPP Futility Limit", 0.01, 0.20, 0.05)

# Safety Rules
with st.sidebar.expander("Safety Rules", expanded=True):
    safe_limit = st.slider("SAE Upper Limit (%)", 0.05, 0.50, 0.15, help="Maximum allowable toxicity rate.")
    safe_conf_req = st.slider("Safety Stop Confidence", 0.5, 0.99, 0.90, help="Confidence required to trigger a safety stop.")

# Sensitivity Prior Settings
with st.sidebar.expander("Sensitivity Prior Settings"):
    opt_p = st.slider("Optimistic Prior Weight", 1, 10, 4)
    skp_p = st.slider("Skeptical Prior Weight", 1, 10, 4)

# --- MATH ENGINE ---
def get_bpp(curr_s, curr_n, m_n, t_eff, s_conf, p_a, p_b):
    """Forecasts trial success using specified priors."""
    rem_n = m_n - curr_n
    if rem_n <= 0:
        return 1.0 if (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf else 0.0
    future_rates = np.random.beta(p_a + curr_s, p_b + curr_n - curr_s, 1000)
    future_successes = np.random.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    return np.mean(final_confs > s_conf)

# Calculations using Adjustable Priors
a_eff, b_eff = prior_alpha + successes, prior_beta + (total_n - successes)
a_safe, b_safe = prior_alpha + saes, prior_beta + (total_n - saes)

eff_mean, eff_ci = a_eff / (a_eff + b_eff), beta.ppf([0.025, 0.975], a_eff, b_eff)
safe_mean, safe_ci = a_safe / (a_safe + b_safe), beta.ppf([0.025, 0.975], a_safe, b_safe)

p_target = 1 - beta.cdf(target_eff, a_eff, b_eff)
p_toxic = 1 - beta.cdf(safe_limit, a_safe, b_safe)
bpp = get_bpp(successes, total_n, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)

# Decision Logic Variables
is_at_min_n = total_n >= min_interim
is_on_cohort_beat = (total_n - min_interim) % check_cohort == 0
is_look_point = is_at_min_n and is_on_cohort_beat

# --- MAIN DASHBOARD ---
st.title("🐍 Hybrid Antivenom Trial Monitor")

# Metrics Row
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample N", f"{total_n}/{max_n_val}")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}")
m3.metric("Eff. Confidence", f"{p_target:.1%}")
m4.metric("Mean Safety", f"{safe_mean:.1%}")
m5.metric("Safety Risk", f"{p_toxic:.1%}")
m6.metric("BPP (Forecast)", f"{bpp:.1%}")

# Decision Status
st.markdown("---")
if p_toxic > safe_conf_req:
    st.error(f"🚨 **CRITICAL SAFETY STOP:** Prob. Toxicity ({p_toxic:.1%}) exceeds the {safe_conf_req:.1%} confidence limit.")
elif is_look_point:
    if bpp < bpp_futility_limit:
        st.warning(f"🛑 **STOP: FUTILITY TRIGGERED:** Forecast ({bpp:.1%}) below limit at N={total_n}.")
    elif p_target > success_conf_req:
        st.success(f"✅ **EFFICACY MET:** Evidence for >{target_eff:.0%} efficacy achieved at N={total_n}.")
    else:
        st.info(f"🛡️ **INTERIM CHECK AT N={total_n}:** No stop triggers hit. Continue enrolling.")
elif not is_at_min_n:
    st.info(f"⏳ **LEAD-IN PHASE:** Enrolling to reach Min N ({min_interim}) for adaptive check.")
else:
    next_check = total_n + (check_cohort - (total_n - min_interim) % check_cohort)
    st.info(f"🧬 **MONITORING:** Next adaptive check scheduled at N={next_check}.")

# Graph Row
st.subheader("Statistical Distributions (Posterior vs. Target)")

x = np.linspace(0, 1, 500)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_eff, b_eff), name="Efficacy PDF", line=dict(color='#2980b9', width=3)))
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_safe, b_safe), name="Safety PDF", line=dict(color='#c0392b', width=3)))
fig.add_vline(x=target_eff, line_dash="dash", line_color="green", annotation_text="Target")
fig.add_vline(x=safe_limit, line_dash="dash", line_color="black", annotation_text="Limit")
fig.update_layout(xaxis=dict(range=[0, 1]), height=400, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# Detailed Stats Breakdown
with st.expander("📊 Full Statistical Breakdown", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Efficacy Summary**")
    c1.write(f"Mean Efficacy: {eff_mean:.1%}")
    c1.write(f"95% Credible Interval: [{eff_ci[0]:.1%} - {eff_ci[1]:.1%}]")
    c1.write(f"Prob > Null ({null_eff:.0%}): {1 - beta.cdf(null_eff, a_eff, b_eff):.1%}")
    c1.write(f"Prob > Target ({target_eff:.0%}): {p_target:.1%}")
    c1.write(f"Prob > Goal ({dream_eff:.0%}): {1 - beta.cdf(dream_eff, a_eff, b_eff):.1%}")
    
    c2.markdown(f"**Safety Summary**")
    c2.write(f"Mean Toxicity: {safe_mean:.1%}")
    c2.write(f"95% Credible Interval: [{safe_ci[0]:.1%} - {safe_ci[1]:.1%}]")
    c2.write(f"Prob > Limit ({safe_limit:.0%}): {p_toxic:.1%}")
    
    c3.markdown(f"**Prior & Forecast Configuration**")
    c3.write(f"Current Prior (Alpha/Beta): {prior_alpha} / {prior_beta}")
    c3.write(f"Forecasted Success (BPP): {bpp:.1%}")
    c3.write(f"Adaptive Status: {'Active' if is_at_min_n else 'Lead-in'}")

# Sensitivity Analysis
st.subheader("🧪 Sensitivity Analysis")

priors_list = [
    (f"Optimistic ({opt_p}:1)", opt_p, 1), 
    ("Neutral (1:1)", 1, 1), 
    (f"Skeptical (1:{skp_p})", 1, skp_p)
]
cols = st.columns(3)
target_probs = []
for i, (name, ap, bp) in enumerate(priors_list):
    ae_s, be_s = ap + successes, bp + (total_n - successes)
    as_s, bs_s = bp + saes, ap + (total_n - saes)
    p_t_s = 1 - beta.cdf(target_eff, ae_s, be_s)
    target_probs.append(p_t_s)
    with cols[i]:
        st.info(f"**{name}**")
        st.write(f"Mean Eff: {ae_s/(ae_s+be_s):.1%}")
        st.write(f"P(>{target_eff:.0%}): {p_t_s:.1%}")
        st.write(f"P(>{safe_limit:.0%}): {1-beta.cdf(safe_limit, as_s, bs_s):.1%}")

spread = max(target_probs) - min(target_probs)
st.markdown(f"**Interpretation:** Results are **{'ROBUST' if spread < 0.15 else 'FRAGILE'}** ({spread:.1%} variance).")
