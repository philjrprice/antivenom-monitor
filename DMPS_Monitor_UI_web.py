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

with st.sidebar.expander("Base Study Priors", expanded=False):
    prior_alpha = st.number_input("Prior Successes (Alpha)", 0.1, 10.0, 1.0)
    prior_beta = st.number_input("Prior Failures (Beta)", 0.1, 10.0, 1.0)

with st.sidebar.expander("Adaptive Timing & Look Points", expanded=True):
    min_interim = st.number_input("Min N before first check", 1, max_n_val, 14)
    check_cohort = st.number_input("Check every X patients (Cohort)", 1, 20, 5)

with st.sidebar.expander("Success & Futility Rules"):
    target_eff = st.slider("Target Efficacy (%)", 0.1, 1.0, 0.60)
    null_eff = st.slider("Null Efficacy (%)", 0.1, 1.0, 0.50)
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

eff_mean, eff_ci = a_eff / (a_eff + b_eff), beta.ppf([0.025, 0.975], a_eff, b_eff)
safe_mean, safe_ci = a_safe / (a_safe + b_safe), beta.ppf([0.025, 0.975], a_safe, b_safe)

p_target = 1 - beta.cdf(target_eff, a_eff, b_eff)
p_toxic = 1 - beta.cdf(safe_limit, a_safe, b_safe)

def get_bpp(curr_s, curr_n, m_n, t_eff, s_conf, p_a, p_b):
    rem_n = m_n - curr_n
    if rem_n <= 0: return 1.0 if (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf else 0.0
    future_rates = np.random.beta(p_a + curr_s, p_b + curr_n - curr_s, 1000)
    future_successes = np.random.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    return np.mean(final_confs > s_conf)

bpp = get_bpp(successes, total_n, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)
is_look_point = (total_n >= min_interim) and ((total_n - min_interim) % check_cohort == 0)

# --- MAIN DASHBOARD ---
st.title("🐍 Hybrid Antivenom Trial Monitor")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample N", f"{total_n}/{max_n_val}")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}")
m3.metric("Eff. Confidence", f"{p_target:.1%}")
m4.metric("Mean Safety", f"{safe_mean:.1%}")
m5.metric("Safety Risk", f"{p_toxic:.1%}")
m6.metric("BPP (Forecast)", f"{bpp:.1%}")

st.markdown("---")
if p_toxic > safe_conf_req:
    st.error(f"🚨 **CRITICAL SAFETY STOP:** Prob. Toxicity ({p_toxic:.1%}) exceeds limit.")
elif is_look_point:
    if bpp < bpp_futility_limit: st.warning(f"🛑 **STOP: FUTILITY TRIGGERED:** Forecast ({bpp:.1%}) below limit.")
    elif p_target > success_conf_req: st.success(f"✅ **EFFICACY MET:** Evidence for >{target_eff:.0%} efficacy achieved.")
    else: st.info(f"🛡️ **INTERIM CHECK AT N={total_n}:** No stop triggers hit.")
elif total_n < min_interim:
    st.info(f"⏳ **LEAD-IN PHASE:** Waiting for Min N ({min_interim}).")
else:
    next_check = total_n + (check_cohort - (total_n - min_interim) % check_cohort)
    st.info(f"🧬 **MONITORING:** Next check at N={next_check}.")

# --- GRAPH WITH 95% CI SHADING AND CENTRAL LEGEND ---
st.subheader("Statistical Distributions with 95% Credible Intervals")



x = np.linspace(0, 1, 500)
y_eff = beta.pdf(x, a_eff, b_eff)
y_safe = beta.pdf(x, a_safe, b_safe)

fig = go.Figure()

# Efficacy Plot & Shading
fig.add_trace(go.Scatter(x=x, y=y_eff, name="Efficacy Belief", line=dict(color='#2980b9', width=3)))
x_ci_eff = np.linspace(eff_ci[0], eff_ci[1], 100)
fig.add_trace(go.Scatter(x=np.concatenate([x_ci_eff, x_ci_eff[::-1]]),
                         y=np.concatenate([beta.pdf(x_ci_eff, a_eff, b_eff), np.zeros(100)]),
                         fill='toself', fillcolor='rgba(41, 128, 185, 0.2)',
                         line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))

# Safety Plot & Shading
fig.add_trace(go.Scatter(x=x, y=y_safe, name="Safety Belief", line=dict(color='#c0392b', width=3)))
x_ci_safe = np.linspace(safe_ci[0], safe_ci[1], 100)
fig.add_trace(go.Scatter(x=np.concatenate([x_ci_safe, x_ci_safe[::-1]]),
                         y=np.concatenate([beta.pdf(x_ci_safe, a_safe, b_safe), np.zeros(100)]),
                         fill='toself', fillcolor='rgba(192, 57, 43, 0.2)',
                         line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))

# Reference Lines
fig.add_vline(x=target_eff, line_dash="dash", line_color="green", annotation_text="Target")
fig.add_vline(x=safe_limit, line_dash="dash", line_color="black", annotation_text="Limit")

# Layout Adjustments: Central Legend
fig.update_layout(
    xaxis=dict(range=[0, 1], title="Rate (Success/SAE)"),
    yaxis=dict(title="Density"),
    height=450,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig, use_container_width=True)

# Detailed Stats
with st.expander("📊 Full Statistical Breakdown", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Efficacy Summary**")
    c1.write(f"Mean: {eff_mean:.1%}")
    c1.write(f"95% CI: [{eff_ci[0]:.1%} - {eff_ci[1]:.1%}]")
    c1.write(f"P(>Target): {p_target:.1%}")
    
    c2.markdown(f"**Safety Summary**")
    c2.write(f"Mean: {safe_mean:.1%}")
    c2.write(f"95% CI: [{safe_ci[0]:.1%} - {safe_ci[1]:.1%}]")
    c2.write(f"P(>Limit): {p_toxic:.1%}")
    
    c3.markdown(f"**Prior & Logic**")
    c3.write(f"Prior α/β: {prior_alpha}/{prior_beta}")
    c3.write(f"BPP: {bpp:.1%}")
    c3.write(f"Look Point: {'YES' if is_look_point else 'NO'}")

# Sensitivity Analysis
st.subheader("🧪 Sensitivity Analysis")
priors_list = [(f"Optimistic ({opt_p}:1)", opt_p, 1), ("Neutral (1:1)", 1, 1), (f"Skeptical (1:{skp_p})", 1, skp_p)]
cols = st.columns(3)
target_probs = []
for i, (name, ap, bp) in enumerate(priors_list):
    ae_s, be_s = ap + successes, bp + (total_n - successes)
    p_t_s = 1 - beta.cdf(target_eff, ae_s, be_s)
    target_probs.append(p_t_s)
    with cols[i]:
        st.info(f"**{name}**")
        st.write(f"Mean Eff: {ae_s/(ae_s+be_s):.1%}")
        st.write(f"P(>{target_eff:.0%}): {p_t_s:.1%}")

spread = max(target_probs) - min(target_probs)
st.markdown(f"**Interpretation:** Results are **{'ROBUST' if spread < 0.15 else 'FRAGILE'}** ({spread:.1%} variance).")
