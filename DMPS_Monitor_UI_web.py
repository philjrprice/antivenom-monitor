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
    null_eff = st.slider("Null Efficacy (p0) (%)", 0.1, 1.0, 0.50)
    target_eff = st.slider("Target Efficacy (p1) (%)", 0.1, 1.0, 0.60)
    dream_eff = st.slider("Goal/Dream Efficacy (%)", 0.1, 1.0, 0.70)
    success_conf_req = st.slider("Success Confidence Req.", 0.5, 0.99, 0.74)
    bpp_futility_limit = st.slider("BPP Futility Limit", 0.01, 0.20, 0.05)

# FIXED: Safety Stop Confidence back inside Safety Rules
with st.sidebar.expander("Safety Rules", expanded=True):
    safe_limit = st.slider("SAE Upper Limit (%)", 0.05, 0.50, 0.15)
    safe_conf_req = st.sidebar.slider("Safety Stop Confidence", 0.5, 0.99, 0.90)

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

# Row 1: Metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample N", f"{total_n}/{max_n_val}")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}")
m3.metric(f"P(>{target_eff:.0%} Target)", f"{p_target:.1%}")
m4.metric("Safety Risk", f"{p_toxic:.1%}")
m5.metric("BPP (Forecast)", f"{bpp:.1%}")
m6.metric(f"P(>{dream_eff:.0%} Goal)", f"{p_goal:.1%}")

st.caption(f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}**")

# Decision Status
st.markdown("---")
if p_toxic > safe_conf_req:
    st.error(f"🚨 **CRITICAL SAFETY STOP:** Prob. Toxicity ({p_toxic:.1%}) exceeds the {safe_conf_req:.1%} confidence limit.")
elif is_look_point:
    if bpp < bpp_futility_limit: st.warning(f"🛑 **STOP: FUTILITY TRIGGERED:** Forecast ({bpp:.1%}) below limit.")
    elif p_target > success_conf_req: st.success(f"✅ **EFFICACY MET:** Evidence for >{target_eff:.0%} efficacy achieved.")
    else: st.info(f"🛡️ **INTERIM CHECK AT N={total_n}:** No stop triggers hit.")
elif total_n < min_interim:
    st.info(f"⏳ **LEAD-IN PHASE:** Enrolling to reach Min N ({min_interim}) before checks.")
else:
    next_check = total_n + (check_cohort - (total_n - min_interim) % check_cohort)
    st.info(f"🧬 **MONITORING:** Adaptive state active. Next check at N={next_check}.")

# Graph Row
st.subheader("Statistical Distributions (95% CI Shaded)")

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
fig.add_vline(x=null_eff, line_dash="dot", line_color="gray", annotation_text="Null")
fig.add_vline(x=target_eff, line_dash="dash", line_color="green", annotation_text="Target")
fig.add_vline(x=dream_eff, line_dash="dashdot", line_color="blue", annotation_text="Goal")
fig.add_vline(x=safe_limit, line_dash="dash", line_color="black", annotation_text="Limit")

fig.update_layout(
    xaxis=dict(range=[0, 1], title="Rate"), yaxis=dict(title="Density"), height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig, use_container_width=True)

# Full Stats Breakdown
with st.expander("📊 Full Statistical Breakdown", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Efficacy Summary**")
        st.write(f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}**")
        st.write(f"Prob > Target ({target_eff:.0%}): **{p_target:.1%}**")
        st.write(f"Prob > Goal ({dream_eff:.0%}): **{p_goal:.1%}**")
        st.write(f"95% CI: [{eff_ci[0]:.1%} - {eff_ci[1]:.1%}]")
    with c2:
        st.markdown("**Safety Summary**")
        st.write(f"Mean Toxicity: {safe_mean:.1%}")
        st.write(f"Prob > Limit ({safe_limit:.0%}): **{p_toxic:.1%}**")
        st.write(f"95% CI: [{safe_ci[0]:.1%} - {safe_ci[1]:.1%}]")
    with c3:
        st.markdown("**Operational Stats**")
        st.write(f"Success Forecast (BPP): {bpp:.1%}")
        st.write(f"Next Check: Patient {total_n if is_look_point else next_check}")
        st.write(f"Adaptive Check Rule: {'READY' if is_at_min_n else 'WAITING'}")

# Sensitivity Analysis
st.subheader("🧪 Sensitivity Analysis")

priors_list = [(f"Optimistic ({opt_p}:1)", opt_p, 1), ("Neutral (1:1)", 1, 1), (f"Skeptical (1:{skp_p})", 1, skp_p)]
cols = st.columns(3)
for i, (name, ap, bp) in enumerate(priors_list):
    ae_s, be_s = ap + successes, bp + (total_n - successes)
    with cols[i]:
        st.info(f"**{name}**")
        st.write(f"P(>Target): {(1 - beta.cdf(target_eff, ae_s, be_s)):.1%}")
