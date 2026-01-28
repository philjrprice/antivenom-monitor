import streamlit as st
import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Universal Trial Monitor: Hybrid", layout="wide")

# --- SIDEBAR: INPUT SECTIONS ---
st.sidebar.header("📋 Current Trial Data")
max_n_val = st.sidebar.number_input("Maximum Sample Size (N)", 10, 500, 70)
total_n = st.sidebar.number_input("Total Patients Enrolled", 0, max_n_val, 20)
successes = st.sidebar.number_input("Total Successes", 0, total_n, value=min(14, total_n))
saes = st.sidebar.number_input("Serious Adverse Events (SAEs)", 0, total_n, value=min(1, total_n))

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Study Parameters")

with st.sidebar.expander("Base Study Priors", expanded=True):
    prior_alpha = st.sidebar.slider("Prior Successes (Alpha)", 0.1, 10.0, 1.0, step=0.1)
    prior_beta = st.sidebar.slider("Prior Failures (Beta)", 0.1, 10.0, 1.0, step=0.1)

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

# --- ADVANCED FEATURE SETTINGS ---
with st.sidebar.expander("Equivalence & Export Settings"):
    equiv_bound = st.slider("Equivalence Bound (+/-)", 0.01, 0.10, 0.05)
    include_heatmap = st.checkbox("Generate Risk-Benefit Heatmap", value=True)

# --- MATH ENGINE ---
a_eff, b_eff = prior_alpha + successes, prior_beta + (total_n - successes)
a_safe, b_safe = prior_alpha + saes, prior_beta + (total_n - saes)

p_null = 1 - beta.cdf(null_eff, a_eff, b_eff)
p_target = 1 - beta.cdf(target_eff, a_eff, b_eff)
p_goal = 1 - beta.cdf(dream_eff, a_eff, b_eff)
p_toxic = 1 - beta.cdf(safe_limit, a_safe, b_safe)

# Equivalence Calculation
p_equiv = beta.cdf(null_eff + equiv_bound, a_eff, b_eff) - beta.cdf(null_eff - equiv_bound, a_eff, b_eff)

eff_mean, eff_ci = a_eff / (a_eff + b_eff), beta.ppf([0.025, 0.975], a_eff, b_eff)
safe_mean, safe_ci = a_safe / (a_safe + b_safe), beta.ppf([0.025, 0.975], a_safe, b_safe)

# NEW: Integrated PPoS and BPP Logic
def get_trial_forecasts(curr_s, curr_n, m_n, t_eff, s_conf, p_a, p_b):
    rem_n = m_n - curr_n
    if rem_n <= 0:
        is_success = (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf
        return 1.0 if is_success else 0.0, 1.0 if is_success else 0.0
    
    # Monte Carlo simulation for future outcomes
    future_rates = np.random.beta(p_a + curr_s, p_b + curr_n - curr_s, 1000)
    future_successes = np.random.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    
    # Final analysis success check (at N_max)
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    ppos = np.mean(final_confs > s_conf)
    
    # BPP uses current success probability
    bpp = np.mean(final_confs > s_conf) 
    return bpp, ppos

bpp, ppos = get_trial_forecasts(successes, total_n, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)
is_look_point = (total_n >= min_interim) and ((total_n - min_interim) % check_cohort == 0)

# --- MAIN DASHBOARD ---
st.title("🐍 Hybrid Antivenom Trial Monitor")

# Header Metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample N", f"{total_n}/{max_n_val}")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}")
m3.metric(f"P(>{target_eff:.0%} Target)", f"{p_target:.1%}")
m4.metric("Safety Risk", f"{p_toxic:.1%}")
m5.metric("PPoS (Final)", f"{ppos:.1%}") # Metric added
m6.metric("ESS (Prior Weight)", f"{prior_alpha + prior_beta:.1f}") # Metric added

st.caption(f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}** | Prob Equivalence: **{p_equiv:.1%}**")

st.markdown("---")
# Action Logic
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
    st.info(f"🧬 **MONITORING:** Next scheduled check at N={next_check}.")

# Graph Row
st.subheader("Statistical Distributions (95% CI Shaded)")
x = np.linspace(0, 1, 500)
y_eff, y_safe = beta.pdf(x, a_eff, b_eff), beta.pdf(x, a_safe, b_safe)
fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y_eff, name="Efficacy Belief", line=dict(color='#2980b9', width=3)))
x_ci_e = np.linspace(eff_ci[0], eff_ci[1], 100)
fig.add_trace(go.Scatter(x=np.concatenate([x_ci_e, x_ci_e[::-1]]), y=np.concatenate([beta.pdf(x_ci_e, a_eff, b_eff), np.zeros(100)]),
                         fill='toself', fillcolor='rgba(41, 128, 185, 0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))

fig.add_trace(go.Scatter(x=x, y=y_safe, name="Safety Belief", line=dict(color='#c0392b', width=3)))
x_ci_s = np.linspace(safe_ci[0], safe_ci[1], 100)
fig.add_trace(go.Scatter(x=np.concatenate([x_ci_s, x_ci_s[::-1]]), y=np.concatenate([beta.pdf(x_ci_s, a_safe, b_safe), np.zeros(100)]),
                         fill='toself', fillcolor='rgba(192, 57, 43, 0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))

fig.add_vline(x=null_eff, line_dash="dot", line_color="gray", annotation_text="Null")
fig.add_vline(x=target_eff, line_dash="dash", line_color="green", annotation_text="Target")
fig.add_vline(x=dream_eff, line_dash="dashdot", line_color="blue", annotation_text="Goal")
fig.add_vline(x=safe_limit, line_dash="dash", line_color="black", annotation_text="Limit")

fig.update_layout(xaxis=dict(range=[0, 1], title="Rate"), height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), margin=dict(l=0, r=0, t=50, b=0))
st.plotly_chart(fig, use_container_width=True)

# Risk-Benefit Heatmap Row
if include_heatmap:
    st.subheader("⚖️ Risk-Benefit Trade-off Heatmap")
    grid_res = 50
    eff_grid = np.linspace(0.2, 0.9, grid_res)
    saf_grid = np.linspace(0.0, 0.4, grid_res)
    E, S = np.meshgrid(eff_grid, saf_grid)
    
    score = E - (2 * S)
    
    fig_heat = px.imshow(score, x=eff_grid, y=saf_grid, labels=dict(x="Efficacy Rate", y="SAE Rate", color="Benefit Score"),
                         color_continuous_scale="RdYlGn", origin="lower")
    
    fig_heat.add_trace(go.Scatter(x=[eff_mean], y=[safe_mean], mode='markers+text', text=["Current Status"], 
                                  textposition="top right", marker=dict(color='white', size=12, symbol='x')))
    
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, use_container_width=True)


# Breakdown
with st.expander("📊 Full Statistical Breakdown", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Efficacy Summary**")
        st.write(f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}**")
        st.write(f"Prob > Target ({target_eff:.0%}): **{p_target:.1%}**")
        st.write(f"Prob > Goal ({dream_eff:.0%}): **{p_goal:.1%}**")
        st.write(f"Equivalence Prob: {p_equiv:.1%}")
        st.write(f"95% CI: [{eff_ci[0]:.1%} - {eff_ci[1]:.1%}]")
    with c2:
        st.markdown("**Safety Summary**")
        st.write(f"Mean Toxicity: {safe_mean:.1%}")
        st.write(f"Prob > Limit ({safe_limit:.0%}): **{p_toxic:.1%}**")
    with c3:
        st.markdown("**Operational Info**")
        st.write(f"BPP Success Forecast: {bpp:.1%}")
        st.write(f"PPoS (Final Analysis): {ppos:.1%}") # Metric added
        st.write(f"Effective Sample Size (ESS): {prior_alpha + prior_beta:.1f}")

# Sensitivity Analysis
st.subheader("🧪 Sensitivity Analysis")
priors_list = [(f"Optimistic ({opt_p}:1)", opt_p, 1), ("Neutral (1:1)", 1, 1), (f"Skeptical (1:{skp_p})", 1, skp_p)]
cols, target_probs = st.columns(3), []
for i, (name, ap, bp) in enumerate(priors_list):
    ae_s, be_s = ap + successes, bp + (total_n - successes)
    p_n_s = 1 - beta.cdf(null_eff, ae_s, be_s)
    p_t_s = 1 - beta.cdf(target_eff, ae_s, be_s)
    p_g_s = 1 - beta.cdf(dream_eff, ae_s, be_s)
    target_probs.append(p_t_s)
    with cols[i]:
        st.info(f"**{name}**")
        st.write(f"Prob > Null: {p_n_s:.1%}")
        st.write(f"Prob > Target: **{p_t_s:.1%}**")
        st.write(f"Prob > Goal: {p_g_s:.1%}")

spread = max(target_probs) - min(target_probs)
st.markdown(f"**Interpretation:** Results are **{'ROBUST' if spread < 0.15 else 'FRAGILE'}** ({spread:.1%} variance between prior mindsets).")

# Regulatory Data Export
st.markdown("---")
if st.button("📥 Generate Regulatory Snapshot"):
    report_data = {
        "Metric": ["Timestamp", "N", "Successes", "SAEs", "Post Mean Eff", "Prob > Target", "Safety Risk", "PPoS", "ESS"],
        "Value": [datetime.now().strftime("%Y-%m-%d %H:%M"), total_n, successes, saes, f"{eff_mean:.2%}", f"{p_target:.2%}", f"{p_toxic:.2%}", f"{ppos:.2%}", f"{prior_alpha+prior_beta:.1f}"]
    }
    df_report = pd.DataFrame(report_data)
    csv = df_report.to_csv(index=False).encode('utf-8')
    st.download_button("Download Snapshot (CSV)", csv, "Trial_Regulatory_Snapshot.csv", "text/csv")
