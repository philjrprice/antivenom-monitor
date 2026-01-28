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
max_n_val = st.sidebar.number_input("Maximum Sample Size (N)", 10, 1000, 70)
total_n = st.sidebar.number_input("Total Patients Enrolled", 0, max_n_val, 20)
successes = st.sidebar.number_input("Total Successes", 0, total_n, value=min(14, total_n))
saes = st.sidebar.number_input("Serious Adverse Events (SAEs)", 0, total_n, value=min(1, total_n))

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Study Parameters")

with st.sidebar.expander("Base Study Priors", expanded=True):
    prior_alpha = st.slider("Prior Successes (Alpha)", 0.1, 10.0, 1.0, step=0.1)
    prior_beta = st.slider("Prior Failures (Beta)", 0.1, 10.0, 1.0, step=0.1)

with st.sidebar.expander("Adaptive Timing & Look Points", expanded=True):
    min_interim = st.number_input("Min N before first check", 1, max_n_val, 14)
    check_cohort = st.number_input("Check every X patients (Cohort)", 1, 50, 5)

with st.sidebar.expander("Success & Futility Rules"):
    null_eff = st.slider("Null Efficacy (p0) (%)", 0.1, 1.0, 0.50)
    target_eff = st.slider("Target Efficacy (p1) (%)", 0.1, 1.0, 0.60)
    dream_eff = st.slider("Goal/Dream Efficacy (%)", 0.1, 1.0, 0.70)
    success_conf_req = st.slider("Success Confidence Req.", 0.5, 0.99, 0.74)
    bpp_futility_limit = st.slider("BPP Futility Limit", 0.01, 0.50, 0.05)

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

p_equiv = beta.cdf(null_eff + equiv_bound, a_eff, b_eff) - beta.cdf(null_eff - equiv_bound, a_eff, b_eff) if 'equiv_bound' in locals() else 0

eff_mean, eff_ci = a_eff / (a_eff + b_eff), beta.ppf([0.025, 0.975], a_eff, b_eff)
safe_mean, safe_ci = a_safe / (a_safe + b_safe), beta.ppf([0.025, 0.975], a_safe, b_safe)

def get_enhanced_forecasts(curr_s, curr_n, m_n, t_eff, s_conf, p_a, p_b):
    np.random.seed(42)
    rem_n = m_n - curr_n
    if rem_n <= 0:
        is_success = (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf
        return 1.0 if is_success else 0.0, [curr_s, curr_s]
    
    future_rates = np.random.beta(p_a + curr_s, p_b + curr_n - curr_s, 1000)
    future_successes = np.random.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    
    ppos = np.mean(final_confs > s_conf)
    s_range = [int(np.percentile(total_proj_s, 5)), int(np.percentile(total_proj_s, 95))]
    return ppos, s_range

ppos, ps_range = get_enhanced_forecasts(successes, total_n, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)

# Evidence Shift Calculation (Bayes Factor Proxy)
skep_a, skep_b = 1 + successes, skp_p + (total_n - successes)
skep_prob = 1 - beta.cdf(target_eff, skep_a, skep_b)
evidence_shift = p_target / skep_prob if skep_prob > 0 else 1.0

is_look_point = (total_n >= min_interim) and ((total_n - min_interim) % check_cohort == 0)

# --- MAIN DASHBOARD ---
st.title("🐍 Hybrid Antivenom Trial Monitor")

# Header Metrics
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample N", f"{total_n}/{max_n_val}")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}")
m3.metric(f"P(>{target_eff:.0%})", f"{p_target:.1%}")
m4.metric("Safety Risk", f"{p_toxic:.1%}")
m5.metric("PPoS (Final)", f"{ppos:.1%}")
m6.metric("Prior ESS", f"{prior_alpha + prior_beta:.1f}")

st.caption(f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}** | Prob Equivalence: **{p_equiv:.1%}**")

st.markdown("---")
# Action Logic
if p_toxic > safe_conf_req:
    st.error(f"🚨 **CRITICAL SAFETY STOP:** Prob. Toxicity ({p_toxic:.1%}) exceeds limit.")
elif is_look_point:
    if ppos < bpp_futility_limit: st.warning(f"🛑 **STOP: FUTILITY TRIGGERED:** Forecast ({ppos:.1%}) below limit.")
    elif p_target > success_conf_req: st.success(f"✅ **EFFICACY MET:** Evidence for >{target_eff:.0%} efficacy achieved.")
    else: st.info(f"🛡️ **INTERIM CHECK AT N={total_n}:** No stop triggers hit.")
else:
    next_check = total_n + (check_cohort - (total_n - min_interim) % check_cohort) if total_n >= min_interim else min_interim
    st.info(f"🧬 **MONITORING:** Next scheduled decision look at N={next_check}.")

# Graph Row: Distributions
st.subheader("Statistical Distributions (95% CI Shaded)")
x = np.linspace(0, 1, 500)
fig = go.Figure()

# Efficacy Plot
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_eff, b_eff), name="Efficacy Posterior", line=dict(color='#2980b9', width=3)))
# Prior overlay (Regulatory requirement)
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, prior_alpha, prior_beta), name="Prior Belief", line=dict(color='gray', dash='dot', width=1)))

# Shade CI
x_ci_e = np.linspace(eff_ci[0], eff_ci[1], 100)
fig.add_trace(go.Scatter(x=np.concatenate([x_ci_e, x_ci_e[::-1]]), y=np.concatenate([beta.pdf(x_ci_e, a_eff, b_eff), np.zeros(100)]),
                         fill='toself', fillcolor='rgba(41, 128, 185, 0.1)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))

fig.add_vline(x=target_eff, line_dash="dash", line_color="green", annotation_text="Target")
fig.add_vline(x=safe_limit, line_dash="dash", line_color="red", annotation_text="SAE Limit")

fig.update_layout(xaxis=dict(range=[0, 1], title="Efficacy/Safety Rate"), height=400, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

# Risk-Benefit Heatmap Row
with st.expander("⚖️ Risk-Benefit & Safety Threshold Analysis"):
    grid_res = 50
    eff_grid = np.linspace(0.2, 0.9, grid_res)
    saf_grid = np.linspace(0.0, 0.4, grid_res)
    E, S = np.meshgrid(eff_grid, saf_grid)
    score = E - (2 * S)
    fig_heat = px.imshow(score, x=eff_grid, y=saf_grid, labels=dict(x="Efficacy", y="SAE Rate", color="Score"),
                         color_continuous_scale="RdYlGn", origin="lower")
    fig_heat.add_trace(go.Scatter(x=[eff_mean], y=[safe_mean], mode='markers', marker=dict(color='white', size=12, symbol='x')))
    fig_heat.add_hline(y=safe_limit, line_dash="dash", line_color="red")
    fig_heat.update_layout(height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

# Breakdown
with st.expander("📊 Full Statistical Breakdown", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Efficacy & Forecasts**")
        st.write(f"Mean Efficacy: **{eff_mean:.1%}**") 
        st.write(f"95% CI: **[{eff_ci[0]:.1%} - {eff_ci[1]:.1%}]**")
        st.write(f"Prob > Null: {p_null:.1%}")
        st.write(f"Prob > Target: {p_target:.1%}")
        st.write(f"Prob > Goal: {p_goal:.1%}")
        st.write(f"Projected Success Range: **{ps_range[0]} - {ps_range[1]}** successes")
    with c2:
        st.markdown("**Regulatory Decision Boundaries**")
        # Pre-calculating success/futility table
        look_points = [min_interim + (i * check_cohort) for i in range(100) if (min_interim + (i * check_cohort)) <= max_n_val]
        boundary_data = []
        for lp in look_points:
            s_req = next((s for s in range(lp+1) if (1 - beta.cdf(target_eff, prior_alpha+s, prior_beta+(lp-s))) > success_conf_req), "N/A")
            f_req = next((s for s in reversed(range(lp+1)) if get_enhanced_forecasts(s, lp, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)[0] > bpp_futility_limit), -1)
            boundary_data.append({"N": lp, "Success S ≥": s_req, "Futility S ≤": f_req if f_req != -1 else "Stop"})
        st.table(pd.DataFrame(boundary_data))
        
    with c3:
        st.markdown("**Sequential Operating Characteristics**")
        if st.button("Simulate Trial (1000 iter)"):
            np.random.seed(42)
            stops = []
            for _ in range(1000):
                sim_s = 0
                stopped = False
                for lp in look_points:
                    curr_batch = lp if lp == min_interim else check_cohort
                    sim_s += np.random.binomial(curr_batch, null_eff)
                    # Check Success
                    if (1 - beta.cdf(target_eff, prior_alpha + sim_s, prior_beta + (lp - sim_s))) > success_conf_req:
                        stops.append("Success (Type I Error)")
                        stopped = True; break
                    # Check Futility
                    p_f, _ = get_enhanced_forecasts(sim_s, lp, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)
                    if p_f < bpp_futility_limit:
                        stops.append("Futility Stop")
                        stopped = True; break
                if not stopped: stops.append("Failed to meet Success")
            
            t1_error = stops.count("Success (Type I Error)") / 1000
            st.warning(f"Simulated Sequential Type I Error: **{t1_error:.2%}**")
            st.caption("Target is typically < 5% or 2.5% for regulatory approval.")

# Sensitivity Analysis
st.subheader("🧪 Sensitivity Analysis")
priors_list = [(f"Optimistic ({opt_p}:1)", opt_p, 1), ("Neutral (1:1)", 1, 1), (f"Skeptical (1:{skp_p})", 1, skp_p)]
cols, target_probs = st.columns(3), []
for i, (name, ap, bp) in enumerate(priors_list):
    ae_s, be_s = ap + successes, bp + (total_n - successes)
    p_t_s = 1 - beta.cdf(target_eff, ae_s, be_s)
    target_probs.append(p_t_s)
    with cols[i]:
        st.info(f"**{name}**")
        st.write(f"Prob > Target: **{p_t_s:.1%}**")
        if "Neutral" in name:
            st.write(f"Bayes Factor (BF₁₀): **{evidence_shift:.2f}x**")
            st.caption("BF > 3 is substantial evidence.")

# Footer & Export
st.markdown("---")
col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    st.caption("🚨 **REGULATORY DISCLAIMER:** This monitor is for clinical decision support. Final regulatory submission requires pre-specified statistical analysis plans (SAP) and locked data environments.")
    st.caption(f"App Version: 2.1.0-Regulatory | System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col_f2:
    if st.button("📥 Full Export"):
        df_report = pd.DataFrame({"Metric": ["N", "Successes", "SAEs", "Bayes Factor"], "Value": [total_n, successes, saes, evidence_shift]})
        st.download_button("Download CSV", df_report.to_csv().encode('utf-8'), "Trial_Report.csv")
