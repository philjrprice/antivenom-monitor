import streamlit as st
import numpy as np
from scipy.stats import beta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Universal Trial Monitor: Airtight Pro", layout="wide")

# --- SIDEBAR: INPUT SECTIONS ---
st.sidebar.header("📋 Current Trial Data")
max_n_val = st.sidebar.number_input("Maximum Sample Size (N)", 10, 500, 70)
total_n = st.sidebar.number_input("Total Patients Enrolled", 0, max_n_val, 20)
successes = st.sidebar.number_input("Total Successes", 0, total_n, value=min(14, total_n))
saes = st.sidebar.number_input("Serious Adverse Events (SAEs)", 0, total_n, value=min(1, total_n))

# --- AIRTIGHT UPGRADE 1: Data Integrity Validation ---
if successes > total_n:
    st.error("⚠️ Data Integrity Error: Successes cannot exceed total patients enrolled.")
    st.stop()
if saes > total_n:
    st.error("⚠️ Data Integrity Error: SAEs cannot exceed total patients enrolled.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Study Parameters")

with st.sidebar.expander("Base Study Priors", expanded=True):
    prior_alpha = st.slider("Prior Successes (Alpha)", 0.1, 10.0, 1.0, step=0.1)
    prior_beta = st.slider("Prior Failures (Beta)", 0.1, 10.0, 1.0, step=0.1)

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

with st.sidebar.expander("Equivalence & Heatmap Settings"):
    equiv_bound = st.slider("Equivalence Bound (+/-)", 0.01, 0.10, 0.05)
    include_heatmap = st.checkbox("Generate Risk-Benefit Heatmap", value=True)

# --- MATH ENGINE ---
a_eff, b_eff = prior_alpha + successes, prior_beta + (total_n - successes)
a_safe, b_safe = prior_alpha + saes, prior_beta + (total_n - saes)

p_null = 1 - beta.cdf(null_eff, a_eff, b_eff)
p_target = 1 - beta.cdf(target_eff, a_eff, b_eff)
p_goal = 1 - beta.cdf(dream_eff, a_eff, b_eff)
p_toxic = 1 - beta.cdf(safe_limit, a_safe, b_safe)
p_equiv = beta.cdf(null_eff + equiv_bound, a_eff, b_eff) - beta.cdf(null_eff - equiv_bound, a_eff, b_eff)

eff_mean, eff_ci = a_eff / (a_eff + b_eff), beta.ppf([0.025, 0.975], a_eff, b_eff)
safe_mean, safe_ci = a_safe / (a_safe + b_safe), beta.ppf([0.025, 0.975], a_safe, b_safe)

@st.cache_data
def get_enhanced_forecasts(curr_s, curr_n, m_n, t_eff, s_conf, p_a, p_b):
    np.random.seed(42) 
    rem_n = m_n - curr_n
    if rem_n <= 0:
        is_success = (1 - beta.cdf(t_eff, p_a + curr_s, p_b + curr_n - curr_s)) > s_conf
        return 1.0 if is_success else 0.0, [curr_s, curr_s]
    
    future_rates = np.random.beta(p_a + curr_s, p_b + curr_n - curr_s, 5000)
    future_successes = np.random.binomial(rem_n, future_rates)
    total_proj_s = curr_s + future_successes
    final_confs = 1 - beta.cdf(t_eff, p_a + total_proj_s, p_b + (m_n - total_proj_s))
    
    ppos = np.mean(final_confs > s_conf)
    s_range = [int(np.percentile(total_proj_s, 5)), int(np.percentile(total_proj_s, 95))]
    return ppos, s_range

bpp, ps_range = get_enhanced_forecasts(successes, total_n, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)

skep_a, skep_b = 1 + successes, skp_p + (total_n - successes)
skep_prob = 1 - beta.cdf(target_eff, skep_a, skep_b)
evidence_shift = p_target / skep_prob if skep_prob > 0 else 1.0

# --- MAIN DASHBOARD ---
st.title("🛡️ Hybrid Antivenom Trial Monitor: Airtight Pro")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Sample N", f"{total_n}/{max_n_val}")
m2.metric("Mean Efficacy", f"{eff_mean:.1%}")
m3.metric(f"P(>{target_eff:.0%})", f"{p_target:.1%}")
m4.metric("Safety Risk", f"{p_toxic:.1%}")
m5.metric("PPoS (Final)", f"{bpp:.1%}")
m6.metric("Prior Weight", f"{prior_alpha + prior_beta:.1f}")

st.caption(f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}** | Prob Equivalence: **{p_equiv:.1%}**")
st.markdown("---")

is_look_point = (total_n >= min_interim) and ((total_n - min_interim) % check_cohort == 0)
if p_toxic > safe_conf_req:
    st.error(f"🛑 **GOVERNING RULE: SAFETY STOP.** Risk ({p_toxic:.1%}) exceeds {safe_conf_req:.0%} threshold.")
elif is_look_point:
    if bpp < bpp_futility_limit: st.warning(f"⚠️ **GOVERNING RULE: FUTILITY STOP.** PPoS ({bpp:.1%}) below floor.")
    elif p_target > success_conf_req: st.success(f"✅ **GOVERNING RULE: EFFICACY SUCCESS.** Evidence achieved at {p_target:.1%}.")
    else: st.info(f"🛡️ **GOVERNING RULE: CONTINUE.** Interim check at N={total_n} is indeterminate.")
elif total_n < min_interim:
    st.info(f"⏳ **STATUS: LEAD-IN.** Enrollment phase; first check at N={min_interim}.")
else:
    next_check = total_n + (check_cohort - (total_n - min_interim) % check_cohort)
    st.info(f"🧬 **STATUS: MONITORING.** Trial between cohorts. Next check at N={next_check}.")

st.subheader("📈 Trial Decision Corridors")
look_points = [min_interim + (i * check_cohort) for i in range(100) if (min_interim + (i * check_cohort)) <= max_n_val]
viz_n = np.array(look_points)
succ_line, fut_line = [], []

for lp in viz_n:
    # Success: Smallest S where confidence > requirement
    s_req = next((s for s in range(lp+1) if (1 - beta.cdf(target_eff, prior_alpha+s, prior_beta+(lp-s))) > success_conf_req), lp+1)
    
    # Futility Refinement: Highest S where BPP is still BELOW the limit
    f_req = next((s for s in reversed(range(lp+1)) if get_enhanced_forecasts(s, lp, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)[0] <= bpp_futility_limit), -1)
    
    succ_line.append(s_req)
    fut_line.append(max(0, f_req))

fig_corr = go.Figure()
fig_corr.add_trace(go.Scatter(x=viz_n, y=succ_line, name="Success Boundary", line=dict(color='green', dash='dash')))
fig_corr.add_trace(go.Scatter(x=viz_n, y=fut_line, name="Futility Boundary", line=dict(color='red', dash='dash')))
fig_corr.add_trace(go.Scatter(x=[total_n], y=[successes], mode='markers+text', text=["Current"], name="Current Data", marker=dict(size=12, color='blue')))
fig_corr.update_layout(xaxis_title="Sample Size (N)", yaxis_title="Successes (S)", height=400, margin=dict(t=20, b=0))
st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("Statistical Distributions (95% CI Shaded)")
x = np.linspace(0, 1, 500)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_eff, b_eff), name="Efficacy Belief", line=dict(color='#2980b9', width=3)))
x_ci_e = np.linspace(eff_ci[0], eff_ci[1], 100)
fig.add_trace(go.Scatter(x=np.concatenate([x_ci_e, x_ci_e[::-1]]), y=np.concatenate([beta.pdf(x_ci_e, a_eff, b_eff), np.zeros(100)]),
                         fill='toself', fillcolor='rgba(41, 128, 185, 0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
fig.add_trace(go.Scatter(x=x, y=beta.pdf(x, a_safe, b_safe), name="Safety Belief", line=dict(color='#c0392b', width=3)))
x_ci_s = np.linspace(safe_ci[0], safe_ci[1], 100)
fig.add_trace(go.Scatter(x=np.concatenate([x_ci_s, x_ci_s[::-1]]), y=np.concatenate([beta.pdf(x_ci_s, a_safe, b_safe), np.zeros(100)]),
                         fill='toself', fillcolor='rgba(192, 57, 43, 0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
fig.add_vline(x=target_eff, line_dash="dash", line_color="green", annotation_text="Target")
fig.add_vline(x=safe_limit, line_dash="dash", line_color="black", annotation_text="Safety Limit")
fig.update_layout(xaxis=dict(range=[0, 1]), height=400, margin=dict(l=0, r=0, t=50, b=0))
st.plotly_chart(fig, use_container_width=True)

if include_heatmap:
    st.subheader("⚖️ Risk-Benefit Trade-off Heatmap")
    eff_grid, saf_grid = np.linspace(0.2, 0.9, 50), np.linspace(0.0, 0.4, 50)
    E, S = np.meshgrid(eff_grid, saf_grid)
    score = E - (2 * S)
    fig_heat = px.imshow(score, x=eff_grid, y=saf_grid, labels=dict(x="Efficacy Rate", y="SAE Rate", color="Benefit Score"), color_continuous_scale="RdYlGn", origin="lower")
    fig_heat.add_trace(go.Scatter(x=[eff_mean], y=[safe_mean], mode='markers+text', text=["Current"], marker=dict(color='white', size=12, symbol='x')))
    st.plotly_chart(fig_heat, use_container_width=True)

with st.expander("📊 Full Statistical Breakdown", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Efficacy Summary**")
        st.write(f"Mean Efficacy: **{eff_mean:.1%}**") 
        st.write(f"95% CI: **[{eff_ci[0]:.1%} - {eff_ci[1]:.1%}]**")
        st.write(f"Prob > Null ({null_eff:.0%}): **{p_null:.1%}**")
        st.write(f"Prob > Target ({target_eff:.0%}): **{p_target:.1%}**")
        st.write(f"Prob > Goal ({dream_eff:.0%}): **{p_goal:.1%}**")
        st.write(f"Prob Equivalence: **{p_equiv:.1%}**")
        st.write(f"Projected Success Range: **{ps_range[0]} - {ps_range[1]} successes**")
    with c2:
        st.markdown("**Safety Summary**")
        st.write(f"Mean Toxicity: **{safe_mean:.1%}**") 
        st.write(f"95% CI: **[{safe_ci[0]:.1%} - {safe_ci[1]:.1%}]**") # NEW
        st.write(f"Prob > Limit ({safe_limit:.0%}): **{p_toxic:.1%}**")
        if st.button("Calculate Sequential Type I Error"):
            np.random.seed(42)
            fp_count = 0
            for _ in range(1000):
                trial_outcomes = np.random.binomial(1, null_eff, max_n_val)
                for lp in look_points:
                    s = sum(trial_outcomes[:lp])
                    if (1 - beta.cdf(target_eff, prior_alpha + s, prior_beta + (lp - s))) > success_conf_req:
                        fp_count += 1
                        break
            st.warning(f"Estimated Sequential Type I Error: **{fp_count/1000:.1%}**")
    with c3:
        st.markdown("**Operational Info**")
        st.write(f"BPP Success Forecast: **{bpp:.1%}**")
        st.write(f"PPoS (Predicted Prob): **{bpp:.1%}**") # NEW (Duplicate)
        st.write(f"ESS (Effective Sample N): **{a_eff + b_eff:.1f}**") # NEW
        st.write(f"Look Points: **N = {', '.join(map(str, look_points))}**")

st.subheader("🧪 Sensitivity Analysis & Robustness")
priors_list = [(f"Optimistic ({opt_p}:1)", opt_p, 1), ("Neutral (1:1)", 1, 1), (f"Skeptical (1:{skp_p})", 1, skp_p)]
cols, target_probs = st.columns(3), []
for i, (name, ap, bp) in enumerate(priors_list):
    ae_s, be_s = ap + successes, bp + (total_n - successes)
    m_eff_s = ae_s / (ae_s + be_s)
    p_n_s = 1 - beta.cdf(null_eff, ae_s, be_s)
    p_t_s = 1 - beta.cdf(target_eff, ae_s, be_s)
    p_g_s = 1 - beta.cdf(dream_eff, ae_s, be_s)
    target_probs.append(p_t_s)
    with cols[i]:
        st.info(f"**{name}**")
        st.write(f"Mean Efficacy: **{m_eff_s:.1%}**")
        st.write(f"Prob > Null: **{p_n_s:.1%}**")
        st.write(f"Prob > Target: **{p_t_s:.1%}**")
        st.write(f"Prob > Goal: **{p_g_s:.1%}**")
        if "Neutral" in name: 
    st.write(f"Bayes Factor (BF₁₀): **{evidence_shift:.2f}x**")
    st.caption("Interpretation: >1 favors Treatment; >10 is strong evidence.")

spread = max(target_probs) - min(target_probs)
st.markdown(f"**Interpretation:** Results are **{'ROBUST' if spread < 0.15 else 'SENSITIVE'}** ({spread:.1%} variance between prior mindsets).")

with st.expander("📋 Regulatory Decision Boundary Table", expanded=True):
    boundary_data = []
    # Ensure this 'for' loop is indented 4 spaces from the 'with' statement
    for lp in look_points:
        if lp <= total_n: 
            continue
        
        # Success threshold
        s_req = next((s for s in range(lp+1) if (1 - beta.cdf(target_eff, prior_alpha+s, prior_beta+(lp-s))) > success_conf_req), "N/A")
        
        # Futility threshold refinement: Highest S that triggers a stop
        f_req = next((s for s in reversed(range(lp+1)) if get_enhanced_forecasts(s, lp, max_n_val, target_eff, success_conf_req, prior_alpha, prior_beta)[0] <= bpp_futility_limit), -1)
        
        # Safety threshold
        safe_req = next((s for s in range(lp+1) if (1 - beta.cdf(safe_limit, prior_alpha+s, prior_beta+(lp-s))) > safe_conf_req), "N/A")
        
        boundary_data.append({
            "N": lp, 
            "Success S ≥": s_req, 
            "Futility S ≤": f_req if f_req != -1 else "No Stop", 
            "Safety SAEs ≥": safe_req
        })
    
    # CRITICAL FIX: Move these lines OUTSIDE the for-loop so they only run once
    if boundary_data: 
        st.table(pd.DataFrame(boundary_data))
    else: 
        st.write("Trial is at the final analysis point.")

st.markdown("---")
if st.button("📥 Prepare Audit-Ready Snapshot"):
    # 1. Capture the data into a dictionary
    report_data = {
        "Metric": [
            "Timestamp", "N", "Successes", "SAEs", 
            "Success Threshold (%)", "Futility Threshold (%)", 
            "PPoS", "ESS"
        ],
        "Value": [
            datetime.now().isoformat(), total_n, successes, saes, 
            f"{success_conf_req:.1%}", f"{bpp_futility_limit:.1%}", 
            f"{bpp:.2%}", f"{a_eff+b_eff:.1f}"
        ]
    }
    
    # 2. Convert to DataFrame
    df_report = pd.DataFrame(report_data)
    
    # 3. Provide the actual Download Button
    st.download_button(
        label="Click here to Download CSV",
        data=df_report.to_csv(index=False).encode('utf-8'),
        file_name=f"Trial_Snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime='text/csv'
    )
    
    # 4. Show a preview so the user knows it worked
    st.table(df_report)






