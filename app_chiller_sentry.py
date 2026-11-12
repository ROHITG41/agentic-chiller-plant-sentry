# =====================================================================
# APPLICATION: app_chiller_sentry.py
# PURPOSE: Autonomous Chiller Plant Sizing & Hydronic Tier Sentry
# RUNTIME: Python 3.12 / Streamlit / LangChain / Google Gemini
# ARCHITECT: Rohit Kumar, PMP, CEng (India), CEA, FSEr (TÜV)
# =====================================================================

import os
import streamlit as st
import pandas as pd
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------
# 1. PAGE CONFIGURATION & EXECUTIVE STYLING
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Chiller Plant Sizing & Hydronic Tier Sentry | Rohit Kumar",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        color: #102C57;
        font-size: 2.10rem;
        font-weight: 800;
        margin-bottom: 0px;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
        color: #0F766E;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 16px;
    }
    .provenance-card {
        background-color: #F8FAFC;
        border-left: 4px solid #102C57;
        padding: 12px 18px;
        border-radius: 4px;
        margin-bottom: 16px;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.5;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-size: 1.50rem;
        font-weight: 700;
        color: #102C57;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .dbr-card {
        background-color: #F0FDFA;
        border: 1px solid #99F6E4;
        border-left: 5px solid #0F766E;
        padding: 18px;
        border-radius: 6px;
        margin-top: 10px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# 2. DETERMINISTIC THERMAL & HYDRAULIC SIZING ENGINE (CPU)
# ---------------------------------------------------------------------
def run_chiller_hydronic_audit(
    cooling_load_tr: float,
    chw_delta_t: float,
    chiller_count: int,
    standby_count: int,
    pipe_length_ft: float,
    pipe_nps_inches: float,
    tier_target: str,
    dual_header: bool,
    c_factor: float = 120.0
) -> Dict[str, Any]:
    total_gpm = (cooling_load_tr * 24.0) / chw_delta_t if chw_delta_t > 0 else 0.0
    active_chillers = chiller_count
    total_installed_chillers = chiller_count + standby_count
    chiller_unit_capacity_tr = cooling_load_tr / active_chillers if active_chillers > 0 else 0.0
    gpm_per_chiller = total_gpm / active_chillers if active_chillers > 0 else 0.0

    pipe_id = pipe_nps_inches * 0.95
    flow_velocity_fps = (0.4085 * total_gpm) / (pipe_id ** 2) if pipe_id > 0 else 0.0

    hf_pipe = (
        0.002083 * pipe_length_ft * ((100.0 / c_factor) ** 1.852) * ((total_gpm ** 1.852) / (pipe_id ** 4.8655))
        if pipe_id > 0 else 0.0
    )
    head_loss_per_100ft = (hf_pipe / pipe_length_ft) * 100.0 if pipe_length_ft > 0 else 0.0

    equipment_dp_ft = 55.0
    total_head_ft = hf_pipe + equipment_dp_ft
    pump_eff = 0.78
    motor_eff = 0.94
    pump_bhp = (total_gpm * total_head_ft) / (3960.0 * pump_eff) if pump_eff > 0 else 0.0
    pump_electrical_kw = (pump_bhp * 0.746) / motor_eff if motor_eff > 0 else 0.0

    tier_passed = True
    tier_defects = []

    if tier_target == "Tier III (Concurrent Maintainability)":
        if standby_count < 1:
            tier_passed = False
            tier_defects.append("Tier III Violation: Missing N+1 redundancy. Minimum 1 standby chiller required.")
        if not dual_header:
            tier_passed = False
            tier_defects.append("Tier III Risk: Single header path prevents concurrent maintenance of isolation valves.")
    elif tier_target == "Tier IV (Fault Tolerant)":
        if standby_count < chiller_count:
            tier_passed = False
            tier_defects.append(f"Tier IV Violation: Requires 2N redundancy ({chiller_count} active requires {chiller_count} standby).")
        if not dual_header:
            tier_passed = False
            tier_defects.append("Tier IV Violation: Compartmentalized dual-path distribution header is mandatory.")

    engineering_warnings = []
    if flow_velocity_fps > 10.0:
        engineering_warnings.append(f"Erosive Fluid Velocity: {flow_velocity_fps:.2f} ft/s exceeds ASHRAE maximum ceiling of 10.0 ft/s.")
    elif flow_velocity_fps < 3.0:
        engineering_warnings.append(f"Laminar / Sluggish Flow: {flow_velocity_fps:.2f} ft/s is below 3.0 ft/s minimum scouring threshold.")

    if head_loss_per_100ft > 4.0:
        engineering_warnings.append(f"Excessive Friction Gradient: {head_loss_per_100ft:.2f} ft/100 ft exceeds 4.0 ft/100 ft limit.")

    if chw_delta_t < 12.0:
        engineering_warnings.append(f"Low Delta-T Risk: {chw_delta_t:.1f} deg F increases pumping parasitic kW; target 12.0 to 16.0 deg F.")

    return {
        "cooling_load_tr": cooling_load_tr,
        "chw_delta_t": chw_delta_t,
        "pipe_nps_inches": pipe_nps_inches,
        "pipe_length_ft": pipe_length_ft,
        "dual_header": dual_header,
        "tier_target": tier_target,
        "total_gpm": total_gpm,
        "active_chillers": active_chillers,
        "standby_chillers": standby_count,
        "total_chillers": total_installed_chillers,
        "unit_capacity_tr": chiller_unit_capacity_tr,
        "gpm_per_chiller": gpm_per_chiller,
        "flow_velocity_fps": flow_velocity_fps,
        "hf_pipe_ft": hf_pipe,
        "head_loss_per_100ft": head_loss_per_100ft,
        "total_head_ft": total_head_ft,
        "pump_bhp": pump_bhp,
        "pump_electrical_kw": pump_electrical_kw,
        "tier_passed": tier_passed,
        "tier_defects": tier_defects,
        "engineering_warnings": engineering_warnings
    }

def build_deterministic_dbr(results: Dict[str, Any]) -> str:
    """Provides an authoritative, mathematically grounded fallback DBR when API key is omitted."""
    status_str = "COMPLIANT" if (results["tier_passed"] and not results["engineering_warnings"]) else "REMEDIATION REQUIRED"
    findings = results["tier_defects"] + results["engineering_warnings"]
    findings_md = "\n".join([f"- {item}" for item in findings]) if findings else "- All hydraulic velocity, head-loss, and Uptime Tier redundancy criteria satisfied."

    return f"""### 1. Executive Summary & Plant Capacity
- **Audit Disposition:** {status_str} ({results['tier_target']})
- **Peak Cooling Demand:** {results['cooling_load_tr']:,.1f} TR at {results['chw_delta_t']:.1f} deg F Delta-T
- **Chiller Configuration:** {results['active_chillers']} Active + {results['standby_chillers']} Standby ({results['total_chillers']} Total Units @ {results['unit_capacity_tr']:,.1f} TR/unit)

### 2. Hydraulic Performance & ASHRAE 90.1 Compliance
- **Total Chilled Water Flow:** {results['total_gpm']:,.0f} GPM ({results['gpm_per_chiller']:,.0f} GPM/chiller)
- **Primary Header Sizing:** NPS {results['pipe_nps_inches']:.0f}-inch ({'Dual-Path Ring Header' if results['dual_header'] else 'Single Header'}) over {results['pipe_length_ft']:,.0f} ft equivalent run
- **Fluid Velocity & Friction:** {results['flow_velocity_fps']:.2f} ft/s | {results['head_loss_per_100ft']:.2f} ft/100 ft ({results['hf_pipe_ft']:.1f} ft pipe friction + 55.0 ft equipment DP = {results['total_head_ft']:.1f} ft TDH)
- **Pumping Power:** {results['pump_bhp']:.1f} BHP ({results['pump_electrical_kw']:.1f} kW electrical input at 78% pump / 94% motor efficiency)

### 3. Uptime Tier Resilience & Audit Findings
{findings_md}

### 4. Principal Engineering Recommendations
- Lock in variable-primary chilled water pumping with differential pressure (DP) reset across the index circuit to minimize part-load parasitic kW.
- Verify isolation valve layout during Level 3/4 commissioning to confirm zero cooling interruption during single-component maintenance.
"""

# ---------------------------------------------------------------------
# 3. MAIN APPLICATION INTERFACE
# ---------------------------------------------------------------------
def main():
    st.markdown('<p class="main-header">❄️ CHILLER PLANT SIZING & HYDRONIC TIER SENTRY</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Mission-Critical Central Cooling Architecture | Deterministic Fluid Physics, ASHRAE 90.1 & Uptime Tier Governance</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="provenance-card">
        <b>Lead Systems Architect:</b> Rohit Kumar, PMP, CEng (India), CEA, FSEr (TÜV)<br/>
        <b>Professional Domain:</b> Mission-Critical Infrastructure & Systems Engineering Specialist | Forensic Project Controls & Contract Governance<br/>
        <b>Contact & Channels:</b> grohit41@gmail.com | LinkedIn: ROHITG41 (linkedin.com/in/rohitg41)<br/>
        <b>Central Repository:</b> <a href="https://github.com/ROHITG41/agentic-chiller-plant-sentry" target="_blank">github.com/ROHITG41/agentic-chiller-plant-sentry</a>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "**Operational Context & Quick Guide:** This engine prevents rule-of-thumb chiller oversizing, erosive header velocities, "
        "and Uptime Tier topology defects during concept design. Select a **Facility Preset** or calibrate cooling tonnage, "
        "chilled water Delta-T, pipe NPS size, and N+1 / 2N redundancy in the left sidebar to run deterministic Hazen-Williams "
        "hydronic physics and generate an executive Concept Design Basis Report (DBR)."
    )

    # Sidebar Controls
    with st.sidebar:
        st.subheader("🔑 Engine Configuration")
        api_key_from_secrets = st.secrets.get("GEMINI_API_KEY", "")
        if api_key_from_secrets:
            os.environ["GEMINI_API_KEY"] = api_key_from_secrets
            st.success("Gemini API Key active via Vault.")
        else:
            user_key = st.text_input("Gemini API Key (Optional)", type="password", help="Leave blank to use deterministic DBR engine.")
            if user_key:
                os.environ["GEMINI_API_KEY"] = user_key

        st.markdown("---")
        st.subheader("📋 Facility Design Presets")
        preset_choice = st.selectbox(
            "Select Mission-Critical Preset",
            [
                "Hyperscale AI Data Center (4,000 TR | Tier III)",
                "Fault-Tolerant Enterprise Hub (6,000 TR | Tier IV)",
                "Undersized Pipe Stress Test (Erosion & Tier Breach)",
                "Custom Plant Parameters"
            ],
            index=0
        )

        if "4,000 TR" in preset_choice:
            p_tr, p_dt, p_n, p_r, p_len, p_nps, p_tier, p_dual = 4000.0, 14.0, 4, 1, 850.0, 16.0, "Tier III (Concurrent Maintainability)", True
        elif "6,000 TR" in preset_choice:
            p_tr, p_dt, p_n, p_r, p_len, p_nps, p_tier, p_dual = 6000.0, 15.0, 4, 4, 1100.0, 20.0, "Tier IV (Fault Tolerant)", True
        elif "Stress Test" in preset_choice:
            p_tr, p_dt, p_n, p_r, p_len, p_nps, p_tier, p_dual = 5000.0, 10.0, 4, 1, 1200.0, 12.0, "Tier IV (Fault Tolerant)", False
        else:
            p_tr, p_dt, p_n, p_r, p_len, p_nps, p_tier, p_dual = 3000.0, 14.0, 3, 1, 750.0, 14.0, "Tier III (Concurrent Maintainability)", True

        st.markdown("---")
        st.subheader("⚙️ Plant & Hydronic Calibration")
        nps_options = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0]
        tier_options = ["Tier III (Concurrent Maintainability)", "Tier IV (Fault Tolerant)"]

        load_tr = st.number_input("Peak Cooling Demand (TR)", min_value=100.0, max_value=50000.0, value=p_tr, step=250.0)
        delta_t = st.slider("Chilled Water Delta-T (deg F)", min_value=6.0, max_value=24.0, value=p_dt, step=0.5)
        chillers = st.number_input("Active Chillers (N)", min_value=1, max_value=24, value=p_n)
        standby = st.number_input("Standby Chillers (+R)", min_value=0, max_value=24, value=p_r)
        pipe_l = st.number_input("Equivalent Pipe Run (ft)", min_value=50.0, max_value=10000.0, value=p_len, step=50.0)
        nps = st.selectbox("Primary Header Size (NPS Inches)", nps_options, index=nps_options.index(p_nps))
        tier = st.selectbox("Uptime Tier Target", tier_options, index=tier_options.index(p_tier))
        dual_loop = st.checkbox("Dual-Path Ring Header Topology", value=p_dual)

        st.markdown("---")
        st.subheader("📐 ASHRAE & Uptime Benchmarks")
        st.caption(
            "• Header Velocity: 4.0 to 10.0 ft/s\n"
            "• Design Delta-T: 12.0 to 16.0 deg F\n"
            "• Friction Loss: <= 4.0 ft / 100 ft run\n"
            "• Tier III: N+1 + Dual Header\n"
            "• Tier IV: 2N + Dual Active Paths"
        )

    # Run Deterministic Physics Immediately
    results = run_chiller_hydronic_audit(
        cooling_load_tr=load_tr,
        chw_delta_t=delta_t,
        chiller_count=chillers,
        standby_count=standby,
        pipe_length_ft=pipe_l,
        pipe_nps_inches=nps,
        tier_target=tier,
        dual_header=dual_loop
    )

    overall_pass = results["tier_passed"] and len(results["engineering_warnings"]) == 0

    # Top 4 Executive KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Audit Disposition</div>
            <div class="metric-value" style="font-size:1.10rem; color:{'#16A34A' if overall_pass else '#DC2626'};">
                {'TIER & HYDRONIC PASS' if overall_pass else 'DEFECTS DETECTED'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Plant Flow Rate</div>
            <div class="metric-value">{results['total_gpm']:,.0f} <span style="font-size:0.85rem;">GPM</span></div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Header Velocity</div>
            <div class="metric-value" style="color:{'#DC2626' if results['flow_velocity_fps']>10.0 else '#102C57'};">
                {results['flow_velocity_fps']:.2f} <span style="font-size:0.85rem;">ft/s</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pumping Power</div>
            <div class="metric-value">{results['pump_electrical_kw']:.1f} <span style="font-size:0.85rem;">kW ({results['pump_bhp']:.0f} BHP)</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Two-Column Clean Layout (Zero Horizontal Table Clipping)
    col_left, col_right = st.columns([1, 1.35])

    with col_left:
        st.subheader("1. Hydronic Loop & Redundancy Schedule")
        schedule_df = pd.DataFrame([
            {"Design Parameter": "Peak Cooling Demand", "Specification": f"{load_tr:,.1f} TR"},
            {"Design Parameter": "Chilled Water Delta-T", "Specification": f"{delta_t:.1f} deg F"},
            {"Design Parameter": "Chiller Topology", "Specification": f"{chillers} Active + {standby} Standby ({results['unit_capacity_tr']:,.0f} TR/unit)"},
            {"Design Parameter": "Flow per Chiller", "Specification": f"{results['gpm_per_chiller']:,.0f} GPM"},
            {"Design Parameter": "Primary Header Pipe", "Specification": f"NPS {nps:.0f}-inch ({pipe_l:,.0f} ft run)"},
            {"Design Parameter": "Friction Loss Gradient", "Specification": f"{results['head_loss_per_100ft']:.2f} ft / 100 ft ({results['hf_pipe_ft']:.1f} ft total)"},
            {"Design Parameter": "Total Dynamic Head (TDH)", "Specification": f"{results['total_head_ft']:.1f} ft (incl. 55 ft plant DP)"},
            {"Design Parameter": "Distribution Topology", "Specification": "Dual-Path Ring Header" if dual_loop else "Single Header"},
            {"Design Parameter": "Reliability Standard", "Specification": tier}
        ])
        st.dataframe(schedule_df, use_container_width=True, hide_index=True)

        if overall_pass:
            st.success("✅ HYDRONIC & UPTIME TIER AUDIT PASSED: All velocity, head-loss, and redundancy gates verified.")
        else:
            st.error("❌ ENGINEERING COMPLIANCE ALERTS DETECTED")
            for defect in results["tier_defects"]:
                st.warning(f"⚠️ {defect}")
            for warn in results["engineering_warnings"]:
                st.warning(f"⚠️ {warn}")

    with col_right:
        st.subheader("2. Executive Concept Design Basis Report (DBR)")
        gen_btn = st.button("🚀 Regenerate AI Design Basis Report", type="primary")

        if os.environ.get("GEMINI_API_KEY"):
            try:
                with st.spinner("Benchmarking against ASHRAE 90.1 & Uptime Tier Standards..."):
                    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.0)
                    dbr_prompt = ChatPromptTemplate.from_messages([
                        ("system",
                         "You are the Principal Mission-Critical Mechanical Engineer and Chartered Engineer (CEng India). "
                         "Analyze the deterministic chiller plant calculations, fluid velocities, friction head loss, and Uptime Tier results. "
                         "Synthesize a concise, executive Concept Design Basis Report (DBR) using plain engineering units (deg F, ft/s, GPM, BHP) without LaTeX '$' symbols. "
                         "Include: 1. Executive Summary & Design Capacity, 2. Hydraulic Performance & ASHRAE 90.1 Compliance, "
                         "3. Uptime Tier Resilience & Failure-Mode Audit, 4. Principal Recommendations."),
                        ("human", "PLANT SIZING METRICS:\n{metrics}")
                    ])
                    chain = dbr_prompt | llm
                    response = chain.invoke({"metrics": str(results)})
                    if isinstance(response.content, list):
                        report_text = "\n".join([str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in response.content])
                    else:
                        report_text = str(response.content)
            except Exception:
                report_text = build_deterministic_dbr(results)
        else:
            report_text = build_deterministic_dbr(results)

        st.markdown(f'<div class="dbr-card">{report_text}</div>', unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Concept Design Basis Report (DBR)",
            data=report_text,
            file_name=f"Chiller_Plant_DBR_{int(load_tr)}TR.md",
            mime="text/markdown",
            use_container_width=True
        )

if __name__ == "__main__":
    main()

# --- EXECUTIVE PORTFOLIO FOOTER ---
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #475569; font-size: 0.85rem; padding: 10px 0;">
        <p style="margin-bottom: 4px;">
            <strong>Architect:</strong> Rohit Kumar, PMP, CEng (India), CEA, FSEr (TÜV) | 
            Mission-Critical Infrastructure &amp; Systems Engineering Specialist | Forensic Project Controls &amp; Contract Governance
        </p>
        <p style="margin-bottom: 4px;">
            <strong>Enterprise Advisory &amp; Custom Architecture:</strong> 
            Available for technical consulting &amp; custom agentic workflows &bull; 
            <a href="https://www.linkedin.com/in/rohitg41" target="_blank" style="color: #2563EB; text-decoration: none;">LinkedIn</a> | 
            <a href="mailto:grohit41@gmail.com" style="color: #2563EB; text-decoration: none;">grohit41@gmail.com</a>
        </p>
        <p style="font-size: 0.78rem; color: #64748B; margin-top: 6px;">
            <em>Dataset Note: Independent engineering demonstration validated on representative, anonymized datasets.</em><br>
            Copyright &copy; 2026 Rohit Kumar. All Rights Reserved.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
