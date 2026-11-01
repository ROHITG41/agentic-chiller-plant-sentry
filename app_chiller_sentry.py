# =====================================================================
# APPLICATION: app_chiller_sentry.py
# PURPOSE: Autonomous Chiller Plant Sizing & Hydronic Tier Sentry
# RUNTIME: Python 3.12 / Streamlit / LangChain / Pydantic / Google Gemini
# ARCHITECT: Rohit Kumar, PMP, CEng | Mission-Critical Infrastructure AI
# =====================================================================

import os
import streamlit as st
import pandas as pd
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------
# UI CONFIGURATION & EXECUTIVE BRANDING
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Mission-Critical Chiller Plant Sizing Sentry",
    page_icon="❄️",
    layout="wide"
)

st.title("❄️ Mission-Critical Chiller Plant Sizing & Hydronic Tier Sentry")
st.caption("Engineered by Rohit Kumar, PMP, CEng | Applied Agentic AI for Hyperscale Infrastructure")
st.markdown("---")

# ---------------------------------------------------------------------
# SIDEBAR CONTROLS & SECRETS GOVERNANCE
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    api_key_from_secrets = st.secrets.get("GEMINI_API_KEY", "")
    if api_key_from_secrets:
        os.environ["GEMINI_API_KEY"] = api_key_from_secrets
        st.success("API Key loaded securely from Secrets.")
    else:
        user_key = st.text_input("Enter Google Gemini API Key:", type="password")
        if user_key:
            os.environ["GEMINI_API_KEY"] = user_key
            
    st.markdown("### 📋 MEP Governance Standards")
    st.info(
        "**Mission-Critical Benchmarks:**\n"
        "- **Fluid Velocity:** 4.0 to 10.0 ft/s (ASHRAE Fundamentals)\n"
        "- **Chilled Water ΔT:** 12.0°F to 16.0°F (Prevents Low-ΔT Syndrome)\n"
        "- **Head Loss Threshold:** <= 4.0 ft / 100 ft run\n"
        "- **Tier III Compliance:** N+1 Concurrent Maintainability\n"
        "- **Tier IV Compliance:** 2N Fault Tolerance (Dual Active Paths)"
    )

# ---------------------------------------------------------------------
# STAGE 1: DETERMINISTIC THERMAL & HYDRAULIC SIZING ENGINE (CPU)
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
    
    hf_pipe = 0.002083 * pipe_length_ft * ((100.0 / c_factor) ** 1.852) * ((total_gpm ** 1.852) / (pipe_id ** 4.8655)) if pipe_id > 0 else 0.0
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
            tier_defects.append("Tier III Risk: Single header path prevents concurrent maintenance of valves.")
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
        engineering_warnings.append(f"Laminar/Sluggish Flow: {flow_velocity_fps:.2f} ft/s is below 3.0 ft/s.")
        
    if head_loss_per_100ft > 4.0:
        engineering_warnings.append(f"Excessive Friction Gradient: {head_loss_per_100ft:.2f} ft/100ft exceeds 4.0 ft/100ft limit.")

    return {
        "cooling_load_tr": cooling_load_tr,
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

# ---------------------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# ---------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Plant Thermal & Hydronic Configuration")
    use_preset = st.checkbox("Load Preset: Hyperscale AI Data Center Facility (4,000 TR)", value=True)
    
    if use_preset:
        load_tr = 4000.0
        delta_t = 14.0
        chillers = 4
        standby = 1
        pipe_l = 850.0
        nps = 16.0
        tier = "Tier III (Concurrent Maintainability)"
        dual_loop = True
    else:
        load_tr = st.number_input("Peak Plant Cooling Demand (TR):", min_value=100.0, max_value=50000.0, value=3000.0, step=100.0)
        delta_t = st.number_input("Chilled Water Delta-T (°F):", min_value=6.0, max_value=24.0, value=12.0, step=1.0)
        c1, c2 = st.columns(2)
        with c1:
            chillers = st.number_input("Active Chillers (N):", min_value=1, max_value=24, value=3)
        with c2:
            standby = st.number_input("Standby Chillers (+R):", min_value=0, max_value=24, value=1)
        pipe_l = st.number_input("Total Equivalent Pipe Run (ft):", min_value=50.0, max_value=10000.0, value=600.0, step=50.0)
        nps = st.selectbox("Primary Header Pipe Size (NPS Inches):", [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0], index=3)
        tier = st.selectbox("Target Reliability Standard:", ["Tier III (Concurrent Maintainability)", "Tier IV (Fault Tolerant)"])
        dual_loop = st.checkbox("Dual-Path Header Distribution Topology", value=True)

    st.write("### Hydronic Loop Parameter Schedule")
    schedule_summary = pd.DataFrame([{
        "Parameter": "Peak Cooling Load", "Value": f"{load_tr:,.1f} TR",
        "Design Delta-T": f"{delta_t:.1f} °F",
        "Header Size": f"NPS {nps:.0f} Inch",
        "Chiller Topology": f"{chillers} Active + {standby} Standby",
        "Redundancy Topology": "Dual Header" if dual_loop else "Single Header"
    }])
    st.dataframe(schedule_summary, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("2. Deterministic Hydronic & Tier Audit")
    if st.button("🚀 Execute Thermal & Hydraulic Audit", type="primary"):
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
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hydronic Flow", f"{results['total_gpm']:,.0f} GPM", f"{results['unit_capacity_tr']:.0f} TR/unit")
        m2.metric("Fluid Velocity", f"{results['flow_velocity_fps']:.2f} ft/s", "Max 10.0 ft/s", delta_color="inverse" if results['flow_velocity_fps'] > 10.0 else "normal")
        m3.metric("Friction Head", f"{results['hf_pipe_ft']:.1f} ft", f"{results['head_loss_per_100ft']:.2f} ft/100ft")
        m4.metric("Pumping Power", f"{results['pump_electrical_kw']:.1f} kW", f"{results['pump_bhp']:.1f} BHP")
        
        if results['tier_passed'] and len(results['engineering_warnings']) == 0:
            st.success("✅ HYDRONIC & TIER INTEGRITY AUDIT PASSED: Design conforms to mission-critical standards.")
        else:
            if not results['tier_passed']:
                st.error("❌ UPTIME TIER COMPLIANCE FAILURE DETECTED")
                for defect in results['tier_defects']:
                    st.warning(f"⚠️ {defect}")
            if results['engineering_warnings']:
                for warn in results['engineering_warnings']:
                    st.warning(f"⚠️ {warn}")

        st.markdown("---")
        st.subheader("3. Cognitive Concept Design Basis Report (Gemini 3.8 Flash)")
        
        if not os.environ.get("GEMINI_API_KEY"):
            st.info("💡 To generate the Executive DBR Memo, enter your Gemini API key in the sidebar or configure Secrets.")
        else:
            with st.spinner("Agentic Loop Benchmarking against ASHRAE 90.1 & Synthesizing DBR..."):
                llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.0)
                
                dbr_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are the Principal Mission-Critical Mechanical Engineer and Uptime Institute Accredited Tier Designer (ATD). "
                     "Analyze the provided deterministic chiller plant calculations, fluid velocities, friction head loss, and tier compliance results. "
                     "Synthesize a formal Executive Concept Design Basis Report (DBR). "
                     "Structure the report with: "
                     "1. Executive Summary & Design Capacity "
                     "2. Hydraulic Performance & Efficiency Compliance (benchmark against ASHRAE 90.1-2022 standards) "
                     "3. Uptime Institute Tier Resilience & Failure-Mode Analysis "
                     "4. Principal Recommendations for Engineering Sign-Off."),
                    ("human", 
                     "PLANT SIZING METRICS:\n{metrics}\n\n"
                     "Output a comprehensive, professional engineering memorandum.")
                ])
                
                chain = dbr_prompt | llm
                response = chain.invoke({"metrics": str(results)})
                
                if isinstance(response.content, list):
                    report_text = "\n".join([str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in response.content])
                else:
                    report_text = str(response.content)

                st.markdown(report_text)
                st.download_button(
                    label="📥 Download Concept Design Basis Report (DBR)",
                    data=report_text,
                    file_name="Chiller_Plant_Concept_DBR.md",
                    mime="text/markdown"
                )
