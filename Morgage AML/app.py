import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import altair as alt
import time

# --- 1. ENTERPRISE CANVAS INITIALIZATION ---
st.set_page_config(page_title="Mortgage AML Analytics Node", layout="wide")

# --- 2. THE BRAND VISUAL HIERARCHY ENGINE ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700&display=swap');
        
        /* Global Reset */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #FFFFFF !important;
            font-family: 'Inter', sans-serif !important;
            color: #1E2D4A;
            scroll-behavior: smooth !important;
        }
        
        /* Hide default header bar completely */
        [data-testid="stHeader"], stAppHeader, .stAppHeader {
            height: 0px !important;
            min-height: 0px !important;
            display: none !important;
        }
        
        /* Calibrated Spacing: Top padding breathing gap */
        .main .block-container, [data-testid="stMainBlockContainer"] {
            padding-top: 2.5rem !important;   
            padding-bottom: 2.5rem !important;
            max-width: 96% !important;
        }
        
        /* Typography structural margins */
        h1 {
            color: #1E2D4A !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em;
            margin: 0px 0px 8px 0px !important; 
            padding: 0px !important;
            font-size: 2.2rem !important;
            line-height: 1.1 !important;
        }
        
        .subtitle-text {
            color: #1E2D4A;
            opacity: 0.8;
            margin-bottom: 1.5rem !important; 
        }
        
        /* -------------------------------------------------------------------
           NATIVE TAB STYLING (Premium Tab Navigation)
        ------------------------------------------------------------------- */
        
        /* Base text styling for all tabs */
        button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
            font-family: 'Inter', sans-serif !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            color: #6B7A99 !important;
        }
        
        /* Active tab text styling */
        button[data-baseweb="tab"][aria-selected="true"] > div[data-testid="stMarkdownContainer"] > p {
            color: #1E2D4A !important;
            font-weight: 600 !important;
        }
        
        /* Hover state for inactive tabs */
        button[data-baseweb="tab"]:hover > div[data-testid="stMarkdownContainer"] > p {
            color: #3A5273 !important;
        }
        
        /* THE FIX: Targeting Streamlit's specific animated underline element */
        div[data-baseweb="tab-highlight"] {
            background-color: #1E2D4A !important;
        }
        
        /* Clean separator lines */
        hr {
            margin-top: 0px !important;
            margin-bottom: 1.5rem !important;
            border-color: #E6ECF5 !important;
        }
        
        /* -------------------------------------------------------------------
           VISUAL HIERARCHY BUTTON STYLING (Includes Download Buttons)
        ------------------------------------------------------------------- */
        div.stButton > button, 
        div.stDownloadButton > button {
            font-family: 'Inter', sans-serif !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            border-radius: 20px !important;         
            padding: 0.5rem 1rem !important;      
            white-space: nowrap !important;         
            transition: all 0.2s ease-in-out !important;
        }
        div.stButton > button *,
        div.stDownloadButton > button * {
            white-space: nowrap !important;         
        }

        /* LEVEL 1: HERO PRIMARY ACTION (Solid Dark Blue) */
        div.stButton > button[kind="primary"] {
            background-color: #3A5273 !important;   
            color: #FFFFFF !important;              
            border: 1px solid #3A5273 !important;   
        }
        div.stButton > button[kind="primary"] * {
            color: #FFFFFF !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #CEDBE5 !important;   
            border-color: #CEDBE5 !important;
            box-shadow: 0 4px 12px rgba(30,45,74,0.1) !important;
        }
        div.stButton > button[kind="primary"]:hover * {
            color: #1E2D4A !important;
        }

        /* LEVEL 2: FILTER ACTION PILLS & DOWNLOADS (Clean Outlines) */
        div.stButton > button[kind="secondary"],
        div.stDownloadButton > button {
            background-color: #FFFFFF !important;   
            color: #3A5273 !important;              
            border: 1px solid #CEDBE5 !important;   
        }
        div.stButton > button[kind="secondary"] *,
        div.stDownloadButton > button * {
            color: #3A5273 !important;
        }
        div.stButton > button[kind="secondary"]:hover,
        div.stDownloadButton > button:hover {
            background-color: #CEDBE5 !important;   
            border-color: #CEDBE5 !important;
        }
        div.stButton > button[kind="secondary"]:hover *,
        div.stDownloadButton > button:hover * {
            color: #1E2D4A !important;
        }

        /* LEVEL 3: UTILITY LINK ACTIONS (Minimalist Text Lines) */
        .utility-container div.stButton > button {
            background-color: transparent !important;
            color: #3A5273 !important;
            border: 1px dashed #CEDBE5 !important;
            border-radius: 6px !important;
            font-size: 13px !important;
            padding: 0.4rem 0.5rem !important;
        }
        .utility-container div.stButton > button * {
            color: #3A5273 !important;
        }
        .utility-container div.stButton > button:hover {
            background-color: #F8FAFC !important;
            border-color: #3A5273 !important;
        }
        .utility-container div.stButton > button:hover * {
            color: #1E2D4A !important;
        }

        /* Modern Minimalist Container Boxes */
        [data-testid="stContainer"] {
            border: 1px solid #E6ECF5 !important;
            background-color: #F8FAFC !important; 
            border-radius: 12px !important;
            padding: 18px !important;
            box-shadow: 0 1px 2px rgba(30,45,74,0.02) !important;
        }
        
        [data-testid="stDataFrame"] {
            border: 1px solid #E6ECF5 !important;
            border-radius: 8px !important;
            overflow: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# --- 3. ENTERPRISE DATA GENERATION ENGINE ---
@st.cache_data
def load_production_matrix(seed=42):
    rng = np.random.default_rng(seed)
    total_records = 100000

    total_flags = int(rng.integers(200, 801))
    splits = rng.dirichlet([1.5, 1.5, 1.5])
    n_layering  = max(1, round(total_flags * splits[0]))
    n_third     = max(1, round(total_flags * splits[1]))
    n_query     = max(1, total_flags - n_layering - n_third)

    n_base = total_records - total_flags
    base_events   = rng.choice(["Standard Monthly Payment", "Account Query"], n_base, p=[0.97, 0.03])
    base_maturity = rng.integers(1, 26, size=n_base)          
    base_payer    = rng.choice([True, False], n_base, p=[1.0, 0.0])  
    base_tx       = rng.integers(1200, 4500, size=n_base)

    lay_maturity  = rng.integers(26, 30, size=n_layering)
    lay_tx        = rng.integers(50000, 500000, size=n_layering)

    tp_tx         = rng.integers(1200, 4500, size=n_third)

    qry_tx        = rng.integers(400001, 500000, size=n_query)

    n_total = n_base + n_layering + n_third + n_query

    events   = np.concatenate([base_events,
                                ["Lump Sum Payoff"]         * n_layering,
                                ["Standard Monthly Payment"] * n_third,
                                ["Account Query"]            * n_query])

    maturity = np.concatenate([base_maturity,
                                lay_maturity,
                                rng.integers(1, 30, size=n_third),
                                rng.integers(1, 30, size=n_query)])

    payer    = np.concatenate([base_payer,
                                [True]  * n_layering,
                                [False] * n_third,
                                [True]  * n_query])

    tx       = np.concatenate([base_tx, lay_tx, tp_tx, qry_tx])

    idx = rng.permutation(n_total)

    data = {
        "Account ID":              [f"MTG_{100000 + i}" for i in range(n_total)],
        "Borrower Name":           rng.choice(["John Doe", "Jane Smith", "Alex Mercer",
                                               "Sarah Jenkins", "Michael Brown"], n_total),
        "Loan Value":              rng.integers(150000, 2000000, size=n_total),
        "Years to Maturity":       maturity[idx],
        "Event Recorded":          events[idx],
        "Transaction Value":       tx[idx],
        "Payer Matches Borrower":  payer[idx],
    }

    return pd.DataFrame(data)

if "data_seed" not in st.session_state:
    st.session_state.data_seed = 42

df = load_production_matrix(st.session_state.data_seed)

# --- 4. HIGH-SCALE VECTOR RISK MASKING & RULES DEFINITION ---
layering_mask = (df['Event Recorded'] == "Lump Sum Payoff") & (df['Years to Maturity'] > 25)
layering_violations = df[layering_mask].copy()
layering_violations['Risk Vector'] = 'Loan Layering'

third_party_mask = (df['Payer Matches Borrower'] == False) & (df['Event Recorded'] == "Standard Monthly Payment")
third_party_violations = df[third_party_mask].copy()
third_party_violations['Risk Vector'] = 'Third-Party Mismatch'

query_mask = (df['Event Recorded'] == "Account Query") & (df['Transaction Value'] > 400000)
query_violations = df[query_mask].copy()
query_violations['Risk Vector'] = 'High-Value Query Anomaly'

master_exceptions_df = pd.concat([layering_violations, third_party_violations, query_violations])
total_risks_found = len(master_exceptions_df)

RULES = [
    {
        "id":          "R-001",
        "name":        "Loan Layering",
        "typology":    "Placement / Layering",
        "description": "Flags accounts where a lump sum payoff is recorded on a mortgage with significant remaining maturity. Early full repayment on a long-horizon loan is inconsistent with normal borrower behaviour.",
        "conditions":  [
            ("Event Recorded", "==", "'Lump Sum Payoff'"),
            ("Years to Maturity", ">", "25"),
        ],
        "alert_count": len(layering_violations),
        "severity":    "High",
        "action":      "Initiate Source of Wealth (SoW) review. Freeze disbursement pending compliance sign-off.",
        "view_key":    "Layering",
    },
    {
        "id":          "R-002",
        "name":        "Third-Party Payer Mismatch",
        "typology":    "Smurfing / Third-Party Funding",
        "description": "Flags standard monthly mortgage payments where the originating payer does not match the registered borrower. Indicates potential structuring arrangements designed to obscure beneficial ownership.",
        "conditions":  [
            ("Payer Matches Borrower", "==", "False"),
            ("Event Recorded", "==", "'Standard Monthly Payment'"),
        ],
        "alert_count": len(third_party_violations),
        "severity":    "Medium",
        "action":      "Cross-reference payer identity against KYC records and internal wire-transfer data.",
        "view_key":    "Third-Party",
    },
    {
        "id":          "R-003",
        "name":        "High-Value Balance Enquiry",
        "typology":    "Pre-Transaction Probing",
        "description": "Flags account queries associated with transaction values exceeding the high-value threshold. Precursor pattern to rapid asset liquidation.",
        "conditions":  [
            ("Event Recorded", "==", "'Account Query'"),
            ("Transaction Value", ">", "$400,000"),
        ],
        "alert_count": len(query_violations),
        "severity":    "Medium",
        "action":      "Place account under enhanced monitoring. If a matching transfer attempt is detected within 30 days, block settlement.",
        "view_key":    "Queries",
    },
]

SEVERITY_COLORS = {
    "High":   ("#7B1E1E", "#FDE8E8"),
    "Medium": ("#7B5E1E", "#FDF3E8"),
    "Low":    ("#1E4D2B", "#E8F5EC"),
}

# --- 5. SESSION STATE ENGINE ---
if "current_view" not in st.session_state:
    st.session_state.current_view = "Complete Ledger"
if "trigger_scroll" not in st.session_state:
    st.session_state.trigger_scroll = False
if "scroll_target" not in st.session_state:
    st.session_state.scroll_target = "analysis-table-workbench"

def route_and_scroll(target_view, target_element="analysis-table-workbench"):
    st.session_state.current_view = target_view
    st.session_state.scroll_target = target_element
    st.session_state.trigger_scroll = True

# --- 5b. FIXED INSTANT SCROLL INJECTOR ---
if st.session_state.trigger_scroll:
    target = st.session_state.scroll_target
    st.session_state.trigger_scroll = False 
    scroll_id = int(time.time() * 1000)
    
    components.html(
        f"""
        <script data-scroll-sync="{scroll_id}">
            window.parent.document.getElementById('{target}').scrollIntoView({{
                behavior: 'smooth',
                block: 'start'
            }});
        </script>
        """,
        height=0,
        width=0
    )

# --- 6. RENDER HEADERS & TABS ---
st.title("Mortgage Portfolios: Systemic AML Risk Analytics")
st.markdown('<p class="subtitle-text">Real-time operational dashboard for automated transaction auditing and pipeline exception handling.</p>', unsafe_allow_html=True)

# Emjois removed from tab titles
tab_dashboard, tab_rules = st.tabs(["Operational Dashboard", "Rule Engine Configuration"])

# ==========================================
# TAB 1: THE OPERATIONAL DASHBOARD
# ==========================================
with tab_dashboard:
    st.write("") 
    
    # --- LIVE OPERATIONS OVERVIEW PANEL ---
    st.markdown("### Live Operations")
    st.caption("High-level overview of total system volume and aggregated exception items.")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("**Threat Intelligence: Patterns Detected**")
            st.markdown(f"## {total_risks_found} Total Flagged Events")
            
            if st.button("Investigate All Flagged Events", key="btn_master_exceptions", type="primary", use_container_width=True):
                route_and_scroll("Master Exceptions")
                
            st.write("---")
            st.caption("Filter directly by pattern sub-type shortcut pills:")
            
            c_lay, c_3rd, c_qry = st.columns(3)
            with c_lay:
                if st.button(f"Loan Layering ({len(layering_violations)})", key="sub_layer", type="secondary", use_container_width=True):
                    route_and_scroll("Layering")
            with c_3rd:
                if st.button(f"Third-Party ({len(third_party_violations)})", key="sub_3party", type="secondary", use_container_width=True):
                    route_and_scroll("Third-Party")
            with c_qry:
                if st.button(f"High-Value Queries ({len(query_violations)})", key="sub_query", type="secondary", use_container_width=True):
                    route_and_scroll("Queries")

    with col2:
        with st.container(border=True):
            st.markdown("**Total Data Processed**")
            st.markdown(f"## {len(df):,}")
            
            st.write("---")
            
            st.markdown('<div class="utility-container">', unsafe_allow_html=True)
            if st.button("Reset View to Base Ledger", key="btn_ledger", use_container_width=True):
                route_and_scroll("Complete Ledger")
            st.markdown('</div>', unsafe_allow_html=True)

    st.write("---")

    # --- CLEAN VISUALIZATION LAYER ---
    st.markdown("### Risk Distribution Charts")
    st.caption("Visualizing anomaly clusters and volume dispersion across the checked portfolio.")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True):
            st.markdown("**Threat Distribution by Pattern Type**")
            
            chart_data = pd.DataFrame({
                'Risk Pattern': ['Loan Layering', 'Third-Party Mismatch', 'High-Value Queries'],
                'Alert Count': [len(layering_violations), len(third_party_violations), len(query_violations)]
            })
            
            bar_chart = alt.Chart(chart_data).mark_bar(color='#3A5273', cornerRadiusEnd=4).encode(
                x=alt.X('Alert Count:Q', title='Volume of Alerts'),
                y=alt.Y('Risk Pattern:N', sort='-x', title=None)
            ).properties(height=180).configure_view(strokeOpacity=0)
            
            st.altair_chart(bar_chart, use_container_width=True)

    with chart_col2:
        with st.container(border=True):
            st.markdown("**Risk Outlier Matrix (Maturity vs. Transaction Value)**")
            
            scatter_plot = alt.Chart(master_exceptions_df).mark_circle(size=70).encode(
                x=alt.X('Years to Maturity:Q', title='Years Left on Mortgage Loan'),
                y=alt.Y('Transaction Value:Q', title='Incoming Transaction Value ($)'),
                color=alt.Color('Risk Vector:N', scale=alt.Scale(range=['#1E2D4A', '#3A5273', '#9BB1D4']), title="Pattern"),
                tooltip=['Account ID', 'Borrower Name', 'Transaction Value', 'Years to Maturity']
            ).properties(height=180).interactive().configure_view(strokeOpacity=0)
            
            st.altair_chart(scatter_plot, use_container_width=True)

    st.write("---")

    # --- POSITION ANCHOR: FULL-WIDTH TABLE WORKBENCH ---
    st.markdown("<div id='analysis-table-workbench'></div>", unsafe_allow_html=True)

    # --- FULL WIDTH TABLE WORKBENCH ---
    if st.session_state.current_view == "Complete Ledger":
        st.markdown(f"#### Complete Ledger View ({len(df):,} Records)")
        st.caption("Displaying baseline platform transaction streams.")
    elif st.session_state.current_view == "Top 100 Preview":
        st.markdown("#### Top 100 Records Preview")
        st.caption("Displaying the first 100 rows of the active dataset for demonstration purposes.")
    elif st.session_state.current_view == "Master Exceptions":
        st.markdown(f"#### Master Exception Queue: Combined Portfolio Flags ({len(master_exceptions_df)} Total Accounts)")
        st.caption("Displaying a complete aggregated overview of all filtered corporate aml threats.")
    elif st.session_state.current_view == "Layering":
        st.markdown(f"#### Loan Layering Exception Worklist ({len(layering_violations)} Accounts)")
        st.caption("Isolating high-value repayments occurring prematurely in the mortgage lifecycle.")
    elif st.session_state.current_view == "Third-Party":
        st.markdown(f"#### Third-Party Queue Exception Worklist ({len(third_party_violations)} Accounts)")
        st.caption("Isolating capital injections originating from unverified outside identities.")
    elif st.session_state.current_view == "Queries":
        st.markdown(f"#### High-Value Query Queue Exception Worklist ({len(query_violations)} Accounts)")
        st.caption("Isolating high-risk liquidation inquiries exceeding compliance safety margins.")

    act_space, btn_col1, btn_col2 = st.columns([2, 1, 1])
    with btn_col1:
        if st.button("Download View (CSV)", key="btn_csv", use_container_width=True):
            if st.session_state.current_view == "Complete Ledger":
                df.to_csv("full_portfolio_ledger.csv", index=False)
            elif st.session_state.current_view == "Top 100 Preview":
                df.head(100).to_csv("top_100_portfolio_preview.csv", index=False)
            elif st.session_state.current_view == "Master Exceptions":
                master_exceptions_df.to_csv("aml_all_combined_exceptions.csv", index=False)
            elif st.session_state.current_view == "Layering":
                layering_violations.to_csv("aml_layering_escalations.csv", index=False)
            elif st.session_state.current_view == "Third-Party":
                third_party_violations.to_csv("aml_third_party_escalations.csv", index=False)
            elif st.session_state.current_view == "Queries":
                query_violations.to_csv("aml_query_escalations.csv", index=False)
            st.success("File generated.")
    with btn_col2:
        if st.button("Bulk Freeze Queue", key="btn_freeze", type="primary", use_container_width=True):
            if st.session_state.current_view in ["Complete Ledger", "Top 100 Preview"]:
                st.warning("Action blocked on base ledger.")
            else:
                st.error("Selected Queue Suspended.")

    if st.session_state.current_view == "Complete Ledger":
        st.dataframe(df, height=360, use_container_width=True)
    elif st.session_state.current_view == "Top 100 Preview":
        st.dataframe(df.head(100), height=360, use_container_width=True)
    elif st.session_state.current_view == "Master Exceptions":
        st.dataframe(master_exceptions_df, height=360, use_container_width=True)
    elif st.session_state.current_view == "Layering":
        st.dataframe(layering_violations, height=360, use_container_width=True)
    elif st.session_state.current_view == "Third-Party":
        st.dataframe(third_party_violations, height=360, use_container_width=True)
    elif st.session_state.current_view == "Queries":
        st.dataframe(query_violations, height=360, use_container_width=True)

    st.write("---")

    # --- DEMO ADMINISTRATION TOOLS ---
    st.markdown("### Demo Tools")
    st.caption("Administrative controls for demonstration and testing purposes.")

    demo_col1, demo_col2, demo_col3 = st.columns(3)

    with demo_col1:
        if st.button("Regenerate Mock Data", key="btn_demo_refresh", type="secondary", use_container_width=True):
            st.session_state.data_seed = np.random.randint(1, 100000)
            st.session_state.current_view = "Complete Ledger"
            load_production_matrix.clear()
            st.rerun()

    with demo_col2:
        if st.button("View Top 100 Rows", key="btn_demo_top100", type="secondary", use_container_width=True):
            route_and_scroll("Top 100 Preview")

    with demo_col3:
        csv_full = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Dataset",
            data=csv_full,
            file_name="complete_mortgage_portfolio.csv",
            mime="text/csv",
            key="btn_demo_download",
            use_container_width=True
        )


# ==========================================
# TAB 2: THE RULE ENGINE CONFIGURATION
# ==========================================
with tab_rules:
    st.write("") 
    
    st.markdown("### Active Rule Definitions")
    st.caption("Active detection rules running against the current portfolio. Click a rule's **Apply Filter to Dashboard** button to prime the queue for investigation.")
    st.write("")

    # 3-Column side-by-side layout fully utilizing the width
    rule_cols = st.columns(3)

    for idx, rule in enumerate(RULES):
        fg, bg = SEVERITY_COLORS[rule["severity"]]
        with rule_cols[idx]:
            with st.container(border=True):
                c_head1, c_head2 = st.columns([3, 1])
                with c_head1:
                    st.markdown(f"**{rule['id']} — {rule['name']}**")
                with c_head2:
                    st.markdown(
                        f"<div style='text-align:right;'>"
                        f"<span style='background:{bg}; color:{fg}; font-size:11px; font-weight:600; "
                        f"padding:3px 10px; border-radius:20px;'>{rule['severity']}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                st.markdown(f"<div style='font-size:12px; color:#6B7A99; margin-top:-10px; margin-bottom:10px;'>{rule['typology']}</div>", unsafe_allow_html=True)
                st.caption(rule["description"])
                st.write("")
                
                st.markdown("**Trigger Logic:**")
                for field, op, val in rule["conditions"]:
                    st.markdown(f"`{field} {op} {val}`")
                
                st.write("---")
                alert_color = "#B91C1C" if rule["alert_count"] > 200 else "#3A5273"
                st.markdown(
                    f"<span style='font-size:1.8rem; font-weight:700; color:{alert_color};'>{rule['alert_count']}</span>"
                    f"<span style='font-size:13px; color:#6B7A99;'> accounts flagged</span>",
                    unsafe_allow_html=True
                )
                
                st.write("")
                if st.button(f"Apply Filter to Dashboard", key=f"rule_view_{rule['id']}", type="secondary", use_container_width=True):
                    st.session_state.current_view = rule["view_key"]
                    st.session_state.scroll_target = "analysis-table-workbench"
                    st.session_state.trigger_scroll = True
                    st.success(f"Dashboard filtered to **{rule['name']}**. Please return to the 'Operational Dashboard' tab to view the results.")
