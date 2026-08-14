
import streamlit as st

st.set_page_config(
    page_title="GenMed AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f4c75, #1b6ca8, #1a7f5e);
        color: white; padding: 20px 28px; border-radius: 12px;
        margin-bottom: 24px;
    }
    .main-header h1 { margin: 0; font-size: 28px; }
    .main-header p  { margin: 4px 0 0; opacity: 0.85; font-size: 14px; }
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 16px; text-align: center;
    }
    .risk-high    { background:#fee2e2; border-left:4px solid #dc2626; padding:12px; border-radius:8px; }
    .risk-moderate{ background:#fef9c3; border-left:4px solid #ca8a04; padding:12px; border-radius:8px; }
    .risk-low     { background:#dcfce7; border-left:4px solid #16a34a; padding:12px; border-radius:8px; }
    .stButton>button { background:#1a7f5e; color:white; border:none; border-radius:8px; }
    .stButton>button:hover { background:#156b4f; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏥 GenMed AI</h1>
    <p>Generative AI-Powered Medical Report Summarizer & Disease Risk Predictor</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("🏥 GenMed AI")
st.sidebar.caption("Healthcare Analytics Platform")

page = st.sidebar.radio("Navigate", [
    "🏠 Home / Dashboard",
    "🔍 Patient Lookup",
    "🤖 Disease Prediction",
    "📊 Analytics",
    "📋 Model Info",
    "📄 Report Summarizer"
])

BACKEND = st.sidebar.text_input("Backend URL", value="http://localhost:8000")

# ── Check backend connectivity ─────────────────────────────────────────────────
import requests

def api(endpoint, method="GET", data=None, params=None):
    try:
        url = f"{BACKEND}{endpoint}"
        if method == "POST":
            r = requests.post(url, json=data, timeout=10)
        else:
            r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Make sure it's running on the URL above.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e.response.json().get('detail', str(e))}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home / Dashboard":
    health = api("/health")
    summary = api("/analytics/summary")

    col1, col2, col3, col4 = st.columns(4)
    if summary:
        with col1:
            st.metric("Total Patients", f"{summary['total_patients']:,}")
        with col2:
            st.metric("Average Age", f"{summary['avg_age']} yrs")
        with col3:
            st.metric("Avg Billing", f"₹{summary['avg_billing']:,.0f}")
        with col4:
            st.metric("Total Revenue", f"₹{summary['total_revenue']/1e6:.2f}M")

    st.divider()

    if health:
        backend_ok = health.get("status") == "healthy"
        if backend_ok:
            st.sidebar.success("✅ Backend Connected")
        else:
            st.sidebar.error("❌ Backend Down")
        st.sidebar.caption(f"Model: {health.get('model_type','—')}")
        st.sidebar.caption(f"Accuracy: {health.get('model_accuracy','—')}")

    if summary:
        import plotly.express as px
        col_a, col_b = st.columns(2)
        with col_a:
            cond_df = {"Condition": list(summary["conditions"].keys()),
                       "Count": list(summary["conditions"].values())}
            fig = px.bar(cond_df, x="Count", y="Condition", orientation="h",
                         title="Patients by Condition", color="Count",
                         color_continuous_scale="teal")
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            gender_data = summary["genders"]
            fig2 = px.pie(names=list(gender_data.keys()),
                          values=list(gender_data.values()),
                          title="Gender Distribution",
                          color_discrete_sequence=["#1a7f5e","#0e5c88"])
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            bt = summary["blood_types"]
            fig3 = px.bar(x=list(bt.keys()), y=list(bt.values()),
                          title="Blood Type Distribution",
                          color=list(bt.values()), color_continuous_scale="blues")
            fig3.update_layout(height=300, showlegend=False,
                               xaxis_title="Blood Type", yaxis_title="Count")
            st.plotly_chart(fig3, use_container_width=True)

        with col_d:
            adm = summary["admission_types"]
            fig4 = px.pie(names=list(adm.keys()), values=list(adm.values()),
                          title="Admission Type Split",
                          color_discrete_sequence=["#ef4444","#f59e0b","#3b82f6"])
            fig4.update_layout(height=300)
            st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PATIENT LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Patient Lookup":
    st.subheader("🔍 Patient Lookup")

    tab1, tab2, tab3 = st.tabs(["Browse Patients", "Search by Name", "Lookup by ID"])

    with tab1:
        st.markdown("**Filter Patients**")
        fc1, fc2, fc3 = st.columns(3)
        opts = api("/meta/options")
        cond_filter = fc1.selectbox("Condition", ["All"] + (opts["conditions"] if opts else []))
        gen_filter  = fc2.selectbox("Gender", ["All", "Male", "Female"])
        adm_filter  = fc3.selectbox("Admission Type", ["All"] + (opts["admission_types"] if opts else []))

        params = {"page": 1, "limit": 20}
        if cond_filter != "All": params["condition"] = cond_filter
        if gen_filter  != "All": params["gender"]    = gen_filter
        if adm_filter  != "All": params["admission_type"] = adm_filter

        result = api("/patients", params=params)
        if result:
            st.caption(f"Showing {len(result['patients'])} of {result['total']} patients")
            import pandas as pd
            df = pd.DataFrame(result["patients"])
            show_cols = ["id","name","age","gender","blood_type","medical_condition",
                         "admission_type","test_results","billing_amount","hospital"]
            st.dataframe(df[show_cols], use_container_width=True)

    with tab2:
        name_query = st.text_input("Search patient name", placeholder="e.g. Arjun")
        if name_query:
            res = api(f"/patients/search/{name_query}")
            if res and res["patients"]:
                import pandas as pd
                st.caption(f"{res['total']} results")
                st.dataframe(pd.DataFrame(res["patients"]), use_container_width=True)
            elif res:
                st.info("No patients found.")

    with tab3:
        pid = st.number_input("Patient ID", min_value=1, max_value=1000, value=1, step=1)
        if st.button("Fetch Patient"):
            p = api(f"/patients/{pid}")
            if p:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Name:** {p['name']}")
                    st.markdown(f"**Age:** {p['age']} | **Gender:** {p['gender']}")
                    st.markdown(f"**Blood Type:** {p['blood_type']}")
                    st.markdown(f"**Condition:** {p['medical_condition']}")
                    st.markdown(f"**Medication:** {p['medication']}")
                with c2:
                    st.markdown(f"**Hospital:** {p['hospital']}")
                    st.markdown(f"**Doctor:** {p['doctor']}")
                    st.markdown(f"**Admission Type:** {p['admission_type']}")
                    st.markdown(f"**Test Results:** {p['test_results']}")
                    st.markdown(f"**Billing:** ₹{p['billing_amount']:,.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DISEASE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Disease Prediction":
    st.subheader("🤖 Disease Risk Prediction")
    st.caption("Enter patient details and get an ML-powered disease risk assessment.")

    opts = api("/meta/options") or {
        "genders":["Male","Female"],
        "blood_types":["A+","A-","B+","B-","O+","O-","AB+","AB-"],
        "admission_types":["Emergency","Elective","Urgent"],
        "test_results":["Normal","Abnormal","Inconclusive"]
    }

    with st.form("predict_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        name   = r1c1.text_input("Patient Name *", placeholder="Full name")
        age    = r1c2.number_input("Age *", min_value=1, max_value=120, value=45)
        gender = r1c3.selectbox("Gender *", opts["genders"])

        r2c1, r2c2, r2c3 = st.columns(3)
        blood_type     = r2c1.selectbox("Blood Type *", opts["blood_types"])
        admission_type = r2c2.selectbox("Admission Type *", opts["admission_types"])
        test_results   = r2c3.selectbox("Test Results *", opts["test_results"])

        r3c1, r3c2, r3c3 = st.columns(3)
        billing        = r3c1.number_input("Billing Amount (₹)", min_value=0.0, value=25000.0, step=500.0)
        symptom_score  = r3c2.slider("Symptom Severity Score (0–100)", min_value=0, max_value=100, value=50,
                                      help="0 = no symptoms, 100 = critical. E.g. Cancer=85, Asthma=40, Anemia=30")
        bmi            = r3c3.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.5,
                                            help="Body Mass Index. Obesity >30, Normal 18.5–24.9")

        r4c1, r4c2, r4c3 = st.columns(3)
        medication = r4c1.text_input("Current Medication (optional)")
        hospital   = r4c2.text_input("Hospital (optional)")

        submitted = st.form_submit_button("🔮 Predict Disease Risk", use_container_width=True)

    if submitted:
        if not name:
            st.warning("Please enter patient name.")
        else:
            payload = {
                "name": name, "age": age, "gender": gender,
                "blood_type": blood_type, "admission_type": admission_type,
                "test_results": test_results, "billing_amount": billing,
                "symptom_score": float(symptom_score), "bmi": float(bmi),
                "medication": medication or None, "hospital": hospital or None
            }
            with st.spinner("Analyzing patient data..."):
                result = api("/predict", method="POST", data=payload)

            if result:
                st.success("✅ Prediction Complete")
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Predicted Condition", result["predicted_condition"])
                pc2.metric("Confidence", f"{result['confidence']*100:.1f}%")
                pc3.metric("Risk Level", result["risk_level"])

                risk_class = f"risk-{result['risk_level'].lower()}"
                st.markdown(f"""
                <div class="{risk_class}">
                    <strong>📋 Clinical Recommendation</strong><br>{result['recommendation']}
                </div>
                """, unsafe_allow_html=True)

                st.divider()
                import plotly.express as px, pandas as pd

                col_prob, col_fi = st.columns(2)
                with col_prob:
                    st.markdown("**Probability by Condition**")
                    probs = result["all_probabilities"]
                    prob_df = pd.DataFrame({
                        "Condition": list(probs.keys()),
                        "Probability": [v*100 for v in probs.values()]
                    }).sort_values("Probability", ascending=True)
                    fig = px.bar(prob_df, x="Probability", y="Condition",
                                 orientation="h", color="Probability",
                                 color_continuous_scale="RdYlGn",
                                 range_color=[0,100])
                    fig.update_layout(height=380, showlegend=False,
                                      xaxis_title="Probability (%)")
                    st.plotly_chart(fig, use_container_width=True)

                with col_fi:
                    st.markdown("**Feature Importances**")
                    fi = result["feature_importances"]
                    fi_df = pd.DataFrame({
                        "Feature": list(fi.keys()),
                        "Importance": list(fi.values())
                    }).sort_values("Importance", ascending=True)
                    fig2 = px.bar(fi_df, x="Importance", y="Feature",
                                  orientation="h", color="Importance",
                                  color_continuous_scale="teal")
                    fig2.update_layout(height=380, showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.subheader("📊 Population Analytics Dashboard")

    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    tab1, tab2, tab3, tab4 = st.tabs(["Conditions", "Age Distribution", "Monthly Trends", "Hospitals"])

    with tab1:
        data = api("/analytics/conditions")
        if data:
            df = pd.DataFrame(data).T.reset_index()
            df.columns = ["condition","count","avg_age","avg_billing","percentage"]
            df = df.sort_values("count", ascending=False)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(df, x="condition", y="count", title="Patient Count by Condition",
                             color="count", color_continuous_scale="teal")
                fig.update_layout(height=350, showlegend=False, xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.bar(df, x="condition", y="avg_billing",
                              title="Average Billing by Condition (₹)",
                              color="avg_billing", color_continuous_scale="reds")
                fig2.update_layout(height=350, showlegend=False, xaxis_tickangle=-30)
                st.plotly_chart(fig2, use_container_width=True)

            fig3 = px.scatter(df, x="avg_age", y="avg_billing", size="count",
                              color="condition", text="condition",
                              title="Avg Age vs Avg Billing (bubble size = patient count)")
            fig3.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        age_data = api("/analytics/age-distribution")
        if age_data:
            bins = age_data["bins"]
            fig = px.bar(x=list(bins.keys()), y=list(bins.values()),
                         title="Age Distribution (10-year bins)",
                         labels={"x":"Age Group","y":"Patients"},
                         color=list(bins.values()), color_continuous_scale="blues")
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        monthly = api("/analytics/monthly-admissions")
        if monthly:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=monthly["months"], y=monthly["admissions"],
                                  name="Admissions", marker_color="#1a7f5e"))
            fig.add_trace(go.Scatter(x=monthly["months"],
                                      y=[r/1000 for r in monthly["revenue"]],
                                      name="Revenue (₹K)", yaxis="y2",
                                      line=dict(color="#f59e0b", width=2),
                                      mode="lines+markers"))
            fig.update_layout(
                title="Monthly Admissions & Revenue",
                yaxis=dict(title="Admissions"),
                yaxis2=dict(title="Revenue (₹K)", overlaying="y", side="right"),
                height=400, legend=dict(orientation="h")
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        hosp_data = api("/analytics/top-hospitals")
        if hosp_data:
            fig = px.bar(x=list(hosp_data.values()), y=list(hosp_data.keys()),
                         orientation="h", title="Top Hospitals by Patient Count",
                         color=list(hosp_data.values()), color_continuous_scale="teal")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Model Info":
    st.subheader("📋 ML Model Information")

    import plotly.express as px
    import pandas as pd

    info = api("/analytics/model-info")
    if info:
        c1, c2, c3 = st.columns(3)
        c1.metric("Model Type", info["model_name"])
        c2.metric("Test Accuracy", f"{info['accuracy']*100:.1f}%")
        c3.metric("Disease Classes", len(info["classes"]))

        st.markdown("**Feature Importances**")
        fi_df = pd.DataFrame({
            "Feature": info["feature_names"],
            "Importance": info["importances"]
        }).sort_values("Importance", ascending=True)
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="teal",
                     title="Feature Importance (Higher = More Influential)")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Supported Disease Classes**")
        cols = st.columns(5)
        for i, cls in enumerate(info["classes"]):
            cols[i % 5].info(cls)

    st.markdown("---")
    st.markdown("**🔗 API Endpoints (Interactive Docs)**")
    st.code(f"{BACKEND}/docs", language="bash")
    st.caption("Open the above URL in your browser to see all API endpoints with Swagger UI.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORT SUMMARIZER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Report Summarizer":
    st.subheader("📄 Medical Report Summarizer")
    st.caption("Generates a full 7-section clinical report using template-based summarization — no external API.")

    # ── PDF generator (defined inside page scope) ──────────────────────────────
    def generate_pdf_report(report):
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
        import io, re

        def strip_html(t): return re.sub(r"<[^>]+>", "", str(t))

        buf = io.BytesIO()
        W   = A4[0] - 36*mm
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=18*mm, rightMargin=18*mm,
                                topMargin=16*mm,  bottomMargin=16*mm)

        DARK  = colors.HexColor("#0F4C75")
        TEAL  = colors.HexColor("#1a7f5e")
        LTEAL = colors.HexColor("#E1F5EE")
        GRAY  = colors.HexColor("#f8fafc")
        GMID  = colors.HexColor("#e2e8f0")
        GTXT  = colors.HexColor("#475569")
        BLK   = colors.HexColor("#1e293b")

        risk     = report.get("prediction", {}).get("risk_level", "Low")
        risk_bg  = {"High": colors.HexColor("#fee2e2"),
                    "Moderate": colors.HexColor("#fef9c3"),
                    "Low": colors.HexColor("#dcfce7")}.get(risk, colors.HexColor("#dcfce7"))
        risk_clr = {"High": colors.HexColor("#dc2626"),
                    "Moderate": colors.HexColor("#92400e"),
                    "Low": colors.HexColor("#166534")}.get(risk, colors.HexColor("#166534"))

        base = getSampleStyleSheet()
        def PS(name, **kw): return ParagraphStyle(name, parent=base["Normal"], **kw)

        sH1    = PS("H1",  fontSize=11, textColor=TEAL,  fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=3)
        sBody  = PS("Bd",  fontSize=9,  textColor=BLK,   fontName="Helvetica", leading=14, spaceAfter=4)
        sLbl   = PS("Lb",  fontSize=8,  textColor=GTXT,  fontName="Helvetica", spaceAfter=1)
        sVal   = PS("Vl",  fontSize=10, textColor=BLK,   fontName="Helvetica-Bold", spaceAfter=5)
        sFoot  = PS("Ft",  fontSize=7.5,textColor=GTXT,  fontName="Helvetica-Oblique", alignment=1)

        snap = report.get("patient_snapshot", {})
        pred = report.get("prediction", {})
        secs = report.get("sections", {})
        conf = pred.get("confidence", 0)

        story = []

        # Header banner
        hdr = Table([[
            Paragraph("GenMed AI", PS("hL", fontSize=14, textColor=colors.white, fontName="Helvetica-Bold")),
            Paragraph("Medical Report", PS("hR", fontSize=14, textColor=colors.white,
                                           fontName="Helvetica", alignment=2))
        ]], colWidths=[W*0.6, W*0.4])
        hdr.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(-1,-1), DARK),
            ("TOPPADDING",    (0,0),(-1,-1), 10), ("BOTTOMPADDING",(0,0),(-1,-1), 10),
            ("LEFTPADDING",   (0,0),(-1,-1), 12), ("RIGHTPADDING", (0,0),(-1,-1), 12),
        ]))
        story.append(hdr)
        story.append(Spacer(1, 5))
        story.append(Paragraph(
            f"Report ID: <b>{report.get('report_id','—')}</b>  |  Generated: {report.get('generated_at','—')}",
            PS("rid", fontSize=8, textColor=GTXT, fontName="Helvetica")
        ))
        story.append(HRFlowable(width=W, thickness=0.5, color=GMID, spaceAfter=8))

        # Patient info table
        rows = [
            [Paragraph("Patient Name", sLbl), Paragraph(snap.get("name","—"), sVal),
             Paragraph("Age / Gender",  sLbl), Paragraph(f"{snap.get('age','—')} yrs / {snap.get('gender','—')}", sVal)],
            [Paragraph("Blood Type",    sLbl), Paragraph(snap.get("blood_type","—"), sVal),
             Paragraph("Hospital",      sLbl), Paragraph(snap.get("hospital","—"), sVal)],
            [Paragraph("Doctor",        sLbl), Paragraph(snap.get("doctor","—"), sVal),
             Paragraph("Condition",     sLbl), Paragraph(snap.get("medical_condition","—"), sVal)],
        ]
        ptbl = Table(rows, colWidths=[W*0.17, W*0.30, W*0.17, W*0.36])
        ptbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1), GRAY), ("BOX",(0,0),(-1,-1),0.5,GMID),
            ("INNERGRID",(0,0),(-1,-1),0.3,GMID),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(ptbl)
        story.append(Spacer(1, 8))

        # Risk banner
        rb = Table([[
            Paragraph(f"Predicted: <b>{pred.get('predicted_condition','—')}</b>",
                      PS("rp", fontSize=11, textColor=risk_clr, fontName="Helvetica-Bold")),
            Paragraph(f"Confidence: <b>{conf*100:.1f}%</b>",
                      PS("rc", fontSize=11, textColor=risk_clr, fontName="Helvetica-Bold", alignment=1)),
            Paragraph(f"Risk Level: <b>{risk}</b>",
                      PS("rr", fontSize=11, textColor=risk_clr, fontName="Helvetica-Bold", alignment=2)),
        ]], colWidths=[W*0.45, W*0.28, W*0.27])
        rb.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),risk_bg),("BOX",(0,0),(-1,-1),1.5,risk_clr),
            ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
            ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(rb)
        story.append(Spacer(1, 10))

        # Report sections
        for key, label in [
            ("patient_overview",         "Patient Overview"),
            ("admission_details",        "Admission Details"),
            ("clinical_findings",        "Clinical Findings"),
            ("ai_diagnosis",             "AI-Assisted Diagnosis"),
            ("risk_assessment",          "Risk Assessment"),
            ("treatment_recommendation", "Treatment Recommendation"),
            ("warning_signs",            "Warning Signs"),
            ("discharge_summary",        "Discharge Summary"),
        ]:
            text = secs.get(key, "")
            if not text: continue
            story.append(HRFlowable(width=W, thickness=0.5, color=LTEAL, spaceAfter=0))
            story.append(Paragraph(f"▌ {label}", sH1))
            story.append(Paragraph(strip_html(text), sBody))

        story.append(Spacer(1, 8))
        story.append(HRFlowable(width=W, thickness=1, color=GMID, spaceAfter=6))

        # Probability table
        story.append(Paragraph("▌ Condition Probability Breakdown", sH1))
        probs = sorted(pred.get("all_probabilities", {}).items(), key=lambda x: -x[1])
        prows = [["Condition", "Probability", "Visual"]]
        for cname, prob in probs:
            bar = "█" * int(prob * 28) + "░" * (28 - int(prob * 28))
            prows.append([cname, f"{prob*100:.1f}%", bar])
        ptb2 = Table(prows, colWidths=[W*0.38, W*0.15, W*0.47])
        ptb2.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),DARK),("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),8),
            ("FONTNAME",(2,1),(2,-1),"Courier"),("TEXTCOLOR",(2,1),(2,-1),TEAL),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[GRAY, colors.white]),
            ("GRID",(0,0),(-1,-1),0.3,GMID),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),
        ]))
        story.append(ptb2)
        story.append(Spacer(1, 10))

        # Disclaimer + footer
        story.append(HRFlowable(width=W, thickness=0.5, color=GMID, spaceAfter=4))
        story.append(Paragraph("⚠  " + report.get("disclaimer",""), sFoot))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"GenMed AI v1.0  |  {report.get('generated_at','—')}  |  For educational purposes only",
            sFoot
        ))

        doc.build(story)
        buf.seek(0)
        return buf.read()

    # ── Report renderer ────────────────────────────────────────────────────────
    def render_report(report):
        snap = report.get("patient_snapshot", {})
        pred = report.get("prediction", {})
        secs = report.get("sections", {})
        risk = pred.get("risk_level", "Low")

        risk_color  = {"High":"#fee2e2","Moderate":"#fef9c3","Low":"#dcfce7"}.get(risk,"#f0f0f0")
        risk_border = {"High":"#dc2626","Moderate":"#ca8a04","Low":"#16a34a"}.get(risk,"#888")

        st.markdown(f"""
<div style="background:{risk_color};border-left:5px solid {risk_border};
            padding:16px 20px;border-radius:10px;margin-bottom:16px;">
  <div style="font-size:11px;font-weight:600;letter-spacing:1px;color:#555;margin-bottom:4px;">
    MEDICAL REPORT &nbsp;·&nbsp; {report.get("report_id","—")} &nbsp;·&nbsp; {report.get("generated_at","—")}
  </div>
  <div style="font-size:20px;font-weight:700;margin-bottom:4px;color:#1e293b;">
    {snap.get("name","—")} &nbsp;·&nbsp; {snap.get("age","—")} yrs &nbsp;·&nbsp; {snap.get("gender","—")} &nbsp;·&nbsp; Blood: {snap.get("blood_type","—")}
  </div>
  <div style="font-size:13px;color:#555;">
    {snap.get("hospital","—")} &nbsp;|&nbsp; {snap.get("doctor","—")} &nbsp;|&nbsp; Risk: <strong>{risk}</strong>
  </div>
</div>
""", unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        k1.metric("Predicted Condition", pred.get("predicted_condition","—"))
        k2.metric("Confidence",           f"{pred.get('confidence',0)*100:.1f}%")
        k3.metric("Risk Level",           risk)

        st.divider()

        icons = {
            "patient_overview":         "👤 Patient Overview",
            "admission_details":        "🏥 Admission Details",
            "clinical_findings":        "🔬 Clinical Findings",
            "ai_diagnosis":             "🤖 AI-Assisted Diagnosis",
            "risk_assessment":          "⚠️ Risk Assessment",
            "treatment_recommendation": "💊 Treatment Recommendation",
            "warning_signs":            "🚨 Warning Signs",
            "discharge_summary":        "📤 Discharge Summary",
        }
        for key, label in icons.items():
            text = secs.get(key, "")
            if text:
                with st.expander(label, expanded=True):
                    st.markdown(text, unsafe_allow_html=True)

        st.divider()
        st.caption(f"⚠️ {report.get('disclaimer','')}")

        pdf_bytes = generate_pdf_report(report)
        st.download_button(
            label="⬇️ Download Report (PDF)",
            data=pdf_bytes,
            file_name=f"{report.get('report_id','report')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    # ── Tabs ───────────────────────────────────────────────────────────────────
    opts = api("/meta/options") or {
        "genders":         ["Male","Female"],
        "blood_types":     ["A+","A-","B+","B-","O+","O-","AB+","AB-"],
        "admission_types": ["Emergency","Elective","Urgent"],
        "test_results":    ["Normal","Abnormal","Inconclusive"]
    }

    tab1, tab2 = st.tabs(["By Patient ID (existing)", "Custom Patient Input"])

    with tab1:
        st.markdown("**Select an existing patient from the dataset (1 – 2000)**")
        pid = st.number_input("Patient ID", min_value=1, max_value=2000, value=1, step=1)
        if st.button("📋 Generate Report", key="gen_id", use_container_width=True):
            with st.spinner("Building report..."):
                report = api(f"/report/generate/{pid}")
            if report:
                render_report(report)

    with tab2:
        with st.form("custom_report_form"):
            r1, r2, r3 = st.columns(3)
            c_name = r1.text_input("Name *",   value="Arjun Kumar")
            c_age  = r2.number_input("Age *",  min_value=1, max_value=120, value=55)
            c_gen  = r3.selectbox("Gender *",  opts["genders"])

            r4, r5, r6 = st.columns(3)
            c_bt   = r4.selectbox("Blood Type *",      opts["blood_types"],     index=4)
            c_adm  = r5.selectbox("Admission Type *",  opts["admission_types"], index=0)
            c_test = r6.selectbox("Test Results *",    opts["test_results"],    index=1)

            r7, r8, r9 = st.columns(3)
            c_bill = r7.number_input("Billing (₹)", min_value=0.0, value=45000.0, step=500.0)
            c_sym  = r8.slider("Symptom Score (0–100)", 0, 100, 75)
            c_bmi  = r9.number_input("BMI", min_value=10.0, max_value=60.0, value=26.0, step=0.5)

            r10, r11, r12 = st.columns(3)
            c_med    = r10.text_input("Medication",   value="Aspirin")
            c_hosp   = r11.text_input("Hospital",     value="Apollo Hospitals Chennai")
            c_doc    = r12.text_input("Doctor",       value="Dr. Ravi Kumar")

            r13, r14 = st.columns(2)
            c_adm_dt = r13.text_input("Admission Date (YYYY-MM-DD)", value="2024-01-10")
            c_dis_dt = r14.text_input("Discharge Date (YYYY-MM-DD)", value="2024-01-17")

            go = st.form_submit_button("📋 Generate Report", use_container_width=True)

        if go:
            payload = {"patient_data": {
                "id": 0, "name": c_name, "age": int(c_age), "gender": c_gen,
                "blood_type": c_bt, "admission_type": c_adm,
                "test_results": c_test, "billing_amount": float(c_bill),
                "symptom_score": float(c_sym), "bmi": float(c_bmi),
                "medication": c_med, "hospital": c_hosp, "doctor": c_doc,
                "room_number": 101, "insurance_provider": "Self-Pay",
                "date_of_admission": c_adm_dt, "date_of_discharge": c_dis_dt
            }}
            with st.spinner("Building report..."):
                report = api("/report/generate", method="POST", data=payload)
            if report:
                render_report(report)


st.sidebar.divider()
st.sidebar.caption("GenMed AI v1.0 | © 2026")
st.sidebar.caption("For educational purposes only.")
