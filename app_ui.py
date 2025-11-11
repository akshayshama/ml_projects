# app_ui.py - Modern Streamlit UI for Two-Input Analysis
import streamlit as st
import requests
import plotly.graph_objects as go
import io

# --- CONFIGURATION & SETUP ---
FASTAPI_BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{FASTAPI_BASE_URL}/api/analyze_submissions"

st.set_page_config(
    page_title="AI & Plagiarism Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- VISUALIZATION FUNCTION (Plotly Gauge) ---
def plot_score_gauge(score: float, title: str):
    """Creates a meter gauge for the originality score (0 to 100)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={'text': title, 'font': {'size': 18}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "#007bff"},
            'steps': [
                {'range': [0, 50], 'color': "#ff4d4f"},  # Red: High Risk
                {'range': [50, 75], 'color': "#ffc107"}, # Yellow: Moderate Risk
                {'range': [75, 100], 'color': "#28a745"} # Green: Original
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score * 100
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(t=30, b=0, l=10, r=10), template="plotly_white")
    return fig

# --- HELPER FUNCTIONS ---
def get_alert_color(verdict):
    """Returns CSS color for verdict banner."""
    if verdict and verdict.startswith('PLAGIARISM'):
        return "#ff4d4f", "#ffe0e6"  # Red, Light Red BG
    if verdict and verdict.startswith('ORIGINAL'):
        return "#28aa45", "#e6ffe6"  # Green, Light Green BG
    return "#007bff", "#f0f4f7"

# --- UI LAYOUT ---
st.title("🤖  AI Code Plagiarism Detector ")
# st.markdown("### **72-Hour Sprint Edition**")
st.info("Upload exactly **two files** (.txt, .py) to compare their **AI generation risk** and **semantic similarity**.")

# File Uploader
uploaded_files = st.file_uploader(
    "Upload File A and File B (max 2 files)",
    type=['txt', 'py', 'js', 'md'],
    accept_multiple_files=True
)

st.markdown("---")
if st.button("🚀 Analyze Submissions", disabled=len(uploaded_files) != 2, use_container_width=True):
    if len(uploaded_files) != 2:
        st.error("Please upload exactly two files to run the comparison.")
    else:
        with st.spinner('Running AI and Semantic Models (Phase 2, 3, 4, 5)...'):
            try:
                # Prepare data for multipart/form-data request
                files_payload = [
                    ('files', (f.name, f.getvalue(), f.type)) for f in uploaded_files
                ]

                # Send POST request to FastAPI
                response = requests.post(
                    API_ENDPOINT,
                    files=files_payload,
                    data={'threshold': 0.60}
                )

                if response.status_code == 200:
                    report = response.json()
                    st.session_state['report'] = report
                    st.session_state['uploaded_files'] = uploaded_files
                else:
                    st.error(f"API Analysis Failed (Status {response.status_code}): {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error(f"Failed to connect to FastAPI backend. Ensure the backend (`app.py`) is running on port 8000.")
            except Exception as e:
                st.exception(e)

# --- RESULTS DISPLAY (Modern Dashboard Layout) ---
if 'report' in st.session_state and st.session_state['report']:
    report = st.session_state['report']

    st.markdown("---")
    st.header("Results Dashboard")

    # --- Download PDF Report ---
    if st.button("Download Full PDF Report", use_container_width=True, key='pdf_download'):
        try:
            response = requests.post(
                f"{FASTAPI_BASE_URL}/api/download_report",
                json=report,
                timeout=10
            )

            if response.status_code == 200:
                st.download_button(
                    label="Click to Download PDF",
                    data=response.content,
                    file_name=f"plagiarism_report_{report.get('report_id', 'session')}.pdf",
                    mime="application/pdf"
                )
            else:
                st.error(f"PDF Generation Failed: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to PDF generation service.")
        except Exception as e:
            st.error(f"An error occurred during PDF preparation: {e}")

    # --- Core Metrics & Gauges ---
    col1, col2, col3 = st.columns([1, 1, 1.2])

    # Calculate core metrics
    avg_plag_prob = (report['file_results'][0]['ai_probability'] + report['file_results'][1]['ai_probability']) / 2
    overall_score = max(0.0, min(1.0, 1 - avg_plag_prob))

    with col1:
        st.subheader("Total Originality")
        st.plotly_chart(plot_score_gauge(overall_score, "Originality Score"), use_container_width=True)

    with col2:
        st.subheader("Cross-File Match")
        st.plotly_chart(plot_score_gauge(report['semantic_similarity_score_A_B'], "Semantic Similarity"), use_container_width=True)

    with col3:
        st.subheader("Key Findings")
        st.metric(
            label="AI Risk Threshold",
            value="60%",
            help="AI probability above this score is flagged as plagiarism."
        )
        st.metric(
            label="Semantic Similarity Threshold",
            value="75.0%",
            help="Similarity between A and B above this score is flagged."
        )

        # ✅ FIXED: Define verdict before using
        verdict = report['file_results'][0].get('verdict', '').upper()

        if 'PLAGIARISM' in verdict:
            st.error("🚨 Action Required: Review AI/Match scores below.")
        else:
            st.success("🎉 Good Submission.")

    st.markdown("---")

    # --- Detailed File Breakdown ---
    st.header("Detailed File Breakdown")
    file_cols = st.columns(2)

    for i, file_report in enumerate(report['file_results']):
        with file_cols[i]:
            is_ai = file_report['is_ai_plagiarism']

            st.markdown(f"#### **File {i+1}:** `{file_report['filename']}`", unsafe_allow_html=True)

            st.metric(
                label="AI Generation Probability",
                value=f"{file_report['ai_probability']*100:.2f}%",
                delta="Flagged" if is_ai else "Safe",
                delta_color="inverse"
            )

            with st.expander("Linguistic & Stylometric Analysis"):
                st.markdown(f"""
                - **Semantic Score (vs. Other File):** `{file_report['semantic_score']*100:.2f}%`
                - **Avg Sentence Length:** {'18.5 words' if i == 0 else '12.1 words'} (Simulated)
                - **Vocabulary Diversity (TTR):** {'0.42' if i == 0 else '0.65'} (Simulated)
                - **Style:** {'Predictable/LLM-like' if is_ai else 'Human/Varied'}
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("AI Detector MVP built using Sentence-Transformers and RoBERTa on FastAPI/Streamlit.")
