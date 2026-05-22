import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Server Failure Prediction",
    page_icon="🖥️",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open('rf_model_100k.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler_100k.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

def add_engineered_features(scaled_row, cpu_pct, mem_pct, temp_c):
    row = list(scaled_row[0])
    row.append(scaled_row[0][0])
    row.append(scaled_row[0][1])
    row.append(scaled_row[0][8])
    row.append(scaled_row[0][0])
    row.append(scaled_row[0][1])
    high_risk = 1 if (cpu_pct > 85 or mem_pct > 90 or temp_c > 85) else 0
    row.append(high_risk)
    return np.array(row).reshape(1, -1)

def get_risk(prob):
    if prob < 0.30:
        return "🟢 LOW RISK", "success"
    elif prob < 0.60:
        return "🟡 MEDIUM RISK", "warning"
    else:
        return "🔴 HIGH RISK", "error"

# ── Header ──────────────────────────────────────────────────
st.title("🖥️ Server Failure Prediction System")
st.markdown("**100K Dataset | Random Forest | F1-Score: 0.9847 | Recall: 1.0000**")
st.markdown("---")

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")
    st.metric("Accuracy",   "99.84%")
    st.metric("F1-Score",   "98.47%")
    st.metric("Recall",     "100.0%")
    st.metric("Precision",  "96.99%")
    st.markdown("---")
    st.markdown("**Risk Levels**")
    st.success("🟢 LOW    : < 30%")
    st.warning("🟡 MEDIUM : 30–60%")
    st.error("🔴 HIGH   : > 60%")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.info("100,000 server records\n5,000 failures (5%)")
    st.markdown("**Top Failure Causes**")
    st.markdown("1. CPU > 85%")
    st.markdown("2. Memory > 90%")
    st.markdown("3. Temperature > 85°C")

# ── Input ────────────────────────────────────────────────────
st.header("📥 Enter Server Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💻 System")
    cpu     = st.slider("CPU Utilization (%)", 0.0, 100.0, 50.0, 0.5)
    memory  = st.slider("Memory Usage (%)", 0.0, 100.0, 55.0, 0.5)
    disk    = st.slider("Disk I/O (MB/s)", 0.0, 50.0, 25.0, 0.5)
    network = st.slider("Network Latency (ms)", 0.0, 200.0, 100.0, 1.0)

with col2:
    st.subheader("⚙️ Process")
    processes  = st.number_input("Process Count", 50, 999, 400, 10)
    threads    = st.number_input("Thread Count", 100, 4999, 1500, 50)
    context_sw = st.number_input("Context Switches", 100, 1999, 900, 50)
    cache_miss = st.slider("Cache Miss Rate", 0.01, 0.20, 0.08, 0.01)

with col3:
    st.subheader("🌡️ Hardware")
    temperature = st.slider("Temperature (°C)", 30.0, 95.0, 60.0, 0.5)
    power       = st.slider("Power (W)", 50.0, 300.0, 175.0, 5.0)
    uptime      = st.number_input("Uptime (hours)", 1.0, 1000.0, 300.0, 10.0)

# ── Live Health Indicators ───────────────────────────────────
st.markdown("---")
st.subheader("⚠️ Live Health Status")
h1, h2, h3, h4 = st.columns(4)
with h1:
    if cpu > 85:      st.error(f"🔴 CPU: {cpu}% — Critical")
    elif cpu > 70:    st.warning(f"🟡 CPU: {cpu}% — High")
    else:             st.success(f"🟢 CPU: {cpu}% — Normal")
with h2:
    if memory > 90:   st.error(f"🔴 Memory: {memory}% — Critical")
    elif memory > 75: st.warning(f"🟡 Memory: {memory}% — High")
    else:             st.success(f"🟢 Memory: {memory}% — Normal")
with h3:
    if temperature > 85:   st.error(f"🔴 Temp: {temperature}°C — Critical")
    elif temperature > 75: st.warning(f"🟡 Temp: {temperature}°C — High")
    else:                  st.success(f"🟢 Temp: {temperature}°C — Normal")
with h4:
    if network > 150:  st.error(f"🔴 Latency: {network}ms — Critical")
    elif network > 100:st.warning(f"🟡 Latency: {network}ms — High")
    else:              st.success(f"🟢 Latency: {network}ms — Normal")

# ── Predict Button ───────────────────────────────────────────
st.markdown("---")
if st.button("🔍 PREDICT FAILURE RISK", type="primary", use_container_width=True):

    raw = np.array([[cpu, memory, disk, network,
                     processes, threads, context_sw, cache_miss,
                     temperature, power, uptime]])
    scaled    = scaler.transform(raw)
    full_inp  = add_engineered_features(scaled, cpu, memory, temperature)
    pred      = model.predict(full_inp)[0]
    prob      = model.predict_proba(full_inp)[0][1]
    risk_label, alert_type = get_risk(prob)

    st.markdown("---")
    st.header("📊 Prediction Result")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Prediction",
                  "⚠️ SERVER FAILURE" if pred == 1 else "✅ NORMAL")
    with r2:
        st.metric("Failure Probability", f"{prob*100:.1f}%")
    with r3:
        st.metric("Risk Level", risk_label)

    st.subheader("📈 Probability Gauge")
    st.progress(float(prob))
    g1, g2, g3 = st.columns(3)
    with g1: st.markdown("🟢 LOW (0–30%)")
    with g2: st.markdown("🟡 MEDIUM (30–60%)")
    with g3: st.markdown("🔴 HIGH (60–100%)")

    if alert_type == "error":
        st.error(f"""
        🚨 HIGH RISK — IMMEDIATE ACTION REQUIRED!
        Failure probability: {prob*100:.1f}%

        ⚡ Actions:
        - Check CPU and Memory immediately
        - Monitor temperature — possible overheating
        - Migrate workloads to another server
        - Alert operations team NOW
        """)
    elif alert_type == "warning":
        st.warning(f"""
        ⚠️ MEDIUM RISK — Monitor Closely
        Failure probability: {prob*100:.1f}%

        📋 Actions:
        - Monitor server metrics continuously
        - Close non-essential processes
        - Check cooling system
        - Prepare backup server
        """)
    else:
        st.success(f"""
        ✅ LOW RISK — Server is Healthy
        Failure probability: {prob*100:.1f}%
        Server is operating within normal parameters.
        """)

    # Input summary
    st.subheader("📋 Input Summary")
    summary = pd.DataFrame({
        "Metric": ["CPU","Memory","Disk I/O","Network Latency",
                   "Processes","Threads","Context Switches","Cache Miss",
                   "Temperature","Power","Uptime"],
        "Value" : [f"{cpu}%", f"{memory}%", f"{disk} MB/s",
                   f"{network} ms", str(processes), str(threads),
                   str(context_sw), str(cache_miss),
                   f"{temperature}°C", f"{power}W", f"{uptime}h"],
        "Status": [
            "🔴 Critical" if cpu>85 else ("🟡 High" if cpu>70 else "🟢 Normal"),
            "🔴 Critical" if memory>90 else ("🟡 High" if memory>75 else "🟢 Normal"),
            "🟢 Normal",
            "🔴 Critical" if network>150 else ("🟡 High" if network>100 else "🟢 Normal"),
            "🟢 Normal","🟢 Normal","🟢 Normal","🟢 Normal",
            "🔴 Critical" if temperature>85 else ("🟡 High" if temperature>75 else "🟢 Normal"),
            "🔴 Critical" if power>250 else ("🟡 High" if power>200 else "🟢 Normal"),
            "🟢 Normal"
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ── Model Comparison ─────────────────────────────────────────
st.markdown("---")
with st.expander("📊 Full Model Comparison — 100K Dataset"):
    st.subheader("All Models Performance")
    comp = pd.DataFrame({
        "Model"    : ["Logistic Regression","Decision Tree","Random Forest ★",
                      "LSTM","GRU","Autoencoder"],
        "Type"     : ["ML","ML","ML","DL","DL","DL"],
        "Accuracy" : ["0.9809","0.9983","0.9984","—","—","—"],
        "Precision": ["0.7244","0.9690","0.9699","—","—","—"],
        "Recall"   : ["0.9990","0.9990","1.0000","—","—","—"],
        "F1-Score" : ["0.8398","0.9838","0.9847","—","—","—"],
        "AUC-ROC"  : ["0.9977","0.9987","0.9992","—","—","—"],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.subheader("Confusion Matrix (TP / TN / FP / FN) — 100K Test Set")
    cm1, cm2, cm3 = st.columns(3)
    with cm1:
        st.markdown("**Logistic Regression**")
        st.markdown("""
| | Pred Normal | Pred Failure |
|---|---|---|
| **Actual Normal** | TN=18,620 | FP=380 |
| **Actual Failure** | FN=1 | TP=999 |
        """)
    with cm2:
        st.markdown("**Decision Tree**")
        st.markdown("""
| | Pred Normal | Pred Failure |
|---|---|---|
| **Actual Normal** | TN=18,968 | FP=32 |
| **Actual Failure** | FN=1 | TP=999 |
        """)
    with cm3:
        st.markdown("**Random Forest ★**")
        st.markdown("""
| | Pred Normal | Pred Failure |
|---|---|---|
| **Actual Normal** | TN=18,969 | FP=31 |
| **Actual Failure** | **FN=0** | **TP=1,000** |
        """)
    st.success("★ Random Forest: FN=0 — Zero missed failures! All 1,000 test failures caught.")

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>"
    "Server Failure Prediction | 100K Dataset | "
    "Random Forest | BITS ZC229T Design Project"
    "</div>",
    unsafe_allow_html=True
)
