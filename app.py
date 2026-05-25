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

def get_failure_reasons(cpu, memory, disk, network, processes, threads,
                        context_sw, cache_miss, temperature, power, uptime):
    reasons = []
    if cpu > 85:
        reasons.append(f"CPU critical ({cpu:.0f}%)")
    elif cpu > 70:
        reasons.append(f"CPU high ({cpu:.0f}%)")
    if memory > 90:
        reasons.append(f"Memory critical ({memory:.0f}%)")
    elif memory > 75:
        reasons.append(f"Memory high ({memory:.0f}%)")
    if temperature > 85:
        reasons.append(f"Overheating ({temperature:.0f}°C)")
    elif temperature > 75:
        reasons.append(f"Temp elevated ({temperature:.0f}°C)")
    if network > 150:
        reasons.append(f"Network latency critical ({network:.0f}ms)")
    elif network > 100:
        reasons.append(f"Network latency high ({network:.0f}ms)")
    if power > 250:
        reasons.append(f"Power draw critical ({power:.0f}W)")
    elif power > 200:
        reasons.append(f"Power draw high ({power:.0f}W)")
    if cache_miss > 0.15:
        reasons.append(f"High cache miss rate ({cache_miss:.2f})")
    if not reasons:
        reasons.append("Metrics within normal range")
    return ", ".join(reasons)

def predict_batch(df):
    results = []
    feature_cols = ["cpu", "memory", "disk", "network",
                    "processes", "threads", "context_sw", "cache_miss",
                    "temperature", "power", "uptime"]
    for _, row in df.iterrows():
        raw = np.array([[row["cpu"], row["memory"], row["disk"], row["network"],
                         row["processes"], row["threads"], row["context_sw"],
                         row["cache_miss"], row["temperature"],
                         row["power"], row["uptime"]]])
        scaled = scaler.transform(raw)
        full_inp = add_engineered_features(scaled, row["cpu"], row["memory"], row["temperature"])
        pred = model.predict(full_inp)[0]
        prob = model.predict_proba(full_inp)[0][1]
        risk_label, _ = get_risk(prob)
        reason = get_failure_reasons(
            row["cpu"], row["memory"], row["disk"], row["network"],
            row["processes"], row["threads"], row["context_sw"],
            row["cache_miss"], row["temperature"], row["power"], row["uptime"]
        )
        results.append({
            "Prediction": "⚠️ FAILURE" if pred == 1 else "✅ NORMAL",
            "Failure Prob": f"{prob*100:.1f}%",
            "Risk Level": risk_label,
            "Failure Reason": reason,
            "_prob": prob,
            "_pred": pred,
        })
    return pd.DataFrame(results)

def generate_sample_fleet(n=20):
    np.random.seed(42)
    data = []
    for i in range(n):
        failure_mode = np.random.choice(
            ["normal", "cpu_overload", "memory_overload", "overheating", "network_issue"],
            p=[0.55, 0.15, 0.12, 0.10, 0.08]
        )
        if failure_mode == "cpu_overload":
            cpu = np.random.uniform(87, 99)
            memory = np.random.uniform(40, 80)
            temp = np.random.uniform(55, 78)
            network = np.random.uniform(20, 90)
        elif failure_mode == "memory_overload":
            cpu = np.random.uniform(40, 75)
            memory = np.random.uniform(92, 99)
            temp = np.random.uniform(55, 78)
            network = np.random.uniform(20, 90)
        elif failure_mode == "overheating":
            cpu = np.random.uniform(60, 85)
            memory = np.random.uniform(50, 80)
            temp = np.random.uniform(87, 95)
            network = np.random.uniform(20, 90)
        elif failure_mode == "network_issue":
            cpu = np.random.uniform(30, 65)
            memory = np.random.uniform(40, 70)
            temp = np.random.uniform(45, 70)
            network = np.random.uniform(155, 200)
        else:
            cpu = np.random.uniform(10, 68)
            memory = np.random.uniform(20, 72)
            temp = np.random.uniform(35, 72)
            network = np.random.uniform(5, 95)

        data.append({
            "server_id": f"SRV-{i+1:03d}",
            "cpu": round(cpu, 1),
            "memory": round(memory, 1),
            "disk": round(np.random.uniform(2, 45), 1),
            "network": round(network, 1),
            "processes": int(np.random.randint(80, 950)),
            "threads": int(np.random.randint(200, 4800)),
            "context_sw": int(np.random.randint(150, 1900)),
            "cache_miss": round(np.random.uniform(0.01, 0.19), 2),
            "temperature": round(temp, 1),
            "power": round(np.random.uniform(60, 290), 1),
            "uptime": round(np.random.uniform(1, 990), 1),
        })
    return pd.DataFrame(data)

def color_risk_row(row):
    prob = row["_prob"]
    if prob >= 0.60:
        return ["background-color: #ffcccc"] * len(row)
    elif prob >= 0.30:
        return ["background-color: #fff3cd"] * len(row)
    else:
        return ["background-color: #d4edda"] * len(row)

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

# ── Title ────────────────────────────────────────────────────
st.title("🖥️ Server Failure Prediction System")
st.markdown("**100K Dataset | Random Forest | F1-Score: 0.9847 | Recall: 1.0000**")
st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────
tab_fleet, tab_single, tab_models = st.tabs([
    "🏢 Fleet Dashboard",
    "🔍 Single Server Prediction",
    "📊 Model Comparison"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — FLEET DASHBOARD
# ════════════════════════════════════════════════════════════
with tab_fleet:
    st.header("🏢 Fleet Overview")
    st.markdown("Monitor all servers, see which are at risk, and understand why.")

    data_source = st.radio(
        "Data source",
        ["Use sample fleet (demo)", "Upload CSV"],
        horizontal=True
    )

    if data_source == "Upload CSV":
        st.info("""
        Upload a CSV with these columns:
        `server_id, cpu, memory, disk, network, processes, threads,
        context_sw, cache_miss, temperature, power, uptime`
        """)
        uploaded = st.file_uploader("Upload server metrics CSV", type=["csv"])
        if uploaded:
            fleet_df = pd.read_csv(uploaded)
        else:
            fleet_df = None
    else:
        fleet_df = generate_sample_fleet(20)

    if fleet_df is not None:
        with st.spinner("Running predictions on all servers…"):
            results_df = predict_batch(fleet_df)
            combined = pd.concat([fleet_df.reset_index(drop=True),
                                   results_df.reset_index(drop=True)], axis=1)

        # ── Summary Cards ────────────────────────────────────
        total  = len(combined)
        high   = (results_df["_prob"] >= 0.60).sum()
        medium = ((results_df["_prob"] >= 0.30) & (results_df["_prob"] < 0.60)).sum()
        low    = (results_df["_prob"] < 0.30).sum()
        failed = (results_df["_pred"] == 1).sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Servers", total)
        c2.metric("🟢 Healthy",    int(low))
        c3.metric("🟡 Medium Risk",int(medium))
        c4.metric("🔴 High Risk",  int(high))
        c5.metric("⚠️ Predicted Failed", int(failed))

        st.markdown("---")

        # ── Filter ───────────────────────────────────────────
        filter_opt = st.selectbox(
            "Filter by risk level",
            ["All servers", "🔴 High Risk only", "🟡 Medium Risk only", "🟢 Healthy only",
             "⚠️ Predicted Failed only"]
        )

        display = combined.copy()
        if filter_opt == "🔴 High Risk only":
            display = combined[combined["_prob"] >= 0.60]
        elif filter_opt == "🟡 Medium Risk only":
            display = combined[(combined["_prob"] >= 0.30) & (combined["_prob"] < 0.60)]
        elif filter_opt == "🟢 Healthy only":
            display = combined[combined["_prob"] < 0.30]
        elif filter_opt == "⚠️ Predicted Failed only":
            display = combined[combined["_pred"] == 1]

        st.markdown(f"**Showing {len(display)} server(s)**")

        # ── Main Table ───────────────────────────────────────
        show_cols = ["server_id", "cpu", "memory", "temperature", "network",
                     "disk", "power", "Prediction", "Failure Prob",
                     "Risk Level", "Failure Reason", "_prob", "_pred"]
        display_show = display[[c for c in show_cols if c in display.columns]]

        styled = (
            display_show.style
            .apply(color_risk_row, axis=1)
            .format({"cpu": "{:.1f}%", "memory": "{:.1f}%",
                     "temperature": "{:.1f}°C", "network": "{:.0f}ms",
                     "disk": "{:.1f} MB/s", "power": "{:.0f}W",
                     "_prob": "{:.1%}"})
            .hide(axis="index")
        )
        st.dataframe(styled, use_container_width=True,
                     column_config={
                         "_prob": st.column_config.ProgressColumn(
                             "Prob Bar", min_value=0, max_value=1, format="%.0f%%"),
                         "_pred": None,
                     })

        st.markdown("---")

        # ── Failed Servers Detail ────────────────────────────
        failed_servers = combined[combined["_pred"] == 1]
        if len(failed_servers) > 0:
            st.subheader(f"🚨 Failed Servers — Detail ({len(failed_servers)} server(s))")
            for _, srv in failed_servers.iterrows():
                with st.expander(
                    f"⚠️ {srv['server_id']}  |  Prob: {float(srv['_prob'])*100:.1f}%  "
                    f"|  {srv['Risk Level']}"
                ):
                    d1, d2, d3, d4 = st.columns(4)
                    d1.metric("CPU",         f"{srv['cpu']}%",
                              delta=f"{srv['cpu']-70:.0f}%" if srv['cpu'] > 70 else None,
                              delta_color="inverse")
                    d2.metric("Memory",      f"{srv['memory']}%",
                              delta=f"{srv['memory']-75:.0f}%" if srv['memory'] > 75 else None,
                              delta_color="inverse")
                    d3.metric("Temperature", f"{srv['temperature']}°C",
                              delta=f"{srv['temperature']-75:.0f}°C" if srv['temperature'] > 75 else None,
                              delta_color="inverse")
                    d4.metric("Network Latency", f"{srv['network']}ms",
                              delta=f"{srv['network']-100:.0f}ms" if srv['network'] > 100 else None,
                              delta_color="inverse")

                    st.error(f"**Why this server is failing:** {srv['Failure Reason']}")

                    st.markdown("**Recommended Actions:**")
                    reasons_text = str(srv["Failure Reason"])
                    if "CPU" in reasons_text:
                        st.markdown("- 🔧 Migrate or kill high-CPU processes immediately")
                    if "Memory" in reasons_text:
                        st.markdown("- 🔧 Clear memory cache or add RAM — restart memory-heavy services")
                    if "Overheating" in reasons_text or "Temp" in reasons_text:
                        st.markdown("- 🔧 Check cooling system / increase fan speed")
                    if "Network" in reasons_text:
                        st.markdown("- 🔧 Inspect network interface and switch ports")
                    if "Power" in reasons_text:
                        st.markdown("- 🔧 Check PSU health and reduce load")
                    st.markdown("- 📢 Alert on-call operations team")
        else:
            st.success("✅ No servers predicted to fail — fleet is healthy!")

        # ── Download ─────────────────────────────────────────
        st.markdown("---")
        csv_out = combined.drop(columns=["_prob", "_pred"]).to_csv(index=False)
        st.download_button(
            "⬇️ Download Report as CSV",
            data=csv_out,
            file_name="server_failure_report.csv",
            mime="text/csv"
        )

# ════════════════════════════════════════════════════════════
# TAB 2 — SINGLE SERVER PREDICTION (original)
# ════════════════════════════════════════════════════════════
with tab_single:
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
            st.metric("Prediction", "⚠️ SERVER FAILURE" if pred == 1 else "✅ NORMAL")
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

        reason = get_failure_reasons(cpu, memory, disk, network,
                                     processes, threads, context_sw,
                                     cache_miss, temperature, power, uptime)
        if alert_type == "error":
            st.error(f"""
🚨 HIGH RISK — IMMEDIATE ACTION REQUIRED!
Failure probability: {prob*100:.1f}%
**Why:** {reason}

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
**Why:** {reason}

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

# ════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON (original)
# ════════════════════════════════════════════════════════════
with tab_models:
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
