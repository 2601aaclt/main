import streamlit as st
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
import folium
from streamlit_folium import st_folium

# =============================
# CONFIG & LAYOUTING
# =============================
st.set_page_config(layout="wide", page_title="AI Traffic Predictor")
st.title("🚗 AI Traffic Travel Time Predictor")

# --- SIDEBAR NAVIGASI UTAMA ---
st.sidebar.title("📌 Main Menu")
menu = st.sidebar.radio("Select View:", ["🗺️ Simulate & Predict", "📊 Data History"])

# =============================
# GLOBAL VARIABLES
# =============================
gate_order = ['Gerbang Tol A','Gerbang Tol B','Gerbang Tol C','Gerbang Tol D','Gerbang Tol E','Gerbang Tol F']
gate_map = {'Gerbang Tol A': 'A', 'Gerbang Tol B': 'B', 'Gerbang Tol C': 'C', 'Gerbang Tol D': 'D', 'Gerbang Tol E': 'E', 'Gerbang Tol F': 'F'}

gate_positions = {
    'Gerbang Tol A': 0, 'Gerbang Tol B': 15, 'Gerbang Tol C': 25,
    'Gerbang Tol D': 35, 'Gerbang Tol E': 45, 'Gerbang Tol F': 55
}

gate_coordinates = {
    "Gerbang Tol A": (3.5952, 98.6722), "Gerbang Tol B": (-6.2088, 106.8456), "Gerbang Tol C": (-7.2575, 112.7521),
    "Gerbang Tol D": (-0.0263, 109.3425), "Gerbang Tol E": (-3.6547, 128.1903), "Gerbang Tol F": (-2.5337, 140.7181),
}

window_size = 6
total_minutes_forecast = 180

# =============================
# LOAD DATA LOKAL (OTOMATIS)
# =============================
@st.cache_data
def load_local_data():
    try:
        data = pd.read_excel("data tol.xlsx") 
        data.columns = data.columns.astype(str)
        df_res = pd.DataFrame()
        
        for col in gate_order:
            if col in data.columns:
                waktu = pd.to_datetime(data[col], errors='coerce').dropna()
                per_menit = waktu.dt.floor('1min').value_counts().sort_index()
                df_res[col] = per_menit
        
        full_time = pd.date_range(start=df_res.index.min(), end=df_res.index.max(), freq='1min')
        return df_res.reindex(full_time).fillna(0).astype(int)
    except Exception as e:
        st.error(f"Gagal membaca file 'data tol.xlsx'. Pastikan nama kolom di Excel benar. Error: {e}")
        st.stop()

volume_data = load_local_data()

# =============================
# LOAD MODEL + SCALER
# =============================
@st.cache_resource
def load_all():
    models, scalers = {}, {}
    for ui_name in gate_order:
        short_name = gate_map[ui_name]
        models[ui_name] = load_model(f"lstm_gate_{short_name}.keras")
        scalers[ui_name] = joblib.load(f"scaler_gate_{short_name}.save")
    return models, scalers

models, scalers = load_all()

# =============================
# LOGIC FUNCTIONS
# =============================
def traffic_label(x):
    if x < 2.5: return "LOW"
    elif x < 4.5: return "MEDIUM"
    else: return "HIGH"

def traffic_to_speed(volume):
    if volume < 10: return 100
    elif volume < 20: return 80
    elif volume < 30: return 60
    else: return 40

def get_route(start_gate, end_gate):
    s_idx, e_idx = gate_order.index(start_gate), gate_order.index(end_gate)
    return gate_order[s_idx:e_idx+1] if s_idx <= e_idx else gate_order[e_idx:s_idx+1][::-1]

# =============================
# SIMULASI FORECAST
# =============================
@st.cache_data
def simulate_all_gates():
    predictions = {gate: [] for gate in gate_order}
    current_sequences = {}

    for ui_name in gate_order:
        scaler = scalers[ui_name]
        last_vals = volume_data[ui_name].values[-window_size:]
        seq = []
        for i, val in enumerate(last_vals):
            seq.append([val, 0, i]) 
        current_sequences[ui_name] = scaler.transform(np.array(seq))

    for m_idx in range(total_minutes_forecast):
        hr = m_idx // 60
        mn = m_idx % 60
        for ui_name in gate_order:
            model, scaler = models[ui_name], scalers[ui_name]
            inp = current_sequences[ui_name].reshape(1, window_size, 3)
            p_scaled = model.predict(inp, verbose=0)
            dummy = np.zeros((1, 3))
            dummy[0, 0] = p_scaled[0, 0]
            real_val = max(0, scaler.inverse_transform(dummy)[0, 0])
            predictions[ui_name].append(real_val)
            new_row_scaled = scaler.transform(np.array([[real_val, hr, mn]]))
            current_sequences[ui_name] = np.vstack([current_sequences[ui_name][1:], new_row_scaled])

    return np.array([[predictions[g][i] for g in gate_order] for i in range(total_minutes_forecast)])

# =========================================================
# LOGIKA TAMPILAN BERDASARKAN MENU SIDEBAR
# =========================================================

if menu == "🗺️ Simulate & Predict":

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Simulation Setting")

    start_node = st.sidebar.selectbox("From", gate_order)
    end_node = st.sidebar.selectbox("To", gate_order)
    dep_time = st.sidebar.time_input("Departure Time")

    if "result" not in st.session_state:
        st.session_state.result = None

    if st.sidebar.button("Calculate 🚀"):

        with st.spinner("AI Process Route..."):

            preds = simulate_all_gates()
            route = get_route(start_node, end_node)

            total_time = 0
            dep_minute = (dep_time.hour * 60 + dep_time.minute) % total_minutes_forecast
            curr_min = dep_minute

            detail_text = "=== TRAFFIC DETAIL ===\n"

            for i in range(len(route)-1):

                g1, g2 = route[i], route[i+1]

                idx1 = gate_order.index(g1)
                idx2 = gate_order.index(g2)

                traffic_seg = (preds[curr_min % 180][idx1] + preds[curr_min % 180][idx2]) / 2

                speed = traffic_to_speed(traffic_seg)

                dist = abs(gate_positions[g2] - gate_positions[g1])

                time_seg = (dist / speed) * 60

                detail_text += f"\n{g1} → {g2}"
                detail_text += f"\nMinute : {curr_min}"
                detail_text += f"\nTraffic : {round(traffic_seg,2)} ({traffic_label(traffic_seg)})"
                detail_text += f"\nSpeed : {speed} km/h"
                detail_text += f"\nTime : {round(time_seg,2)} minutes\n"

                total_time += time_seg
                curr_min += int(time_seg)

            detail_text += f"\nTotal Travel Time: {round(total_time,2)} minutes"

            st.session_state.result = {
                "total": round(total_time,2),
                "detail": detail_text,
                "start": start_node,
                "end": end_node
            }

    # ==============================
    # MAP SECTION (HARUS DI DALAM MENU)
    # ==============================

    st.subheader("🗺️ Toll Map")

    m = folium.Map(location=[-2.5,118], zoom_start=5)

    if st.session_state.get("result"):

        route = get_route(
            st.session_state.result["start"],
            st.session_state.result["end"]
        )

        route_coords = [gate_coordinates[g] for g in route]

        folium.PolyLine(
            route_coords,
            color="blue",
            weight=5,
            opacity=0.8
        ).add_to(m)

    for gate, coord in gate_coordinates.items():

        color = "blue"

        if st.session_state.get("result"):

            if gate == st.session_state.result["start"]:
                color = "green"

            elif gate == st.session_state.result["end"]:
                color = "red"

        folium.Marker(
            location=coord,
            popup=gate,
            icon=folium.Icon(color=color)
        ).add_to(m)

    st_folium(m, width=1200, height=450)

    if st.session_state.get("result"):

        st.code(st.session_state.result["detail"])

        st.success(
            f"Estimated Travel Time: {st.session_state.result['total']} Minutes"
        )

elif menu == "📊 Data History":

    st.header("📊 Historical Data & AI Forecast")

    tab_past, tab_future = st.tabs(
    ["🕒 Historical Data", "🔮 AI Forecast"]
    )

    # =================================================
    # TAB 1 : DATA EXCEL
    # =================================================
    with tab_past:

        st.subheader("Vehicle Transactions Per Minute")

        def total_traffic_label(x):
            if x <= 15:
                return "LOW"
            elif x <= 29:
                return "MEDIUM"
            else:
                return "HIGH"

        df_display = volume_data.copy()

        df_display["Total Volume"] = df_display[gate_order].sum(axis=1)

        df_display["Overall Status"] = df_display["Total Volume"].apply(
            total_traffic_label
        )

        st.dataframe(df_display, use_container_width=True, height=400)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Menit Tercatat", f"{len(df_display)} Menit")

        col2.metric(
            "Rata-rata Volume", f"{round(df_display['Total Volume'].mean(),2)}"
        )

        col3.metric("Puncak Volume", f"{df_display['Total Volume'].max()}")

    # =================================================
    # TAB 2 : AI FORECAST
    # =================================================
    with tab_future:

        st.subheader("Prediksi AI untuk 180 Menit ke Depan")

        if "forecast_data" not in st.session_state:
            st.session_state.forecast_data = None

        if "rounded_data" not in st.session_state:
            st.session_state.rounded_data = None

        # =========================
        # Tombol Forecast
        # =========================
        if st.button("🚀 Jalankan AI Forecast"):

            with st.spinner("AI is calculating future traffic trends..."):

                preds_array = simulate_all_gates()

                df_forecast = pd.DataFrame(
                    preds_array,
                    columns=gate_order
                )

                df_forecast.index.name = "Menit ke-Depan"

                df_forecast["Total Volume"] = df_forecast.sum(axis=1)

                df_forecast["Overall Status"] = df_forecast[
                    "Total Volume"
                ].apply(total_traffic_label)

                st.session_state.forecast_data = df_forecast
                st.session_state.rounded_data = None

        # =========================
        # Jika forecast sudah ada
        # =========================
        if st.session_state.forecast_data is not None:

            df_forecast = st.session_state.forecast_data

            st.line_chart(df_forecast[gate_order])

            st.write("### Tabel Prediksi Per Menit")

            st.dataframe(df_forecast, use_container_width=True, height=400)

            # =========================
            # Tombol Pembulatan
            # =========================
            if st.button("🔢 Bulatkan Nilai Prediksi"):

                df_rounded = df_forecast.copy()

                df_rounded[gate_order] = (
                    df_rounded[gate_order]
                    .round()
                    .astype(int)
                )

                df_rounded["Total Volume"] = df_rounded[
                    gate_order
                ].sum(axis=1)

                st.session_state.rounded_data = df_rounded

            if st.session_state.rounded_data is not None:

                st.write("### Tabel Prediksi (Sudah Dibulatkan)")

                st.dataframe(
                    st.session_state.rounded_data,
                    use_container_width=True,
                    height=400,
                )

        st.info(
            "💡 Grafik menunjukkan prediksi volume kendaraan "
            "di setiap gerbang selama 3 jam ke depan berdasarkan "
            "pola data historis."
        )