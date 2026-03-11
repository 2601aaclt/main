import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import folium
from streamlit_folium import st_folium
# =============================
# CONFIG
# =============================
st.set_page_config(layout="wide")
st.title("🚗 AI Traffic Travel Time Predictor")

gate_order = ['A','B','C','D','E','F']

gate_positions = {
    'A': 0, 'B': 15, 'C': 25,
    'D': 35, 'E': 45, 'F': 55
}

gate_coordinates = {
    "A": (-6.200, 106.816),
    "B": (-6.210, 106.820),
    "C": (-6.220, 106.830),
    "D": (-6.230, 106.840),
    "E": (-6.240, 106.850),
    "F": (-6.250, 106.860),
}

window_size = 6
total_minutes = 180

# =============================
# LOAD MODEL + SCALER + LAST SEQ
# =============================
@st.cache_resource
def load_all():
    models = {}
    scalers = {}
    last_sequences = {}

    for gate in gate_order:
        models[gate] = load_model(f"lstm_gate_{gate}.keras")
        scalers[gate] = joblib.load(f"scaler_gate_{gate}.save")
        last_sequences[gate] = joblib.load(f"last_seq_{gate}.save")

    return models, scalers, last_sequences

models, scalers, last_sequences = load_all()

# =============================
# TRAFFIC LOGIC (SAMA DENGAN COLLAB)
# =============================
def traffic_label(x):
    if x < 2.5:
        return "LOW"
    elif x < 4.5:
        return "MEDIUM"
    else:
        return "HIGH"

def traffic_to_speed(volume):
    if volume < 10:
        return 100
    elif volume < 20:
        return 80
    elif volume < 30:
        return 60
    else:
        return 40

def get_route(start_gate, end_gate):
    start_idx = gate_order.index(start_gate)
    end_idx = gate_order.index(end_gate)

    if start_idx <= end_idx:
        return gate_order[start_idx:end_idx+1]
    else:
        return gate_order[end_idx:start_idx+1][::-1]

# =============================
# SIMULASI FORECAST 180 MENIT
# =============================
@st.cache_data
def simulate_all_gates():

    predictions = {gate: [] for gate in gate_order}

    # pakai last sequence asli dari training
    current_sequences = {
        gate: last_sequences[gate].copy()
        for gate in gate_order
    }

    for minute in range(total_minutes):

        hour = minute // 60
        minute_only = minute % 60

        for gate in gate_order:

            model = models[gate]
            scaler = scalers[gate]

            seq_scaled = current_sequences[gate]

            input_data = seq_scaled.reshape(1, window_size, 3)

            pred_scaled = model.predict(input_data, verbose=0)

            # inverse scaling
            dummy = np.zeros((1,3))
            dummy[0,0] = pred_scaled[0,0]
            real_value = scaler.inverse_transform(dummy)[0,0]

            predictions[gate].append(real_value)

            # buat row baru (volume_pred, hour, minute)
            new_row = np.array([[real_value, hour, minute_only]])
            new_row_scaled = scaler.transform(new_row)

            # rolling window (autoregressive BENAR)
            seq_scaled = np.vstack([seq_scaled[1:], new_row_scaled])
            current_sequences[gate] = seq_scaled

    final_preds = []

    for i in range(total_minutes):
        row = [predictions[g][i] for g in gate_order]
        final_preds.append(row)

    return np.array(final_preds)

# =============================
# PREDIKSI PERJALANAN
# =============================
def predict_travel_time_ai(start_gate, end_gate, departure_minute):

    preds = simulate_all_gates()
    st.write(preds[:5])
    route = get_route(start_gate, end_gate)

    total_time = 0
    current_minute = departure_minute
    max_minutes = len(preds)

    output_text = "=== DETAIL PERJALANAN ===\n"

    for i in range(len(route)-1):

        g1 = route[i]
        g2 = route[i+1]

        idx1 = gate_order.index(g1)
        idx2 = gate_order.index(g2)

        traffic_segment = (
            preds[current_minute % max_minutes][idx1] +
            preds[current_minute % max_minutes][idx2]
        ) / 2

        speed = traffic_to_speed(traffic_segment)
        distance = abs(gate_positions[g2] - gate_positions[g1])
        time_minutes = (distance / speed) * 60

        label = traffic_label(traffic_segment)

        output_text += f"""
{g1} → {g2}
   Menit ke-{current_minute}
   Traffic: {round(traffic_segment,2)} ({label})
   Kecepatan: {speed} km/jam
   Waktu segmen: {round(time_minutes,2)} menit
"""

        total_time += time_minutes
        current_minute += int(time_minutes)

    output_text += f"\nTotal waktu tempuh: {round(total_time,2)} menit"

    return round(total_time,2), output_text

# =============================
# SIDEBAR
# =============================
st.sidebar.header("⚙️ Pengaturan Simulasi")

start = st.sidebar.selectbox("Gerbang Asal", gate_order)
end = st.sidebar.selectbox("Gerbang Tujuan", gate_order)
departure_time = st.sidebar.time_input("Jam Berangkat")

total_minutes_input = departure_time.hour * 60 + departure_time.minute

if "result" not in st.session_state:
    st.session_state.result = None

if st.sidebar.button("Hitung Estimasi 🚀"):

    if start == end:
        st.sidebar.warning("Gerbang asal dan tujuan tidak boleh sama.")
    else:
        travel_time, detail_text = predict_travel_time_ai(
            start,
            end,
            total_minutes_input % 180
        )

        st.session_state.result = {
            "start": start,
            "end": end,
            "departure_time": departure_time,
            "travel_time": travel_time,
            "detail": detail_text
        }

# =============================
# PETA
# =============================
st.subheader("🗺️ Peta Gerbang Tol")

m = folium.Map(location=[-6.225, 106.835], zoom_start=12)

for gate, coord in gate_coordinates.items():

    color = "blue"

    if st.session_state.result is not None:
        if gate == st.session_state.result["start"]:
            color = "green"
        elif gate == st.session_state.result["end"]:
            color = "red"

    folium.Marker(
        location=coord,
        popup=f"Gerbang {gate}",
        tooltip=f"Gerbang {gate}",
        icon=folium.Icon(color=color)
    ).add_to(m)

if st.session_state.result is not None:

    route = get_route(
        st.session_state.result["start"],
        st.session_state.result["end"]
    )

    route_coords = [gate_coordinates[g] for g in route]
    folium.PolyLine(route_coords, color="orange", weight=6).add_to(m)

st_folium(m, width=1200, height=500)

# =============================
# HASIL
# =============================
if st.session_state.result is not None:

    st.markdown("---")
    st.markdown("## 📊 Hasil Simulasi")

    result = st.session_state.result

    st.write(f"Gerbang asal: {result['start']}")
    st.write(f"Gerbang tujuan: {result['end']}")
    st.write(f"Jam berangkat: {result['departure_time'].strftime('%H:%M')}")

    st.code(result["detail"], language="text")
    st.success(f"Estimasi waktu tempuh: {result['travel_time']} menit")
