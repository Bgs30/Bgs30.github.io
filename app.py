import streamlit as st
import numpy as np
import joblib

# Fungsi untuk memuat model
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file {model_path} tidak ditemukan.")
        st.stop()

# Fungsi prediksi
def predict_with_model(model, input_data, label_mapping):
    try:
        prediction = model.predict(input_data)
        return label_mapping[prediction[0]]
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {str(e)}")

# Label mapping
label_mapping_fish = {
    0: "Anabas testudineus",
    1: "Coilia dussumieri",
    2: "Otolithoides biauritus",
    3: "Otolithoides pama",
    4: "Pethia conconius",
    5: "Polynemus paradiseus",
    6: "Puntius lateristriga",
    7: "Setipinna taty",
    8: "Sillaginopsis panijus"
}
label_mapping_fruit = {
    0: "Orange",
    1: "Grapefruit"
}

# Sidebar untuk jenis pembelajaran
learning_type = st.sidebar.selectbox(
    "Pilih Tipe Pembelajaran",
    ["Supervised Learning", "Unsupervised Learning"]
)

if learning_type == "Supervised Learning":
    # Pilih model
    model_choice = st.sidebar.selectbox(
        "Pilih Model",
        ["Logistic Regression", "Perceptron", "SVM"]
    )

    # Pilih jenis prediksi
    prediksi_choice = st.sidebar.selectbox(
        "Pilih Prediksi",
        ["Ikan", "Buah"]
    )

    if prediksi_choice == "Ikan":
        st.title(f"Prediksi Jenis Ikan dengan {model_choice}")

        # Input data untuk ikan
        panjang = st.number_input("Panjang (cm)", min_value=1.0, step=0.1)
        lebar = st.number_input("Lebar (cm)", min_value=0.0, step=0.1)
        wlRatio = st.number_input("Rata Rata (cm)", min_value=0.0, step=0.1)

        # Path model
        model_path = {
            "Logistic Regression": "Dataset/fish_Lg.pkl",
            "Perceptron": "Dataset/fish_Pc.pkl",
            "SVM": "Dataset/fish_Svm.pkl"
        }

        if st.button("Prediksi Jenis Ikan"):
            model = load_model(model_path.get(model_choice))
            input_data = np.array([[panjang, lebar, wlRatio]])
            hasil_prediksi = predict_with_model(model, input_data, label_mapping_fish)
            st.write(f"Prediksi Spesies Ikan: **{hasil_prediksi}**")

    elif prediksi_choice == "Buah":
        st.title(f"Prediksi Jenis Buah dengan {model_choice}")

        # Input data untuk buah
        diameter = st.number_input("Diameter", min_value=0.0, step=0.1)
        weight = st.number_input("Weight", min_value=0.0, step=0.1)
        red = st.number_input("Red (RGB value)", min_value=0.0, step=1.0)
        green = st.number_input("Green (RGB value)", min_value=0.0, step=1.0)
        blue = st.number_input("Blue (RGB value)", min_value=0.0, step=1.0)

        # Path model
        model_path = {
            "Logistic Regression": "Dataset/fruit_Lr.pkl",
            "Perceptron": "Dataset/fruit_Pc.pkl",
            "SVM": "Dataset/fruit_Svm.pkl"
        }
        if st.button("Prediksi Buah"):
            model = load_model(model_path.get(model_choice))
            input_data = np.array([[diameter, weight, red, green, blue]])
            hasil_prediksi = predict_with_model(model, input_data, label_mapping_fruit)
            st.write(f"Prediksi: **{hasil_prediksi}**")

elif learning_type == "Unsupervised Learning":
    st.title("Prediksi dengan K-Means")

    # Input data untuk K-Means

else:
    st.write("Pilih tipe pembelajaran untuk melanjutkan.")
