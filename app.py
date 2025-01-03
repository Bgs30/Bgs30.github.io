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
            "SVM": "Dataset/fish_SVM.pkl"
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
            "SVM": "Dataset/fruit_svm.pkl"
        }
        if st.button("Prediksi Buah"):
            model = load_model(model_path.get(model_choice))
            input_data = np.array([[diameter, weight, red, green, blue]])
            hasil_prediksi = predict_with_model(model, input_data, label_mapping_fruit)
            st.write(f"Prediksi: **{hasil_prediksi}**")


elif learning_type == "Unsupervised Learning":
    st.title("Prediksi dengan K-Means")
    
    # Dataset atau data untuk clustering
    # Pastikan `data_to_cluster` adalah DataFrame atau array numpy yang valid
    data_to_cluster = np.random.rand(100, 3)  # Contoh data dummy untuk clustering

    # Hitung nilai K optimal menggunakan metode elbow
    if st.button("Hitung Nilai K Optimal"):
        def calculate_inertia(data):
            from sklearn.cluster import KMeans
            inertias = []
            k_range = range(1, 11)  # Nilai k dari 1 hingga 10
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            return k_range, inertias

        k_range, inertias = calculate_inertia(data_to_cluster)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(k_range, inertias, marker='o')
        plt.xlabel("Jumlah Cluster (k)")
        plt.ylabel("Inertia")
        plt.title("Metode Elbow untuk Menentukan k")
        st.pyplot(plt)
        plt.close()  # Tutup plot setelah ditampilkan

    # Slider untuk menentukan jumlah cluster
    n_clusters = st.slider("Jumlah cluster:", min_value=2, max_value=10, value=3)

    # Melakukan clustering dengan K-Means
    def kmeans_clustering(data, n_clusters):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data)
        return clusters, kmeans

    clusters, kmeans_model = kmeans_clustering(data_to_cluster, n_clusters)

    st.subheader(f"Hasil Clustering dengan {n_clusters} Cluster")
    st.write("Cluster Labels:", clusters)
    st.write("Centroid Cluster:")
    st.write(kmeans_model.cluster_centers_)
else:
    st.write("Pilih tipe pembelajaran untuk melanjutkan.")

