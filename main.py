import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt

# Configuracion de la app
st.set_page_config(page_title="K-Means con PCA y Comparativa", layout="wide")
st.title("Clustering Interactivo con K-Means y PCA")
st.write(
    """
Sube tus datos, aplica **K-Means**, y observa cómo el algoritmo agrupa los puntos en un espacio reducido con **PCA (2D o 3D)**.
**Utiliza la barra lateral para ajustar todos los parámetros del modelo.**
"""
)

# --- Subir archivo ---
st.sidebar.header("Subir datos")
uploaded_file = st.sidebar.file_uploader("Selecciona tu archivo CSV", type=["csv"])

# Inicializar variables para usar en la función de descarga si no hay archivo cargado
data = None

if uploaded_file is not None:
    # 1. Cargar y preprocesar
    data = pd.read_csv(uploaded_file)
    st.success("Archivo cargado correctamente.")
    st.write("### Vista previa de los datos:")
    st.dataframe(data.head())

    # Filtrar columnas numéricas
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("El archivo debe contener al menos dos columnas numéricas.")
    else:
        st.sidebar.header("Configuración del modelo")

        # Seleccionar columnas a usar
        selected_cols = st.sidebar.multiselect(
            "1. Selecciona las columnas numéricas para el clustering:",
            numeric_cols,
            default=numeric_cols,
        )

        # Parámetros básicos
        k = st.sidebar.slider("2. Número de clusters (k):", 1, 10, 3)
        n_components = st.sidebar.radio("3. Visualización PCA:", [2, 3], index=0)

        # --- Parámetros Avanzados de K-Means ---
        st.sidebar.subheader("Parámetros K-Means")
        
        kmeans_init = st.sidebar.selectbox(
            "4. Método de Inicialización (init):",
            ("k-means", "random"),
            index=0,
            help="k-means: Selecciona centros iniciales de forma inteligente. " \
            "Random: selecciona centros al azar."
        )

        kmeans_max_iter = st.sidebar.slider(
            "5. Máximo de Iteraciones (max_iter):",
            50, 1000, 300, 50,
            help="Número máximo de veces que el algoritmo reajustará los centroides."
        )

        kmeans_n_init = st.sidebar.slider(
            "6. Número de ejecuciones con distintas semillas (n_init):",
            1, 50, 10,
            help="Número de veces que el algoritmo K-Means se ejecutará con diferentes inicializaciones de centroides. El mejor resultado es el final."
        )

        # Control para Random State
        use_random_state = st.sidebar.checkbox(
            "7. Usar Semilla Fija (Random State)", 
            value=True,
            help="Fija el valor para asegurar la reproducibilidad de los resultados."
        )
        random_state_val = None
        if use_random_state:
            random_state_val = st.sidebar.number_input(
                "Valor de la Semilla (Random State):",
                min_value=0,
                value=42,
                step=1
            )
        # --------------------------------------------------

        # --- Datos y modelo ---
        X = data[selected_cols]
        # Asignamos los parámetros seleccionados
        kmeans = KMeans(
            n_clusters=k,
            init=kmeans_init,
            max_iter=kmeans_max_iter,
            n_init=kmeans_n_init,
            random_state=random_state_val,
        )
        
        # Manejo de excepción por si el número de clusters es mayor que el número de muestras
        try:
            kmeans.fit(X)
            data["Cluster"] = kmeans.labels_
        except ValueError as e:
            st.error(f"Error al ejecutar K-Means: {e}. Asegúrate de que el número de clusters (k={k}) no exceda el número de filas en tus datos.")
            st.stop() # Usar st.stop() para detener la ejecución en Streamlit
            
        # --- PCA ---
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        pca_cols = [f"PCA{i+1}" for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols, index=data.index)
        pca_df["Cluster"] = data["Cluster"]

        # --- Visualización antes del clustering ---
        st.subheader("Distribución original (antes de K-Means)")
        if n_components == 2:
            fig_before = px.scatter(
                pca_df,
                x="PCA1",
                y="PCA2",
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"],
            )
        else:
            fig_before = px.scatter_3d(
                pca_df,
                x="PCA1",
                y="PCA2",
                z="PCA3",
                title="Datos originales proyectados con PCA (sin agrupar)",
                color_discrete_sequence=["gray"],
            )
        st.plotly_chart(fig_before, use_container_width=True)

        # --- Visualización después del clustering ---
        st.subheader(f"Datos agrupados con K-Means (k = {k})")
        if n_components == 2:
            fig_after = px.scatter(
                pca_df,
                x="PCA1",
                y="PCA2",
                color=pca_df["Cluster"].astype(str),
                title=f"Clusters visualizados en 2D (PCA). Varianza Explicada Total: {pca.explained_variance_ratio_.sum():.2f}",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
        else:
            fig_after = px.scatter_3d(
                pca_df,
                x="PCA1",
                y="PCA2",
                z="PCA3",
                color=pca_df["Cluster"].astype(str),
                title=f"Clusters visualizados en 3D (PCA). Varianza Explicada Total: {pca.explained_variance_ratio_.sum():.2f}",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
        st.plotly_chart(fig_after, use_container_width=True)

        # --- Centroides ---
        st.subheader("📌 Centroides de los clusters (en espacio PCA)")
        centroides_pca = pd.DataFrame(
            pca.transform(kmeans.cluster_centers_), columns=pca_cols
        )
        centroides_pca.index.name = "Cluster ID"
        st.dataframe(centroides_pca)

        # --- Método del Codo ---
        st.subheader("Método del Codo (Elbow Method)")
        st.write("Herramienta para estimar el valor óptimo de k (clusters).")
        if st.button("Calcular Inercia para K=1 a K=10"):
            inertias = []
            K_range = range(1, min(11, len(X) + 1))
            
            # Reutilizamos los nuevos parámetros de K-Means para el cálculo del Codo
            for i in K_range:
                # El n_clusters es la variable de la iteración (i)
                km = KMeans(
                    n_clusters=i,
                    init=kmeans_init,
                    max_iter=kmeans_max_iter,
                    n_init=kmeans_n_init,
                    random_state=random_state_val,
                    # Suprimir el warning de n_init para versiones más nuevas de sklearn
                    # Aunque estamos usando la versión 1.7.2, es una buena práctica:
                    # n_init='auto' if sklearn.__version__ >= '1.4' else kmeans_n_init
                )
                km.fit(X)
                inertias.append(km.inertia_)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            plt.plot(K_range, inertias, "bo-")
            plt.title("Método del Codo (Inercia vs. Número de Clusters)")
            plt.xlabel("Número de Clusters (k)")
            plt.ylabel("Inercia (SSE)")
            plt.grid(True)
            st.pyplot(fig2)

        # --- Descarga de resultados ---
        st.subheader("⬇Descargar datos con clusters asignados")
        buffer = BytesIO()
        # Aseguramos que la columna 'Cluster' se guarde
        data.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="Descargar CSV con Clusters",
            data=buffer,
            file_name="datos_clusterizados.csv",
            mime="text/csv",
        )

else:
    st.info("Carga un archivo CSV en la barra lateral para comenzar.")
    st.write(
        """
    **Ejemplo de formato:**
    | Ingreso_Anual | Gasto_Tienda | Edad |
    |---------------|--------------|------|
    | 45000         | 350          | 28   |
    | 72000         | 680          | 35   |
    | 28000         | 210          | 22   |
    """
    )