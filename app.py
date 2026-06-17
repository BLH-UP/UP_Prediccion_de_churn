# ==============================================================================
# APP STREAMLIT - PREDICCIÓN DE CHURN EN TELECOMUNICACIONES
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================

st.set_page_config(
    page_title="Predicción de Churn",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Predicción de Churn en Telecomunicaciones")

st.markdown("""
Esta aplicación utiliza un modelo de **Random Forest** para estimar la probabilidad
de que un cliente abandone el servicio.

El usuario puede cargar una base de clientes en formato CSV y la aplicación devuelve:

- Probabilidad de churn por cliente.
- Predicción de churn.
- Segmento de riesgo.
- Acción sugerida de retención.
- Archivo descargable con resultados.
""")

# ==============================================================================
# 2. CARGA DEL MODELO Y OBJETOS NECESARIOS
# ==============================================================================

@st.cache_resource
def cargar_objetos():
    modelo = joblib.load("modelo_churn_random_forest.pkl")
    columnas_modelo = joblib.load("columnas_modelo_churn.pkl")
    scaler = joblib.load("scaler_churn.pkl")
    return modelo, columnas_modelo, scaler


try:
    modelo, columnas_modelo, scaler = cargar_objetos()
    st.success("Modelo y objetos de preprocesamiento cargados correctamente.")
except FileNotFoundError as e:
    st.error(
        "No se encontraron todos los archivos necesarios. "
        "Asegúrate de subir al repositorio los siguientes archivos: "
        "`modelo_churn_random_forest.pkl`, `columnas_modelo_churn.pkl` y `scaler_churn.pkl`."
    )
    st.code(str(e))
    st.stop()

# ==============================================================================
# 3. FUNCIONES AUXILIARES
# ==============================================================================

def clasificar_riesgo(probabilidad):
    """
    Clasifica a cada cliente según su probabilidad estimada de churn.
    """
    if probabilidad < 0.30:
        return "Bajo"
    elif probabilidad < 0.60:
        return "Medio"
    elif probabilidad < 0.80:
        return "Alto"
    else:
        return "Crítico"


def asignar_accion(segmento):
    """
    Asigna una acción comercial sugerida según el nivel de riesgo.
    """
    acciones = {
        "Bajo": "Comunicación general y monitoreo regular.",
        "Medio": "Seguimiento preventivo y encuesta de satisfacción.",
        "Alto": "Oferta personalizada, revisión de plan y beneficios adicionales.",
        "Crítico": "Contacto prioritario del equipo de retención y propuesta comercial directa."
    }
    return acciones.get(segmento, "Sin acción sugerida.")


def preprocesar_datos(df_original, columnas_modelo, scaler):
    """
    Preprocesa una base nueva de clientes para que tenga la misma estructura
    que los datos usados durante el entrenamiento del modelo en Colab.
    """

    df = df_original.copy()

    # --------------------------------------------------------------------------
    # 1. Eliminar identificador si existe
    # --------------------------------------------------------------------------

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # --------------------------------------------------------------------------
    # 2. Convertir TotalCharges a numérico
    # --------------------------------------------------------------------------

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        if df["TotalCharges"].isna().all():
            df["TotalCharges"] = df["TotalCharges"].fillna(0)
        else:
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # --------------------------------------------------------------------------
    # 3. Guardar Churn real si viene en el archivo
    # --------------------------------------------------------------------------

    churn_real = None

    if "Churn" in df.columns:
        churn_real = df["Churn"].map({"No": 0, "Yes": 1})
        df = df.drop(columns=["Churn"])

    # --------------------------------------------------------------------------
    # 4. Codificación binaria igual que en Colab
    # --------------------------------------------------------------------------

    binary_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling"
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # En el notebook: Female -> 0, Male -> 1
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1})

    # SeniorCitizen ya viene como 0/1, pero aseguramos formato numérico
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # --------------------------------------------------------------------------
    # 5. One-Hot Encoding para variables categóricas restantes
    # --------------------------------------------------------------------------

    df = pd.get_dummies(df, drop_first=True)

    # Convertir booleanos a enteros
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # --------------------------------------------------------------------------
    # 6. Alinear columnas con las columnas del modelo
    # --------------------------------------------------------------------------

    df = df.reindex(columns=columnas_modelo, fill_value=0)

    # --------------------------------------------------------------------------
    # 7. Aplicar StandardScaler a variables numéricas
    # --------------------------------------------------------------------------

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

    cols_a_escalar = [col for col in numeric_features if col in df.columns]

    if len(cols_a_escalar) > 0:
        df[cols_a_escalar] = scaler.transform(df[cols_a_escalar])

    return df, churn_real


def generar_predicciones(df_clientes):
    """
    Genera predicciones de churn para una base de clientes cargada por el usuario.
    """

    X_nuevo, churn_real = preprocesar_datos(
        df_original=df_clientes,
        columnas_modelo=columnas_modelo,
        scaler=scaler
    )

    probabilidades = modelo.predict_proba(X_nuevo)[:, 1]
    predicciones = (probabilidades >= 0.50).astype(int)

    resultados = df_clientes.copy()

    resultados["Probabilidad_Churn"] = probabilidades
    resultados["Prediccion_Churn"] = predicciones
    resultados["Segmento_Riesgo"] = resultados["Probabilidad_Churn"].apply(clasificar_riesgo)
    resultados["Accion_Sugerida"] = resultados["Segmento_Riesgo"].apply(asignar_accion)

    if churn_real is not None:
        resultados["Churn_Real"] = churn_real

    return resultados


def convertir_a_csv(df):
    """
    Convierte un DataFrame a CSV descargable.
    """
    return df.to_csv(index=False).encode("utf-8")


# ==============================================================================
# 4. CARGA DE ARCHIVO CSV
# ==============================================================================

st.sidebar.header("Carga de datos")

archivo = st.sidebar.file_uploader(
    "Sube un archivo CSV con clientes",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "El archivo debe tener las variables del dataset Telco Customer Churn o una estructura compatible."
)

# ==============================================================================
# 5. EJECUCIÓN PRINCIPAL DE LA APP
# ==============================================================================

if archivo is not None:

    try:
        df_clientes = pd.read_csv(archivo)
    except Exception as e:
        st.error("No se pudo leer el archivo CSV. Verifica que el formato sea correcto.")
        st.code(str(e))
        st.stop()

    st.subheader("Vista previa de la base cargada")
    st.dataframe(df_clientes.head(), use_container_width=True)

    try:
        resultados = generar_predicciones(df_clientes)
    except Exception as e:
        st.error("Ocurrió un error al generar las predicciones.")
        st.write("Revisa que el archivo tenga las columnas esperadas por el modelo.")
        st.code(str(e))
        st.stop()

    # --------------------------------------------------------------------------
    # 5.1 Métricas principales
    # --------------------------------------------------------------------------

    st.subheader("Resultados de predicción")

    total_clientes = len(resultados)

    clientes_alto_critico = resultados[
        resultados["Segmento_Riesgo"].isin(["Alto", "Crítico"])
    ].shape[0]

    probabilidad_promedio = resultados["Probabilidad_Churn"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Clientes analizados", f"{total_clientes:,}")
    col2.metric("Clientes alto/crítico", f"{clientes_alto_critico:,}")
    col3.metric("Probabilidad promedio", f"{probabilidad_promedio:.2%}")

    # --------------------------------------------------------------------------
    # 5.2 Tabla de clientes priorizados
    # --------------------------------------------------------------------------

    st.markdown("### Tabla de clientes priorizados")

    columnas_mostrar = [
        "Probabilidad_Churn",
        "Prediccion_Churn",
        "Segmento_Riesgo",
        "Accion_Sugerida"
    ]

    if "customerID" in resultados.columns:
        columnas_mostrar = ["customerID"] + columnas_mostrar

    if "Churn_Real" in resultados.columns:
        columnas_mostrar.append("Churn_Real")

    st.dataframe(
        resultados[columnas_mostrar].sort_values(
            by="Probabilidad_Churn",
            ascending=False
        ),
        use_container_width=True
    )

    # --------------------------------------------------------------------------
    # 5.3 Resumen visual
    # --------------------------------------------------------------------------

    st.subheader("Resumen visual")

    col_graf1, col_graf2 = st.columns(2)

    with col_graf1:
        st.markdown("#### Clientes por segmento de riesgo")

        orden_segmentos = ["Bajo", "Medio", "Alto", "Crítico"]

        conteo_segmentos = (
            resultados["Segmento_Riesgo"]
            .value_counts()
            .reindex(orden_segmentos)
            .fillna(0)
        )

        st.bar_chart(conteo_segmentos)

    with col_graf2:
        st.markdown("#### Distribución de probabilidad de churn")

        hist_data = pd.DataFrame({
            "Probabilidad_Churn": resultados["Probabilidad_Churn"]
        })

        st.bar_chart(hist_data)

    # --------------------------------------------------------------------------
    # 5.4 Tabla resumen por segmento
    # --------------------------------------------------------------------------

    st.markdown("### Resumen por segmento de riesgo")

    resumen_segmentos = (
        resultados
        .groupby("Segmento_Riesgo")
        .agg(
            Clientes=("Segmento_Riesgo", "count"),
            Probabilidad_Promedio=("Probabilidad_Churn", "mean")
        )
        .reindex(["Bajo", "Medio", "Alto", "Crítico"])
        .fillna(0)
    )

    resumen_segmentos["Probabilidad_Promedio"] = resumen_segmentos["Probabilidad_Promedio"].map(
        lambda x: f"{x:.2%}"
    )

    st.dataframe(resumen_segmentos, use_container_width=True)

    # --------------------------------------------------------------------------
    # 5.5 Descarga de resultados
    # --------------------------------------------------------------------------

    csv_resultados = convertir_a_csv(resultados)

    st.download_button(
        label="⬇️ Descargar resultados en CSV",
        data=csv_resultados,
        file_name="predicciones_churn.csv",
        mime="text/csv"
    )

else:
    st.warning("Sube un archivo CSV para generar predicciones.")

    st.markdown("""
    ### Flujo de uso de la aplicación

    1. Subir una base de clientes en formato CSV.
    2. La app aplica el mismo preprocesamiento usado durante el entrenamiento.
    3. El modelo Random Forest calcula la probabilidad de churn.
    4. Cada cliente se clasifica en un segmento de riesgo.
    5. Se sugiere una acción de retención.
    6. Se descargan los resultados para uso operativo.
    """)

    st.markdown("""
    ### Archivos necesarios en el repositorio

    Para que la aplicación funcione correctamente, el repositorio debe contener:

    - `app.py`
    - `requirements.txt`
    - `modelo_churn_random_forest.pkl`
    - `columnas_modelo_churn.pkl`
    - `scaler_churn.pkl`
    """)
