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
- Segmento de riesgo.
- Acciones sugeridas de retención.
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
        "Asegúrate de subir al repositorio: "
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
    Asigna acciones de retención según el nivel de riesgo.
    Las acciones están basadas en la estrategia de retención de la presentación.
    """

    acciones = {
        "Bajo": (
            "Fidelización y cross-selling: programa de referidos con incentivos, "
            "servicios premium a precio preferencial y reconocimiento de lealtad en aniversarios."
        ),
        "Medio": (
            "Comunicación preventiva y engagement: newsletter con tips de uso del servicio, "
            "invitación a eventos y beneficios exclusivos, y programa de puntos por antigüedad."
        ),
        "Alto": (
            "Campaña de retención semi-personalizada: email/SMS con ofertas de servicios adicionales, "
            "beneficios de contratos anuales vs. mensuales, 1 mes gratis de OnlineSecurity o TechSupport "
            "y recordatorio de beneficios no utilizados."
        ),
        "Crítico": (
            "Intervención inmediata y personalizada: llamada de agente de retención dedicado, "
            "descuento temporal del 20% en cargos mensuales, upgrade gratuito de plan por 3 meses "
            "y migración a contrato anual con incentivo económico."
        )
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
    # 4. Codificación binaria
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

    # Codificación de género
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1})

    # Asegurar SeniorCitizen como numérico
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = (
            pd.to_numeric(df["SeniorCitizen"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    # --------------------------------------------------------------------------
    # 5. One-Hot Encoding para variables categóricas restantes
    # --------------------------------------------------------------------------

    df = pd.get_dummies(df, drop_first=True)

    # Convertir columnas booleanas a enteros
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
    # 5.1 TABLA RESUMEN DE HALLAZGOS
    # --------------------------------------------------------------------------

    st.subheader("Resultados de predicción")

    total_clientes = len(resultados)

    clientes_bajo = resultados[resultados["Segmento_Riesgo"] == "Bajo"].shape[0]
    clientes_medio = resultados[resultados["Segmento_Riesgo"] == "Medio"].shape[0]
    clientes_alto = resultados[resultados["Segmento_Riesgo"] == "Alto"].shape[0]
    clientes_critico = resultados[resultados["Segmento_Riesgo"] == "Crítico"].shape[0]

    clientes_alto_critico = clientes_alto + clientes_critico
    probabilidad_promedio = resultados["Probabilidad_Churn"].mean()

    resumen_hallazgos = pd.DataFrame({
        "Hallazgo": [
            "Clientes analizados",
            "Clientes en riesgo bajo",
            "Clientes en riesgo medio",
            "Clientes en riesgo medio-alto",
            "Clientes en riesgo alto/crítico",
            "Clientes prioritarios",
            "Probabilidad promedio de churn"
        ],
        "Resultado": [
            f"{total_clientes:,}",
            f"{clientes_bajo:,}",
            f"{clientes_medio:,}",
            f"{clientes_alto:,}",
            f"{clientes_critico:,}",
            f"{clientes_alto_critico:,}",
            f"{probabilidad_promedio:.2%}"
        ],
        "Interpretación": [
            "Total de clientes evaluados por el modelo.",
            "Clientes con baja probabilidad de abandonar el servicio.",
            "Clientes que requieren seguimiento preventivo.",
            "Clientes que conviene atender con campañas de retención semi-personalizadas.",
            "Clientes con mayor urgencia de intervención comercial.",
            "Clientes en riesgo medio-alto o alto/crítico que deben atenderse primero.",
            "Riesgo promedio estimado en la base cargada."
        ]
    })

    st.dataframe(
        resumen_hallazgos,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------------------------
    # 5.2 GRÁFICA ÚTIL: CLIENTES POR SEGMENTO
    # --------------------------------------------------------------------------

    st.subheader("Distribución de clientes por segmento de riesgo")

    conteo_segmentos = (
        resultados["Segmento_Riesgo"]
        .value_counts()
        .reindex(["Bajo", "Medio", "Alto", "Crítico"])
        .fillna(0)
    )

    st.bar_chart(conteo_segmentos)

    # --------------------------------------------------------------------------
    # 5.3 TABLAS SEPARADAS POR SEGMENTO DE RIESGO
    # --------------------------------------------------------------------------

    st.subheader("Clientes priorizados por segmento de riesgo")

    st.markdown("""
    Las siguientes secciones separan a los clientes por nivel de riesgo.  
    Así se evita repetir la misma acción sugerida en cada fila y se facilita la priorización comercial.
    """)

    columnas_base = [
        "Probabilidad_Churn",
        "Prediccion_Churn"
    ]

    if "customerID" in resultados.columns:
        columnas_base = ["customerID"] + columnas_base

    if "Churn_Real" in resultados.columns:
        columnas_base.append("Churn_Real")

    segmentos_info = {
        "Crítico": {
            "titulo": "🔴 Riesgo Alto / Crítico",
            "metodo": "Intervención inmediata y personalizada",
            "acciones": [
                "Llamada de agente de retención dedicado.",
                "Descuento temporal del 20% en cargos mensuales.",
                "Upgrade gratuito de plan por 3 meses.",
                "Migración a contrato anual con incentivo económico."
            ]
        },
        "Alto": {
            "titulo": "🟠 Riesgo Medio-Alto",
            "metodo": "Campaña de retención semi-personalizada",
            "acciones": [
                "Email/SMS con ofertas de servicios adicionales.",
                "Beneficios de contratos anuales vs. mensuales.",
                "1 mes gratis de OnlineSecurity o TechSupport.",
                "Recordatorio de beneficios no utilizados."
            ]
        },
        "Medio": {
            "titulo": "🟡 Riesgo Medio",
            "metodo": "Comunicación preventiva y engagement",
            "acciones": [
                "Newsletter con tips de uso del servicio.",
                "Invitación a eventos y beneficios exclusivos.",
                "Programa de puntos por antigüedad."
            ]
        },
        "Bajo": {
            "titulo": "🟢 Riesgo Bajo",
            "metodo": "Fidelización y cross-selling",
            "acciones": [
                "Programa de referidos con incentivos.",
                "Servicios premium a precio preferencial.",
                "Reconocimiento de lealtad en aniversarios."
            ]
        }
    }

    orden_segmentos = ["Crítico", "Alto", "Medio", "Bajo"]

    for segmento in orden_segmentos:

        info = segmentos_info[segmento]

        df_segmento = (
            resultados[resultados["Segmento_Riesgo"] == segmento]
            .sort_values(by="Probabilidad_Churn", ascending=False)
        )

        st.markdown(f"## {info['titulo']}")

        col_a, col_b, col_c = st.columns(3)

        col_a.metric("Clientes", f"{len(df_segmento):,}")

        if len(df_segmento) > 0:
            col_b.metric(
                "Probabilidad promedio",
                f"{df_segmento['Probabilidad_Churn'].mean():.2%}"
            )
            col_c.metric(
                "Probabilidad máxima",
                f"{df_segmento['Probabilidad_Churn'].max():.2%}"
            )
        else:
            col_b.metric("Probabilidad promedio", "0.00%")
            col_c.metric("Probabilidad máxima", "0.00%")

        st.markdown(f"**Método:** {info['metodo']}")

        st.markdown("**Acciones sugeridas:**")
        for accion in info["acciones"]:
            st.markdown(f"- {accion}")

        if len(df_segmento) > 0:
            st.dataframe(
                df_segmento[columnas_base],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No hay clientes en este segmento para la base cargada.")

        st.markdown("---")

    # --------------------------------------------------------------------------
    # 5.4 DESCARGA DE RESULTADOS
    # --------------------------------------------------------------------------

    st.subheader("Descarga de resultados")

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
    5. Se sugieren acciones de retención por segmento.
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
