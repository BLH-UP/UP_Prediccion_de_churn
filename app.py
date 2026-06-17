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

# ==============================================================================
# 2. ESTILOS VISUALES
# ==============================================================================

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #0B1F3A;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #374151;
        margin-bottom: 25px;
    }

    .risk-card {
        padding: 24px;
        border-radius: 18px;
        margin-top: 25px;
        margin-bottom: 18px;
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
    }

    .risk-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 16px;
    }

    .risk-method {
        font-size: 17px;
        margin-bottom: 10px;
        color: #111827;
    }

    .risk-actions {
        font-size: 16px;
        line-height: 1.7;
        color: #1F2937;
    }

    .metric-box {
        background: rgba(255,255,255,0.65);
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.05);
        text-align: center;
    }

    .metric-label {
        font-size: 14px;
        color: #4B5563;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #111827;
    }

    .finance-box {
        padding: 18px;
        border-radius: 14px;
        margin-top: 14px;
        margin-bottom: 20px;
        border-left: 7px solid;
        font-size: 18px;
        font-weight: 700;
    }

    .small-note {
        color: #6B7280;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📡 Predicción de Churn en Telecomunicaciones</div>', unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Esta aplicación utiliza un modelo de <b>Random Forest</b> para estimar la probabilidad de que un cliente abandone el servicio.
Permite cargar una base CSV, segmentar clientes por nivel de riesgo y proponer acciones de retención.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. CARGA DEL MODELO Y OBJETOS NECESARIOS
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
# 4. FUNCIONES AUXILIARES
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


def formato_moneda(valor):
    """
    Da formato de moneda a valores financieros.
    """
    return f"${valor:,.2f}"


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
    # 2. Convertir TotalCharges y MonthlyCharges a numérico
    # --------------------------------------------------------------------------

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        if df["TotalCharges"].isna().all():
            df["TotalCharges"] = df["TotalCharges"].fillna(0)
        else:
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

        if df["MonthlyCharges"].isna().all():
            df["MonthlyCharges"] = df["MonthlyCharges"].fillna(0)
        else:
            df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].median())

    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

        if df["tenure"].isna().all():
            df["tenure"] = df["tenure"].fillna(0)
        else:
            df["tenure"] = df["tenure"].fillna(df["tenure"].median())

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

    # Asegurar MonthlyCharges numérico en resultados para cálculos financieros
    if "MonthlyCharges" in resultados.columns:
        resultados["MonthlyCharges"] = pd.to_numeric(
            resultados["MonthlyCharges"],
            errors="coerce"
        ).fillna(0)

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


def render_risk_section(segmento, info, df_segmento, columnas_base):
    """
    Renderiza una sección visual por segmento de riesgo.
    """

    clientes = len(df_segmento)

    if clientes > 0:
        prob_promedio = df_segmento["Probabilidad_Churn"].mean()
        prob_maxima = df_segmento["Probabilidad_Churn"].max()
    else:
        prob_promedio = 0
        prob_maxima = 0

    if "MonthlyCharges" in df_segmento.columns:
        representacion_financiera = df_segmento["MonthlyCharges"].sum()
    else:
        representacion_financiera = 0

    acciones_html = "".join([f"<li>{accion}</li>" for accion in info["acciones"]])

    st.markdown(
        f"""
        <div class="risk-card" style="background:{info['bg']}; border-color:{info['border']};">
            <div class="risk-title" style="color:{info['color']};">
                {info['icon']} {info['titulo']}
            </div>

            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; margin-bottom:20px;">
                <div class="metric-box">
                    <div class="metric-label">Clientes</div>
                    <div class="metric-value">{clientes:,}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Probabilidad promedio</div>
                    <div class="metric-value">{prob_promedio:.2%}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Probabilidad máxima</div>
                    <div class="metric-value">{prob_maxima:.2%}</div>
                </div>
            </div>

            <div class="risk-method">
                <b>Método:</b> {info['metodo']}
            </div>

            <div class="risk-actions">
                <b>Acciones sugeridas:</b>
                <ul>
                    {acciones_html}
                </ul>
            </div>

            <div class="finance-box" style="background:{info['finance_bg']}; border-left-color:{info['color']}; color:{info['color']};">
                Representación financiera del grupo de {info['titulo'].lower()}: {formato_moneda(representacion_financiera)}
            </div>

            <div class="small-note">
                La representación financiera corresponde a la suma de los cargos mensuales de los clientes clasificados en este segmento.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if clientes > 0:
        st.dataframe(
            df_segmento[columnas_base],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No hay clientes en este segmento para la base cargada.")

    st.markdown("---")


# ==============================================================================
# 5. CARGA DE ARCHIVO CSV
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
# 6. EJECUCIÓN PRINCIPAL DE LA APP
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
    # 6.1 TABLA RESUMEN DE HALLAZGOS
    # --------------------------------------------------------------------------

    st.subheader("Resultados de predicción")

    total_clientes = len(resultados)

    clientes_bajo = resultados[resultados["Segmento_Riesgo"] == "Bajo"].shape[0]
    clientes_medio = resultados[resultados["Segmento_Riesgo"] == "Medio"].shape[0]
    clientes_alto = resultados[resultados["Segmento_Riesgo"] == "Alto"].shape[0]
    clientes_critico = resultados[resultados["Segmento_Riesgo"] == "Crítico"].shape[0]

    clientes_prioritarios = clientes_alto + clientes_critico
    probabilidad_promedio = resultados["Probabilidad_Churn"].mean()

    if "MonthlyCharges" in resultados.columns:
        representacion_total = resultados["MonthlyCharges"].sum()
        representacion_prioritaria = resultados[
            resultados["Segmento_Riesgo"].isin(["Alto", "Crítico"])
        ]["MonthlyCharges"].sum()
    else:
        representacion_total = 0
        representacion_prioritaria = 0

    resumen_hallazgos = pd.DataFrame({
        "Hallazgo": [
            "Clientes analizados",
            "Clientes en riesgo bajo",
            "Clientes en riesgo medio",
            "Clientes en riesgo medio-alto",
            "Clientes en riesgo alto/crítico",
            "Clientes prioritarios",
            "Probabilidad promedio de churn",
            "Representación financiera total mensual",
            "Representación financiera prioritaria mensual"
        ],
        "Resultado": [
            f"{total_clientes:,}",
            f"{clientes_bajo:,}",
            f"{clientes_medio:,}",
            f"{clientes_alto:,}",
            f"{clientes_critico:,}",
            f"{clientes_prioritarios:,}",
            f"{probabilidad_promedio:.2%}",
            formato_moneda(representacion_total),
            formato_moneda(representacion_prioritaria)
        ],
        "Interpretación": [
            "Total de clientes evaluados por el modelo.",
            "Clientes con baja probabilidad de abandonar el servicio.",
            "Clientes que requieren seguimiento preventivo.",
            "Clientes que conviene atender con campañas de retención semi-personalizadas.",
            "Clientes con mayor urgencia de intervención comercial.",
            "Clientes en riesgo medio-alto o alto/crítico que deben atenderse primero.",
            "Riesgo promedio estimado en la base cargada.",
            "Suma de cargos mensuales de todos los clientes cargados.",
            "Suma de cargos mensuales de los clientes que requieren mayor prioridad."
        ]
    })

    st.dataframe(
        resumen_hallazgos,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------------------------
    # 6.2 GRÁFICA ÚTIL: CLIENTES POR SEGMENTO
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
    # 6.3 TABLAS SEPARADAS POR SEGMENTO DE RIESGO
    # --------------------------------------------------------------------------

    st.subheader("Clientes priorizados por segmento de riesgo")

    st.markdown("""
    Las siguientes secciones separan a los clientes por nivel de riesgo.  
    Cada bloque incluye el método de retención, las acciones sugeridas, la tabla de clientes y su representación financiera mensual.
    """)

    columnas_base = []

    if "customerID" in resultados.columns:
        columnas_base.append("customerID")

    columnas_base.append("Probabilidad_Churn")
    columnas_base.append("Prediccion_Churn")

    if "MonthlyCharges" in resultados.columns:
        columnas_base.append("MonthlyCharges")

    if "Churn_Real" in resultados.columns:
        columnas_base.append("Churn_Real")

    segmentos_info = {
        "Crítico": {
            "titulo": "Riesgo Alto / Crítico",
            "icon": "🔴",
            "color": "#E84855",
            "bg": "#FFF1F2",
            "border": "#FDA4AF",
            "finance_bg": "#FFE4E6",
            "metodo": "Intervención inmediata y personalizada",
            "acciones": [
                "Llamada de agente de retención dedicado.",
                "Descuento temporal del 20% en cargos mensuales.",
                "Upgrade gratuito de plan por 3 meses.",
                "Migración a contrato anual con incentivo económico."
            ]
        },
        "Alto": {
            "titulo": "Riesgo Medio-Alto",
            "icon": "🟠",
            "color": "#F97316",
            "bg": "#FFF7ED",
            "border": "#FDBA74",
            "finance_bg": "#FFEDD5",
            "metodo": "Campaña de retención semi-personalizada",
            "acciones": [
                "Email/SMS con ofertas de servicios adicionales.",
                "Beneficios de contratos anuales vs. mensuales.",
                "1 mes gratis de OnlineSecurity o TechSupport.",
                "Recordatorio de beneficios no utilizados."
            ]
        },
        "Medio": {
            "titulo": "Riesgo Medio",
            "icon": "🟡",
            "color": "#D97706",
            "bg": "#FFFBEB",
            "border": "#FCD34D",
            "finance_bg": "#FEF3C7",
            "metodo": "Comunicación preventiva y engagement",
            "acciones": [
                "Newsletter con tips de uso del servicio.",
                "Invitación a eventos y beneficios exclusivos.",
                "Programa de puntos por antigüedad."
            ]
        },
        "Bajo": {
            "titulo": "Riesgo Bajo",
            "icon": "🟢",
            "color": "#059669",
            "bg": "#ECFDF5",
            "border": "#6EE7B7",
            "finance_bg": "#D1FAE5",
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

        render_risk_section(
            segmento=segmento,
            info=info,
            df_segmento=df_segmento,
            columnas_base=columnas_base
        )

    # --------------------------------------------------------------------------
    # 6.4 DESCARGA DE RESULTADOS
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
    6. Se calcula la representación financiera mensual por grupo de riesgo.
    7. Se descargan los resultados para uso operativo.
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
