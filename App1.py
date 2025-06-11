# /opt/anaconda3/bin/python -m streamlit run App1.py
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from scipy.stats import pearsonr, shapiro, kruskal, spearmanr


# Encabezado
st.title("Proyecto final de Análisis de Datos Masivos")
st.write("**Autores:** Christian Hernández • Andrés Romero")

df = pd.read_csv("indicadores_clean.csv")

# Descripción del estudio
st.header("Descripción del conjunto de datos")
st.markdown(
    """Este análisis se basa en el **Legatum Prosperity Index 2023**, informe que clasifica a más de 160 paísesen 12 dimensiones de prosperidad y desarrollo sostenible.
    """
)

# Objetivos del análisis

st.header("Objetivos")
st.markdown(
    """ 1. **Factores clave de prosperidad**  
    Identificar los pilares que más influyen en el puntaje global.

    2. **Comparación regional**  
    Explorar brechas y coincidencias entre continentes.

    3. **Relaciones entre indicadores**  
    Investigar asociaciones relevantes entre las 12 dimensiones.

    4. **Países atípicos**  
    Detectar naciones que sobresalen o se rezagan en indicadores específicos. """
)


# OBJETIVO 1 – Factores clave

st.subheader("Objetivo 1 | Factores clave de la prosperidad global")

corr_matrix_12 = df.select_dtypes(["float64", "int64"]).corr()
fig_corr_12 = px.imshow(
    corr_matrix_12, text_auto=True, color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1, title="Mapa de calor • Correlación entre indicadores"
)
st.plotly_chart(fig_corr_12, use_container_width=True)

indicadores = [
    "SafetySecurity", "PersonelFreedom", "Governance", "SocialCapital",
    "InvestmentEnvironment", "EnterpriseConditions", "MarketAccessInfrastructure",
    "EconomicQuality", "LivingConditions", "Health", "Education", "NaturalEnvironment"
]

if st.toggle("Mostrar coeficientes de Pearson", value=False):
    res = [
        {
            "Indicador": i,
            "r": round(pearsonr(df[i], df["AveragScore"])[0], 3),
            "p-valor": f'{pearsonr(df[i], df["AveragScore"])[1]:.5f}'
        } for i in indicadores
    ]
    st.dataframe(pd.DataFrame(res), use_container_width=True)

st.markdown(
    """
    **Conclusión 1** Todos los indicadores presentan correlaciones positivas y significativas con la prosperidad global p < 0.001. Los de mayor impacto son *Investment Environment*, *Market Access & Infrastructure*, *Governance*, *Education* y *Economic Quality*, confirmando su papel estratégico en el desarrollo sostenible.
    """
)

st.markdown("---")

# OBJETIVO 2 – Comparación regional
st.subheader("Objetivo 2 | Comparación regional de la prosperidad")

pros = (
    df.groupby("Continent")["AveragScore"]
    .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
    .sort_values(by="mean", ascending=False)
)
st.dataframe(
    pros.style.format({"mean": "{:.2f}", "median": "{:.2f}", "std": "{:.2f}",
                    "min": "{:.2f}", "max": "{:.2f}"}),
    use_container_width=True
)

fig_bar = px.bar(
    pros["mean"].reset_index().rename(columns={"mean": "AveragScore"}),
    x="Continent", y="AveragScore", color="AveragScore",
    color_continuous_scale="Oranges",
    title="Media de prosperidad por continente",
    labels={"AveragScore": "Índice medio", "Continent": "Continente"}
)
st.plotly_chart(fig_bar, use_container_width=True)

fig_box = px.box(
    df, x="Continent", y="AveragScore", color="Continent",
    title="Distribución de prosperidad por continente",
    labels={"AveragScore": "Índice de prosperidad", "Continent": "Continente"}
)
fig_box.update_layout(showlegend=False)
st.plotly_chart(fig_box, use_container_width=True)

if st.toggle("Normalidad y Kruskal–Wallis", value=False):
    conts = ["Africa", "America", "Asia", "Europe", "Oceania"]
    groups = {c: df[df["Continent"] == c]["AveragScore"] for c in conts}
    shapiro_res = [
        {"Continente": c,
        "Estadístico": round(shapiro(groups[c])[0], 3),
        "p-valor": f'{shapiro(groups[c])[1]:.5f}'}
        for c in conts
    ]
    st.dataframe(pd.DataFrame(shapiro_res), use_container_width=True)
    kw_stat, kw_p = kruskal(*groups.values())
    st.markdown(f"**Kruskal–Wallis:** estadístico = {kw_stat:.3f} • p = {kw_p:.2e}")

promedios = df.groupby("Continent")[indicadores].mean().round(2)
fig_heat2 = px.imshow(
    promedios, text_auto=True, color_continuous_scale="Oranges",
    title="Promedio de indicadores por continente",
    labels={"x": "Indicador", "y": "Continente", "color": "Promedio"}
)
fig_heat2.update_xaxes(side="bottom", tickangle=30)
st.plotly_chart(fig_heat2, use_container_width=True)

st.markdown(
    """
    **Conclusión 2**  
    Europa lidera la prosperidad (72.6), seguida de Oceanía (68.8) y América (59.3). El test de Kruskal–Wallis (p ≈ 8.3×10⁻¹⁹) corrobora diferencias significativas entre continentes, destacando una brecha de ~25 puntos entre Europa y África.
    """
)

st.markdown("---")

# OBJETIVO 3 – Relaciones entre indicadores

st.subheader("Objetivo 3 | Relaciones entre indicadores")

st.markdown(
    """
    Para explorar cómo interactúan las 12 dimensiones de prosperidad entre sí, se generó una matriz de correlación y se analizaron cuatro pares particularmente reveladores.
    """
)

# Matriz de correlación de los 12 indicadores
corr_matrix = df[indicadores].corr()
fig_corr3 = px.imshow(
    corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
    labels={"x": "Indicador", "y": "Indicador", "color": "Corr."},
    title="Correlaciones entre indicadores"
)
fig_corr3.update_xaxes(side="bottom", tickangle=30)
st.plotly_chart(fig_corr3, use_container_width=True)

pares = [
    ("InvestmentEnvironment", "EnterpriseConditions"),
    ("LivingConditions", "Education"),
    ("PersonelFreedom", "NaturalEnvironment"),
    ("SocialCapital", "EconomicQuality")
]

def corr_res(a, b):
    r_p, p_p = pearsonr(df[a], df[b])
    r_s, p_s = spearmanr(df[a], df[b])
    return {
        "Par": f"{a} vs {b}",
        "Pearson r": round(r_p, 3),
        "p (Pearson)": f"{p_p:.5f}",
        "Spearman ρ": round(r_s, 3),
        "p (Spearman)": f"{p_s:.5f}"
    }

results = [corr_res(a, b) for a, b in pares]

if st.toggle("Mostrar tabla de Pearson / Spearman", value=False):
    st.dataframe(pd.DataFrame(results), use_container_width=True)

# Conclusión del objetivo 3
st.markdown(
    """
    **Conclusión 3**  
    Los pares analizados presentan correlaciones fuertes y altamente significativas p < 0.001:  
    * **Investment Environment vs Enterprise Conditions** r ≈ 0.93 
    * **Living Conditions vs Education** r ≈ 0.94  
    * **Personel Freedom vs Natural Environment** r ≈ 0.72 
    * **Social Capital vs Economic Quality** r ≈ 0.68  

    Estos hallazgos confirman que zonas de inversión favorables se asocian a ecosistemas empresariales robustos, la calidad de vida va de la mano de una educación sólida y, aunque menos obvio, la libertad individual guarda relación con un desempeño ambiental positivo.
    """
)

st.markdown("---")

# OBJETIVO 4 – Países con desempeños atípicos

st.subheader("Objetivo 4 | Países con desempeños atípicos")
st.markdown(
    """
    Identificamos naciones cuyo comportamiento en un indicador resulta inesperadamente mejor o peor que en otro, explorando tanto distribuciones por continente como diferencias de ranking.
    """
)

# Pregunta 1 – Salud vs Calidad Económica

st.markdown("#### 1) ¿Países con **mala calidad económica** pero **excelente salud**?")

# Boxplots de Salud y Economía
fig_h = px.box(
    df, x="Continent", y="Health", points="all",
    color="Continent",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title="Distribución de Salud por Continente"
)
st.plotly_chart(fig_h, use_container_width=True)

fig_e = px.box(
    df, x="Continent", y="EconomicQuality", points="all",
    color="Continent",
    color_discrete_sequence=px.colors.qualitative.Vivid,
    title="Distribución de Calidad Económica por Continente"
)
st.plotly_chart(fig_e, use_container_width=True)

# Normalidad
with st.expander("Pruebas de normalidad"):
    normas = []
    for c in df["Continent"].unique():
        for var in ["Health", "EconomicQuality"]:
            stat, p = shapiro(df[df["Continent"] == c][var])
            normas.append({
                "Continente": c,
                "Variable": var,
                "Estadístico": round(stat, 3),
                "p-valor": f"{p:.4f}",
                "Normalidad": "Sí" if p >= 0.05 else "No"
            })
    st.dataframe(pd.DataFrame(normas), use_container_width=True)

# Ranking diferencial
df["Health_rank"] = df["Health"].rank(ascending=False)
df["Econ_rank"]   = df["EconomicQuality"].rank(ascending=False)
df["Diff_rank"]   = df["Health_rank"] - df["Econ_rank"]
top_health = df.sort_values("Diff_rank", ascending=False).head(10)

with st.expander("TOP salud vs economía"):
    st.dataframe(
        top_health[["Country", "Continent", "Health", "EconomicQuality", "Diff_rank"]],
        use_container_width=True
    )

st.markdown(
    """
    **Conclusión 1**  
    Europa y Oceanía muestran las medianas de salud más altas y poca dispersión, mientras
    que África se ubica con los valores más bajos y mayor variabilidad. Los tests de Shapiro-Wilk
    confirman distribuciones no normales en la mayoría de los casos p < 0.05, lo que justifica
    el uso de análisis no paramétricos.  

    El ranking diferencial destaca a *Botswana*, *Equatorial Guinea* y *Côte d'Ivoire* como
    ejemplos de países con sistemas de salud robustos pese a una calidad económica moderada,
    probablemente gracias a políticas públicas sanitarias focalizadas o apoyos internacionales.
    """
)

st.markdown("---")

# Pregunta 2 – Educación vs Infraestructura de Mercado
st.markdown("#### 2) ¿Países con **infraestructura limitada** pero **alto nivel educativo**?")

# Boxplots de Educación y Mercado
fig_ed = px.box(
    df, x="Continent", y="Education", points="all",
    color="Continent",
    color_discrete_sequence=px.colors.qualitative.Light24,
    title="Distribución de Educación por Continente"
)
st.plotly_chart(fig_ed, use_container_width=True)

fig_mi = px.box(
    df, x="Continent", y="MarketAccessInfrastructure", points="all",
    color="Continent",
    color_discrete_sequence=px.colors.qualitative.Antique,
    title="Distribución de Infraestructura de Mercado por Continente"
)
st.plotly_chart(fig_mi, use_container_width=True)

# Normalidad 
with st.expander("Pruebas de normalidad"):
    normas2 = []
    for c in df["Continent"].unique():
        for var in ["Education", "MarketAccessInfrastructure"]:
            stat, p = shapiro(df[df["Continent"] == c][var])
            normas2.append({
                "Continente": c,
                "Variable": var,
                "Estadístico": round(stat, 3),
                "p-valor": f"{p:.4f}",
                "Normalidad": "Sí" if p >= 0.05 else "No"
            })
    st.dataframe(pd.DataFrame(normas2), use_container_width=True)

# Ranking Educación vs Infra

df["Edu_rank"]        = df["Education"].rank(ascending=False)
df["Infra_rank"]      = df["MarketAccessInfrastructure"].rank(ascending=False)
df["DiffEduInfra"]    = df["Edu_rank"] - df["Infra_rank"]
top_edu = df.sort_values("DiffEduInfra", ascending=False).head(10)

with st.expander("TOP educación vs infraestructura"):
    st.dataframe(
        top_edu[["Country", "Continent", "Education",
                "MarketAccessInfrastructure", "DiffEduInfra"]],
        use_container_width=True
    )

st.markdown(
    """
    **Conclusión 2**  
    Europa y Oceanía lideran en educación e infraestructura, con distribuciones homogéneas.
    América y Asia presentan mayor dispersión, y África muestra medianas bajas en ambos indicadores.
    Los tests de Shapiro-Wilk indican no normalidad en varios grupos p < 0.05, por lo que
    el ranking diferencial aporta un enfoque sólido para detectar outliers.  

    Países como *Marruecos*, *Panamá* y *Guatemala* destacan por superar las limitaciones de
    infraestructura con sistemas educativos avanzados, evidenciando estrategias nacionales
    focalizadas en capital humano y políticas sociales innovadoras.
    """
)