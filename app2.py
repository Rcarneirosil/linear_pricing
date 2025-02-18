import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats
import altair as alt
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.title("📊 Dashboard Avançado de Análise de Pricing")

# Upload de dados
st.subheader("📂 Carregue seus dados")
uploaded_file = st.file_uploader(
    "Carregue um arquivo CSV com colunas numéricas (preço, quantidade, e outras variáveis)",
    type=["csv"]
)

if uploaded_file is not None:
    # Ler o CSV
    data = pd.read_csv(uploaded_file, sep=";", decimal=",", encoding="utf-8")
    
    # Seleção de variáveis
    st.subheader("🔧 Configuração do Modelo")
    cols = data.columns.tolist()
    
    target = st.selectbox("Selecione a variável TARGET (quantidade demandada):", cols)
    features = st.multiselect("Selecione as FEATURES (inclua o preço):", [c for c in cols if c != target])
    price_var = st.selectbox("Qual variável representa o PREÇO?", features)

    if len(features) >= 1 and target:
        # Preparar dados
        X = data[features]
        y = data[target]
        
        # ======================= MODELAGEM =======================
        model = LinearRegression()
        model.fit(X, y)
        
        # Coeficientes
        intercept = model.intercept_
        coefficients = dict(zip(features, model.coef_))
        
        # ======================= CÁLCULOS CHAVE =======================
        # Elasticidade-Preço
        mean_price = X[price_var].mean()
        mean_quantity = y.mean()
        elasticity = coefficients[price_var] * (mean_price / mean_quantity)
        
        # Preço Ótimo (considerando custo médio)
        custo_medio = st.number_input("💰 Custo variável médio por unidade:", value=0.0)
        if coefficients[price_var] != 0:
            numerator = -intercept - sum([coefficients[f] * X[f].mean() for f in features if f != price_var]) + coefficients[price_var] * custo_medio
            price_optimal = numerator / (2 * coefficients[price_var])
        else:
            price_optimal = np.nan
        
        # ======================= DIAGNÓSTICOS =======================
        # Métricas de performance
        y_pred = model.predict(X)
        r_squared = model.score(X, y)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Testes estatísticos
        residuals = y - y_pred
        shapiro_p = stats.shapiro(residuals).pvalue
        X_with_const = add_constant(X)
        try:
            _, bp_p, _, _ = het_breuschpagan(residuals, X_with_const)
        except:
            bp_p = np.nan
        
        # Multicolinearidade (VIF)
        vif_data = pd.DataFrame()
        vif_data["Variável"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        # ======================= VISUALIZAÇÃO =======================
        st.subheader("📌 Principais Indicadores")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Elasticidade-Preço", f"{elasticity:.2f}")
        with col2:
            st.metric("Preço Ótimo Estimado", f"R$ {price_optimal:.2f}" if not np.isnan(price_optimal) else "N/A")
        with col3:
            st.metric("R²", f"{r_squared:.2%}")
        
        # Gráfico parcial para preço
        if len(features) == 1:
            chart_data = pd.DataFrame({'Preço': X[price_var], 'Demanda': y})
            st.altair_chart(alt.Chart(chart_data).mark_circle().encode(
                x='Preço', y='Demanda', tooltip=['Preço', 'Demanda']
            ).interactive(), use_container_width=True)
        else:
            st.write("🔍 Visualização parcial (apenas relação preço-demanda):")
            partial_data = pd.DataFrame({'Preço': X[price_var], 'Demanda Real': y, 'Demanda Prevista': y_pred})
            st.line_chart(partial_data.set_index('Preço'))
        
        # ======================= DETALHES TÉCNICOS =======================
        st.subheader("🔍 Diagnóstico do Modelo")
        
        # Coeficientes
        st.write("**Coeficientes:**")
        coef_df = pd.DataFrame.from_dict(coefficients, orient='index', columns=['Valor'])
        st.dataframe(coef_df.style.format("{:.2f}"))
        
        # Multicolinearidade
        st.write("**Multicolinearidade (VIF):**")
        st.write(vif_data.style.format({"VIF": "{:.1f}"})
                  .highlight_between(subset=["VIF"], low=0, high=5, color="lightgreen")
                  .highlight_between(subset=["VIF"], low=5, high=10, color="orange")
                  .highlight_between(subset=["VIF"], low=10, high=None, color="red"))
        
        # Pressupostos
        st.write("**Testes Estatísticos:**")
        stats_list = f"""
        - **Normalidade dos Resíduos (Shapiro-Wilk):** {'✅' if shapiro_p > 0.05 else '❌'} (p = {shapiro_p:.3f})
        - **Homocedasticidade (Breusch-Pagan):** {'✅' if bp_p > 0.05 else '❌'} (p = {bp_p:.3f})
        - **MAE:** {mae:.2f}
        - **RMSE:** {rmse:.2f}
        """
        st.markdown(stats_list)
        
        # Dados brutos
        st.subheader("📋 Dados Completos")
        st.write(data)
    else:
        st.error("⚠️ Selecione pelo menos uma feature e um target!")
