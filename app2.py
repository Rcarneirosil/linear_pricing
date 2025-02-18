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

st.title("📊 Dashboard de Análise de Elasticidade e Preço Ótimo")

# ======================= 📂 CARREGAMENTO DE DADOS =======================
st.subheader("📂 Carregue seus dados")
uploaded_file = st.file_uploader(
    "Carregue um arquivo CSV com colunas numéricas (preço, demanda e outras variáveis)",
    type=["csv"]
)
st.write("🔹 Formato requerido: CSV com **ponto e vírgula** como separador de colunas e **vírgula** como decimal.")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=";", decimal=",", encoding="utf-8")
    
    # ======================= 🔧 SELEÇÃO DE VARIÁVEIS =======================
    st.subheader("🔧 Configuração do Modelo")
    cols = data.columns.tolist()
    
    target = st.selectbox("Selecione a variável TARGET (demanda):", cols)
    features = st.multiselect("Selecione as FEATURES (inclua o preço):", [c for c in cols if c != target])
    price_var = st.selectbox("Qual variável representa o PREÇO?", features)

    if len(features) >= 1 and target and price_var:
        # ======================= 📈 MODELAGEM =======================
        X = data[features]
        y = data[target]
        
        model = LinearRegression()
        model.fit(X, y)
        
        # ======================= 📊 CÁLCULOS-CHAVE =======================
        # Coeficientes
        intercept = model.intercept_
        coefficients = model.coef_
        coef_dict = dict(zip(features, coefficients))
        
        # Elasticidade-Preço
        mean_price = X[price_var].mean()
        mean_quantity = y.mean()
        elasticity = coef_dict[price_var] * (mean_price / mean_quantity)
        
        # Preço Ótimo (considerando custo)
        custo = st.number_input("💰 Custo variável médio por unidade:", value=0.0)
        numerator = -intercept - sum([coef_dict[f] * X[f].mean() for f in features if f != price_var]) + coef_dict[price_var] * custo
        price_optimal = numerator / (2 * coef_dict[price_var]) if coef_dict[price_var] != 0 else np.nan
        
        # ======================= 🧪 DIAGNÓSTICOS =======================
        # Métricas de performance
        y_pred = model.predict(X)
        r_squared = model.score(X, y)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Testes estatísticos
        residuals = y - y_pred
        shapiro_test = stats.shapiro(residuals)
        shapiro_p = shapiro_test.pvalue
        
        # Heterocedasticidade (Breusch-Pagan)
        X_with_const = add_constant(X)
        try:
            _, bp_p, _, _ = het_breuschpagan(residuals, X_with_const)
        except:
            bp_p = np.nan
        
        # Multicolinearidade (VIF)
        vif_data = pd.DataFrame()
        vif_data["Variável"] = features
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(features))]
        
        # ======================= 📉 VISUALIZAÇÃO =======================
        st.subheader("📌 Indicadores Principais")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Preço Ótimo (E = -1)", f"R$ {price_optimal:,.2f}" if not np.isnan(price_optimal) else "N/A")
        with col2:
            st.metric("Elasticidade-Preço", f"{elasticity:,.2f}")
        with col3:
            st.write('<div style="font-size:16px;display:flex;align-items:center;gap:5px;">Qualidade do Modelo</div>', unsafe_allow_html=True)
            score = sum([r_squared >= 0.2, r_squared >= 0.4, r_squared >= 0.6, r_squared >= 0.8])
            st.write(f'<div style="display:flex;gap:5px;">{"🟢"*score}{"⚪"*(5-score)}</div>', unsafe_allow_html=True)
        
        # Gráfico (adaptado para múltiplas variáveis)
        if len(features) == 1:
            regression_df = pd.DataFrame({
                'Price': np.linspace(X.min()[0], X.max()[0], 100),
                'Predicted': model.predict(np.linspace(X.min()[0], X.max()[0], 100).reshape(-1, 1))
            })
            chart = alt.Chart(data).mark_circle().encode(
                x=alt.X(features[0], title="Preço"),
                y=alt.Y(target, title="Demanda")
            ) + alt.Chart(regression_df).mark_line(color='red').encode(
                x='Price',
                y='Predicted'
            )
        else:
            chart = alt.Chart(data).mark_circle().encode(
                x=alt.X(price_var, title="Preço"),
                y=alt.Y(target, title="Demanda"),
                tooltip=features
            )
        st.altair_chart(chart, use_container_width=True)
        
        # ======================= 📊 ESTATÍSTICAS COMPLEMENTARES =======================
        st.subheader("📊 Estatísticas Complementares")
        
        # Lista de métricas
        stats_list = f"""
        - **Intercepto (α):** {intercept:,.2f}
        - **Coeficiente do Preço (β):** {coef_dict[price_var]:,.2f}
        - **R²:** {r_squared:,.2f}
        - **MAE:** {mae:,.2f}
        - **RMSE:** {rmse:,.2f}
        - **Teste de Normalidade (Shapiro):** {'✅' if shapiro_p > 0.05 else '❌'} (p={shapiro_p:.3f})
        - **Heterocedasticidade (Breusch-Pagan):** {'✅' if bp_p > 0.05 else '❌'} (p={bp_p:.3f})
        """
        st.markdown(stats_list)
        
        # Tabela de VIF
        st.write("**Multicolinearidade (VIF):**")
        st.dataframe(
            vif_data.style.format({"VIF": "{:.1f}"})
            .highlight_between(subset=["VIF"], left=0, right=5, color="#C6EFCE")
            .highlight_between(subset=["VIF"], left=5, right=10, color="#FFEB9C")
            .highlight_between(subset=["VIF"], left=10, right=np.inf, color="#FFC7CE")
        )
        
        # Dados brutos
        st.subheader("📋 Dados Completos")
        st.write(data)
    else:
        st.error("Selecione pelo menos uma feature e identifique a variável de preço!")
