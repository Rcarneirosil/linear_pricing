import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats
import altair as alt

st.title("📊 Dashboard de Análise de Elasticidade")

# Upload de dados
st.header("📂 Carregue seus dados")
uploaded_file = st.file_uploader("Carregue um arquivo CSV com duas colunas: Preço (P) e Quantidade (Q) - use separador ';'", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=";")

    # Verifica se há duas colunas
    if data.shape[1] != 2:
        st.error("❌ O arquivo deve ter exatamente duas colunas: Preço (P) e Quantidade (Q).")
    else:
        # Renomear colunas para padronizar
        data.columns = ['Price', 'Quantity']

        # Variáveis
        X = data['Price'].values.reshape(-1, 1)  
        y = data['Quantity'].values.reshape(-1, 1)  

        # Regressão Linear
        model = LinearRegression()
        model.fit(X, y)

        # Coeficientes da regressão
        intercept = model.intercept_[0]
        slope = model.coef_[0][0]

        # Elasticidade e preço ótimo
        mean_price = data['Price'].mean()
        mean_quantity = data['Quantity'].mean()
        elasticity = (slope * mean_price) / mean_quantity
        price_optimal = -intercept / (2 * slope)

        # Estatísticas do modelo
        r_squared = model.score(X, y)
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Correlação de Pearson
        correlation, p_value = stats.pearsonr(data['Price'], data['Quantity'])

        # Resíduos
        residuals = y - y_pred
        residuals_mean = residuals.mean()
        shapiro_test = stats.shapiro(residuals.flatten())
        shapiro_p_value = shapiro_test.pvalue

        # Criando um intervalo contínuo de preços para a reta de regressão
        price_range = np.linspace(data['Price'].min(), data['Price'].max(), 100).reshape(-1, 1)
        predicted_range = model.predict(price_range)

        regression_df = pd.DataFrame({'Price': price_range.flatten(), 'Predicted': predicted_range.flatten()})

        # Ajustando os limites automáticos para os eixos
        x_min, x_max = data['Price'].min(), data['Price'].max()
        y_min, y_max = data['Quantity'].min(), data['Quantity'].max()

        # ======================= 🟢 1. Indicadores Principais =======================
        st.subheader("📌 Indicadores Principais")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Preço Ótimo (E = -1)", f"{price_optimal:,.2f}".replace(".", "X").replace(",", ".").replace("X", ","))
        with col2:
            st.metric("Elasticidade-Preço da Demanda", f"{elasticity:,.2f}".replace(".", "X").replace(",", ".").replace("X", ","))

        # ======================= 🔵 2. Gráfico de Regressão =======================
        st.subheader("📈 Regressão Linear: Preço vs Quantidade")

        scatter_chart = alt.Chart(data).mark_circle(size=60, color="blue").encode(
            x=alt.X('Price', title="Preço", scale=alt.Scale(domain=(x_min, x_max))),
            y=alt.Y('Quantity', title="Quantidade", scale=alt.Scale(domain=(y_min, y_max))),
            tooltip=['Price', 'Quantity']
        )

        line_chart = alt.Chart(regression_df).mark_line(color='red', strokeWidth=2).encode(
            x=alt.X('Price', scale=alt.Scale(domain=(x_min, x_max))),
            y=alt.Y('Predicted', scale=alt.Scale(domain=(y_min, y_max)))
        )

        final_chart = (scatter_chart + line_chart).properties(
            width=700,
            height=400
        )

        st.altair_chart(final_chart, use_container_width=True)

        # ======================= 🟠 3. Estatísticas Complementares =======================
        st.subheader("📊 Estatísticas Complementares")

# Avaliação do R² com ícones 🔴🟡🟢
if r_squared < 0.3:
    r2_status = "🔴 - <span style='color:blue;'>Baixo</span>"
elif 0.3 <= r_squared < 0.7:
    r2_status = "🟡 - <span style='color:blue;'>Regular</span>"
else:
    r2_status = "🟢 - <span style='color:blue;'>Excelente</span>"

# P-valor da correlação com ✔️ caso seja significativo
p_status = "<span style='color:blue;'>✅ Normal</span>" if p_value < 0.05 else "<span style='color:blue;'>❌ Não Normal</span>"

# Lista formatada com estatísticas (valores no padrão brasileiro)
stats_list = f"""
- **Coeficiente de Determinação (R²):** {r_squared:,.4f}.replace(".", "X").replace(",", ".").replace("X", ",") {r2_status}  
- **Intercepto (α):** {intercept:,.2f}.replace(".", "X").replace(",", ".").replace("X", ",")  
- **Coeficiente Angular (β):** {slope:,.2f}.replace(".", "X").replace(",", ".").replace("X", ",")  
- **Correlação de Pearson:** {correlation:,.4f}.replace(".", "X").replace(",", ".").replace("X", ",")  
- **P-valor da Correlação:** {p_value:,.4f}.replace(".", "X").replace(",", ".").replace("X", ",") {p_status}  
- **Erro Absoluto Médio (MAE):** {mae:,.2f}.replace(".", "X").replace(",", ".").replace("X", ",")  
- **Erro Padrão dos Resíduos (RMSE):** {rmse:,.2f}.replace(".", "X").replace(",", ".").replace("X", ",")  
- **Média dos Resíduos:** {residuals_mean:,.2e}.replace(".", "X").replace(",", ".").replace("X", ",")  
- **Teste de Normalidade dos Resíduos (Shapiro-Wilk):**  
  **P-valor:** {shapiro_p_value:,.4f}.replace(".", "X").replace(",", ".").replace("X", ",") <span style='color:blue;'>{'✅ Normal' if shapiro_p_value > 0.05 else '❌ Não Normal'}</span>
"""

# Exibir estatísticas complementares com separadores formatados
st.markdown(stats_list.replace(".", "X").replace(",", ".").replace("X", ","), unsafe_allow_html=True)

        # ======================= 🟡 4. Exibição dos Dados =======================
        st.subheader("📋 Tabela de Dados Carregados")
        st.write(data)
