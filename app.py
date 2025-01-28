import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import altair as alt

st.title("📊 Dashboard de Elasticidade via Regressão Linear")

# Upload de dados
st.header("Carregue seus dados 📂")
uploaded_file = st.file_uploader("Carregue um arquivo CSV com duas colunas: Preço (P) e Quantidade (Q)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=";")
    st.write("📌 **Dados carregados:**")
    st.write(data)

    # Verifica se há duas colunas
    if data.shape[1] != 2:
        st.error("❌ O arquivo deve ter exatamente duas colunas: Preço (P) e Quantidade (Q).")
    else:
        # Renomear colunas para padronizar
        data.columns = ['Price', 'Quantity']

        # Variáveis
        X = data['Price'].values.reshape(-1, 1)  # Variável independente (Preço)
        y = data['Quantity'].values.reshape(-1, 1)  # Variável dependente (Quantidade)

        # Regressão Linear
        model = LinearRegression()
        model.fit(X, y)

        # Coeficientes da regressão
        intercept = model.intercept_[0]
        slope = model.coef_[0][0]

        # Cálculo da elasticidade-preço da demanda no ponto médio
        mean_price = data['Price'].mean()
        mean_quantity = data['Quantity'].mean()
        elasticity = (slope * mean_price) / mean_quantity

        # Cálculo do preço ótimo (E = -1)
        price_optimal = -intercept / (2 * slope)

        # R² (Coeficiente de Determinação)
        r_squared = model.score(X, y)

        # Correlação de Pearson
        correlation, p_value = stats.pearsonr(data['Price'], data['Quantity'])

        # Criando um intervalo contínuo de preços para a reta de regressão
        price_range = np.linspace(data['Price'].min(), data['Price'].max(), 100).reshape(-1, 1)
        predicted_range = model.predict(price_range)

        regression_df = pd.DataFrame({'Price': price_range.flatten(), 'Predicted': predicted_range.flatten()})

        # Ajustando os limites automáticos para os eixos
        x_min, x_max = data['Price'].min(), data['Price'].max()
        y_min, y_max = data['Quantity'].min(), data['Quantity'].max()

        # 📊 Criar gráfico de dispersão com escala dinâmica
        scatter_chart = alt.Chart(data).mark_circle(size=60, color="blue").encode(
            x=alt.X('Price', title="Preço", scale=alt.Scale(domain=(x_min, x_max))),
            y=alt.Y('Quantity', title="Quantidade", scale=alt.Scale(domain=(y_min, y_max))),
            tooltip=['Price', 'Quantity']
        )

        # 📈 Criar linha da regressão com escala dinâmica
        line_chart = alt.Chart(regression_df).mark_line(color='red', strokeWidth=2).encode(
            x=alt.X('Price', scale=alt.Scale(domain=(x_min, x_max))),
            y=alt.Y('Predicted', scale=alt.Scale(domain=(y_min, y_max)))
        )

        # 📊 Combinar gráficos e exibir
        final_chart = (scatter_chart + line_chart).properties(
            title="Regressão Linear: Preço vs Quantidade",
            width=700,
            height=400
        )

        # Exibir métricas calculadas
        st.subheader("📌 Resultados Estatísticos")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Intercepto (α)", f"{intercept:.2f}")
            st.metric("Coeficiente Angular (β)", f"{slope:.2f}")
            st.metric("Elasticidade-Preço da Demanda", f"{elasticity:.2f}")

        with col2:
            st.metric("Preço Ótimo (E = -1)", f"{price_optimal:.2f}")
            st.metric("Coeficiente de Determinação (R²)", f"{r_squared:.4f}")
            st.metric("Correlação de Pearson", f"{correlation:.4f}")
            st.metric("P-valor", f"{p_value:.4f}")

        st.altair_chart(final_chart, use_container_width=True)
