import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import altair as alt

st.title("App de Elasticidade via Regressão Linear")

# Upload de dados
st.header("Carregue seus dados")
uploaded_file = st.file_uploader("Carregue um arquivo CSV com duas colunas: Preço (P) e Quantidade (Q)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=";")
    st.write("Dados carregados:")
    st.write(data)

    # Verifica se há duas colunas
    if data.shape[1] != 2:
        st.error("O arquivo deve ter exatamente duas colunas: Preço (P) e Quantidade (Q).")
    else:
        # Renomear colunas para padronizar
        data.columns = ['Price', 'Quantity']

        # Regressão Linear sem aplicar logaritmos
        X = data['Price'].values.reshape(-1, 1)  # Variável independente (Preço)
        y = data['Quantity'].values.reshape(-1, 1)  # Variável dependente (Quantidade)

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

        st.write(f"Intercepto (α): {intercept:.2f}")
        st.write(f"Coeficiente Angular (β): {slope:.2f}")
        st.write(f"Elasticidade-Preço da Demanda: {elasticity:.2f}")
        st.write(f"Preço Ótimo (E = -1): {price_optimal:.2f}")

        # Previsões para a linha de regressão
        data['Predicted'] = model.predict(X)

        # Gráfico interativo com Altair
        chart = alt.Chart(data).mark_circle(size=60, color="blue").encode(
            x=alt.X('Price', title="Preço"),
            y=alt.Y('Quantity', title="Quantidade"),
            tooltip=['Price', 'Quantity', 'Predicted']
        ).properties(
            title="Regressão Linear: Preço vs Quantidade",
            width=700,
            height=400
        ) + alt.Chart(data).mark_line(color='red', strokeWidth=2).encode(
            x='Price',
            y='Predicted'
        )

        st.altair_chart(chart, use_container_width=True)
