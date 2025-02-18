import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.stats as stats
import altair as alt

st.title("📊 Dashboard de Análise de Elasticidade e Preço Ótimo")

# Upload de dados
st.subheader("📂 Carregue seus dados")
uploaded_file = st.file_uploader(
    "Carregue um arquivo CSV contendo duas colunas: **Preço (P)** e **Quantidade (Q)**.",
    type=["csv"]
)
st.write("🔹 O arquivo deve estar no formato **CSV**, com colunas separadas por **ponto e vírgula (`;`)** e números com **vírgula (`14,50`)** como separador decimal.")

if uploaded_file is not None:
    # Ler o CSV corretamente (separador de colunas = ; e decimal = ,)
    data = pd.read_csv(uploaded_file, sep=";", decimal=",", encoding="utf-8")

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
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Preço Ótimo (E = -1)", f"{price_optimal:,.2f}".replace(".", "X").replace(",", ".").replace("X", ","))
        with col2:
            st.metric("Elasticidade-Preço da Demanda", f"{elasticity:,.2f}".replace(".", "X").replace(",", ".").replace("X", ","))
        with col3:
            st.write('<div style="font-size: 16px; font-weight: normal; display: flex; align-items: center; gap: 5px;">'
             'Qualidade do Modelo'
             '</div>', unsafe_allow_html=True)

            # Definir a pontuação do modelo de 1 a 5 baseado no R² e P-valor
            if r_squared < 0.2 or p_value > 0.05:
                score = 1
                color = "🔴"  # Vermelho
            elif r_squared < 0.4:
                score = 2
                color = "🟠"  # Laranja
            elif r_squared < 0.6:
                score = 3
                color = "🟡"  # Amarelo
            elif r_squared < 0.8:
                score = 4
                color = "🟢"  # Verde claro
            else:
                score = 5
                color = "🟢"  # Verde forte

            # Construção das bolinhas: preenchidas à esquerda e vazias à direita
            filled_circles = "".join([color] * score)  # Evita erro de multiplicação de emoji
            empty_circles = "".join(["⚪"] * (5 - score))

            # Criando um layout flexível para alinhamento
            st.write(f"""
            <div style="display: flex; align-items: center; gap: 5px; margin-top: 10px">
                {filled_circles}{empty_circles}
            </div>
            """, unsafe_allow_html=True)

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
            r2_status = "🔴 Baixo"
        elif 0.3 <= r_squared < 0.7:
            r2_status = "🟡 Regular"
        else:
            r2_status = "🟢 Excelente"

        # P-valor da correlação com ✔️ caso seja significativo
        p_status = "✅ Normal" if p_value < 0.05 else "❌ Não Normal"

        # Teste de Heteroccedasticidade (Breusch-Pagan)
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.tools import add_constant

        # Adiciona constante ao X para o teste
        X_with_const = add_constant(X)
        bp_lm, bp_p_value, _, _ = het_breuschpagan(residuals.flatten(), X_with_const)

        # Lista formatada com estatísticas
        stats_list = f"""
        - **Intercepto (α):** {intercept:,.2f}  
        - **Coeficiente Angular (β):** {slope:,.2f}  
        - **P-valor da Correlação:** {p_value:,.4f} {p_status}  
        - **Erro Absoluto Médio (MAE):** {mae:,.2f}  
        - **Erro Padrão dos Resíduos (RMSE):** {rmse:,.2f}  
        - **Coeficiente de Determinação (R²):** {r_squared:,.4f} {r2_status}  
        - **Correlação de Pearson:** {correlation:,.4f}  
        - **Média dos Resíduos:** {residuals_mean:,.2e}  
        - **Teste de Normalidade dos Resíduos (Shapiro-Wilk):**  
          **P-valor:** {shapiro_p_value:,.4f} {'✅ Significativo' if shapiro_p_value > 0.05 else '❌ Não Significativo'}  
        - **Teste de Heterocedasticidade (Breusch-Pagan):**  
          **P-valor:** {bp_p_value:,.4f} {'✅ Homocedástico' if bp_p_value > 0.05 else '❌ Heterocedástico'}
        """

        st.markdown(stats_list.replace(".", "X").replace(",", ".").replace("X", ","))

        # ======================= 🟡 4. Exibição dos Dados =======================
        st.subheader("📋 Tabela de Dados Carregados")
        st.write(data)
