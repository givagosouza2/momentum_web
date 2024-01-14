import streamlit as st


def app():
    st.markdown("<h1 style='text-align: center;color: blue;'>Atualizações do Momentum Web</h1>",
                unsafe_allow_html=True)
    # Primeiro parágrafo
    texto_markdown = """
    9 de dezembro de 2023 - Publicação da página na internet
        """
    st.markdown(
        f"<div style='text-align: justify;'>{texto_markdown}</div>", unsafe_allow_html=True)
