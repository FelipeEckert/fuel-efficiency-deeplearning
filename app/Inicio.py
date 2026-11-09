"""Roteador do app: navegação no topo entre as páginas."""

import streamlit as st

st.set_page_config(layout="wide", page_title="Eficiência energética")
st.navigation(
    [
        st.Page("pages/0_Inicial.py", title="Início", default=True),
        st.Page("pages/1_Veiculo.py", title="Veículo"),
        st.Page("pages/2_Portfolio.py", title="Portfólio"),
        st.Page("pages/3_Comparacao.py", title="Comparação"),
    ],
    position="top",
).run()
