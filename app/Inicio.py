"""App router: top navigation between the pages."""

import streamlit as st

st.set_page_config(layout="wide", page_title="Vehicle energy efficiency")
st.navigation(
    [
        st.Page("pages/0_Inicial.py", title="Home", default=True),
        st.Page("pages/1_Veiculo.py", title="Vehicle"),
        st.Page("pages/2_Portfolio.py", title="Portfolio"),
        st.Page("pages/3_Comparacao.py", title="Comparison"),
    ],
    position="top",
).run()
