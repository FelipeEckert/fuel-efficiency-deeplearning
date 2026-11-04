"""Página inicial do estimador de eficiência energética de veículos."""

import streamlit as st

st.set_page_config(page_title="Eficiência energética", layout="wide")
st.title("Eficiência energética de veículos")
st.write("Estimativa de consumo (MJ/km e km/l) a partir das especificações técnicas, com faixa de incerteza.")
st.page_link(
    "pages/1_Veiculo.py", label="Veículo: estimativa, cenários e 'e se'", icon=":material/directions_car:"
)
