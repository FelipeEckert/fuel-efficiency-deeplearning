"""Página inicial do estimador de eficiência energética de veículos."""

import streamlit as st

st.title("Eficiência energética de veículos")
st.write("Estimativa de consumo (MJ/km e km/l) a partir das especificações técnicas, com faixa de incerteza.")

CARTOES = [
    (
        "pages/1_Veiculo.py",
        "Veículo",
        ":material/directions_car:",
        "Estime o consumo de um veículo e teste o 'e se'.",
    ),
    (
        "pages/2_Portfolio.py",
        "Portfólio",
        ":material/query_stats:",
        "Veja o consumo estimado de vários veículos.",
    ),
    (
        "pages/3_Comparacao.py",
        "Comparação",
        ":material/compare_arrows:",
        "Compare os modelos candidatos do último treino.",
    ),
]
for coluna, (pagina, rotulo, icone, frase) in zip(st.columns(3), CARTOES, strict=True):
    with coluna, st.container(border=True):
        st.page_link(pagina, label=rotulo, icon=icone)
        st.write(frase)
