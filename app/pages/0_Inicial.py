"""Home page of the vehicle energy efficiency estimator."""

import streamlit as st

st.title("Vehicle energy efficiency")
st.write(
    "Estimates energy consumption (MJ/km and km/l) from technical specifications, with an uncertainty range."
)

CARTOES = [
    (
        "pages/1_Veiculo.py",
        "Vehicle",
        ":material/directions_car:",
        "Estimate one vehicle and try what-if changes.",
    ),
    (
        "pages/2_Portfolio.py",
        "Portfolio",
        ":material/query_stats:",
        "Get the sales-weighted consumption of several vehicles.",
    ),
    (
        "pages/3_Comparacao.py",
        "Comparison",
        ":material/compare_arrows:",
        "Compare the candidate models from the latest training run.",
    ),
]
for coluna, (pagina, rotulo, icone, frase) in zip(st.columns(3), CARTOES, strict=True):
    with coluna, st.container(border=True):
        st.page_link(pagina, label=rotulo, icon=icone)
        st.write(frase)
