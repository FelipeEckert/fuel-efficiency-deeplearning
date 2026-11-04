"""Página Comparação: candidatos do último treino, com erro por propulsão, monotonia e figuras do relatório."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from eficiencia.treino import PASTA_RELATORIOS

NOMES = {
    "linear": "Regressão linear",
    "floresta": "Random Forest",
    "xgboost": "XGBoost",
    "rede_por_alvo": "Rede neural por alvo",
    "rede_multi": "Rede neural multi-saída",
}
PROPULSOES = {
    "gasolina": "Gasolina",
    "etanol": "Etanol",
    "eletrico_eq": "Elétrico equivalente",
    "diesel": "Diesel",
}
FIGURAS = {"erro_por_propulsao.png": "Erro por propulsão", "previsto_real.png": "Previsto × real do vencedor"}
MAX_VIOLACOES = 10


def _pct(valor: float) -> str:
    return f"{100 * valor:.1f}".replace(".", ",")


def _tabela(candidatos: dict, vencedor: str) -> pd.DataFrame:
    propulsoes = list(next(iter(candidatos.values()))["erro_por_propulsao"])
    linhas = []
    for nome, c in sorted(candidatos.items(), key=lambda item: item[1]["erro_mj"]):
        modelo = NOMES.get(nome, nome)
        linha = {
            "Modelo": f"🏆 {modelo}" if nome == vencedor else modelo,
            "Erro geral (%)": _pct(c["erro_mj"]),
        }
        linha.update({f"{PROPULSOES.get(p, p)} (%)": _pct(c["erro_por_propulsao"][p]) for p in propulsoes})
        linha["Monotonia"] = "✅ aprovado" if c["monotonia"] == "aprovado" else "❌ reprovado"
        linha["Tentativas"] = c["n_tentativas"]
        linhas.append(linha)
    return pd.DataFrame(linhas)


st.set_page_config(page_title="Comparação", layout="wide")
st.title("Comparação dos candidatos")

# Lido a cada execução: um treino novo aparece após recarregar a página.
pasta = Path(os.environ.get("EFICIENCIA_RELATORIOS") or PASTA_RELATORIOS)
arquivo = pasta / "metricas.json"
if not arquivo.exists():
    st.info("Nenhum treino encontrado. Rode: uv run eficiencia-treinar")
    st.stop()

metricas = json.loads(arquivo.read_text(encoding="utf-8"))
candidatos, vencedor = metricas["candidatos"], metricas["vencedor"]

st.success(
    f"Vencedor: {NOMES.get(vencedor, vencedor)} — {_pct(candidatos[vencedor]['erro_mj'])} % de erro médio no MJ/km"
)
st.subheader("Candidatos")
st.dataframe(_tabela(candidatos, vencedor), hide_index=True)
st.caption(
    "Os erros vêm de validação cruzada aninhada por grupo de modelo: cada veículo é previsto por um modelo "
    "que não o viu no treino. O erro geral é o percentual médio no MJ/km combinado (urbano e estrada). "
    "Só modelos aprovados no teste de monotonia podem vencer."
)

for nome, c in candidatos.items():
    if c["monotonia"] == "reprovado":
        with st.expander(f"{NOMES.get(nome, nome)}: reprovado na monotonia"):
            violacoes = c["violacoes"]
            st.text("\n".join(violacoes[:MAX_VIOLACOES]))
            if len(violacoes) > MAX_VIOLACOES:
                st.write(f"e mais {len(violacoes) - MAX_VIOLACOES}")

figuras = [
    (pasta / "figuras" / arq, titulo) for arq, titulo in FIGURAS.items() if (pasta / "figuras" / arq).exists()
]
if figuras:
    st.subheader("Figuras")
    for caminho, titulo in figuras:
        st.image(str(caminho), caption=titulo)
