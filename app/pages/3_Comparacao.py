"""Comparison page: candidates from the latest training run, with error by powertrain, monotonicity and the report figures."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pasta app/, para importar visual

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from visual import CORES, NOMES

from eficiencia.dados import RAIZ
from eficiencia.esquema import nome_valor
from eficiencia.formato import numero

ORDEM_PROPULSAO = ["Combustão", "Híbrido", "Plug-in", "Elétrico"]  # as demais vêm depois, em ordem alfabética
FIGURAS = {
    "erro_por_propulsao.png": "Error by powertrain",
    "previsto_real.png": "Predicted vs. label MJ/km for the selected model",
}
MAX_VIOLACOES = 10
MSG_CORROMPIDO = (
    "The training report is incomplete or corrupted. Run the training again: uv run eficiencia-treinar"
)


def _carregar(arquivo: Path) -> dict:
    """Métricas do treino; se o arquivo estiver corrompido ou incompleto (treino interrompido), mostra erro e para."""
    try:
        metricas = json.loads(arquivo.read_text(encoding="utf-8"))
        candidatos = metricas["candidatos"]
        ok = (
            isinstance(candidatos, dict)
            and len(candidatos) > 0
            and (metricas["vencedor"] is None or metricas["vencedor"] in candidatos)
            and all(
                isinstance(c, dict) and isinstance(c.get("erro_mj"), (int, float))
                for c in candidatos.values()
            )
        )
    except OSError, ValueError, KeyError, TypeError:  # ValueError inclui JSONDecodeError
        ok = False
    if not ok:
        st.error(MSG_CORROMPIDO)
        st.stop()
    return metricas


def _propulsoes(candidatos: dict) -> list[str]:
    """União das propulsões de todos os candidatos, na ordem de ORDEM_PROPULSAO."""
    existentes = {p for c in candidatos.values() for p in c.get("erro_por_propulsao", {})}
    return sorted(
        existentes,
        key=lambda p: (ORDEM_PROPULSAO.index(p) if p in ORDEM_PROPULSAO else len(ORDEM_PROPULSAO), p),
    )


def _tabela(candidatos: dict, vencedor: str) -> pd.DataFrame:
    propulsoes = _propulsoes(candidatos)
    rotulos = {"aprovado": "✅ pass", "reprovado": "❌ fail"}
    linhas = []
    for nome, c in sorted(candidatos.items(), key=lambda item: item[1]["erro_mj"]):
        modelo = NOMES.get(nome, nome)
        linha = {
            "Model": f"🏆 {modelo}" if nome == vencedor else modelo,
            "Overall error (%)": numero(100 * c["erro_mj"], 1),
        }
        erros = c.get("erro_por_propulsao", {})
        linha.update(
            {
                f"{nome_valor('propulsao', p)} (%)": numero(100 * erros[p], 1) if p in erros else "—"
                for p in propulsoes
            }
        )
        linha["Monotonicity"] = rotulos.get(c.get("monotonia"), "—")
        linha["Trials"] = str(c.get("n_tentativas", "—"))
        linhas.append(linha)
    return pd.DataFrame(linhas)


st.set_page_config(page_title="Comparison", layout="wide")
st.title("Model comparison")

# Lido a cada execução: um treino novo aparece após recarregar a página.
pasta = Path(os.environ.get("EFICIENCIA_RELATORIOS") or RAIZ / "relatorios")
arquivo = pasta / "metricas.json"
if not arquivo.exists():
    st.info("No training run found. Run: uv run eficiencia-treinar")
    st.stop()

metricas = _carregar(arquivo)
candidatos, vencedor = metricas["candidatos"], metricas["vencedor"]

ordem = sorted(
    candidatos, key=lambda n: candidatos[n]["erro_mj"], reverse=True
)  # menor erro no topo do gráfico
fig = go.Figure(
    go.Bar(
        x=[100 * candidatos[n]["erro_mj"] for n in ordem],
        y=[NOMES.get(n, n) for n in ordem],
        orientation="h",
        marker_color=[CORES["primaria"] if n == vencedor else "#C4B5FD" for n in ordem],
        texttemplate="%{x:.1f} %",
        hovertemplate="%{y}: %{x:.1f} %<extra></extra>",
    )
)
fig.update_layout(xaxis_title="Overall error (%)", height=80 + 50 * len(ordem), margin={"t": 30})
st.plotly_chart(fig)
with st.container(border=True):
    if vencedor:
        st.success(
            f"Selected: {NOMES.get(vencedor, vencedor)}, {numero(100 * candidatos[vencedor]['erro_mj'], 1)} % mean error on MJ/km"
        )
    else:  # treino sem candidato aprovado: relatório e métricas existem, mas nenhum pacote foi salvo
        st.warning("No model passed the monotonicity check in this training run.")
    st.subheader("Candidates")
    st.dataframe(_tabela(candidatos, vencedor), hide_index=True)
st.caption(
    "Errors come from nested cross-validation grouped by model family: each vehicle is predicted by a model "
    "that did not see it in training. The overall error is the mean percentage error on combined MJ/km "
    "(city and highway). Only models that pass the monotonicity check can be selected."
)

for nome, c in candidatos.items():
    violacoes = c.get("violacoes", [])
    if c.get("monotonia") != "aprovado" and violacoes:
        with st.expander(f"{NOMES.get(nome, nome)}: monotonicity violations"):
            st.text("\n".join(violacoes[:MAX_VIOLACOES]))
            if len(violacoes) > MAX_VIOLACOES:
                st.write(f"and {len(violacoes) - MAX_VIOLACOES} more")

figuras = [
    (pasta / "figuras" / arq, titulo) for arq, titulo in FIGURAS.items() if (pasta / "figuras" / arq).exists()
]
if figuras:
    st.subheader("Figures")
    for caminho, titulo in figuras:
        st.image(str(caminho), caption=titulo)
