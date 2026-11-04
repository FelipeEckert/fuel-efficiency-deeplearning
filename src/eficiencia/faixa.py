"""Faixas observadas na base por propulsão e avisos para valores fora delas."""

from collections.abc import Mapping

import pandas as pd

from eficiencia.esquema import CATEGORICAS, NUMERICAS, ROTULOS
from eficiencia.formato import numero


def calcular_faixas(df: pd.DataFrame) -> dict:
    numericas = [c for c in NUMERICAS if c != "bateria_estimada"]
    faixas: dict = {"numericas": {}, "categoricas": {}}
    for propulsao, grupo in df.groupby("propulsao"):
        faixas["numericas"][propulsao] = {
            c: (float(grupo[c].min()), float(grupo[c].max())) for c in numericas if grupo[c].notna().any()
        }
        faixas["categoricas"][propulsao] = {
            c: set(grupo[c].dropna()) for c in CATEGORICAS if grupo[c].notna().any()
        }
    return faixas


def avisos(veiculo: Mapping, faixas: dict) -> list[str]:
    propulsao = veiculo.get("propulsao")
    numericas = faixas["numericas"].get(propulsao, {})
    categoricas = faixas["categoricas"].get(propulsao, {})
    mensagens: list[str] = []
    for coluna in NUMERICAS + CATEGORICAS:  # numéricas primeiro: ordem exigida pelo teste da tarefa 7
        valor = veiculo.get(coluna)
        if valor is None or pd.isna(valor):
            continue
        rotulo = ROTULOS[coluna]
        if coluna in numericas:
            minimo, maximo = numericas[coluna]
            if valor < minimo:
                mensagens.append(
                    f"{rotulo} = {numero(valor)} abaixo de qualquer {propulsao} da base (mínimo {numero(minimo)})"
                )
            elif valor > maximo:
                mensagens.append(
                    f"{rotulo} = {numero(valor)} acima de qualquer {propulsao} da base (máximo {numero(maximo)})"
                )
        elif coluna in categoricas and coluna != "propulsao" and valor not in categoricas[coluna]:
            mensagens.append(f"{rotulo} = {valor} nunca vista em {propulsao} na base")
    return mensagens
