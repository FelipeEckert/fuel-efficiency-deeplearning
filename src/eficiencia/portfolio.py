"""Portfólio de veículos: MJ/km médio ponderado pelo volume, com valores reais ou estimados pelo modelo."""

import io
import zipfile
from dataclasses import dataclass
from math import nan
from typing import BinaryIO

import numpy as np
import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException

from eficiencia.artefatos import Pacote, carregar
from eficiencia.entrada import _campo, _chave, _numero, _vazio, validar
from eficiencia.esquema import ENTRADAS, ROTULOS
from eficiencia.prever import prever

COLUNAS_PORTFOLIO = ["nome", "volume", "mj_km_conhecido"] + [c for c in ENTRADAS if c != "bateria_estimada"]
ROTULOS_PORTFOLIO = {"nome": "Nome", "volume": "Volume", "mj_km_conhecido": "MJ/km conhecido", **ROTULOS}
OBRIGATORIAS = ("nome", "volume")
NAO_INFORMADO = "não informado"
_CABECALHOS = {_chave(t): c for c in COLUNAS_PORTFOLIO for t in (c, ROTULOS_PORTFOLIO[c])}
_COLUNAS_TABELA = [
    "nome",
    "volume",
    "mj_km",
    "origem",
    "mj_km_min",
    "mj_km_max",
    "avisos",
    "propulsao",
    "categoria",
]


@dataclass
class ResultadoPortfolio:
    tabela: pd.DataFrame
    media_mj_km: float
    por_propulsao: pd.DataFrame
    por_categoria: pd.DataFrame
    erros: list[str]


def modelo_planilha() -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(columns=[ROTULOS_PORTFOLIO[c] for c in COLUNAS_PORTFOLIO]).to_excel(buf, index=False)
    return buf.getvalue()


def ler_planilha(arquivo: bytes | BinaryIO) -> tuple[pd.DataFrame, list[str]]:
    """Lê a planilha pelo rótulo ou pelo nome da coluna; colunas opcionais ausentes viram vazias."""
    try:
        bruto = pd.read_excel(io.BytesIO(arquivo) if isinstance(arquivo, bytes) else arquivo)
    except ValueError, zipfile.BadZipFile, InvalidFileException:
        return pd.DataFrame(columns=COLUNAS_PORTFOLIO), ["arquivo não é uma planilha Excel (.xlsx) válida"]
    origem: dict[str, object] = {}
    for coluna in bruto.columns:
        canonico = _CABECALHOS.get(_chave(str(coluna)))
        if canonico is not None and canonico not in origem:
            origem[canonico] = coluna
    erros = [f"coluna obrigatória ausente: {ROTULOS_PORTFOLIO[c]}" for c in OBRIGATORIAS if c not in origem]
    if erros:
        return pd.DataFrame(columns=COLUNAS_PORTFOLIO), erros
    df = pd.DataFrame(
        {c: bruto[origem[c]] if c in origem else np.nan for c in COLUNAS_PORTFOLIO}, index=bruto.index
    )
    return df, []


def _quebra(tabela: pd.DataFrame, coluna: str) -> pd.DataFrame:
    pesos = pd.DataFrame(
        {coluna: tabela[coluna], "volume": tabela.volume, "pv": tabela.volume * tabela.mj_km}
    )
    g = pesos.groupby(coluna, as_index=False).sum()
    g["media_mj_km"] = g.pv / g.volume
    return g[[coluna, "volume", "media_mj_km"]]


def media_portfolio(df: pd.DataFrame, pacote: Pacote | None = None) -> ResultadoPortfolio:
    """Média ponderada por volume. Linhas com MJ/km conhecido usam o valor real; as demais são estimadas."""
    erros: list[str] = []
    linhas: dict[int, dict] = {}  # posição (1-based) na tabela -> linha do resultado
    a_estimar: list[pd.DataFrame] = []
    tem_volume = False
    for pos, bruto in enumerate(df.to_dict("records"), start=1):
        volume = _numero(bruto.get("volume"))
        if volume is None or volume <= 0:
            erros.append(f"linha {pos}: volume deve ser maior que zero")
            continue
        tem_volume = True
        base = {
            "nome": bruto.get("nome"),
            "volume": volume,
            "propulsao": _campo("propulsao", bruto.get("propulsao"), False)[0] or NAO_INFORMADO,
            "categoria": _campo("categoria", bruto.get("categoria"), False)[0] or NAO_INFORMADO,
        }
        if not _vazio(bruto.get("mj_km_conhecido")):
            real = _numero(bruto.get("mj_km_conhecido"))
            if real is None or real <= 0:
                erros.append(f"linha {pos}: MJ/km conhecido deve ser número maior que zero")
                continue
            linhas[pos] = {
                **base,
                "mj_km": real,
                "origem": "real",
                "mj_km_min": real,
                "mj_km_max": real,
                "avisos": [],
            }
            continue
        ok, erros_linha = validar(pd.DataFrame([bruto], index=[pos]), primeira_linha=pos)
        if erros_linha:
            erros += erros_linha
            continue
        a_estimar.append(ok)
        linhas[pos] = base
    if not tem_volume:
        erros.append("nenhum veículo com volume válido")
    if a_estimar:
        if pacote is None:
            pacote = carregar()
        previstos = prever(pd.concat(a_estimar), pacote)
        for pos, previsto in previstos.iterrows():
            linhas[pos] |= {
                "mj_km": previsto["mj_km"],
                "origem": "estimado",
                "mj_km_min": previsto["mj_km_min"],
                "mj_km_max": previsto["mj_km_max"],
                "avisos": previsto["avisos"],
            }
    tabela = pd.DataFrame([linhas[p] for p in sorted(linhas)], columns=_COLUNAS_TABELA)
    tabela = tabela.astype({"volume": float, "mj_km": float, "mj_km_min": float, "mj_km_max": float})
    total = tabela.volume.sum()
    media = float((tabela.volume * tabela.mj_km).sum() / total) if total > 0 else nan
    return ResultadoPortfolio(
        tabela=tabela,
        media_mj_km=media,
        por_propulsao=_quebra(tabela, "propulsao"),
        por_categoria=_quebra(tabela, "categoria"),
        erros=erros,
    )
