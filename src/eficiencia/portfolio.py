"""Vehicle portfolio: sales-weighted average MJ/km, using label values where known and model estimates otherwise."""

import io
import re
from dataclasses import dataclass
from math import nan
from typing import BinaryIO

import numpy as np
import pandas as pd

from eficiencia.artefatos import Pacote, carregar
from eficiencia.entrada import _CANONICOS, _campo, _chave, _numero, _vazio, validar
from eficiencia.esquema import CATEGORICAS, ENTRADAS, ROTULOS
from eficiencia.formato import numero
from eficiencia.prever import prever

COLUNAS_PORTFOLIO = ["nome", "volume", "mj_km_conhecido"] + ENTRADAS
ROTULOS_PORTFOLIO = {"nome": "Name", "volume": "Volume", "mj_km_conhecido": "Known MJ/km", **ROTULOS}
OBRIGATORIAS = ("nome", "volume")
NAO_INFORMADO = "not given"
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
_MILHARES_PT = re.compile(r"\d{1,3}(\.\d{3})+(,\d+)?")  # "10.000" e "1.000,5"; "1.5" segue decimal
_MILHARES_EN = re.compile(r"\d{1,3}(,\d{3})+(\.\d+)?")  # "10,000" e "1,000.5"; "1,5" segue decimal


def _volume(valor) -> float | None:
    """Volume: thousands separators (10.000 or 10,000) are dropped; the rest follows _numero."""
    if isinstance(valor, str):
        texto = valor.strip()
        if _MILHARES_PT.fullmatch(texto):
            valor = texto.replace(".", "")
        elif _MILHARES_EN.fullmatch(texto):
            valor = texto.replace(",", "")
    return _numero(valor)


def volume_texto(valor: float) -> str:
    """Texto do volume para o editor, lido de volta pelo mesmo número: 1.234 vira "1.2340", nunca mil e poucos."""
    texto = numero(valor)
    return texto + "0" if _MILHARES_PT.fullmatch(texto) else texto


@dataclass
class ResultadoPortfolio:
    tabela: pd.DataFrame
    media_mj_km: float
    por_propulsao: pd.DataFrame
    por_categoria: pd.DataFrame
    erros: list[str]
    n_incluidos: int
    n_total: int
    volume_incluido: float
    volume_total: float
    media_min: float = nan  # faixa de 90 % da média: NaN sem erros_oof ou sem linhas estimadas
    media_max: float = nan


def modelo_planilha() -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(columns=[ROTULOS_PORTFOLIO[c] for c in COLUNAS_PORTFOLIO]).to_excel(buf, index=False)
    return buf.getvalue()


def ler_planilha(arquivo: bytes | BinaryIO) -> tuple[pd.DataFrame, list[str]]:
    """Lê a planilha pelo rótulo ou pelo nome da coluna; colunas opcionais ausentes viram vazias."""
    try:
        # dtype=object: sem isso o pandas lê o texto "10.000" como 10 e "0012" como 12
        bruto = pd.read_excel(io.BytesIO(arquivo) if isinstance(arquivo, bytes) else arquivo, dtype=object)
    except Exception as erro:  # noqa: BLE001 - qualquer falha de leitura vira mensagem, nunca exceção
        return pd.DataFrame(columns=COLUNAS_PORTFOLIO), [
            f"could not read the spreadsheet: {type(erro).__name__}"
        ]
    origem: dict[str, object] = {}
    erros = []
    for coluna in bruto.columns:
        canonico = _CABECALHOS.get(_chave(str(coluna).strip()))
        if canonico in origem:
            erros.append(
                f'columns "{origem[canonico]}" and "{coluna}" are both {ROTULOS_PORTFOLIO[canonico]}'
            )
        elif canonico is not None:
            origem[canonico] = coluna
    erros += [f"required column missing: {ROTULOS_PORTFOLIO[c]}" for c in OBRIGATORIAS if c not in origem]
    if erros:
        return pd.DataFrame(columns=COLUNAS_PORTFOLIO), erros
    df = pd.DataFrame(
        {c: bruto[origem[c]] if c in origem else np.nan for c in COLUNAS_PORTFOLIO}, index=bruto.index
    )
    outras = [
        c for c in COLUNAS_PORTFOLIO if c not in ("nome", "volume")
    ]  # nome e volume ficam como na célula
    df[outras] = df[outras].infer_objects()
    for coluna in CATEGORICAS:  # "Hibrido" -> "Híbrido": o editor só mostra as opções canônicas
        df[coluna] = df[coluna].map(
            lambda v, c=coluna: _CANONICOS[c].get(_chave(v.strip()), v) if isinstance(v, str) else v
        )
    return df, []


def _quebra(tabela: pd.DataFrame, coluna: str) -> pd.DataFrame:
    pesos = pd.DataFrame(
        {coluna: tabela[coluna], "volume": tabela.volume, "pv": tabela.volume * tabela.mj_km}
    )
    g = pesos.groupby(coluna, as_index=False).sum()
    g["media_mj_km"] = g.pv / g.volume
    return g[[coluna, "volume", "media_mj_km"]]


def totais_portfolio(df: pd.DataFrame) -> tuple[int, float]:
    """Número de veículos e volume da tabela inteira; volume inválido conta como zero."""
    volumes = [_volume(v) for v in df["volume"]]
    return len(df), float(sum(v for v in volumes if v is not None and v > 0))


def _faixa_media(tabela: pd.DataFrame, pacote: Pacote | None) -> tuple[float, float]:
    """Faixa de 90 % da média ponderada: reamostra os erros fora da amostra das linhas estimadas."""
    est = (tabela.origem == "estimado").to_numpy()
    if not est.any():
        media = float((tabela.volume * tabela.mj_km).sum() / tabela.volume.sum())
        return media, media
    erros = getattr(pacote, "erros_oof", None)
    todos = np.concatenate([np.asarray(v, float) for v in (erros or {}).values()] or [np.empty(0)])
    if todos.size == 0:  # sem erros fora da amostra (ausentes ou listas vazias): não há faixa
        return nan, nan
    rng = np.random.default_rng(0)
    n = 2000
    mj = np.tile(tabela.mj_km.to_numpy(), (n, 1))
    # linhas com a mesma previsão são o mesmo carro: um sorteio só, senão dividir uma linha estreita a faixa.
    # shortcut: carros diferentes sorteiam independente; erros de versões irmãs correlacionados alargariam a
    # faixa. Com mais dados, sortear por grupo (modelo/marca).
    chaves = list(zip(tabela.propulsao, tabela.mj_km))
    for chave in sorted({chaves[j] for j in np.flatnonzero(est)}):  # ordenado: a ordem das linhas não importa
        fonte = np.asarray(erros.get(chave[0], todos), float)
        sorteio = 1 + rng.choice(fonte if fonte.size else todos, n)
        for j in np.flatnonzero(est):
            if chaves[j] == chave:
                mj[:, j] *= sorteio
    medias = mj @ tabela.volume.to_numpy() / tabela.volume.sum()
    return tuple(float(x) for x in np.percentile(medias, [5, 95]))


def media_portfolio(df: pd.DataFrame, pacote: Pacote | None = None) -> ResultadoPortfolio:
    """Média ponderada por volume. Linhas com MJ/km conhecido usam o valor real; as demais são estimadas."""
    erros: list[str] = []
    linhas: dict[int, dict] = {}  # posição (1-based) na tabela -> linha do resultado
    a_estimar: list[pd.DataFrame] = []
    tem_volume = False
    for pos, bruto in enumerate(df.to_dict("records"), start=1):
        volume = _volume(bruto.get("volume"))
        if volume is None or volume <= 0:
            erros.append(f"row {pos}: volume must be greater than zero")
            continue
        tem_volume = True
        base = {"nome": bruto.get("nome"), "volume": volume}
        problemas_spec: list[str] = []  # especificações opcionais: só valor presente e inválido é reportado
        for coluna in ("propulsao", "categoria"):
            valor, problema = _campo(coluna, bruto.get(coluna), False)
            base[coluna] = valor or NAO_INFORMADO
            if problema:
                problemas_spec.append(f"row {pos}: {ROTULOS[coluna]} {problema}")
        if not _vazio(bruto.get("mj_km_conhecido")):
            real = _numero(bruto.get("mj_km_conhecido"))
            if real is None or real <= 0:
                erros.append(f"row {pos}: known MJ/km must be a number greater than zero")
                continue
            erros += problemas_spec
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
        erros.append("no vehicle with a valid volume")
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
    n_total, volume_total = totais_portfolio(df)
    media_min, media_max = _faixa_media(tabela, pacote) if total > 0 else (nan, nan)
    return ResultadoPortfolio(
        tabela=tabela,
        media_mj_km=media,
        por_propulsao=_quebra(tabela, "propulsao"),
        por_categoria=_quebra(tabela, "categoria"),
        erros=erros,
        n_incluidos=len(tabela),
        n_total=n_total,
        volume_incluido=float(total),
        volume_total=volume_total,
        media_min=media_min,
        media_max=media_max,
    )
