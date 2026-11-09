"""Input validation and normalization: one error message per problem and row, never an exception or a silently wrong number."""

import math
import unicodedata

import numpy as np
import pandas as pd

from eficiencia.esquema import CATEGORICAS, ENTRADAS, NOMES_VALORES, ROTULOS, VALORES, nome_valor
from eficiencia.formato import numero

LIMITES_FISICOS = {
    "massa_om_kg": (300, 10000),
    "classe_inercia_kg": (300, 10000),
    "pbt_kg": (300, 15000),
    "f0_n": (1, 2000),
    "f2_n_kmh2": (0.001, 1),
    "potencia_kw": (1, 2000),
    "cilindrada_cm3": (0, 10000),
    "n_marchas": (0, 12),
    "bateria_kwh": (0, 300),
}
OPCIONAIS = ("n_marchas", "bateria_estimada")  # bateria_kwh é obrigatória só fora de Combustão


def _chave(texto: str) -> str:
    """Forma comparável: sem acento e sem diferença de maiúsculas."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().casefold()


# extra spellings accepted in uploaded tables, besides the PBEV codes and their English names
SINONIMOS = {
    "propulsao": {
        "ice": "Combustão",
        "ev": "Elétrico",
        "bev": "Elétrico",
        "hev": "Híbrido",
        "phev": "Plug-in",
    },
    "combustivel": {"petrol": "gasolina", "electric": "eletrico", "flex": "flex"},
    "transmissao": {"automatic": "automatica", "dual clutch": "dct", "direct": "direta"},
}

_CANONICOS = {
    c: {_chave(v): v for v in VALORES[c]}
    | {_chave(nome): v for v, nome in NOMES_VALORES.get(c, {}).items()}
    | {_chave(apelido): v for apelido, v in SINONIMOS.get(c, {}).items()}
    for c in CATEGORICAS
}


def _vazio(valor) -> bool:
    if isinstance(valor, str):
        return not valor.strip()
    return valor is None or (pd.api.types.is_scalar(valor) and bool(pd.isna(valor)))


def _numero(valor) -> float | None:
    if isinstance(valor, (bool, np.bool_)):  # True viraria 1.0 sem ser um número digitado
        return None
    if isinstance(valor, str):
        valor = valor.strip().replace(",", ".")
    try:
        x = float(valor)
    except TypeError, ValueError:
        return None
    return x if math.isfinite(x) else None


def _campo(coluna: str, valor, obrigatorio: bool) -> tuple[object, str | None]:
    """Valor normalizado e o problema (sem o prefixo de linha), se houver."""
    if _vazio(valor):
        if obrigatorio:
            return None, "is required"
        return (None if coluna in CATEGORICAS else math.nan), None
    if coluna in CATEGORICAS:
        texto = str(valor).strip()
        canonico = _CANONICOS[coluna].get(_chave(texto))
        return canonico, None if canonico is not None else f"is invalid: {texto}"
    x = _numero(valor)
    if x is None:
        return math.nan, "is not a number"
    if coluna in LIMITES_FISICOS:
        minimo, maximo = LIMITES_FISICOS[coluna]
        if not minimo <= x <= maximo:
            return x, f"is outside the physical limits ({numero(minimo)} to {numero(maximo)})"
    return x, None


def validar(df: pd.DataFrame, primeira_linha: int = 1) -> tuple[pd.DataFrame, list[str]]:
    """Valid rows (normalized, original index) and every error message."""
    erros: list[str] = []
    linhas: list[dict] = []
    indices: list = []
    for pos, (indice, bruto) in enumerate(zip(df.index, df.to_dict("records"))):
        n = primeira_linha + pos
        problemas: list[str] = []
        limpo: dict = {}
        for coluna in ENTRADAS:
            if coluna == "bateria_estimada":  # sempre falsa: a bateria é informada, nunca estimada aqui
                continue
            if coluna == "bateria_kwh":
                obrigatorio = limpo.get("propulsao") not in (None, "Combustão")
            elif coluna == "cilindrada_cm3":
                obrigatorio = limpo.get("propulsao") not in (None, "Elétrico")
            else:
                obrigatorio = coluna not in OPCIONAIS
            limpo[coluna], problema = _campo(coluna, bruto.get(coluna), obrigatorio)
            if problema:
                problemas.append(f"{ROTULOS[coluna]} {problema}")
        prop, comb = limpo.get("propulsao"), limpo.get("combustivel")
        if prop == "Elétrico" and comb is not None and comb != "eletrico":
            problemas.append("Fuel must be Electric for an electric powertrain")
        if comb == "eletrico" and prop is not None and prop != "Elétrico":
            problemas.append("Powertrain must be Electric when the fuel is Electric")
        # mínimos por propulsão; valor ausente já virou "obrigatório" (NaN ou None, não float positivo)
        for campo, propulsoes in (
            ("cilindrada_cm3", ("Combustão", "Híbrido", "Plug-in")),
            ("bateria_kwh", ("Híbrido", "Plug-in", "Elétrico")),
        ):
            valor = limpo.get(campo)
            if prop in propulsoes and isinstance(valor, float) and not math.isnan(valor) and valor <= 0:
                problemas.append(
                    f"{ROTULOS[campo]} must be greater than zero for a {nome_valor('propulsao', prop)} powertrain"
                )
        if prop == "Combustão":
            limpo["bateria_kwh"] = 0.0
        if prop == "Elétrico":
            limpo["cilindrada_cm3"] = 0.0
        limpo["bateria_estimada"] = False
        erros += [f"row {n}: {p}" for p in problemas]
        if not problemas:
            linhas.append(limpo)
            indices.append(indice)
    return pd.DataFrame(linhas, index=indices, columns=ENTRADAS), erros
