"""API pública de previsão: valida as entradas e devolve MJ/km, km/l e faixas de incerteza por combustível."""

from collections.abc import Mapping
from math import nan

import pandas as pd

from eficiencia.artefatos import Pacote, carregar
from eficiencia.entrada import validar
from eficiencia.esquema import ENTRADAS
from eficiencia.faixa import avisos
from eficiencia.fisica import FATOR_ELETRICO, combinar_kml, combinar_mj, combustiveis_aplicaveis, kml_diesel
from eficiencia.incerteza import aplicar

CICLOS = ("urbano", "estrada")
SAIDAS = ["mj_km", "mj_km_urbano", "mj_km_estrada"] + [
    f"kml_{c}{s}"
    for c in ("gasolina", "etanol", "eletrico_eq", "diesel")
    for s in ("", "_urbano", "_estrada")
]


def _quantil(pacote: Pacote, propulsao: str, saida: str) -> float:
    """Quantil da propulsão; se faltar, o maior dos outros; se ainda faltar, NaN (faixa vazia)."""
    if saida in pacote.quantis.get(propulsao, {}):
        return pacote.quantis[propulsao][saida]
    return max((q[saida] for q in pacote.quantis.values() if saida in q), default=nan)


def _saida(r: dict, nome: str, valor: float, faixa: tuple[float, float]) -> None:
    r[nome] = valor
    r[f"{nome}_min"], r[f"{nome}_max"] = faixa


def _registro(pacote: Pacote, veiculo: dict, previsto: dict) -> dict:
    p = veiculo["propulsao"]
    aplicaveis = combustiveis_aplicaveis(p, veiculo["combustivel"])
    r: dict = {"avisos": avisos(veiculo, pacote.faixas)}

    mj = {c: float(previsto[f"mj_km_{c}"]) for c in CICLOS}
    faixa_mj = {c: aplicar(mj[c], _quantil(pacote, p, f"mj_km_{c}")) for c in CICLOS}
    for c in CICLOS:
        _saida(r, f"mj_km_{c}", mj[c], faixa_mj[c])
    mj_total = combinar_mj(mj["urbano"], mj["estrada"])
    _saida(r, "mj_km", mj_total, aplicar(mj_total, _quantil(pacote, p, "mj_km")))

    # gasolina e etanol: previstos pelo modelo, só onde o combustível se aplica
    for comb in ("gasolina", "etanol"):
        km = {}
        for c in CICLOS:
            nome = f"kml_{comb}_{c}"
            km[c] = float(previsto[nome]) if comb in aplicaveis else nan
            _saida(r, nome, km[c], aplicar(km[c], _quantil(pacote, p, nome)))
        total = combinar_kml(km["urbano"], km["estrada"])
        _saida(r, f"kml_{comb}", total, aplicar(total, _quantil(pacote, p, f"kml_{comb}")))

    # eletrico_eq = kml_diesel com FATOR_ELETRICO; faixa do km/l = fator / faixa do MJ/km (inverte mín. e máx.)
    for comb, (fator_u, fator_e) in (
        ("eletrico_eq", (FATOR_ELETRICO, FATOR_ELETRICO)),
        ("diesel", pacote.fatores_diesel),
    ):
        ativo = comb in aplicaveis
        km = {}
        for c, fator in zip(CICLOS, (fator_u, fator_e)):
            valor = kml_diesel(mj[c], fator) if ativo else nan
            minimo, maximo = faixa_mj[c]
            faixa = (kml_diesel(maximo, fator), kml_diesel(minimo, fator)) if ativo else (nan, nan)
            _saida(r, f"kml_{comb}_{c}", valor, faixa)
            km[c] = (valor, faixa)
        (v_u, (min_u, max_u)), (v_e, (min_e, max_e)) = km["urbano"], km["estrada"]
        _saida(
            r, f"kml_{comb}", combinar_kml(v_u, v_e), (combinar_kml(min_u, min_e), combinar_kml(max_u, max_e))
        )
    return r


def prever(veiculos: pd.DataFrame | Mapping, pacote: Pacote | None = None) -> pd.DataFrame:
    """Previsões por veículo (uma linha cada), com _min e _max por saída e a coluna avisos.

    Levanta ValueError com todas as mensagens de validação se alguma linha for inválida.
    """
    if pacote is None:
        pacote = carregar()
    if isinstance(veiculos, Mapping):
        veiculos = pd.DataFrame([veiculos])
    ok, erros = validar(veiculos)
    if erros:
        raise ValueError("\n".join(erros))
    colunas = [c for s in SAIDAS for c in (s, f"{s}_min", f"{s}_max")] + ["avisos"]
    if ok.empty:
        return pd.DataFrame(columns=colunas)
    previstos = pacote.modelo.predict(ok[ENTRADAS])
    registros = [
        _registro(pacote, veiculo, previsto)
        for veiculo, previsto in zip(ok.to_dict("records"), previstos.to_dict("records"))
    ]
    return pd.DataFrame(registros, index=ok.index, columns=colunas)
