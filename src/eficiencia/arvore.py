"""Caminho real de um veículo numa árvore do XGBoost, para desenhar a árvore no app."""

import numpy as np
import pandas as pd


def caminho_arvore(modelo, veiculo: dict, alvo: str, indice: int = 0) -> list[dict]:
    """Nós dos níveis 0-2 da árvore `indice` mais o caminho do veículo (regra do XGBoost: x < Split vai a Yes)."""
    prep = modelo.prep
    xt = prep.transform(pd.DataFrame([veiculo]))[0]
    booster = modelo.estimadores[alvo].get_booster()
    t = booster[indice : indice + 1].trees_to_dataframe().set_index("ID")  # só a árvore pedida (ids "0-N")

    sim = {}  # nó do caminho -> o veículo foi para "Yes"? (None na folha)
    atual = "0-0"
    while True:
        r = t.loc[atual]
        if r.Feature == "Leaf":
            sim[atual] = None
            break
        # float32 como o XGBoost: em float64 o Split impresso (8 dígitos) diverge da folha em ~5 % dos veículos
        sim[atual] = bool(np.float32(xt[int(r.Feature[1:])]) < np.float32(r.Split))
        atual = r.Yes if sim[atual] else r.No

    saida, fila = [], [("0-0", None, 0)]
    while fila:
        id_, pai, nivel = fila.pop(0)
        r = t.loc[id_]
        folha = r.Feature == "Leaf"
        nome = None if folha else prep.nomes[int(r.Feature[1:])]
        categorica = nome is not None and "=" in nome
        coluna, categoria = nome.split("=", 1) if categorica else (nome, None)
        limite = None if folha or categorica else float(r.Split * prep._desvio[nome] + prep._media[nome])
        saida.append(
            {
                "id": int(id_.split("-")[1]),
                "pai": pai,
                "nivel": nivel,
                "folha": folha,
                "coluna": coluna,
                "categoria": categoria,
                "limite": limite,
                "valor": float(r.Gain) if folha else None,  # na folha o XGBoost guarda o valor em Gain
                "carros": round(r.Cover),
                "no_caminho": id_ in sim,
                "sim": sim.get(id_),
            }
        )
        if not folha:
            for filho in (r.Yes, r.No):
                if nivel + 1 <= 2 or filho in sim:
                    fila.append((filho, int(id_.split("-")[1]), nivel + 1))
    return saida
