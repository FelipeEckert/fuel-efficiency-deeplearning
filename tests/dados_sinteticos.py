"""Synthetic stand-in for the clean PBEV table, so the test suite and CI run without the real data.

The real spreadsheets stay on the developer's machine (see data/README.md). This module builds a
table with the same columns, types and value ranges. Consumption comes from a simple road-load
model, so it rises with mass, F0 and F2 by construction and the monotonicity checks hold.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROPULSOES = {"Combustão": 0.56, "Elétrico": 0.21, "Híbrido": 0.12, "Plug-in": 0.11}
CATEGORIAS = ["EXG", "SUVC", "M", "PIC", "G", "E", "SUVG", "CL", "SUVG4", "FEG", "C", "PC", "SC"]
MARCAS = ["ALFA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL", "INDIA", "JULIET"]

# energy content per liter (MJ/l) used to turn MJ/km into km/l
MJ_POR_LITRO = {"gasolina": 30.0, "etanol": 21.3, "diesel": 35.9}
KML_ELETRICO_EQ = 20.29  # km/l equivalent = 20.29 / MJ/km, as on the PBEV label

# share of road-load energy that reaches the wheels, per powertrain
EFICIENCIA = {"Combustão": 0.24, "Híbrido": 0.33, "Plug-in": 0.45, "Elétrico": 0.82}


def gerar(n: int = 420, semente: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    propulsao = rng.choice(list(PROPULSOES), size=n, p=list(PROPULSOES.values()))
    propulsao[0] = "Combustão"  # tests use the first row as a typical gasoline car, like the real table
    linhas = []
    for i, prop in enumerate(propulsao):
        grupo = i // 2  # two homologations per model group, like body or trim variants
        marca = MARCAS[grupo % len(MARCAS)]
        categoria = "G" if i == 0 else CATEGORIAS[rng.integers(len(CATEGORIAS))]
        picape = categoria in ("PIC", "CL")

        if prop == "Elétrico":
            combustivel, bateria = "eletrico", float(rng.uniform(40, 100))
        elif prop == "Plug-in":
            combustivel, bateria = "gasolina", float(rng.uniform(10, 25))
        elif prop == "Híbrido":
            combustivel, bateria = str(rng.choice(["gasolina", "flex"])), float(rng.uniform(0.9, 2.0))
        else:
            opcoes = [0.4, 0.4, 0.2] if picape else [0.5, 0.5, 0.0]
            combustivel = "gasolina" if i == 0 else str(rng.choice(["gasolina", "flex", "diesel"], p=opcoes))
            bateria = 0.0

        massa = float(rng.uniform(950, 2400) + (1.0 if picape else 0.0) * 300 + bateria * 4)
        f0 = float(0.0085 * massa * 9.81 * rng.uniform(0.85, 1.15))
        f2 = float(rng.uniform(0.028, 0.06) * (1.25 if picape else 1.0))
        potencia = float(rng.uniform(55, 260) * (1.3 if prop in ("Elétrico", "Plug-in") else 1.0))

        if prop == "Elétrico":
            transmissao, marchas, cilindrada = "direta", 1.0, 0.0
        elif prop == "Híbrido" and rng.random() < 0.3:
            transmissao, marchas, cilindrada = "dht", np.nan, float(rng.uniform(1400, 2500))
        else:
            transmissao = str(rng.choice(["automatica", "manual", "cvt", "dct"], p=[0.55, 0.15, 0.15, 0.15]))
            marchas = np.nan if transmissao == "cvt" else float(rng.integers(5, 10))
            cilindrada = float(rng.uniform(999, 3000))

        # road-load energy per km on a city-like and a highway-like cycle, in MJ/km
        rodovia = (f0 + f2 * 80**2) * 1e-3 + 0.00012 * massa
        cidade = (f0 + f2 * 30**2) * 1e-3 + 0.00035 * massa
        ruido = rng.normal(1.0, 0.03, size=2)
        mj_urbano = float(cidade / EFICIENCIA[prop] * ruido[0] * (1 + 0.0004 * potencia))
        mj_estrada = float(rodovia / EFICIENCIA[prop] * ruido[1] * (1 + 0.0002 * potencia))
        mj = 0.55 * mj_urbano + 0.45 * mj_estrada

        kml = dict.fromkeys(
            [
                f"kml_{c}_{t}"
                for c in ("etanol", "gasolina", "diesel", "eletrico_eq")
                for t in ("urbano", "estrada")
            ],
            np.nan,
        )
        if combustivel in ("gasolina", "flex"):
            kml["kml_gasolina_urbano"] = MJ_POR_LITRO["gasolina"] / mj_urbano
            kml["kml_gasolina_estrada"] = MJ_POR_LITRO["gasolina"] / mj_estrada
        if combustivel == "flex":
            kml["kml_etanol_urbano"] = MJ_POR_LITRO["etanol"] / mj_urbano
            kml["kml_etanol_estrada"] = MJ_POR_LITRO["etanol"] / mj_estrada
        if combustivel == "diesel":
            kml["kml_diesel_urbano"] = MJ_POR_LITRO["diesel"] / mj_urbano
            kml["kml_diesel_estrada"] = MJ_POR_LITRO["diesel"] / mj_estrada
        if prop in ("Elétrico", "Plug-in"):
            kml["kml_eletrico_eq_urbano"] = KML_ELETRICO_EQ / mj_urbano
            kml["kml_eletrico_eq_estrada"] = KML_ELETRICO_EQ / mj_estrada

        modelo = f"MODEL {grupo:03d}"
        linhas.append(
            {
                "lcvm": f"SY{i:05d}",
                "nome_homologado": f"{marca} {modelo} {i}",
                "marca": marca,
                "modelo": modelo,
                "propulsao": prop,
                "combustivel": combustivel,
                "categoria": categoria,
                "categoria_proconve": str(rng.choice(["CL2", "CL3"])) if picape else "PL",
                "massa_om_kg": round(massa),
                "classe_inercia_kg": round(massa + 136, -1),
                "pbt_kg": round(massa + 450),
                "f0_n": round(f0, 3),
                "f2_n_kmh2": round(f2, 5),
                "area_m2": round(float(rng.uniform(6, 13)), 2),
                "potencia_kw": round(potencia, 1),
                "cilindrada_cm3": round(cilindrada),
                "transmissao": transmissao,
                "n_marchas": marchas,
                "direcao": str(rng.choice(["eletrica", "hidraulica"], p=[0.93, 0.07])),
                "bateria_kwh": round(bateria, 1),
                **kml,
                "mj_km_urbano": mj_urbano,
                "mj_km_estrada": mj_estrada,
                "mj_km": mj,
                "bateria_estimada": False,
                "versoes": f"{marca} {modelo} {i}",
                "n_versoes": 1,
                "grupo_cv": f"{marca} {modelo}",
            }
        )
    return pd.DataFrame(linhas)


def salvar(caminho: Path, n: int = 420, semente: int = 0) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    gerar(n, semente).to_csv(caminho, index=False, encoding="utf-8", lineterminator="\n")
    return caminho
