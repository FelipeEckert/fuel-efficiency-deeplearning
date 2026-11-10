"""Limpa a aba "Dados de Entrada" da planilha PBEV e gera um CSV com uma linha por homologação.

Uso:  uv run eficiencia-limpar [caminho.xlsm] [saida.csv]

Lê o .xlsm direto do XML (valores já calculados pelo Excel); não executa macros
e não precisa de openpyxl.
"""

import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
ENTRADA = RAIZ / "data/raw/PED - PLANILHA DE ENTRADA DE DADOS PBEV 2026_05-05-2026.xlsm"
COMPLEMENTO_BATERIAS = RAIZ / "data/raw/complemento_baterias.csv"
CHAVE_COMPLEMENTO = ["marca", "modelo", "potencia_kw", "massa_om_kg"]
LCVM_VALIDO = r"(?:LC|SL)?\d+(?:/\d{2})?"
SAIDA = RAIZ / "data/processed/pbev_2026_limpo.csv"
ABA = "xl/worksheets/sheet1.xml"  # "Dados de Entrada"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# índice da coluna na aba (0 = coluna A) -> nome no CSV
COLUNAS = {
    1: "nacional_importado",
    2: "acima_limite_vendas",
    3: "categoria",
    4: "area_m2",
    5: "marca",
    6: "modelo",
    7: "motor",
    8: "potencia_txt",
    9: "transmissao_txt",
    10: "versao",
    11: "propulsao",
    14: "direcao",
    15: "combustivel",
    16: "cilindrada_txt",
    17: "massa_om_kg",
    20: "f0_n",
    22: "f2_n_kmh2",
    24: "pbt_kg",
    25: "classe_inercia_kg",
    48: "categoria_proconve",
    49: "lcvm",
    50: "nome_homologado",
    52: "bateria_kwh",
    78: "autonomia_eletrica_km",  # "Autonomia no modo Elétrico Corrigida (km)"
    # alvos (valores corrigidos = os publicados na etiqueta)
    60: "kml_etanol_urbano",
    61: "kml_gasolina_urbano",
    62: "kml_etanol_estrada",
    63: "kml_gasolina_estrada",
    64: "kml_diesel_urbano",
    65: "kml_diesel_estrada",
    66: "kml_eletrico_eq_urbano",
    67: "kml_eletrico_eq_estrada",
    70: "mj_km_urbano",
    71: "mj_km_estrada",
    72: "mj_km",
}
ALVOS = [c for c in COLUNAS.values() if c.startswith(("kml_", "mj_km"))]
FEATURES = [
    "propulsao",
    "combustivel",
    "categoria",
    "categoria_proconve",
    "marca",
    "massa_om_kg",
    "classe_inercia_kg",
    "pbt_kg",
    "f0_n",
    "f2_n_kmh2",
    "area_m2",
    "potencia_kw",
    "cilindrada_cm3",
    "transmissao",
    "n_marchas",
    "direcao",
    "bateria_kwh",
    "autonomia_eletrica_km",
]
CV_PARA_KW = 0.7355


def ler_aba(caminho, aba=ABA):
    """Retorna dict {linha: {coluna: valor}} com os valores em cache do Excel."""
    with zipfile.ZipFile(caminho) as z:
        shared = [
            "".join(t.text or "" for t in si.iter(NS + "t"))
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si")
        ]
        linhas = {}
        with z.open(aba) as f:
            for _, el in ET.iterparse(f):
                if el.tag != NS + "row":
                    continue
                vals = {}
                for c in el.findall(NS + "c"):
                    v, t = c.find(NS + "v"), c.get("t")
                    if t == "inlineStr":
                        vals[_col(c.get("r"))] = "".join(e.text or "" for e in c.iter(NS + "t"))
                    elif v is not None:
                        vals[_col(c.get("r"))] = shared[int(v.text)] if t == "s" else v.text
                linhas[int(el.get("r"))] = vals
                el.clear()
    return linhas


def _col(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def sem_acento(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def num(x):
    """Converte texto em float; vazio, 'NA', '#N/A', '-', '\\' etc. viram NaN."""
    if x is None:
        return np.nan
    try:
        return float(str(x).strip().replace(",", "."))
    except ValueError:
        return np.nan


def potencia_kw(txt, marca):
    t = (txt or "").replace("\n", " ").replace(",", ".").lower()
    if m := re.search(r"(\d+(?:\.\d+)?)\s*kw", t):
        return float(m.group(1))
    if cvs := re.findall(r"(\d+(?:\.\d+)?)\s*cv", t):
        return round(max(map(float, cvs)) * CV_PARA_KW, 1)  # flex: maior valor (G ou E)
    if m := re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*", t):
        return float(m.group(1)) + float(m.group(2))  # Mercedes: motor + elétrico, em kW
    if m := re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*", t):
        return float(m.group(2))  # JAC: "cv/kW"
    if m := re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", t):
        # shortcut: unidade inferida pela marca (VW declara cv, Mercedes kW); revisar se surgirem outras marcas
        return round(float(m.group(1)) * CV_PARA_KW, 1) if marca == "VOLKSWAGEN" else float(m.group(1))
    return np.nan


def transmissao(txt, propulsao):
    t = sem_acento(txt or "").replace(" ", "")
    marchas = num(m.group(1)) if (m := re.search(r"(\d+)", t)) else np.nan
    if t.startswith("cvt"):
        tipo, marchas = "cvt", np.nan
    elif t.startswith("dct"):
        tipo = "dct"
    elif t.startswith("dht"):
        tipo = "dht"
    elif t.startswith(("m-", "mt")) or t == "m":
        tipo = "manual"
    elif t.startswith(("a", "automatica")):
        tipo = "automatica"
    else:
        tipo = "direta" if propulsao == "Elétrico" else np.nan
    if propulsao == "Elétrico" and tipo in ("automatica", "direta") and np.isnan(marchas):
        marchas = 1.0
    return tipo, marchas


def padronizar_propulsao(s):
    s = sem_acento(s)
    if s.startswith("plug"):
        return "Plug-in"
    if s.startswith("hibrid"):
        return "Híbrido"
    if s.startswith("eletr"):
        return "Elétrico"
    return "Combustão"  # "combustao" e o caso "Gasolina" digitado no campo errado


MARCAS = {"VW": "VOLKSWAGEN", "CAOACHANGAN": "CAOA CHANGAN"}

# Variantes do mesmo carro (tração, câmbio, combustível, acabamento, ano-modelo) ficam no mesmo grupo
# da validação cruzada. Carrocerias diferentes (Onix × Onix Plus, Defender 90 × 110) continuam separadas.
VARIANTES = re.compile(r"\b(?:4X2|4X4|2WD|AT|MT|DIESEL|FLEX|TREMOR|DARK HORSE|4M)\b|\(MOD\.? ?\d+\)")
GRUPOS_MANUAIS = {"RENAULT MASTER FUGÃO PRÓ": "RENAULT MASTER FURGÃO"}  # erro de digitação + acabamento


def grupo_cv(marca, modelo):
    g = f"{marca} {VARIANTES.sub(' ', modelo.upper())}"
    g = re.sub(r"\s+", " ", g).strip()
    return GRUPOS_MANUAIS.get(g, g)


def limpar(caminho=ENTRADA):
    linhas = ler_aba(caminho)
    df = pd.DataFrame(
        [
            {nome: linhas[r].get(i) for i, nome in COLUNAS.items()}
            for r in sorted(linhas)
            if r > 1 and linhas[r].get(5) and linhas[r].get(6)
        ]
    )
    n_bruto = len(df)
    txt = lambda c: df[c].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    df["marca"] = txt("marca").str.upper().replace(MARCAS)
    df["modelo"] = txt("modelo")
    df["versao"] = txt("versao")
    df["propulsao"] = txt("propulsao").map(padronizar_propulsao)
    df["combustivel"] = txt("combustivel").map({"G": "gasolina", "F": "flex", "D": "diesel", "E": "eletrico"})
    df["direcao"] = txt("direcao").map(
        {"E": "eletrica", "Elétrica": "eletrica", "H": "hidraulica", "E-H": "eletro-hidraulica"}
    )
    df["nacional_importado"] = txt("nacional_importado").map(
        lambda s: {"I": "importado", "N": "nacional"}.get(s, "ambos")
    )
    df["acima_limite_vendas"] = txt("acima_limite_vendas").eq("S")
    df["categoria_proconve"] = txt("categoria_proconve").where(lambda s: s.str.match(r"^(PL|CL\d?)$"))

    for c in [
        "area_m2",
        "massa_om_kg",
        "f0_n",
        "f2_n_kmh2",
        "pbt_kg",
        "classe_inercia_kg",
        "bateria_kwh",
        "autonomia_eletrica_km",
        *ALVOS,
    ]:
        df[c] = df[c].map(num)

    df["potencia_kw"] = [potencia_kw(p, m) for p, m in zip(df["potencia_txt"], df["marca"])]
    df[["transmissao", "n_marchas"]] = [
        transmissao(t, p) for t, p in zip(df["transmissao_txt"], df["propulsao"])
    ]

    cil = df["cilindrada_txt"].fillna("").astype(str).str.strip()
    cil = cil.where(
        ~cil.str.fullmatch(r"\d\.\d{3}"), cil.str.replace(".", "", regex=False)
    )  # "1.798" -> 1798
    df["cilindrada_cm3"] = cil.map(num)
    # cilindrada vazia: usa a do nome do motor ("2.0-16V" -> 2000), ex.: GAC GS4 híbrido
    litros = txt("motor").str.extract(r"(\d\.\d)", expand=False).map(num) * 1000
    df["cilindrada_cm3"] = df["cilindrada_cm3"].fillna(litros)
    df.loc[df["propulsao"].eq("Elétrico"), "cilindrada_cm3"] = 0.0
    # combustão não tem bateria de tração (a Honda informou os 60 Ah da bateria 12 V nesse campo)
    df.loc[df["propulsao"].eq("Combustão"), "bateria_kwh"] = 0.0
    # só Plug-in e Elétrico rodam em modo elétrico: nos demais a autonomia é 0
    df.loc[df["propulsao"].isin(["Combustão", "Híbrido"]), "autonomia_eletrica_km"] = 0.0
    # formatos válidos: LC12233, 19437, 10945/24, SL64402; "N.A.", "Isento", "xxxx" e o "0.3248…" do Tank 300 viram vazio
    df["lcvm"] = txt("lcvm").where(lambda s: s.str.fullmatch(LCVM_VALIDO))

    # F2 declarado em N/(m/s)² em vez de N/(km/h)²: converte (÷3,6²). Confirmado pelo PRR80 da Oroch.
    f2_corrigido = df["f2_n_kmh2"] > 0.12
    df.loc[f2_corrigido, "f2_n_kmh2"] /= 3.6**2

    # sem F0/F2/massa/alvo não dá para usar a linha
    essenciais = ["f0_n", "f2_n_kmh2", "massa_om_kg", "mj_km"]
    sem_essencial = df[essenciais].isna().any(axis=1)
    df = df[~sem_essencial].copy()

    df, n_complemento = completar_baterias(df)

    # uma linha por homologação: versões com dados técnicos idênticos viram uma só.
    # marca fica fora da chave: vans Stellantis (Jumper/Ducato/Boxer etc.) são o mesmo ensaio em 3 marcas.
    chave = [c for c in FEATURES + ALVOS if c != "marca"]
    # map(str), não astype(str): pandas 3 mantém NaN no astype e o join quebra
    chave_txt = df[chave].round(4).map(str).agg("|".join, axis=1)
    agrupado = df.groupby(chave_txt, sort=False)
    limpo = agrupado.first()[["lcvm", "nome_homologado", "marca", "modelo", *chave, "bateria_estimada"]]
    limpo["versoes"] = agrupado.apply(
        lambda g: "; ".join(
            dict.fromkeys(f"{ma} {m} {v}".strip() for ma, m, v in zip(g["marca"], g["modelo"], g["versao"]))
        )
    )
    limpo["n_versoes"] = agrupado.size()
    limpo["grupo_cv"] = [grupo_cv(ma, mo) for ma, mo in zip(limpo["marca"], limpo["modelo"])]
    limpo = limpo.reset_index(drop=True)

    relatorio = {
        "linhas lidas": n_bruto,
        "F2 convertido de N/(m/s)² para N/(km/h)²": int(f2_corrigido.sum()),
        "removidas por falta de F0/F2/massa/MJ_km": int(sem_essencial.sum()),
        "bateria preenchida pelo complemento (versões)": n_complemento,
        "bateria estimada (homologações)": int(limpo["bateria_estimada"].sum()),
        "homologações únicas (linhas no CSV)": len(limpo),
        "potência não identificada": int(limpo["potencia_kw"].isna().sum()),
    }
    return limpo, relatorio


def completar_baterias(df, caminho=COMPLEMENTO_BATERIAS):
    """Preenche bateria faltante com a pesquisa em fontes públicas; o que faltar vira mediana da propulsão.

    Nunca sobrescreve valor do PBEV. `bateria_estimada` = True quando o valor veio de inferência
    (confiança baixa) ou da mediana.
    """
    df["bateria_estimada"] = False
    n = 0
    if caminho.exists():
        comp = pd.read_csv(caminho)
        comp = comp[comp["bateria_kwh"].notna()]
        df = df.merge(
            comp[CHAVE_COMPLEMENTO + ["bateria_kwh", "confianca"]],
            on=CHAVE_COMPLEMENTO,
            how="left",
            suffixes=("", "_comp"),
            validate="many_to_one",
        )
        usar = df["bateria_kwh"].isna() & df["bateria_kwh_comp"].notna()
        df.loc[usar, "bateria_kwh"] = df.loc[usar, "bateria_kwh_comp"]
        df.loc[usar & df["confianca"].eq("baixa"), "bateria_estimada"] = True
        n = int(usar.sum())
        df = df.drop(columns=["bateria_kwh_comp", "confianca"])
    falta = df["bateria_kwh"].isna()
    mediana = df[~falta & ~df["bateria_estimada"]].groupby("propulsao")["bateria_kwh"].median()
    df.loc[falta, "bateria_kwh"] = df.loc[falta, "propulsao"].map(mediana)
    df.loc[falta, "bateria_estimada"] = True
    return df, n


def checar(df):
    """Invariantes do arquivo limpo."""
    assert df["f2_n_kmh2"].between(0.015, 0.12).all(), "F2 fora da faixa"
    assert df["f0_n"].between(50, 500).all(), "F0 fora da faixa"
    assert df["massa_om_kg"].between(600, 4500).all(), "massa fora da faixa"
    assert df["mj_km"].between(0.2, 6).all(), "MJ/km fora da faixa"
    assert set(df["propulsao"]) <= {"Combustão", "Híbrido", "Plug-in", "Elétrico"}
    assert df["combustivel"].notna().all() and df["propulsao"].notna().all()
    assert (df.loc[df["propulsao"].eq("Combustão"), "bateria_kwh"] == 0).all()
    assert df["bateria_kwh"].notna().all(), "bateria sem valor"
    aut = df["autonomia_eletrica_km"]
    assert (aut[~df["propulsao"].isin(["Plug-in", "Elétrico"])] == 0).all(), (
        "autonomia fora de Plug-in/Elétrico"
    )
    assert aut.dropna().between(0, 2000).all(), "autonomia fora da faixa"
    assert df["cilindrada_cm3"].notna().all(), "cilindrada sem valor"
    assert (df.loc[df["propulsao"].ne("Elétrico"), "cilindrada_cm3"] > 500).all()
    assert (df.loc[df["propulsao"].eq("Elétrico"), "bateria_kwh"] > 10).all()
    assert (df.loc[df["propulsao"].ne("Combustão"), "bateria_kwh"] > 0).all()
    assert df["lcvm"].dropna().str.fullmatch(LCVM_VALIDO).all()
    tecnico = [c for c in FEATURES + ALVOS if c != "marca"]
    assert not df[tecnico].round(4).map(str).duplicated().any(), "homologação repetida"
    assert np.allclose(df["mj_km"], 0.55 * df["mj_km_urbano"] + 0.45 * df["mj_km_estrada"], atol=1e-6)
    pw = df["potencia_kw"].dropna()
    assert pw.between(30, 1000).all(), f"potência estranha: {pw[~pw.between(30, 1000)].tolist()}"
    # exemplos reais de cada formato de potência
    assert potencia_kw("176 cv (G) e 176 cv (E)", "JEEP") == round(176 * CV_PARA_KW, 1)
    assert potencia_kw("170cv (125kW)", "FORD") == 125
    assert potencia_kw("120 + 10", "MERCEDES-BENZ") == 130
    assert potencia_kw("150/110", "JAC") == 110
    assert potencia_kw("116", "VOLKSWAGEN") == round(116 * CV_PARA_KW, 1)
    assert potencia_kw("58,8 kW", "HYUNDAI") == 58.8
    assert transmissao("A - 6", "Combustão") == ("automatica", 6.0)
    assert transmissao("N.A.", "Elétrico") == ("direta", 1.0)
    assert (
        grupo_cv("TOYOTA", "HILUX DIESEL 4X4 AT")
        == grupo_cv("TOYOTA", "Hilux Diesel 4x4 MT")
        == "TOYOTA HILUX"
    )
    assert grupo_cv("HONDA", "HR-V (mod. 26)") == "HONDA HR-V"
    assert grupo_cv("CHEVROLET", "ONIX PLUS") != grupo_cv("CHEVROLET", "ONIX")


def salvar(df, saida=SAIDA):
    saida.parent.mkdir(parents=True, exist_ok=True)
    # LF em qualquer sistema, igual ao arquivo versionado
    df.to_csv(saida, index=False, encoding="utf-8", lineterminator="\n")


def main():
    entrada = Path(sys.argv[1]) if len(sys.argv) > 1 else ENTRADA
    saida = Path(sys.argv[2]) if len(sys.argv) > 2 else SAIDA
    df, rel = limpar(entrada)
    checar(df)
    salvar(df, saida)
    for k, v in rel.items():
        print(f"{k}: {v}")
    print(f"salvo em {saida}")
