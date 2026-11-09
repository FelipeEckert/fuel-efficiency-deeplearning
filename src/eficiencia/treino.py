"""Treino de todos os candidatos, escolha do vencedor, artefatos e relatório. Uso: uv run eficiencia-treinar"""

import argparse
import json
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from eficiencia.artefatos import PASTA_MODELO, Pacote, salvar
from eficiencia.dados import RAIZ
from eficiencia.esquema import ALVOS, ENTRADAS, carregar_base
from eficiencia.faixa import calcular_faixas
from eficiencia.fisica import combinar_mj, fatores_kml
from eficiencia.formato import numero
from eficiencia.incerteza import calcular_quantis
from eficiencia.modelos import CANDIDATOS
from eficiencia.monotonia import checar_monotonia
from eficiencia.validacao import buscar_params, cv_aninhada, hash8, prefixo_estudo

PASTA_RELATORIOS = RAIZ / "relatorios"
STORAGE = f"sqlite:///{(RAIZ / 'optuna' / 'estudos.db').as_posix()}"
SEMENTE = 0
BIBLIOTECAS = ["pandas", "numpy", "scikit-learn", "xgboost", "torch", "optuna"]


def _pastas_padrao(rapido: bool) -> tuple[Path, Path]:
    """O modo rápido não sobrescreve o modelo nem os relatórios reais: usa pastas de desenvolvimento."""
    if rapido:
        return RAIZ / "modelos/rapido", RAIZ / "relatorios/rapido"
    return PASTA_MODELO, PASTA_RELATORIOS


def treinar(
    n_tentativas: int = 50,
    candidatos: list[str] | None = None,
    rapido: bool = False,
    storage: str | None = STORAGE,
    pasta_modelo: Path | None = None,
    pasta_relatorios: Path | None = None,
) -> dict:
    nomes = candidatos or list(CANDIDATOS)
    n_externas, n_internas = 5, 3
    if rapido:
        n_tentativas, n_externas, n_internas, storage = 2, 2, 2, None
    padrao_modelo, padrao_relatorios = _pastas_padrao(rapido)
    if pasta_modelo is None:
        pasta_modelo = padrao_modelo
    if pasta_relatorios is None:
        pasta_relatorios = padrao_relatorios
    if storage is not None:
        (RAIZ / "optuna").mkdir(exist_ok=True)
        # uma conexão para todos os estudos; com a URL, o Optuna abriria uma por estudo e deixaria abertas
        storage = optuna.storages.RDBStorage(storage)
    df = carregar_base()
    h = hash8(df)

    resultados, modelos, params, tentativas, violacoes = {}, {}, {}, {}, {}
    for nome in nomes:
        r = cv_aninhada(
            nome,
            df,
            n_tentativas,
            storage,
            n_externas=n_externas,
            n_internas=n_internas,
            rapido=rapido,
            semente=SEMENTE,
        )
        nome_final = f"{prefixo_estudo(nome, h)}-i{n_internas}-s{SEMENTE}-final"
        p = buscar_params(
            nome,
            df,
            n_tentativas,
            storage,
            nome_final,
            n_dobras=n_internas,
            rapido=rapido,
            semente=SEMENTE,
        )
        modelo = CANDIDATOS[nome](semente=SEMENTE, **p).fit(df[ENTRADAS], df[ALVOS])
        # sem storage o estudo final é novo: as tentativas são as desta chamada
        n_final = (
            len(optuna.load_study(study_name=nome_final, storage=storage).trials) if storage else n_tentativas
        )
        resultados[nome], modelos[nome], params[nome] = r, modelo, p
        tentativas[nome] = r.n_tentativas + n_final
        violacoes[nome] = checar_monotonia(modelo, df)
    if storage is not None:  # fecha a conexão com o banco antes de gravar os artefatos
        storage.remove_session()
        storage.engine.dispose()

    aprovados = [n for n in nomes if not violacoes[n]]
    vencedor = min(aprovados, key=lambda n: resultados[n].erro_mj) if aprovados else None

    metadados = {
        "algoritmo": vencedor,
        "params": params[vencedor] if vencedor else None,
        "data": date.today().isoformat(),
        "hash_csv": h,
        "rapido": rapido,
        "versoes": {b: version(b) for b in BIBLIOTECAS},
        "tentativas": tentativas,
    }
    metricas = {
        "vencedor": vencedor,
        "candidatos": {
            n: {
                "erro_mj": resultados[n].erro_mj,
                "erro_por_propulsao": resultados[n].erro_por_propulsao,
                "erro_por_alvo": resultados[n].erro_por_alvo,
                "n_tentativas": tentativas[n],
                "monotonia": "reprovado" if violacoes[n] else "aprovado",
                "violacoes": violacoes[n],
            }
            for n in nomes
        },
    }
    # o relatório é gravado mesmo sem vencedor: mostra por que cada candidato foi reprovado
    figuras = pasta_relatorios / "figuras"
    figuras.mkdir(parents=True, exist_ok=True)
    (pasta_relatorios / "metricas.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _figura_erro_por_propulsao(metricas, figuras / "erro_por_propulsao.png")
    if vencedor:
        _figura_previsto_real(df, resultados[vencedor].oof, vencedor, figuras / "previsto_real.png")
    (pasta_relatorios / "relatorio.md").write_text(_relatorio(metricas, metadados), encoding="utf-8")
    if vencedor is None:
        detalhe = "\n".join(f"{n}: " + "; ".join(violacoes[n]) for n in nomes)
        raise RuntimeError(f"nenhum candidato passou no teste de monotonia:\n{detalhe}")

    pacote = Pacote(
        algoritmo=vencedor,
        modelo=modelos[vencedor],
        quantis=calcular_quantis(df, resultados[vencedor].oof),
        faixas=calcular_faixas(df),
        fatores_kml=fatores_kml(df),
        referencias=_referencias(df),
        metadados=metadados,
    )
    salvar(pacote, pasta_modelo)
    return metricas


def _referencias(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    medianas = df.groupby("categoria")[["f0_n", "f2_n_kmh2"]].median()
    return {cat: {c: float(v) for c, v in linha.items()} for cat, linha in medianas.iterrows()}


def _figura_erro_por_propulsao(metricas: dict, caminho: Path) -> None:
    fig = Figure(figsize=(8, 4.5))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    nomes = list(metricas["candidatos"])
    propulsoes = list(metricas["candidatos"][nomes[0]]["erro_por_propulsao"])
    largura = 0.8 / len(nomes)
    x = np.arange(len(propulsoes))
    for i, nome in enumerate(nomes):
        erros = metricas["candidatos"][nome]["erro_por_propulsao"]
        ax.bar(x + i * largura, [100 * erros[p] for p in propulsoes], largura, label=nome)
    ax.set_xticks(x + largura * (len(nomes) - 1) / 2, propulsoes)
    ax.set_ylabel("Erro médio MJ/km (%)")
    ax.set_title("Erro fora da amostra por propulsão")
    ax.legend()
    fig.savefig(caminho, dpi=120, bbox_inches="tight")


def _figura_previsto_real(df: pd.DataFrame, oof: pd.DataFrame, nome: str, caminho: Path) -> None:
    fig = Figure(figsize=(5, 5))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    previsto = combinar_mj(oof["mj_km_urbano"], oof["mj_km_estrada"])
    for propulsao, grupo in df.groupby("propulsao"):
        ax.scatter(grupo["mj_km"], previsto[grupo.index], s=8, label=propulsao)
    limite = [df["mj_km"].min(), df["mj_km"].max()]
    ax.plot(limite, limite, color="gray", linewidth=1)
    ax.set_xlabel("MJ/km real")
    ax.set_ylabel("MJ/km previsto (fora da amostra)")
    ax.set_title(f"Vencedor: {nome}")
    ax.legend()
    fig.savefig(caminho, dpi=120, bbox_inches="tight")


def _relatorio(metricas: dict, metadados: dict) -> str:
    candidatos = metricas["candidatos"]
    vencedor = metricas["vencedor"]
    propulsoes = list(next(iter(candidatos.values()))["erro_por_propulsao"])
    if vencedor:
        resumo = [
            f"**Vencedor: {vencedor}**, o de menor erro entre os que passam no teste de monotonia.",
            f"Hiperparâmetros: `{metadados['params']}`",
        ]
    else:
        resumo = ["**Nenhum candidato passou no teste de monotonia.** Nenhum modelo foi salvo."]
    linhas = [
        "# Relatório de treino: eficiência energética",
        "",
        f"Data: {metadados['data']} · hash da base: `{metadados['hash_csv']}`",
        "",
        *resumo,
        "",
        "## Candidatos",
        "",
        "| Candidato | Erro MJ/km | Monotonia | Tentativas |",
        "|---|---|---|---|",
    ]
    for nome, c in candidatos.items():
        linhas.append(
            f"| {nome} | {numero(100 * c['erro_mj'], 2)} % | {c['monotonia']} | {c['n_tentativas']} |"
        )
    linhas += [
        "",
        "## Erro por propulsão (MJ/km)",
        "",
        "| Candidato | " + " | ".join(propulsoes) + " |",
        "|---|" + "---|" * len(propulsoes),
    ]
    for nome, c in candidatos.items():
        celulas = [numero(100 * c["erro_por_propulsao"][p], 2) + " %" for p in propulsoes]
        linhas.append(f"| {nome} | " + " | ".join(celulas) + " |")
    reprovados = {n: c["violacoes"] for n, c in candidatos.items() if c["violacoes"]}
    if reprovados:
        linhas += ["", "## Violações de monotonia", ""]
        linhas += [f"- {n}: {len(v)} violações, por exemplo: {v[0]}" for n, v in reprovados.items()]
    linhas += ["", "## Figuras", "", "![Erro por propulsão](figuras/erro_por_propulsao.png)", ""]
    if vencedor:
        linhas += ["![Previsto × real do vencedor](figuras/previsto_real.png)", ""]
    return "\n".join(linhas)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Treina os candidatos, escolhe o vencedor e gera os artefatos."
    )
    parser.add_argument("--tentativas", type=int, default=50, help="tentativas do Optuna por candidato")
    parser.add_argument(
        "--candidatos", type=lambda s: s.split(","), help="ex.: linear,xgboost (padrão: todos)"
    )
    parser.add_argument(
        "--rapido",
        action="store_true",
        help="modo rápido, para desenvolvimento e CI; grava em modelos/rapido e relatorios/rapido",
    )
    args = parser.parse_args(argv)
    metricas = treinar(n_tentativas=args.tentativas, candidatos=args.candidatos, rapido=args.rapido)
    for nome, c in metricas["candidatos"].items():
        erro = numero(100 * c["erro_mj"], 2)
        print(f"{nome}: erro MJ/km {erro} %, monotonia {c['monotonia']}, tentativas {c['n_tentativas']}")
    print(f"relatório em {_pastas_padrao(args.rapido)[1] / 'relatorio.md'}")
    print(f"vencedor: {metricas['vencedor']}")
