"""Experimentos de otimização dos modelos (XGBoost, outros boostings e redes). Uso: uv run eficiencia-experimento {samplers,cerca,sementes,redes,boosting}

Nunca gravam em optuna/estudos.db: só leem de lá os parâmetros finais do treino oficial.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import optuna
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from eficiencia.dados import RAIZ
from eficiencia.esquema import ALVOS, ENTRADAS, carregar_base
from eficiencia.modelos import CANDIDATOS
from eficiencia.modelos.xgb import FAIXAS_AMPLO
from eficiencia.treino import STORAGE
from eficiencia.validacao import (
    SAMPLERS,
    ResultadoCV,
    cv_aninhada,
    cv_fixa,
    dobras_por_grupo,
    hash8,
    prefixo_estudo,
)

STORAGE_EXPERIMENTOS = f"sqlite:///{(RAIZ / 'optuna' / 'experimentos.db').as_posix()}"
PASTA_EXPERIMENTOS = RAIZ / "relatorios/experimentos"


def _com_storage(storage: str | None, f):
    """Roda f(storage) com uma única conexão, fechada no fim (no Windows o sqlite aberto trava a limpeza)."""
    if storage is None:
        return f(None)
    arm = optuna.storages.RDBStorage(storage)
    try:
        return f(arm)
    finally:
        arm.remove_session()
        arm.engine.dispose()


def melhores_params(nome: str, storage: str = STORAGE) -> dict:
    """best_params do estudo final do treino oficial para esta base."""
    if storage.startswith("sqlite:///") and not Path(storage.removeprefix("sqlite:///")).exists():
        raise ValueError(f"database not found: {storage.removeprefix('sqlite:///')}")  # não cria um vazio
    nome_estudo = f"{prefixo_estudo(nome, hash8(carregar_base()))}-i3-s0-final"

    def ler(arm):
        try:
            return dict(optuna.load_study(study_name=nome_estudo, storage=arm).best_params)
        except KeyError:
            raise ValueError(f"study not found in {storage}: {nome_estudo}") from None

    return _com_storage(storage, ler)


def na_borda(params: dict, faixas: dict) -> list[str]:
    """Parâmetros a menos de 5 % da largura da faixa de um dos limites (em log10 nos de escala log)."""
    saida = []
    for nome, valor in params.items():
        if nome not in faixas:
            continue
        lo, hi, log = faixas[nome]
        if log:
            valor, lo, hi = math.log10(valor), math.log10(lo), math.log10(hi)
        folga = 0.05 * (hi - lo)
        if valor - lo < folga or hi - valor < folga:
            saida.append(nome)
    return saida


def _resumo(r: ResultadoCV) -> dict:
    return {"erro_mj": r.erro_mj, "erro_por_propulsao": r.erro_por_propulsao}


def _gravar(nome: str, resultado: dict, rapido: bool) -> None:
    pasta = PASTA_EXPERIMENTOS / "rapido" if rapido else PASTA_EXPERIMENTOS
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / f"{nome}.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")


def _dobras(rapido: bool) -> dict:
    return {"n_externas": 2, "n_internas": 2} if rapido else {"n_externas": 5, "n_internas": 3}


def experimento_samplers(
    n_tentativas: int = 50, storage: str | None = STORAGE_EXPERIMENTOS, rapido: bool = False
) -> dict:
    df = carregar_base()

    def rodar(arm):
        def cv(**kw):
            return _resumo(
                cv_aninhada("xgboost", df, n_tentativas, arm, rapido=rapido, **_dobras(rapido), **kw)
            )

        r = {s: cv(sampler=s) for s in SAMPLERS}
        # mesmas dobras, só a semente do sampler muda: mede o ruído da busca (tpe já é a semente 0)
        r |= {f"tpe_ss{k}": cv(semente_sampler=k) for k in (1, 2)}
        tpes = [r["tpe"], r["tpe_ss1"], r["tpe_ss2"]]
        return r | {"ruido_busca_tpe": {"erros": [x["erro_mj"] for x in tpes]} | _estatisticas(tpes)}

    resultado = _com_storage(storage, rodar)
    _gravar("samplers", resultado, rapido)
    return resultado


def experimento_cerca(
    n_tentativas: int = 100, storage: str | None = STORAGE_EXPERIMENTOS, rapido: bool = False
) -> dict:
    df = carregar_base()
    r = _com_storage(
        storage,
        lambda arm: cv_aninhada("xgboost_amplo", df, n_tentativas, arm, rapido=rapido, **_dobras(rapido)),
    )
    resultado = _resumo(r) | {
        "params_por_dobra": r.params_por_dobra,
        "bordas": [na_borda(p, FAIXAS_AMPLO) for p in r.params_por_dobra],
    }
    _gravar("cerca", resultado, rapido)
    return resultado


def experimento_boosting(
    n_tentativas: int = 100, storage: str | None = STORAGE_EXPERIMENTOS, rapido: bool = False
) -> dict:
    """CatBoost e LightGBM na mesma CV aninhada do XGBoost oficial (100 tentativas, 5 dobras externas)."""
    df = carregar_base()

    def rodar(arm):
        r = {}
        for nome in ("catboost", "lightgbm"):
            cv = cv_aninhada(nome, df, n_tentativas, arm, rapido=rapido, **_dobras(rapido))
            r[nome] = _resumo(cv) | {"params_por_dobra": cv.params_por_dobra}
        return r

    resultado = _com_storage(storage, rodar)
    _gravar("boosting", resultado, rapido)
    return resultado


def experimento_sementes(params: dict, n: int = 5, rapido: bool = False) -> dict:
    df = carregar_base()
    if rapido:
        params = params | CANDIDATOS["xgboost"].PARAMS_RAPIDOS
    n_externas = _dobras(rapido)["n_externas"]
    por_semente = [
        _resumo(cv_fixa("xgboost", df, params, n_externas=n_externas, semente_modelo=s)) for s in range(n)
    ]
    erros = [x["erro_mj"] for x in por_semente]
    resultado = {
        "params": params,
        "media": float(np.mean(erros)),
        "desvio": float(np.std(erros, ddof=1)) if n > 1 else 0.0,
        "por_semente": por_semente,
    }
    _gravar("sementes", resultado, rapido)
    return resultado


def _estatisticas(resumos: list[dict]) -> dict:
    erros = [x["erro_mj"] for x in resumos]
    return {"media": float(np.mean(erros)), "desvio": float(np.std(erros, ddof=1)) if len(erros) > 1 else 0.0}


def _figura_curvas(historicos: dict[str, list], caminho) -> None:
    """Um painel por modo de validação: perda de treino e de validação por época."""
    fig = Figure(figsize=(10, 4))
    FigureCanvasAgg(fig)
    for ax, (titulo, historico) in zip(fig.subplots(1, len(historicos), sharey=True), historicos.items()):
        treino, validacao = zip(*historico)
        ax.plot(treino, label="treino")
        ax.plot(validacao, label="validação")
        ax.set_title(titulo)
        ax.set_xlabel("Época")
        ax.legend()
    fig.axes[0].set_ylabel("Perda (alvo padronizado)")
    fig.savefig(caminho, dpi=120, bbox_inches="tight")


def experimento_redes(
    params: dict, n: int = 5, lrs=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2), rapido: bool = False
) -> dict:
    df = carregar_base()
    if rapido:
        params = params | CANDIDATOS["rede_por_alvo"].PARAMS_RAPIDOS
    n_externas = _dobras(rapido)["n_externas"]
    com_grupo = ENTRADAS + ["grupo_cv"]

    def cv(p=params, **kw):
        return _resumo(cv_fixa("rede_por_alvo", df, p, n_externas=n_externas, **kw))

    resultado = {"params": params}
    for k in range(n):
        resultado[f"linha_s{k}"] = cv(semente_modelo=k)
        resultado[f"grupo_s{k}"] = cv(semente_modelo=k, colunas=com_grupo)
    resultado["media5"] = cv(p=params | {"n_redes": 5})
    for lr in lrs:
        resultado[f"lr_{lr}"] = cv(p=params | {"lr": lr})
    for prefixo in ("linha", "grupo"):
        resultado[f"{prefixo}_resumo"] = _estatisticas([resultado[f"{prefixo}_s{k}"] for k in range(n)])

    # curvas de perda: dobra externa 0, alvo mj_km_urbano, nos dois modos de validação
    tr, _ = next(iter(dobras_por_grupo(df["grupo_cv"], n_externas, 0)))
    alvo = "mj_km_urbano"
    historicos = {}
    for titulo, colunas in (("Validação por linha", ENTRADAS), ("Validação por grupo", com_grupo)):
        modelo = CANDIDATOS["rede_por_alvo"](semente=0, **params).fit(
            df.iloc[tr][colunas], df.iloc[tr][ALVOS]
        )
        historicos[titulo] = modelo.estimadores[alvo].mlp.historicos[0]
    pasta = PASTA_EXPERIMENTOS / "rapido" if rapido else PASTA_EXPERIMENTOS
    pasta.mkdir(parents=True, exist_ok=True)
    _figura_curvas(historicos, pasta / "curvas_perda.png")
    _gravar("redes", resultado, rapido)
    return resultado


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Model optimization experiments (XGBoost, other boosting libraries and networks)."
    )
    parser.add_argument("experimento", choices=["samplers", "cerca", "sementes", "redes", "boosting"])
    parser.add_argument(
        "--tentativas", type=int, help="Optuna trials (default: 50 for samplers, 100 for cerca and boosting)"
    )
    parser.add_argument("--rapido", action="store_true", help="2 folds, fast parameters, no database")
    args = parser.parse_args(argv)
    if args.experimento in ("sementes", "redes"):
        nome = "xgboost" if args.experimento == "sementes" else "rede_por_alvo"
        params = {} if args.rapido else melhores_params(nome)
        f = experimento_sementes if args.experimento == "sementes" else experimento_redes
        resultado = f(params, rapido=args.rapido)
    else:
        f = {"samplers": experimento_samplers, "cerca": experimento_cerca, "boosting": experimento_boosting}[
            args.experimento
        ]
        kw = {"n_tentativas": args.tentativas} if args.tentativas else {}
        if args.rapido:
            kw |= {"n_tentativas": 2, "storage": None}
        else:
            (RAIZ / "optuna").mkdir(exist_ok=True)
        resultado = f(rapido=args.rapido, **kw)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
