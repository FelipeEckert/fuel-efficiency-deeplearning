"""Experimentos de otimização do XGBoost. Uso: uv run eficiencia-experimento {samplers,cerca,sementes}

Nunca gravam em optuna/estudos.db: só leem de lá os parâmetros finais do treino oficial.
"""

import argparse
import json
import math

import numpy as np
import optuna

from eficiencia.dados import RAIZ
from eficiencia.esquema import carregar_base
from eficiencia.modelos import CANDIDATOS
from eficiencia.modelos.xgb import FAIXAS_AMPLO
from eficiencia.treino import STORAGE
from eficiencia.validacao import SAMPLERS, ResultadoCV, cv_aninhada, cv_fixa, hash8, prefixo_estudo

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
    nome_estudo = f"{prefixo_estudo(nome, hash8(carregar_base()))}-i3-s0-final"

    def ler(arm):
        try:
            return dict(optuna.load_study(study_name=nome_estudo, storage=arm).best_params)
        except KeyError:
            raise ValueError(f"estudo não encontrado em {storage}: {nome_estudo}") from None

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
        return {
            s: _resumo(
                cv_aninhada("xgboost", df, n_tentativas, arm, rapido=rapido, sampler=s, **_dobras(rapido))
            )
            for s in SAMPLERS
        }

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Experimentos de otimização do XGBoost.")
    parser.add_argument("experimento", choices=["samplers", "cerca", "sementes"])
    parser.add_argument(
        "--tentativas", type=int, help="tentativas do Optuna (padrão: 50 samplers, 100 cerca)"
    )
    parser.add_argument("--rapido", action="store_true", help="2 dobras, parâmetros rápidos, sem banco")
    args = parser.parse_args(argv)
    if args.experimento == "sementes":
        params = {} if args.rapido else melhores_params("xgboost")
        resultado = experimento_sementes(params, rapido=args.rapido)
    else:
        f = experimento_samplers if args.experimento == "samplers" else experimento_cerca
        kw = {"n_tentativas": args.tentativas} if args.tentativas else {}
        if args.rapido:
            kw |= {"n_tentativas": 2, "storage": None}
        else:
            (RAIZ / "optuna").mkdir(exist_ok=True)
        resultado = f(rapido=args.rapido, **kw)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
