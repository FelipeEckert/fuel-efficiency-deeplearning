"""Validação aninhada por grupo (grupo_cv) com busca de hiperparâmetros no Optuna, retomável via storage."""

import hashlib
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd

from eficiencia.esquema import ALVOS, ENTRADAS
from eficiencia.fisica import combinar_mj
from eficiencia.modelos import CANDIDATOS, EXPERIMENTAIS

optuna.logging.set_verbosity(optuna.logging.WARNING)

SAMPLERS: dict[str, type[optuna.samplers.BaseSampler]] = {
    "tpe": optuna.samplers.TPESampler,
    "random": optuna.samplers.RandomSampler,
    "cmaes": optuna.samplers.CmaEsSampler,
    "qmc": optuna.samplers.QMCSampler,
    "gp": optuna.samplers.GPSampler,
}


def erro_percentual(real, previsto) -> np.ndarray:
    return np.abs(np.asarray(previsto) / np.asarray(real) - 1)


def dobras_por_grupo(grupos: pd.Series, n: int, semente: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    unicos = np.array(grupos.unique(), dtype=object)
    np.random.default_rng(semente).shuffle(unicos)
    dobras = []
    for parte in np.array_split(unicos, n):
        teste = grupos.isin(parte).to_numpy()
        dobras.append((np.flatnonzero(~teste), np.flatnonzero(teste)))
    return dobras


def erro_mj_combinado(df: pd.DataFrame, previsto: pd.DataFrame) -> np.ndarray:
    combinado = combinar_mj(previsto["mj_km_urbano"], previsto["mj_km_estrada"])
    return erro_percentual(df["mj_km"].to_numpy(), combinado.to_numpy())


def hash8(df: pd.DataFrame) -> str:
    dados = pd.util.hash_pandas_object(df[ENTRADAS + ALVOS], index=False).to_numpy().tobytes()
    return hashlib.sha256(dados).hexdigest()[:8]


def prefixo_estudo(nome: str, h: str) -> str:
    """Início do nome dos estudos de um candidato nesta base; a versão 1 não leva sufixo, para manter os estudos já feitos."""
    versao = (CANDIDATOS | EXPERIMENTAIS)[nome].VERSAO
    return f"{nome}-{h}" if versao == 1 else f"{nome}-{h}-v{versao}"


def _estudo(
    nome: str,
    df: pd.DataFrame,
    n_tentativas: int,
    storage: str | None,
    nome_estudo: str,
    n_dobras: int,
    semente: int,
    rapido: bool = False,
    sampler: str = "tpe",
    semente_sampler: int | None = None,
) -> optuna.Study:
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler: {sampler}")
    cls = (CANDIDATOS | EXPERIMENTAIS)[nome]
    estudo = optuna.create_study(
        study_name=nome_estudo,
        storage=storage,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    # semente deslocada pelos trials já feitos: retomar não repete as mesmas amostras
    base = semente if semente_sampler is None else semente_sampler  # as dobras seguem com `semente`
    estudo.sampler = SAMPLERS[sampler](seed=base + len(estudo.trials))
    dobras = dobras_por_grupo(df["grupo_cv"], n_dobras, semente)

    def objetivo(trial: optuna.Trial) -> float:
        params = cls.sugerir(trial)
        if rapido:  # a busca precisa treinar com os mesmos parâmetros do refit rápido
            params = {**params, **cls.PARAMS_RAPIDOS}
        erros = []
        for k, (tr, te) in enumerate(dobras):
            modelo = cls(semente=semente, **params).fit(df.iloc[tr][ENTRADAS], df.iloc[tr][ALVOS])
            previsto = modelo.predict(df.iloc[te][ENTRADAS])
            erros.append(erro_mj_combinado(df.iloc[te], previsto))
            trial.report(float(np.concatenate(erros).mean()), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.concatenate(erros).mean())

    # n_tentativas é o total do estudo: retomar completa só as que faltam (uma tentativa interrompida não conta)
    feitas = sum(t.state.is_finished() for t in estudo.trials)
    estudo.optimize(objetivo, n_trials=max(0, n_tentativas - feitas))
    return estudo


def _params_finais(estudo: optuna.Study, nome: str, rapido: bool) -> dict:
    params = dict(estudo.best_params)
    if rapido:
        params = {**params, **(CANDIDATOS | EXPERIMENTAIS)[nome].PARAMS_RAPIDOS}
    return params


def buscar_params(
    nome: str,
    df: pd.DataFrame,
    n_tentativas: int,
    storage: str | None,
    nome_estudo: str,
    n_dobras: int = 3,
    rapido: bool = False,
    semente: int = 0,
    sampler: str = "tpe",
) -> dict:
    estudo = _estudo(nome, df, n_tentativas, storage, nome_estudo, n_dobras, semente, rapido, sampler)
    return _params_finais(estudo, nome, rapido)


@dataclass
class ResultadoCV:
    nome: str
    oof: pd.DataFrame
    params_por_dobra: list[dict]
    erro_mj: float
    erro_por_propulsao: dict[str, float]
    erro_por_alvo: dict[str, float]
    n_tentativas: int


def _resultado(
    nome: str, df: pd.DataFrame, oof: pd.DataFrame, params_por_dobra: list[dict], n_tentativas: int
) -> ResultadoCV:
    erro = erro_mj_combinado(df, oof)
    erro_por_propulsao = pd.Series(erro, index=df["propulsao"].to_numpy()).groupby(level=0).mean()
    erro_por_alvo = {
        alvo: float(erro_percentual(df[alvo][df[alvo].notna()], oof[alvo][df[alvo].notna()]).mean())
        for alvo in ALVOS
    }
    return ResultadoCV(
        nome=nome,
        oof=oof,
        params_por_dobra=params_por_dobra,
        erro_mj=float(erro.mean()),
        erro_por_propulsao={str(p): float(e) for p, e in erro_por_propulsao.items()},
        erro_por_alvo=erro_por_alvo,
        n_tentativas=n_tentativas,
    )


def cv_aninhada(
    nome: str,
    df: pd.DataFrame,
    n_tentativas: int,
    storage: str | None,
    n_externas: int = 5,
    n_internas: int = 3,
    rapido: bool = False,
    semente: int = 0,
    sampler: str = "tpe",
    semente_sampler: int | None = None,
) -> ResultadoCV:
    cls = (CANDIDATOS | EXPERIMENTAIS)[nome]
    h = hash8(df)
    oof = pd.DataFrame(np.nan, index=df.index, columns=ALVOS)
    params_por_dobra: list[dict] = []
    n_total = 0
    sufixo_sampler = "" if sampler == "tpe" else f"-{sampler}"  # o TPE mantém o nome antigo
    if semente_sampler is not None:
        sufixo_sampler += f"-ss{semente_sampler}"
    for k, (tr, te) in enumerate(dobras_por_grupo(df["grupo_cv"], n_externas, semente)):
        treino = df.iloc[tr]
        nome_estudo = (
            f"{prefixo_estudo(nome, h)}{sufixo_sampler}-e{n_externas}-i{n_internas}-s{semente}-ext{k}"
        )
        estudo = _estudo(
            nome,
            treino,
            n_tentativas,
            storage,
            nome_estudo,
            n_internas,
            semente,
            rapido,
            sampler,
            semente_sampler,
        )
        n_total += len(estudo.trials)
        params = _params_finais(estudo, nome, rapido)
        modelo = cls(semente=semente, **params).fit(treino[ENTRADAS], treino[ALVOS])
        oof.iloc[te] = modelo.predict(df.iloc[te][ENTRADAS])[ALVOS].to_numpy()
        params_por_dobra.append(params)

    return _resultado(nome, df, oof, params_por_dobra, n_total)


def cv_fixa(
    nome: str,
    df: pd.DataFrame,
    params: dict,
    n_externas: int = 5,
    semente_dobras: int = 0,
    semente_modelo: int = 0,
    colunas: list[str] = ENTRADAS,
) -> ResultadoCV:
    """CV externa sem busca: treina com os mesmos params em cada dobra (as dobras são as de cv_aninhada)."""
    cls = (CANDIDATOS | EXPERIMENTAIS)[nome]
    oof = pd.DataFrame(np.nan, index=df.index, columns=ALVOS)
    for tr, te in dobras_por_grupo(df["grupo_cv"], n_externas, semente_dobras):
        modelo = cls(semente=semente_modelo, **params).fit(df.iloc[tr][colunas], df.iloc[tr][ALVOS])
        oof.iloc[te] = modelo.predict(df.iloc[te][colunas])[ALVOS].to_numpy()
    return _resultado(nome, df, oof, [params] * n_externas, 0)
