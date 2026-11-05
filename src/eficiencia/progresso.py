"""Progresso do treino, lido do banco do Optuna. Uso: uv run eficiencia-progresso --tentativas 100"""

import argparse

import optuna
from optuna.trial import TrialState

from eficiencia.esquema import carregar_base
from eficiencia.formato import numero
from eficiencia.modelos import CANDIDATOS
from eficiencia.treino import STORAGE
from eficiencia.validacao import hash8, prefixo_estudo

ESTUDOS_POR_MODELO = 6  # 5 dobras externas + o estudo final, como em treino.treinar


def _rodando(trials: list) -> bool:
    """Uma tentativa cortada por queda do processo fica RUNNING para sempre; se outra depois dela já
    terminou, o treino foi retomado e ela está abandonada."""
    abertas = [t.number for t in trials if t.state == TrialState.RUNNING]
    terminadas = [t.number for t in trials if t.state.is_finished()]
    # shortcut: se o processo caiu e ainda não foi retomado, a última tentativa aparece como "em treino"
    return bool(abertas) and max(abertas) > max(terminadas, default=-1)


def progresso(storage: str, meta: int, h: str) -> list[dict]:
    """Uma linha por candidato: tentativas concluídas, meta, melhor erro da busca e situação."""
    banco = optuna.storages.RDBStorage(storage)
    try:
        estudos: dict[str, list] = {nome: [] for nome in CANDIDATOS}
        prefixos = {nome: prefixo_estudo(nome, h) + "-" for nome in CANDIDATOS}
        for resumo in optuna.get_all_study_summaries(banco):
            # o prefixo deixa de fora estudos de outra base ou de outra versão do modelo
            nome = resumo.study_name.split("-")[0]
            if nome in prefixos and resumo.study_name.startswith(prefixos[nome]):
                estudos[nome].append(optuna.load_study(study_name=resumo.study_name, storage=banco).trials)
    finally:
        banco.remove_session()
        banco.engine.dispose()

    linhas = []
    for nome, lista in estudos.items():
        feitas = sum(min(meta, sum(t.state.is_finished() for t in trials)) for trials in lista)
        valores = [t.value for trials in lista for t in trials if t.state == TrialState.COMPLETE]
        total = meta * ESTUDOS_POR_MODELO
        if any(_rodando(trials) for trials in lista):
            situacao = "em treino"
        else:
            situacao = "concluído" if feitas >= total else "pendente"
        linhas.append(
            {
                "modelo": nome,
                "feitas": feitas,
                "meta": total,
                "melhor": min(valores) if valores else None,
                "situacao": situacao,
            }
        )
    return linhas


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mostra quanto do treino já foi feito.")
    parser.add_argument(
        "--tentativas", type=int, default=50, help="o mesmo valor passado ao eficiencia-treinar"
    )
    parser.add_argument("--banco", default=STORAGE, help="URL do banco do Optuna")
    parser.add_argument("--hash", help="hash da base (padrão: o da base atual)")
    args = parser.parse_args(argv)

    linhas = progresso(args.banco, args.tentativas, args.hash or hash8(carregar_base()))
    print(f"{'modelo':<15}{'tentativas':>12}{'melhor na busca':>17}  situação")
    for linha in linhas:
        melhor = "-" if linha["melhor"] is None else numero(100 * linha["melhor"], 2) + " %"
        tentativas = f"{linha['feitas']}/{linha['meta']}"
        print(f"{linha['modelo']:<15}{tentativas:>12}{melhor:>17}  {linha['situacao']}")
    print(f"faltam {sum(linha['meta'] - linha['feitas'] for linha in linhas)} tentativas")
