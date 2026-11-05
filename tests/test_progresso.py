import optuna
import pytest

from eficiencia.progresso import main, progresso


@pytest.fixture
def banco(tmp_path):
    # uma conexão só, fechada no fim: o Windows não apaga a pasta temporária com o banco aberto
    banco = optuna.storages.RDBStorage(f"sqlite:///{(tmp_path / 'estudos.db').as_posix()}")
    yield banco
    banco.remove_session()
    banco.engine.dispose()


def _estudo(banco, nome, valores, rodando=False, abandonada=False):
    estudo = optuna.create_study(study_name=nome, storage=banco)
    if abandonada:  # tentativa cortada por uma queda, antes da retomada
        estudo.ask()
    for v in valores:
        estudo.add_trial(optuna.trial.create_trial(params={}, distributions={}, value=v))
    if rodando:
        estudo.ask()
    return estudo


def test_progresso_por_modelo(banco):
    for k in range(5):
        _estudo(banco, f"linear-abc-e5-i3-s0-ext{k}", [0.08, 0.07])
    _estudo(banco, "linear-abc-i3-s0-final", [0.06, 0.09, 0.10], abandonada=True)  # passa da meta: conta só 2
    _estudo(banco, "xgboost-abc-e5-i3-s0-ext0", [0.05], rodando=True)
    _estudo(banco, "xgboost-velho-e5-i3-s0-ext0", [0.01])  # outra base: ignorado
    _estudo(banco, "rede_multi-abc-e5-i3-s0-ext0", [0.01])  # arquitetura antiga da rede: ignorado
    _estudo(banco, "rede_por_alvo-abc-v2-e5-i3-s0-ext0", [0.02])

    linhas = {linha["modelo"]: linha for linha in progresso(banco.url, meta=2, h="abc")}

    assert linhas["linear"] == {
        "modelo": "linear",
        "feitas": 12,
        "meta": 12,
        "melhor": 0.06,
        "situacao": "concluído",
    }
    assert linhas["xgboost"]["feitas"] == 1
    assert linhas["xgboost"]["melhor"] == 0.05
    assert linhas["xgboost"]["situacao"] == "em treino"
    assert linhas["rede_multi"]["feitas"] == 0
    assert linhas["rede_por_alvo"]["feitas"] == 1
    assert linhas["floresta"] == {
        "modelo": "floresta",
        "feitas": 0,
        "meta": 12,
        "melhor": None,
        "situacao": "pendente",
    }


def test_main_mostra_o_que_falta(banco, capsys):
    _estudo(banco, "linear-abc-e5-i3-s0-ext0", [0.0712])
    main(["--tentativas", "2", "--banco", banco.url, "--hash", "abc"])
    saida = capsys.readouterr().out
    assert "linear" in saida and "1/12" in saida and "7,12 %" in saida
    assert "faltam 59 tentativas" in saida
