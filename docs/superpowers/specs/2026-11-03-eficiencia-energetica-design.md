# Estimador de eficiência energética veicular — design

Data: 2026-11-03 · Status: escopo aprovado; revisado por leitor independente; aguardando aprovação do documento

## 1. Intenção

Um time técnico pequeno (planejamento de produto / engenharia, 2–10 pessoas, todos sabem o que é
F0/F2 de coastdown) precisa estimar o consumo energético de **veículos que ainda não existem**
para tomar decisões de portfólio com foco em CAFE (média corporativa de eficiência).

A ferramenta recebe as especificações técnicas de um veículo (ou de um portfólio inteiro) e
estima o consumo no ciclo PBEV — MJ/km e km/l — com faixa de incerteza, avisando quando o
veículo está fora do que a base de dados cobre, e permitindo simular cenários ("e se eu tirar
80 kg?").

O projeto também é uma comparação honesta entre modelos clássicos de regressão e deep learning
sobre a mesma base.

## 2. Objetivos

1. Prever, a partir de especificações técnicas, os valores de etiqueta PBEV (valores *corrigidos*).
2. Comparar regressão linear, Random Forest, XGBoost e duas formas de rede neural em PyTorch
   na mesma validação, e usar o vencedor no app.
3. App Streamlit (MVP) com: estimativa de um veículo, cenários "e se", portfólio ponderado por
   volume, página de comparação de modelos.
4. Motor (dados, treino, previsão, incerteza, checagem de faixa) separado da interface, para
   trocar o frontend no futuro sem mexer nele.
5. Repositório privado com CI, testes, e o modelo vencedor versionado.

**Meta de qualidade** (meta, não bloqueio): erro percentual absoluto médio ≤ 5 % no MJ/km
combinado, reportado **por propulsão**. Referência já medida: regressão linear de log(MJ/km)
sem tuning, CV por grupo, dá 7,7 % no geral (Combustão 5,9 %, Elétrico 7,8 %, Híbrido 8,9 %,
Plug-in 16,6 %). Plug-in (50 homologações) será o grupo mais difícil.

## 3. Não-objetivos (fora do MVP)

- Meta CAFE / Mover e comparação da média do portfólio com a meta.
- Bônus de off-cycle (será um desconto calculado fora do modelo, em versão futura).
- "E se" aplicado ao portfólio inteiro (ex.: −5 % de massa em todos os SUVs).
- Gráfico de sensibilidade (tornado).
- Estimar F0/F2 quando o usuário não souber.
- Servidor / deploy; API HTTP; tradução para inglês.
- Ver previsões de todos os modelos no app (só o vencedor; comparação fica numa página de métricas).
- Ferramenta de tracking externa (MLflow etc.).

## 4. Anti-objetivos (seria fracasso mesmo funcionando)

- **Vazamento de resposta**: versões do mesmo modelo (ou o mesmo veículo em marcas diferentes)
  em treino e teste ao mesmo tempo; colunas que são resultado do ensaio (autonomia, CO₂,
  poluentes, nota verde, fatores EHR, colunas de auditoria) usadas como entrada; métrica
  reportada no mesmo laço que escolheu hiperparâmetros.
- Número seco sem faixa de incerteza.
- Extrapolar em silêncio: veículo fora da faixa da base sem aviso.
- Modelo que depende da marca em vez da física.
- "E se" incoerente: mais massa, F0 ou F2 reduzindo o consumo.

## 5. Restrições

- Máquina de desenvolvimento: Windows 11, Python 3.14, CPU (não assumir GPU).
- Todas as bibliotecas necessárias têm wheel para Python 3.14 / Windows (verificado em
  nov/2026: torch 2.15, xgboost 3.4, scikit-learn 1.9, pyarrow 26, streamlit 1.66).
- Gerenciamento: `uv` + `pyproject.toml`.
- Dados ficam no repositório (privado; o usuário autorizou).
- Interface e nomes no código em **português** (como `scripts/limpar_pbev.py`).
- Base pequena: 510 linhas, 258 grupos de modelo. Previsão no app instantânea; treino na CPU sem
  limite de tempo (ver 7.2).

## 6. Dados (pronto)

Já implementado em `scripts/limpar_pbev.py`; documentação completa em `data/README.md`.

- Fonte: aba "Dados de Entrada" de `data/raw/PED - PLANILHA DE ENTRADA DE DADOS PBEV 2026_05-05-2026.xlsm`
  (lida direto do XML, sem macros) + `data/raw/complemento_baterias.csv` (bateria de 57
  homologações pesquisada em fontes públicas, com link e confiança).
- Saída: `data/processed/pbev_2026_limpo.csv`, uma linha por homologação (510), `grupo_cv` para
  validação por grupo (258 grupos). O grupo junta variantes do mesmo carro (tração, câmbio,
  combustível, acabamento, ano-modelo: Hilux AT/MT, Ranger 4X2/4X4, HR-V/HR-V mod. 26…) e mantém
  separadas carrocerias diferentes (Onix × Onix Plus, Corolla × Corolla Cross, Defender 90 × 110)
  — minha decisão: um veículo futuro pode ser "irmão" de carroceria de um existente, então essa
  proximidade é realista.
- O script já verifica suas invariantes com `assert` ao rodar; o builder deve transformar essa
  checagem em teste pytest e mover a lógica para o pacote (o script pode virar um atalho).
- `lcvm` e `nome_homologado` são só identificação do governo; não entram no modelo.
- Atualização anual: trocar a planilha em `data/raw/`, rodar limpeza, rodar treino (2 comandos
  documentados no README).

### 6.1 Entradas do modelo

| coluna | tipo | observação |
|---|---|---|
| `propulsao` | cat | Combustão, Híbrido, Plug-in, Elétrico |
| `combustivel` | cat | gasolina, flex, diesel, eletrico |
| `categoria` | cat | 16 códigos PBEV |
| `categoria_proconve` | cat | PL, CL, CL1, CL2, CL3 (1 linha vazia) |
| `massa_om_kg` | num | |
| `classe_inercia_kg` | num | 1 linha vazia |
| `pbt_kg` | num | 1 linha vazia |
| `f0_n` | num | N |
| `f2_n_kmh2` | num | N/(km/h)² |
| `potencia_kw` | num | |
| `cilindrada_cm3` | num | 0 em elétrico |
| `transmissao` | cat | automatica, manual, cvt, dct, dht, direta |
| `n_marchas` | num | vazio em CVT/DHT (72 linhas) |
| `direcao` | cat | eletrica, hidraulica, eletro-hidraulica |
| `bateria_kwh` | num | 0 em combustão |
| `bateria_estimada` | bool | True em 17 híbridos (valor inferido ou mediana) |

**Fora**: `marca`, `modelo`, `area_m2` (testado: não melhora; é comprimento × largura, não área
frontal), `lcvm`, `nome_homologado`, e todo resultado de ensaio.

### 6.2 Alvos e conversões

Modelados (6), só nas linhas em que existem:

| alvo | linhas com valor |
|---|---|
| `mj_km_urbano`, `mj_km_estrada` | 510 |
| `kml_gasolina_urbano`, `kml_gasolina_estrada` | 344 (gasolina, flex, híbridos e plug-in a gasolina) |
| `kml_etanol_urbano`, `kml_etanol_estrada` | 168 (flex) |

Calculados (fórmulas verificadas nos dados):

- `mj_km` (combinado) = 0,55 × urbano + 0,45 × estrada — exato nas 510 linhas.
- km/l combinado = 1 / (0,55 / urbano + 0,45 / estrada) — média **harmônica**; bate em 100 %
  das 423 comparações feitas na planilha de especificações do PBEV.
- km/l equivalente elétrico (BEV e PHEV) = 20,29 / MJ/km (urbano e estrada separadamente);
  exato exceto no único PHEV flex (19,95).
- km/l diesel = fator / MJ/km, com fator ≈ 27,9 (urbano) — variação ±1,5 % na base; o builder
  calcula os fatores urbano/estrada como mediana dos dados (57 homologações com km/l diesel).

O app mostra, por veículo: MJ/km combinado (principal), urbano e estrada; km/l urbano, estrada e
combinado de **cada combustível aplicável** (flex: gasolina e etanol; elétrico/PHEV: km/l
equivalente; diesel: diesel).

## 7. Modelagem

### 7.1 Validação

- **Validação aninhada por grupo** (`grupo_cv`): 5 partes externas × 3 internas, grupos nunca
  divididos entre treino e teste.
- O laço interno escolhe hiperparâmetros (Optuna); o externo mede o erro reportado.
- Métrica principal: erro percentual absoluto médio do **MJ/km combinado**, calculado a partir
  das previsões urbano + estrada. Reportar também por propulsão e por alvo.
- As previsões fora-da-amostra do laço externo alimentam a incerteza (7.4) e o relatório.

### 7.2 Candidatos

1. Regressão linear regularizada (tuning só da força de regularização).
2. Random Forest (grade pequena e fixa).
3. XGBoost com **restrição monotônica**: MJ/km não diminui com `massa_om_kg`,
   `classe_inercia_kg`, `f0_n`, `f2_n_kmh2` (e km/l não aumenta com elas). ~50 tentativas Optuna.
4. Rede neural PyTorch, **uma rede por alvo**. ~50 tentativas Optuna.
5. Rede neural PyTorch **multi-saída**: uma rede para os 6 alvos, perda mascarada (alvo
   ausente na linha não contribui). ~50 tentativas Optuna.

Todos comparados na mesma validação. Optuna guarda os estudos em SQLite local (fora do git),
visíveis com `optuna-dashboard`.

**Orçamento de tempo** (decisão do usuário): **sem limite**. O usuário pode deixar a máquina
otimizando durante a noite, todos os dias. Consequências para o build:
- número de tentativas por candidato configurável na linha de comando (padrão ~50);
- estudos Optuna **retomáveis**: rodar de novo continua o estudo salvo no SQLite, acumulando
  tentativas noite após noite, em vez de recomeçar;
- poda (pruning) e parada antecipada nas redes continuam desejáveis para render mais tentativas
  por noite;
- um modo rápido separado para desenvolvimento e para o treino de fumaça do CI;
- o relatório registra quantas tentativas cada candidato acumulou.

### 7.3 Escolha e treino final

- Um **único algoritmo** para todos os alvos: o de menor erro no MJ/km combinado (laço externo).
  Motivo: o "e se" fica coerente entre MJ/km e km/l.
- Treino final: rerodar a busca do vencedor com CV por grupo em todos os dados, retreinar com
  os melhores hiperparâmetros em todos os dados.
- Teste automático de monotonia (aumentar massa/F0/F2 de um veículo nunca reduz MJ/km),
  aplicado a **todos os candidatos** sobre um conjunto fixo de veículos da base.
- **Só concorre ao posto de vencedor quem passa no teste de monotonia** (minha decisão; o "e se"
  incoerente é anti-objetivo). O vencedor é o de menor erro entre os que passam; o relatório mostra
  todos, inclusive os reprovados. O teste roda no CI contra o artefato versionado e precisa estar verde.
- **Todos os candidatos são monotônicos por construção** (decisão do usuário, 2026-11-04, após o
  treino rápido mostrar que só o XGBoost passava): regressão linear com sinal dos coeficientes
  restrito; Random Forest com `monotonic_cst`; redes com arquitetura monotônica (caminho de pesos
  positivos para massa, classe de inércia, F0 e F2, modulado pelas demais entradas). Busca das redes
  ampliada: 1–5 camadas ocultas, 32–512 neurônios. Assim os cinco competem de igual para igual.

### 7.4 Incerteza

Conformal por propulsão: a partir dos erros relativos fora-da-amostra do laço externo, o quantil
de 90 % por propulsão define a faixa (± %) aplicada à previsão. Exibido como
"1,62 MJ/km (1,54 – 1,70)". Teste: cobertura empírica perto de 90 %.

### 7.5 Aviso de fora da faixa

Para cada campo numérico, comparar com mínimo e máximo dos veículos **da mesma propulsão** na
base de treino; listar os campos fora, com o limite (ex.: "F2 = 0,025 abaixo de qualquer elétrico
da base: 0,029"). Só avisa, não bloqueia. Categorias nunca vistas para aquela propulsão também
geram aviso.

### 7.6 Artefatos (versionados no git)

- Modelo vencedor com pesos, pré-processamento, quantis conformal, faixas por propulsão e
  metadados (algoritmo, hiperparâmetros, data, hash do CSV de treino).
- `metricas.json` (todos os candidatos, por alvo e por propulsão) e relatório Markdown com
  gráficos (previsto × real, erro por propulsão).
- Fora do git: banco SQLite do Optuna.

## 8. Motor (pacote Python)

Interface mínima que o app usa (nomes exatos a critério do builder, em português):

- `prever(veiculos) -> tabela` — aceita um ou vários veículos com as entradas de 6.1; devolve,
  por veículo, cada saída de 6.2 com limite inferior e superior, e a lista de avisos de faixa.
- `classe_inercia(massa_om_kg) -> kg` — tabela NBR 6601 (abaixo).
- `media_portfolio(veiculos_com_volume) -> resultado` — MJ/km ponderado por volume, e quebra por
  propulsão e categoria; usa "MJ/km conhecido" quando informado, senão a previsão.
- Comandos: limpar dados; treinar/comparar (gera artefatos e relatório).

Sem API HTTP no MVP. O frontend futuro chamará o mesmo pacote.

### 8.1 Classe de inércia (NBR 6601)

Massa de ensaio = `massa_om_kg + 136`. Classe = classe da maior faixa cujo mínimo ≤ massa de
ensaio. Bate em 92,5 % da base; o resto declara classe fora da tabela, por isso o app deixa
editar. Tabela (aba "Tabela MOM 6601" da planilha PED; mínimo da faixa → classe):

```
0→454  483→510  539→567  596→624  653→680  710→737  766→794  823→850  880→907
936→964  993→1021  1050→1077  1106→1134  1163→1191  1220→1247  1277→1304
1333→1361  1390→1417  1447→1474  1503→1531  1560→1588  1617→1644  1673→1701
1730→1758  1787→1814  1872→1928  1985→2041  2099→2155  2212→2268  2326→2381
2439→2495
```

(A planilha tem uma última linha "2608→2495", que parece erro de digitação; ignorar.) Acima de
~2.500 kg a base tem classes até 2.650 que a tabela não cobre — por isso o campo é editável.

## 9. App Streamlit (MVP)

Roda localmente: `uv run streamlit run <app>`. Em português.

### 9.1 Veículo

- Formulário com as entradas de 6.1 (exceto `bateria_estimada`, sempre False no app).
- F0 e F2 **obrigatórios**; ao lado, o valor típico (mediana) da categoria escolhida como
  referência.
- Classe de inércia preenchida automaticamente pela massa (8.1), editável.
- Bateria obrigatória para Híbrido, Plug-in e Elétrico; 0 fixo em Combustão.
- Campos coerentes com a propulsão (ex.: cilindrada 0 e transmissão direta sugeridas em
  elétrico; combustível "eletrico" só com propulsão Elétrico) — regras exatas a critério do builder.
- Resultado: saídas de 6.2 com faixa de incerteza + avisos de faixa.

### 9.2 "E se" e cenários

- Controles deslizantes ±20 % para massa, F0, F2 e potência a partir do veículo do formulário;
  mostrar novo MJ/km e diferença para o original. Ao mexer na massa, a classe de inércia
  recalcula (minha decisão).
- Salvar cenários com nome ("base", "−80 kg", "pneu de baixa resistência"); gráfico de barras
  lado a lado com a faixa de incerteza e a diferença para o "base"; exportar cenários em Excel.
- Prioridade do usuário: intuitivo e fácil de visualizar.

### 9.3 Portfólio

- Tabela editável no app (adicionar/remover veículos, coluna de volume) **e** upload/download de
  planilha Excel modelo com as mesmas colunas. Sem banco de dados; salvar = baixar o arquivo.
- Coluna opcional **"MJ/km conhecido"** (decisão do usuário) para veículos já homologados que
  continuam no portfólio: quando preenchida, o valor real é usado na média (sem faixa de
  incerteza) e as especificações técnicas daquela linha passam a ser opcionais; quando vazia, o
  modelo estima. A tabela mostra a origem de cada valor (real / estimado).
- Resultado: MJ/km médio ponderado por volume; quebra por propulsão e categoria; previsão e faixa
  de cada veículo estimado na tabela. Recalcula ao editar.
- Não mostrar km/l médio do portfólio (mistura combustíveis).
- Faixa de incerteza da média do portfólio: fora do MVP (minha decisão).

### 9.4 Comparação de modelos

Página com `metricas.json` e o relatório: erro de cada candidato por alvo e por propulsão,
gráfico previsto × real do vencedor.

## 10. Engenharia

- Repositório **privado** `fuel-efficiency-deeplearning` na conta GitHub FelipeEckert (o `gh`
  local está autenticado com escopos `repo` e `workflow`). `main` protegida exigindo CI verde.
  No GitHub Free, repositório privado pessoal pode não permitir proteção de branch; nesse caso
  (minha decisão) vale a convenção: nenhum commit direto na `main`, tudo por PR, merge só pelo
  usuário — e o usuário é avisado.
- `.gitignore`: ambiente, caches, banco Optuna. Planilhas brutas e CSV limpo **vão** para o git.
- **CI (GitHub Actions, Ubuntu)** em cada PR: `ruff` (lint + formato) e `pytest`:
  regras da limpeza; contrato do `prever()`; monotonia; aviso de faixa; cobertura da incerteza;
  soma do portfólio; classe de inércia; treino de fumaça (poucos dados, poucas tentativas). Treino
  completo só local.
- **Fluxo**: cada tarefa do plano = uma branch + uma PR; CI verde; o **usuário** revisa e faz o
  merge.
- **Execução do plano**: `superpowers:subagent-driven-development` — implementação por subagentes
  Haiku 5.5, revisão de código por Sonnet 5.5, orquestração por Opus 5.5. Tarefas independentes
  podem rodar em paralelo.
- Primeira tarefa do plano: `pyproject.toml` com uv (pandas, numpy, scikit-learn, xgboost, torch
  CPU, optuna, optuna-dashboard, streamlit, openpyxl para Excel, plotting a critério do builder;
  dev: pytest, ruff), git init, repositório, CI.

## 11. A critério do builder

- Arquitetura e espaço de busca das redes e do XGBoost/RF; uso de transformação log nos alvos.
- Codificação de categóricas; tratamento de `n_marchas` vazio (CVT/DHT) — consistente entre
  modelos e entre treino e app.
- Estrutura de pastas do pacote, app e testes; biblioteca de gráficos; layout das páginas.
- Rótulos legíveis das 16 categorias PBEV (aba "Critérios Classificação" da planilha PED).
- Formato de serialização dos artefatos.

## 12. Limitações conhecidas

- Sem tração 4x2/4x4 (o campo da planilha descreve o dinamômetro).
- "Híbrido" mistura mild hybrid e híbrido completo; a bateria (kWh) é o que os diferencia.
- Baterias de mild hybrid vêm em parte de bases de dados de imprensa; 17 marcadas como estimadas.
- Plug-in tem poucas linhas e o MJ/km depende da autonomia elétrica — erro maior esperado.
