# Validação aprofundada do estimador veicular

**18 de novembro de 2026.** A PR [#37](https://github.com/FelipeEckert/fuel-efficiency-deeplearning/pull/37) estava integrada à `main` em `2045372`. O servidor local usava o código de `155a9c3`. Conferimos que a diferença entre as revisões se restringia aos dois relatórios de auditoria.

## Decisões de produto sustentadas pela rodada

### 1. Preservar identificadores textuais no Excel — defeito reproduzido

Importar duas linhas cujas células **Name** contêm os textos `0012` e `0142`. Exportar o portfólio e inspecionar os tipos das células. O arquivo baixado contém os **números 12 e 142**. Em outro ensaio, os nomes literais `NA` e `N/A` viraram ausentes. Isso compromete o cruzamento posterior com cadastros mesmo quando o consumo e o volume estão corretos. O arquivo original permanece íntegro. Um revisor independente confirmou a perda também pelo leitor do projeto.

**Aceite.** Conservar o texto original de **Name** da leitura à exportação. Cobrir zeros iniciais, `NA`, `N/A`, um nome alfanumérico e uma célula realmente vazia na mesma bateria de regressão. [Entrada com nomes textuais](../docs/validacao-aprofundada/evidencias/qa-deep-xlsx-names.xlsx) · [Arquivo baixado da interface](../docs/validacao-aprofundada/evidencias/qa-deep-xlsx-confirm-export-names.xlsx)

**Como conferir o tipo de célula.** Nos dois arquivos acima, a coluna `Name` pode ser inspecionada com `openpyxl`. Na entrada, os pares `(valor, data_type)` são `("0012", "s")` e `("0142", "s")`. Na exportação, são `(12, "n")` e `(142, "n")`. O teste de regressão deve conferir tipo e valor após salvar e reabrir o XLSX. Um formato visual de quatro dígitos pode mascarar a perda do texto. Isso importa para fontes como o MLIT japonês, que identifica um subtipo do Daihatsu Thor por `0001`. [Exemplo japonês](fontes-internacionais.md#japão)

### 2. Explicitar quando uma célula de consumo conhecido vira estimativa — decisão de contrato

Em uma carteira com duas linhas de mesmo volume, uma das fichas técnicas estava completa. Com **Known MJ/km** igual ao texto literal `NA` nessa linha e 3 na outra, o portfólio incluiu ambas, exibiu **2,43 MJ/km** e uma faixa de **2,36–2,52**. A exportação gravou a célula como vazia. Trocando apenas `NA` por `oops`, o aplicativo informou valor inválido e calculou **3 MJ/km**, com uma linha e 50% do volume. A diferença já nasce na leitura da planilha.

Tratar `NA` como “sem etiqueta disponível” pode ser intencional. O risco é um usuário não perceber que a média agora inclui uma estimativa. **Aceite.** Documentar os marcadores aceitos, identificar no resumo quantas linhas e quanto volume vieram de estimativas e testar separadamente vazio, marcador de ausência e texto inválido. [Resultado com marcador](../docs/validacao-aprofundada/evidencias/qa-deep-xlsx-known-na.png) · [Controle com texto inválido](../docs/validacao-aprofundada/evidencias/qa-deep-xlsx-known-invalid.png)

### 3. Identificar o ciclo na largura do intervalo da ficha — problema de documentação

A tabela da ficha informa meia largura “típica” de ±30,8% para plug-ins. No pacote distribuído, esse número coincide com a **estrada**. A API usa um quantil específico para o combinado. Ler essa tabela como se descrevesse o resultado combinado leva a uma interpretação incorreta.

| Propulsão | Combinado | Cidade | Estrada |
|---|---:|---:|---:|
| Combustão | ±9,04% | ±12,40% | ±9,78% |
| Híbrido | ±19,98% | ±24,59% | ±15,68% |
| Elétrico | ±10,41% | ±14,29% | ±9,60% |
| Plug-in | **±24,62%** | **±33,31%** | **±30,83%** |

**Aceite.** Gerar as três colunas a partir do mesmo pacote usado pela interface e identificar o ciclo. Os resíduos de calibração gravados no pacote recuperam 46/50 plug-ins dentro da faixa combinada. Para testar a promessa de cobertura em famílias novas é necessária uma amostra independente.

### 4. Delimitar a interpretação de diesel em plug-ins — lacuna de validação empírica

O sistema aceita a combinação, emite aviso de combustível ausente no treino daquele grupo e mostra km/l. Três fichas sintéticas produziram **42,74–43,71 km/l de diesel**. O caminho aplica a MJ/km um fator de diesel calculado em combustão e híbridos sem tomada. A descrição dos modelos dedicados de combustível do plug-in cobre gasolina e etanol; não explica essa exceção. Há um teste explícito que exige resultado finito e aviso para a combinação.

**Aceite.** Tornar visível ao lado do km/l a origem desse cálculo e registrar a exceção na ficha. Uma medição de referência seria necessária para estabelecer o erro real ou justificar um limiar de uso. Os valores acima demonstram o comportamento da implementação, sem quantificar sua exatidão. [Resultados da API por combinação](../docs/validacao-aprofundada/evidencias/qa-deep-model-probe-321.csv)

## Fluxos e propriedades verificados

| Ensaio | Controle e resultado |
|---|---|
| Duas sessões independentes de Chrome | A abriu 25 configurações estimadas; B começou vazia e importou um portfólio conhecido com média de 1,18 MJ/km. Um upload sem coluna obrigatória preservou o resultado anterior em B. Recarregar B esvaziou apenas B. A continuou em **1,04 MJ/km** e exportou 25 identificadores distintos, volume total 325, na ordem esperada. |
| Cálculo com oráculo independente para a agregação | Prever linha a linha cinco configurações de propulsão/combustível em cinco níveis de massa (−20%, −10%, base, +10%, +20%), recalculando a classe de inércia. Agregar separadamente com pesos de 1 a 25. A média esperada foi **1,0363497046 MJ/km**; o navegador apresentou 1,04, inclusive após inverter a planilha. Isso verifica agregação e integração, sem aferir a verdade física das previsões. |
| Invariantes nos mesmos 25 registros | A maior diferença entre inferência individual e em lote para MJ/km combinado foi **zero**. Nos cinco perfis, aumentar a massa e recalcular a inércia não reduziu o resultado bruto. Em todas as linhas, o combinado correspondeu a 55% cidade + 45% estrada; os três valores centrais ficaram entre limites positivos e ordenados. |
| Efeito da ordem sobre a faixa de portfólio | Com as mesmas 25 linhas, a API retornou **[0,99662; 1,06842]**; em ordem inversa, **[0,99504; 1,06958] MJ/km**. A média se manteve. Os limites das duas faixas arredondam para os mesmos centésimos nesta amostra. É sensibilidade à reamostragem, distinta do estreitamento por duplicação já registrado na PR #36. |
| Simulador × portfólio | No veículo flex com −10% de massa, o cenário exportado usa o resultado suavizado de **1,86277 MJ/km**. Para a mesma ficha, a previsão direta do portfólio foi **1,86478**. Ambos exibem 1,86. Nas 25 combinações, a maior diferença absoluta entre caminhos foi **0,0380 MJ/km**. Identificar a suavização na exportação evita que consumidores interpretem as diferenças de método como corrupção dos números. |
| Cenários com nomes repetidos e navegação | Salvar `  Urban test  `, depois `Urban test` e `urban test` com o ajuste de −10%. O XLSX trouxe `Urban test`, `Urban test (2)` e `urban test`, respectivamente **1,93965**, **1,86277** e **1,86277 MJ/km**. Ir a Comparison e voltar preservou as células de todos os cenários exportados. Os controles correntes e o nome em rascunho voltaram aos padrões. |
| Aplicabilidade e sensibilidade | Testar 10 pares aceitos de propulsão e combustível em três condições cada. Repetir índices de DataFrame preservou a ordem e as previsões do lote. Três variações isoladas de bateria/autonomia deram o mesmo resultado de etanol no plug-in flex. A contraprova com **240** fichas variadas gerou **240 valores distintos**, de **6,11 a 8,32 km/l**; 216 passaram sem aviso de extrapolação. Assim, essa pequena amostra inicial não sustenta a hipótese de saída constante. |

**Planilhas para repetir o ensaio de carteira:** [25 fichas e pesos](../docs/validacao-aprofundada/evidencias/deep25.xlsx), [mesmas linhas em ordem inversa](../docs/validacao-aprofundada/evidencias/deep25reverse.xlsx), [exportação de A após recarregar B](../docs/validacao-aprofundada/evidencias/deep-persisted.xlsx). **Outras evidências:** [cenários salvos](../docs/validacao-aprofundada/evidencias/qa-deep-vehicle-duplicates.xlsx) e [240 resultados de plug-ins flex](../docs/validacao-aprofundada/evidencias/qa-deep-model-support-321.csv).

## Recuperação depois de um erro

No ensaio do agente de veículo, 301 kg com redução de 10% gerou um ajuste inválido de 270,9 kg. O botão permitiu salvar e exportar **301 kg e 1,72548 MJ/km**, reproduzindo o defeito já registrado. Depois de restaurar o original, informar **1.500 kg** e aplicar novamente −10%, a exportação trouxe corretamente **1.350 kg e 1,85856 MJ/km**. Remover a referência promoveu o cenário seguinte e preservou os valores das quatro linhas restantes. Recarregar completamente a página apagou essas quatro linhas e retirou o botão de exportação. Portanto, o botão **Save scenario** mantém resultados na sessão; a exportação é a forma verificada de guardá-los após a recarga.

O agente de Excel também confirmou que um campo de consumo realmente vazio e o marcador `NA` produziram os mesmos números e a mesma faixa no navegador. Um arquivo corrompido e uma planilha sem cabeçalhos obrigatórios preservaram a tabela anterior. Em arquivo de múltiplas abas, o leitor usou a primeira mesmo quando a segunda estava ativa no Excel; a interface não oferece escolha de aba. [Ensaios adicionais de Excel](../docs/validacao-aprofundada/evidencias/qa-deep-xlsx-report.md) · [Roteiro de cenários e recuperação](../docs/validacao-aprofundada/evidencias/qa-deep-vehicle-report.md)

## O que permitiria ampliar a evidência

- Para os fluxos da interface, o Chrome e o Playwright locais já permitem controlar sessões separadas, anexar XLSX, editar a grade e inspecionar os arquivos exportados. Uma planilha anonimizada de uso habitual, com o resultado esperado e as convenções de campos vazios e identificadores, permitiria testar o contrato praticado pela equipe.
- Para aferir o erro e a cobertura de 90%, são necessárias fichas com resultados reais de homologação **independentes do treino e da calibração**, de preferência separadas por família e safra. Os 240 casos sintéticos exercitam o software, mas não acrescentam 240 medições reais.
- Para classificar lentidão como regressão, definir um volume de carteira e uma latência aceitáveis. Os testes de isolamento usaram duas sessões concorrentes; a rodada não estabeleceu o limite de usuários simultâneos nem de linhas suportadas.

A rodada usou a skill de desenvolvimento com subagentes para três frentes de testes e uma revisão independente dos achados centrais. O servidor e o perfil de navegador do usuário permaneceram separados dos perfis sintéticos dos testes.

## Como repetir o teste das duas sessões no Windows

Na raiz de uma cópia do repositório, execute `uv sync` e `uv run streamlit run app/Inicio.py`. Mantenha o servidor aberto na porta 8501. Em um segundo PowerShell, também na raiz, execute o trecho abaixo. Ele cria uma pasta temporária exclusiva para entradas e downloads. O roteiro usa o Google Chrome instalado e a interface em inglês. A execução original da automação usou Python 3.13 e Playwright 1.65.2.

```powershell
$evidencias = (Resolve-Path 'docs/validacao-aprofundada/evidencias').Path
$tempAnterior = $env:TEMP
try {
    $env:TEMP = (New-Item -ItemType Directory -Path (Join-Path $tempAnterior ([guid]::NewGuid().ToString()))).FullName
    $destino = New-Item -ItemType Directory -Path (Join-Path $env:TEMP 'fuel-ui-20261118')
    Copy-Item "$evidencias/*" -Destination $destino.FullName
    uv --system-certs run --no-project --python 3.13 --with playwright==1.65.2 python "$evidencias/qa-deep-root.py"
} finally {
    $env:TEMP = $tempAnterior
}
```

O roteiro verifica a média de cada carteira, a ordem inversa, o upload inválido e o isolamento após recarga. O exemplar publicado da automação aguarda que a métrica termine de renderizar após trocar de arquivo. As planilhas que entram no teste e as exportações originais estão na [pasta de evidências](../docs/validacao-aprofundada/evidencias). Os logs foram convertidos para UTF-8 para leitura no GitHub. O arquivo `qa-deep-xlsx-corrupt.xlsx` é intencionalmente ilegível e exercita a recuperação de um upload corrompido.

Para planejar a comparação com veículos homologados, consulte a [pesquisa de fontes internacionais](fontes-internacionais.md).
