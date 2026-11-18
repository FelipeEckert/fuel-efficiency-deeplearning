# Casos de uso do estimador no Chrome

**18 de novembro de 2026.** Testes de ponta a ponta no aplicativo Streamlit com o modelo XGBoost distribuído. A referência é a [main após a PR #36](https://github.com/FelipeEckert/fuel-efficiency-deeplearning/commit/1b5dc235f87db67ad49505c10f1757ad93cf6429). Entre a versão do servidor local e essa revisão, o único arquivo alterado foi o [primeiro relatório de auditoria](auditoria-do-produto.md).

## Parecer

A jornada principal funciona. Foi possível abrir um veículo, calcular o efeito de reduzir a massa, salvar duas alternativas, ler os resultados no Excel e comparar os modelos. **O maior risco de decisão continua no portfólio.** Uma importação aparentemente bem-sucedida pode alterar os pesos sem alertar o usuário. Além dos casos de separadores já documentados, o navegador confirmou um caso novo em que só a ordem de duas colunas altera o resultado. A prioridade seguinte é impedir que um cenário inválido seja salvo com o resultado do veículo original.

## Ambiente e critério de verificação

O conector de controle do Windows não conseguiu abrir o canal nativo e a integração de navegador não encontrou uma aba conectada. A alternativa foi Chrome **157.0.8103.79** instalado no Windows, em modo headless, com contextos de navegador temporários. O Playwright dirigiu o Chrome contra o servidor local já em execução. As ações passaram pelos controles da interface, incluindo navegação, entrada de dados, upload, edição de célula e download. As leituras de XLSX foram feitas sobre os arquivos efetivamente baixados. As planilhas de teste contêm apenas veículos fictícios.

As verificações usaram área de visualização de **1440 × 900** e, para a navegação em largura reduzida, **390 × 844**. Os valores abaixo transcrevem a precisão exibida na interface ou indicam explicitamente quando foram lidos do XLSX. Os casos quantitativos aguardaram o resultado da nova execução do Streamlit para evitar atribuir à planilha atual o resultado da anterior.

## Roteiro executado

| Caso | Ação pela interface | Resultado observado |
|---|---|---|
| UC01 · Entrada | Abrir Home e seguir o link Vehicle. | A página inicial apresentou os três destinos. O veículo padrão indicou 1,94 MJ/km combinados, 2,15 na cidade e 1,69 na estrada. |
| UC02 · Comparação de cenários | Salvar a configuração inicial, aplicar **10 % lighter**, salvar novamente e baixar o Excel. | A estimativa ajustada caiu para 1,86 MJ/km (−4% na tela). O XLSX trouxe dois cenários, massas de 1.495 e 1.345,5 kg e resultados de 1,93965 e 1,86277 MJ/km. |
| UC03 · Explicações e restauração | Selecionar o ciclo rodoviário, abrir **Neural network** e clicar em **Back to original**. | A explicação passou a identificar o alvo rodoviário, a aba exibiu a descrição da rede ilustrativa e o resultado ajustado voltou a 1,94 MJ/km. |
| UC04 · Limite físico | Informar massa de 301 kg e aplicar redução de 10%, para 270,9 kg. Tentar salvar e exportar. | A página acusou o limite mínimo de 300 kg. Permitiu salvar e exportar 1,73 MJ/km, valor do veículo original. Ver F03. |
| UC05 · Troca de propulsão | Na largura de 390 px, trocar o veículo padrão para elétrico. | A tela mostrou combustível elétrico, transmissão direta de uma marcha, **6,00 marchas** e estimativa de 0,54 MJ/km. Um aviso informou máximo de uma marcha na amostra elétrica. Em sessão separada, a seleção passou por híbrido, plug-in, elétrico e combustão. Ao voltar a combustão, a transmissão direta herdada também gerou aviso. |
| UC06 · Portfólio conhecido | Baixar o modelo de planilha. Enviar dois veículos com volumes 100 e 10 e consumos conhecidos 1 e 3 MJ/km. | O modelo XLSX tinha 19 colunas e zero veículos. A média exibida foi 1,18 MJ/km, coerente com 130/110. |
| UC07 · Edição e exportação | Na grade de UC06, editar o volume da primeira linha de 100 para 10 e confirmar. Baixar a planilha. | A média passou para 2 MJ/km. O XLSX baixado continha o volume atualizado na linha Alpha. Em um ensaio de ida e volta com os volumes originais, a reimportação preservou a média de 1,18. |
| UC08 · Separadores e tipos | Repetir UC06 com a primeira célula de volume numérica fracionária, texto com ponto e texto com vírgula. | Dois dos três casos produziram média incorreta. Ver os dados e o valor de controle em F01. |
| UC09 · Cabeçalhos ambíguos | Enviar duas planilhas com os mesmos dados e as colunas `Volume` e `volume` em ordem inversa. | A média passou de 2 para 1 MJ/km na tela. Não apareceu alerta em nenhuma das importações. Ver F02. |
| UC10 · Entradas inválidas | Enviar volume zero, consumo conhecido com texto inválido e, em seguida, planilha sem a coluna de volume. | A página identificou a linha com erro e calculou a parte válida. No caso de consumo inválido, informou **1 de 2 veículos e 9,1% do volume**. A falta do cabeçalho obrigatório gerou mensagem e preservou a tabela anterior. |
| UC11 · Adição pelo formulário | Usar **Add** com o volume inicial zero. Depois preencher 50 e adicionar o veículo padrão ao portfólio de UC06. | A primeira ação inseriu uma linha inválida. A segunda inseriu o veículo estimado e a média exibida foi 1,42 MJ/km. A tela informou **3 de 4 veículos** e manteve o erro da linha criada com zero. |
| UC12 · Navegação e sessão | Abrir Comparison e voltar a Vehicle ou Portfolio. Em ensaio separado, recarregar a página do portfólio. | As duas alternativas salvas permaneceram. O ajuste ativo de −10% voltou ao valor de origem. O portfólio importado sobreviveu à navegação interna; uma recarga completa abriu uma tabela vazia. |
| UC13 · Comparison | Abrir a página de comparação, verificar o vencedor, a tabela e as figuras. | Exibiu XGBoost com 5,6%, tabela de candidatos e duas imagens decodificadas no navegador. Ver F07 sobre o rótulo da métrica. |
| UC14 · Largura reduzida | Abrir Home e Vehicle com largura de 390 px, navegar pelo cartão e operar o seletor de propulsão. | Os cartões e o formulário foram utilizáveis. Nessas duas páginas a largura do documento permaneceu em 390 px, sem rolagem horizontal global. |

## Achados e critérios de aceite

**F01 · Alta · O tipo da célula e o separador alteram o peso de vendas.** O ensaio fixa os consumos em 1 e 3 MJ/km e o volume do segundo veículo em 10. A tabela mostra apenas a primeira célula alterada. O caso com vírgula é o controle positivo. O cálculo esperado usa a média ponderada, e o valor de tela é arredondado.

| Primeira célula de volume | Média esperada (MJ/km) | Tela (MJ/km) |
|---|---:|---:|
| Número decimal 1,234 | 2,78031 | **1,02** |
| Texto `10.000` representando dez mil | 1,00200 | **2** |
| Texto `10,000` representando dez mil | 1,00200 | 1 |

Preservar o tipo da célula na importação e definir uma interpretação única para os separadores ao editar. **Aceite:** os três casos devem manter os volumes e a média após importar, editar outra coluna, exportar e reimportar. A interface deve rejeitar frações explicitamente caso o produto passe a exigir unidades inteiras. [Leitor de volumes e XLSX](../src/eficiencia/portfolio.py) · [Conversão para o editor](../app/pages/2_Portfolio.py)

**F02 · Alta · Cabeçalhos equivalentes e conflitantes são aceitos em silêncio.** Na linha A, preencher `volume = 10`, `Volume = 10000` e consumo 1. Na linha B, preencher as duas colunas com 10 e consumo 3. Com `volume` primeiro, a média exibida foi 2; com `Volume` primeiro, foi 1. O leitor guarda a primeira ocorrência do cabeçalho normalizado. Tabelas produzidas pela junção de fontes podem chegar nesse formato. **Aceite:** detectar o campo duplicado antes de substituir a tabela e indicar ambos os cabeçalhos. Inverter a ordem deve levar à mesma mensagem e preservar o portfólio em uso. [Mapeamento de cabeçalhos](../src/eficiencia/portfolio.py#L77)

**F03 · Alta para decisões de cenário · O botão de salvar permanece disponível após um ajuste inválido.** Em UC04, o alerta de massa apareceu, mas **Save scenario** criou uma alternativa com o resultado original. No arquivo baixado, a linha continha 1,72548 MJ/km. Quem recebe a planilha pode atribuir esse valor à redução pedida. **Aceite:** enquanto o ajuste for inválido, bloquear o salvamento da alternativa. Ao corrigir a massa, salvar exatamente a configuração e o resultado exibidos. [Caminho de salvamento](../app/pages/1_Veiculo.py#L301)

**F04 · Média · O próprio formulário cria uma combinação contraditória no elétrico.** UC05 reproduz no navegador o achado 13 do primeiro relatório. O aviso detecta seis marchas fora da amostra depois de o formulário ter escolhido transmissão direta de uma marcha. **Aceite:** atualizar transmissão e número de marchas em conjunto, tanto ao entrar quanto ao sair da configuração elétrica, e conferir a ficha exportada. [Troca de propulsão](../app/formulario.py#L91)

**F05 · Média · O trabalho em andamento tem persistência diferente conforme a ação.** Em UC12, as alternativas salvas sobreviveram à troca de página, enquanto o ajuste atual voltou de 1,86 para 1,94 MJ/km. Uma recarga completa também esvaziou o portfólio de teste. Para quem consulta Comparison no meio de uma análise, isso exige reconstruir os ajustes. **Aceite:** conservar a edição atual durante a navegação interna. Se o projeto mantiver o portfólio apenas durante a sessão, informar isso junto à tabela e tornar a exportação fácil de localizar antes da recarga. [Estado do veículo](../app/pages/1_Veiculo.py) · [Estado do portfólio](../app/pages/2_Portfolio.py#L145)

**F06 · Baixa · Add insere uma linha sabidamente inválida.** Basta abrir o formulário e clicar no botão, pois o volume inicial é zero. A tentativa seguinte, com 50 unidades, não corrige a linha anterior. **Aceite:** só acrescentar uma linha quando o volume for positivo. Explicar o campo pendente junto ao botão, antes de alterar a tabela. [Botão de inclusão](../app/pages/2_Portfolio.py)

**F07 · Baixa · A comparação omite “absoluto” no nome do erro.** O texto da página denomina o resultado *mean percentage error*. O treino usa erro percentual **absoluto** médio (MAPE); erros de sinais opostos teriam interpretações diferentes nas duas métricas. **Aceite:** identificar o vencedor e o eixo como MAPE do MJ/km combinado e manter o mesmo termo no rodapé e no README. [Texto da página](../app/pages/3_Comparacao.py#L123) · [Definição no README](../README.md#L13)

## Ordem de correção e limites

1. Proteger a fidelidade dos números em F01/F02 e o salvamento em F03. Usar UC04 e UC06–UC10 como regressões com arquivos reais do Excel.
2. Corrigir a troca de propulsão e tornar previsível a navegação entre páginas. Repetir UC05 e UC12 nos dois tamanhos de tela.
3. Impedir a criação de linhas com volume zero e uniformizar o nome da métrica na comparação.

Os testes cobriram o Chrome isolado e planilhas sintéticas. O tamanho reduzido foi simulado em desktop; não houve teste em telefone físico, com leitor de tela, em outros navegadores ou com portfólios de grande porte. Esta rodada verificou o comportamento entregue ao usuário. Aferição estatística na safra PBEV real e resultados da suíte Python estão documentados separadamente no [relatório anterior](auditoria-do-produto.md).
