# QA de importação e exportação XLSX de portfólio

Ambiente local 127.0.0.1:8501, Chrome headless pt-BR, planilhas sintéticas.

## Evidências

- Identificadores em células de texto: 0012 e 0142 foram exportados como células de número 12 e 142. NA e N/A foram exportados em branco. Reimportar as duas exportações manteve a contagem e a média, mas não recuperou as identificações. Dados das células em qa-deep-xlsx-export-cells.txt.
- Known MJ/km com ficha completa e texto literal NA: 2.43 MJ/km, faixa 2.36–2.52, 2/2 linhas, zero alertas. A mesma ficha com texto literal oops: 3 MJ/km, 1/2 linhas, erro de Known MJ/km inválido. O leitor trata NA como ausência e passa a linha para o modelo. Evidência visual em qa-deep-xlsx-known-na.png e qa-deep-xlsx-known-invalid.png.
- Known MJ/km decimal pt-BR: volumes 100 e 300 com 1,25 e 2,75 renderam 2.38 MJ/km, 2/2 incluídos.
- XLSX de várias abas: só a primeira aba foi lida mesmo com a segunda ativa. Sem requisito multiabas identificado.
- XLSX sem Name e Volume, primeira aba apenas de instruções e XLSX corrompido: erros exibidos, média anterior 1 MJ/km e linha anterior preservadas. Upload válido posterior mudou a média para 2 MJ/km.
- Reconfirmação de comportamento já registrado nas PR36/37: volumes numéricos 1.125 e 0.125 são reinterpretados como 1125 e 125 na média ponderada.

Scripts em qa-deep-xlsx-fixtures.py, qa-deep-xlsx-replay.py, qa-deep-xlsx-na-replay.py e qa-deep-xlsx-read.py. Capturas, arquivos de entrada e saídas baixadas têm o mesmo prefixo.
