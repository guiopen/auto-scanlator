### TEXTO:
- Adicionar figuras ao texto
- Aderessar "problema de pesquisa" e distinção de objetivos sugeridos pelo fileto
- Rever o tamanho da fonte: assumimos que ele não importa para a estética, mas posição e espaço ocupado importam; falta fundamentar essa hierarquia

### CODIGO:
Principal (afeta os resultados da pesquisa):
- Preservar negrito (feito), caixa alta (feito), cor, angulo (feito), posição (feito) e bordas do texto
- Ter um balanço entre legibilidade (tamanho de fonte) e posição/espaço do texto.
- REVERTER AS ÁREAS DESCARTADAS PELA FILTRAGEM E COMPOR UMA IMAGEM MISTA: HOJE O INPAINTING APAGA TODOS OS BLOCOS DETECTADOS, INCLUSIVE AS ONOMATOPEIAS E OS TEXTOS DE CENA QUE A TRADUÇÃO DESCARTA. A IDEIA É MANTER A PÁGINA LIMPA COMO REFERÊNCIA DA ANÁLISE E, DEPOIS DA FILTRAGEM, RESTAURAR AS ÁREAS DESCARTADAS A PARTIR DA PÁGINA ORIGINAL.

Secundario (melhora a funcionalidade ou a usabilidade):
- Salvar a imagem de saída — o pipeline nunca persiste resultado, só mostra debug. Não existe saída em disco nenhuma
- Instruções personalizadas do usuario

Coisas pra verificar:
- Filtrar por diferença de tamanho no agrupamento
- Ver como palavras compostas (guarda-chuva) sao tratadas
- tratamento de paginas muito longas (webtoon)