Principal (afeta os resultados da pesquisa):
- Tradução de onomatopeias (com fonte alternativa e opção de on/off)
- Preservar negrito, itálico, caixa alta, cor, angulo, posição e bordas do texto

Secundario (melhora a funcionalidade ou a usabilidade):
- Salvar a imagem de saída — o pipeline nunca persiste resultado, só mostra debug. Não existe saída em disco nenhuma
- salvar output
- Instruções personalizadas do usuario

Terciario (melhora qualidade do sistema):
- Retry da LLM com fallback de temperatura quando JSON malformado (hoje _parse_llm_response só retorna [])
- Interface web
- testes e prevenção de erros

Performance:
- Resize da imagem pra acelerar OCR (e nao estourar llm)
- Suporte GPU/CUDA exposto na config
- ONNX runtime p/ deploy barato

Coisas pra verificar:
- Filtrar por diferença de tamanho no agrupamento
- Ver como palavras compostas (guarda-chuva) sao tratadas
- Alternativas ao lama inpaint
- tratamento de paginas muito longas (webtoon)