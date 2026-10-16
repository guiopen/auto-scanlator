# Auto-Scanlator

Ferramenta de tradução automática de quadrinhos.

> **Nota:** O programa não suporta idiomas com escrita vertical (ex: japonês).  
> Apenas idiomas com orientação horizontal são compatíveis.
> O programa é um TCC em desenvolvimento, ainda esta na fase de protótipo

## Setup

1. Instale o uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Instale as dependências:
```bash
uv sync
```

3. Certifique-se de ter um LLM rodando em `http://127.0.0.1:8080`.

4. Coloque uma fonte TrueType na pasta `fonts/` na raiz do projeto:
- `fonts/font.ttf` (obrigatória)
- `fonts/font-bold.ttf` (opcional; sem ela, o negrito é renderizado com a fonte regular)

A fonte precisa conter os caracteres do idioma de destino. Fontes gratuitas podem ser baixadas do [Google Fonts](https://fonts.google.com/) e renomeadas para `font.ttf` e `font-bold.ttf`. A pasta `fonts/` é ignorada pelo git, então precisa ser criada manualmente.

Recomendo o uso da fonte Komika para a maioria dos quadrinhos: https://www.fontsquirrel.com/fonts/Komika-Hand

## Uso

```bash
# Traduzir um diretório de quadrinhos
uv run main.py comic/ --source-lang pt --target-lang en

# Traduzir uma página específica
uv run main.py comic/page1.png --source-lang pt --target-lang en
```
