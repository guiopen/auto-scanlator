# Auto-Scanlator

Ferramenta de tradução automática de quadrinhos.

> **Nota:** O programa não suporta idiomas com escrita vertical (ex: japonês).  
> Apenas idiomas com orientação horizontal são compatíveis.

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

## Uso

```bash
# Traduzir um diretório de quadrinhos
uv run main.py comic/ --source-lang pt --target-lang en

# Traduzir uma página específica
uv run main.py comic/page1.png --source-lang pt --target-lang en
```
