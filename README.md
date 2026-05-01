# Auto-Scanlator

Ferramenta de tradução automática de quadrinhos.

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
uv run main.py comics/ --source-lang "pt" --target-lang "ENGLISH"
```
