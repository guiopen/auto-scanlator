# Auto-Scanlator

Ferramenta de tradução automática de quadrinhos.

## Setup

1. Crie o ambiente virtual com Python 3.12 e ative:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

2. Instale as dependências:
```bash
uv sync
```

3. Certifique-se de ter um LLM rodando em `http://127.0.0.1:8080`.

## Uso

```bash
python main.py comics/ --source-lang "pt" --target-lang "ENGLISH"
```
