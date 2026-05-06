# Classificador de Textos com Ollama

Classifica textos e datasets via LLM local (Ollama) usando FastAPI + Streamlit.

## Pré-requisitos

- Python 3.11 → `brew install python@3.11`
- Ollama → `brew install ollama`

## Configuração do ambiente

```bash
python3.11 -m venv .venv          # cria o ambiente virtual
source .venv/bin/activate         # ativa (Mac/Linux)
pip install -r requirements.txt   # instala as dependências
```

## Rodar

```bash
./run_app.sh          # inicia Ollama + API + Streamlit
./run_app.sh --stop   # encerra tudo
```

O `run_app.sh` cria o `.venv` e instala as dependências automaticamente na primeira execução.

| Serviço | URL |
|---|---|
| Interface | http://localhost:8501 |
| API | http://localhost:8000 |
| Docs (Swagger) | http://localhost:8000/docs |
| Logs | `./logs/` |

## Uso manual

```bash
source .venv/bin/activate
ollama serve                                         # terminal 1
uvicorn src.api.main:app --reload --port 8000        # terminal 2
streamlit run streamlit_app.py                       # terminal 3
```

## Endpoints principais

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/classify/text` | Classifica um texto |
| `POST` | `/classify/dataset` | Classifica dataset CSV/Excel |
| `GET` | `/models` | Lista modelos instalados |
| `POST` | `/models/pull/stream` | Download com progresso em tempo real |
| `DELETE` | `/models/{name}` | Remove um modelo |
| `GET` | `/health` | Status da API e do Ollama |

## Estrutura

```
├── run_app.sh              # script de inicialização
├── streamlit_app.py        # interface web
├── requirements.txt
└── src/
    ├── api/                # FastAPI (rotas, dependências)
    ├── services/           # OllamaService, ClassificationService
    ├── models/             # schemas Pydantic
    └── interfaces/         # abstrações SOLID
```
