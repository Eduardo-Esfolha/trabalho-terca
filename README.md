# Análise de Sentimento com PLN

Atividade avaliativa de Fundamentos de Inteligência Artificial sobre Processamento de Linguagem Natural (PLN) e Aprendizado de Máquina Supervisionado.

## Como Instalar

Requisitos: Python 3.10+

1. Clone o repositório ou navegue até a pasta do projeto:
```bash
git clone https://github.com/Eduardo-Esfolha/trabalho-terca.git
cd trabalho-terca
```

2. Crie e ative o ambiente virtual:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / Mac
python3 -m venv .venv
source .venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como Executar

Para rodar todo o pipeline do zero e gerar os arquivos na pasta `resultados/`:

```bash
python main.py
```

## Estrutura do Repositório

```text
├── data/
│   ├── avaliacoes.csv
│   └── casos_dificeis.csv
├── src/
│   ├── exploracao.py
│   ├── preprocessamento.py
│   ├── modelos.py
│   ├── avaliacao.py
│   └── casos_dificeis.py
├── resultados/
│   ├── exploracao.txt
│   ├── comparacao.csv
│   ├── matriz_confusao.csv
│   ├── relatorio_classificacao.txt
│   ├── erros.csv
│   ├── palavras_influentes.csv
│   └── casos_dificeis_predicoes.csv
├── main.py
├── README.md
├── RELATORIO.md
├── requirements.txt
└── .gitignore
```
