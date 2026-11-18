import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

try:
    from src.preprocessamento import preprocessar_texto
except ModuleNotFoundError:
    from preprocessamento import preprocessar_texto


def preparar_dados():
    # 1. Carregar as avaliações
    df = pd.read_csv("data/avaliacoes.csv")

    # 2. Separar features (X = texto) e alvo (y = sentimento)
    X = df["texto"]
    y = df["sentimento"]

    # 3. Divisão única: 75% treino e 25% teste (estratificada e com semente 42)
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    return X_treino, X_teste, y_treino, y_teste


def executar_experimentos():
    X_treino, X_teste, y_treino, y_teste = preparar_dados()

    # Preparamos duas versões do conjunto de treino:
    # - Uma com stopwords mantidas
    # - Outra com stopwords removidas
    X_treino_com_stop = X_treino.apply(
        lambda t: preprocessar_texto(t, remover_stopwords=False)
    )
    X_treino_sem_stop = X_treino.apply(
        lambda t: preprocessar_texto(t, remover_stopwords=True)
    )

    # Definimos as 4 configurações pedidas no enunciado
    configuracoes = [
        {
            "config": "A",
            "stopwords": "mantidas",
            "vetorizador_nome": "Bag of Words",
            "vetorizador": CountVectorizer(),
            "X": X_treino_com_stop,
        },
        {
            "config": "B",
            "stopwords": "mantidas",
            "vetorizador_nome": "TF-IDF",
            "vetorizador": TfidfVectorizer(),
            "X": X_treino_com_stop,
        },
        {
            "config": "C",
            "stopwords": "removidas",
            "vetorizador_nome": "Bag of Words",
            "vetorizador": CountVectorizer(),
            "X": X_treino_sem_stop,
        },
        {
            "config": "D",
            "stopwords": "removidas",
            "vetorizador_nome": "TF-IDF",
            "vetorizador": TfidfVectorizer(),
            "X": X_treino_sem_stop,
        },
    ]

    # Validação cruzada estratificada em 5 partes (folds) apenas no treino
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    linhas_tabela = []

    for item in configuracoes:
        # Encadeamos vetorizador e classificador em um Pipeline.
        # Isso garante que NÃO haja vazamento de dados (o vetorizador só vê o treino de cada fold).
        pipeline = Pipeline(
            [
                ("vetorizador", item["vetorizador"]),
                (
                    "classificador",
                    LogisticRegression(max_iter=1000, random_state=42),
                ),
            ]
        )

        # Medimos acurácia e f1 com validação cruzada
        # Como as classes são "positivo" e "negativo", usamos pos_label="positivo" para o f1
        scoring = ["accuracy", "f1_macro"]
        scores = cross_validate(
            pipeline, item["X"], y_treino, cv=cv, scoring=scoring
        )

        # Ajustamos o vetorizador no treino inteiro para registrar o tamanho do vocabulário
        item["vetorizador"].fit(item["X"])
        tamanho_vocab = len(item["vetorizador"].vocabulary_)

        # Guardamos as médias e desvios padrão
        linhas_tabela.append(
            {
                "config": item["config"],
                "stopwords": item["stopwords"],
                "vetorizador": item["vetorizador_nome"],
                "vocabulario": tamanho_vocab,
                "acc_cv_media": round(scores["test_accuracy"].mean(), 4),
                "acc_cv_desvio": round(scores["test_accuracy"].std(), 4),
                "f1_cv_media": round(scores["test_f1_macro"].mean(), 4),
                "f1_cv_desvio": round(scores["test_f1_macro"].std(), 4),
            }
        )

    # 4. Criar o DataFrame e salvar em resultados/comparacao.csv
    df_comparacao = pd.DataFrame(linhas_tabela)
    df_comparacao.to_csv("resultados/comparacao.csv", index=False)

    return df_comparacao


if __name__ == "__main__":
    tabela = executar_experimentos()
    print("Experimentos concluídos com sucesso!")
    print(tabela.to_string(index=False))
