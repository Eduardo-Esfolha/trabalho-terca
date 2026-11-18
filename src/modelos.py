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
    df = pd.read_csv("data/avaliacoes.csv")
    X = df["texto"]
    y = df["sentimento"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    return X_treino, X_teste, y_treino, y_teste


def executar_experimentos():
    X_treino, X_teste, y_treino, y_teste = preparar_dados()

    X_treino_com_stop = X_treino.apply(
        lambda t: preprocessar_texto(t, remover_stopwords=False)
    )
    X_treino_sem_stop = X_treino.apply(
        lambda t: preprocessar_texto(t, remover_stopwords=True)
    )

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    linhas_tabela = []

    for item in configuracoes:
        pipeline = Pipeline(
            [
                ("vetorizador", item["vetorizador"]),
                (
                    "classificador",
                    LogisticRegression(max_iter=1000, random_state=42),
                ),
            ]
        )

        scoring = ["accuracy", "f1_macro"]
        scores = cross_validate(
            pipeline, item["X"], y_treino, cv=cv, scoring=scoring
        )

        item["vetorizador"].fit(item["X"])
        tamanho_vocab = len(item["vetorizador"].vocabulary_)

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

    df_comparacao = pd.DataFrame(linhas_tabela)
    df_comparacao.to_csv("resultados/comparacao.csv", index=False)
    return df_comparacao


if __name__ == "__main__":
    tabela = executar_experimentos()
    print("Experimentos concluídos com sucesso!")
    print(tabela.to_string(index=False))
