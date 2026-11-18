import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

try:
    from src.preprocessamento import preprocessar_texto
except ModuleNotFoundError:
    from preprocessamento import preprocessar_texto

try:
    from src.modelos import preparar_dados
except ModuleNotFoundError:
    from modelos import preparar_dados


def avaliar_modelo_final():
    X_treino, X_teste, y_treino, y_teste = preparar_dados()

    X_treino_limpo = X_treino.apply(
        lambda t: preprocessar_texto(t, remover_stopwords=False)
    )
    X_teste_limpo = X_teste.apply(
        lambda t: preprocessar_texto(t, remover_stopwords=False)
    )

    vetorizador = CountVectorizer()
    X_treino_vetorizado = vetorizador.fit_transform(X_treino_limpo)
    X_teste_vetorizado = vetorizador.transform(X_teste_limpo)

    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_treino_vetorizado, y_treino)

    y_pred = modelo.predict(X_teste_vetorizado)
    y_proba = modelo.predict_proba(X_teste_vetorizado)

    labels = ["positivo", "negativo"]
    cm = confusion_matrix(y_teste, y_pred, labels=labels)
    df_cm = pd.DataFrame(
        cm,
        index=["Real: Positivo", "Real: Negativo"],
        columns=["Previsto: Positivo", "Previsto: Negativo"],
    )
    df_cm.to_csv("resultados/matriz_confusao.csv")

    relatorio = classification_report(y_teste, y_pred, digits=4)
    with open(
        "resultados/relatorio_classificacao.txt", "w", encoding="utf-8"
    ) as f:
        f.write("RELATÓRIO DE CLASSIFICAÇÃO NO TESTE (CONFIGURAÇÃO A)\n\n")
        f.write(relatorio)

    erros = []
    for texto_original, real, previsto, probas in zip(
        X_teste, y_teste, y_pred, y_proba
    ):
        if real != previsto:
            prob_prevista = max(probas)
            erros.append(
                {
                    "texto": texto_original,
                    "real": real,
                    "previsto": previsto,
                    "probabilidade": round(prob_prevista, 4),
                }
            )

    df_erros = pd.DataFrame(erros)
    df_erros.to_csv("resultados/erros.csv", index=False)

    palavras = np.array(vetorizador.get_feature_names_out())
    coeficientes = modelo.coef_[0]

    indices_ordenados = np.argsort(coeficientes)
    top_10_negativas_idx = indices_ordenados[:10]
    top_10_positivas_idx = indices_ordenados[-10:][::-1]

    influentes = []
    for idx in top_10_positivas_idx:
        influentes.append(
            {
                "palavra": palavras[idx],
                "coeficiente": round(coeficientes[idx], 4),
                "sentimento": "positivo",
            }
        )
    for idx in top_10_negativas_idx:
        influentes.append(
            {
                "palavra": palavras[idx],
                "coeficiente": round(coeficientes[idx], 4),
                "sentimento": "negativo",
            }
        )

    df_influentes = pd.DataFrame(influentes)
    df_influentes.to_csv("resultados/palavras_influentes.csv", index=False)

    return df_cm, relatorio, df_erros, df_influentes


if __name__ == "__main__":
    cm, relatorio, erros, influentes = avaliar_modelo_final()
    print("Avaliação final concluída com sucesso!")
    print(cm)
    print(relatorio)
