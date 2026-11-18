import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

try:
    from src.preprocessamento import preprocessar_texto
except ModuleNotFoundError:
    from preprocessamento import preprocessar_texto

try:
    from src.modelos import preparar_dados
except ModuleNotFoundError:
    from modelos import preparar_dados


def avaliar_casos_dificeis():
    X_treino, _, y_treino, _ = preparar_dados()
    X_treino_limpo = X_treino.apply(
        lambda t: preprocessar_texto(t, remover_stopwords=False)
    )

    vetorizador = CountVectorizer()
    X_treino_vec = vetorizador.fit_transform(X_treino_limpo)

    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_treino_vec, y_treino)

    df_dificeis = pd.read_csv("data/casos_dificeis.csv")
    textos_limpos = df_dificeis["texto"].apply(
        lambda t: preprocessar_texto(t, remover_stopwords=False)
    )
    X_dificeis_vec = vetorizador.transform(textos_limpos)

    previsoes = modelo.predict(X_dificeis_vec)
    probas = modelo.predict_proba(X_dificeis_vec)
    idx_positivo = list(modelo.classes_).index("positivo")

    resultados = []
    for i, row in df_dificeis.iterrows():
        prob_pos = probas[i][idx_positivo]
        resultados.append(
            {
                "texto": row["texto"],
                "tipo": row["tipo"],
                "esperado": row["sentimento_esperado"],
                "previsto": previsoes[i],
                "probabilidade_positivo": round(prob_pos, 4),
            }
        )

    df_resultado = pd.DataFrame(resultados)
    df_resultado.to_csv("resultados/casos_dificeis_predicoes.csv", index=False)
    return df_resultado


if __name__ == "__main__":
    tabela = avaliar_casos_dificeis()
    print("Casos difíceis avaliados com sucesso!")
    print(tabela.to_string(index=False))
