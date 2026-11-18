import pandas as pd


def explorar():
    df = pd.read_csv("data/avaliacoes.csv")

    numerolnhas = len(df)

    contagem = df["sentimento"].value_counts()
    df["tamanho"] = df["texto"].apply(lambda x: len(x.split()))
    tamanho_medio = df.groupby("sentimento")["tamanho"].mean()

    exemplos_positivos = df[df["sentimento"] == "positivo"].head(3)
    exemplos_negativos = df[df["sentimento"] == "negativo"].head(3)

    with open("resultados/exploracao.txt", "w", encoding="utf-8") as f:
        f.write("=== EXPLORACAO DOS DADOS ===\n\n")

        f.write(f"1. Total de linhas: {numerolnhas}\n\n")

        f.write("2. Distribuicao das classes:\n")
        f.write(f"- Positivos: {contagem['positivo']}\n")
        f.write(f"- Negativos: {contagem['negativo']}\n\n")

        f.write("3. Tamanho medio dos textos (em palavras):\n")
        f.write(f"- Positivos: {tamanho_medio['positivo']:.2f} palavras\n")
        f.write(f"- Negativos: {tamanho_medio['negativo']:.2f} palavras\n\n")

        f.write("4. Exemplos de textos positivos:\n")
        for texto in exemplos_positivos["texto"]:
            f.write(f"- \"{texto}\"\n")

        f.write("\n5. Exemplos de textos negativos:\n")
        for texto in exemplos_negativos["texto"]:
            f.write(f"- \"{texto}\"\n")


if __name__ == "__main__":
    explorar()
    print("Arquivo resultados/exploracao.txt gerado com sucesso!")
