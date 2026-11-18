import re

STOPWORDS = [
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
    "com", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como"
]


def preprocessar_texto(texto: str, remover_stopwords: bool = False) -> str:
    texto = texto.lower()
    palavras = re.findall(r"[a-záàâãéêíóôõúçü]+", texto)
    if remover_stopwords:
        palavras = [p for p in palavras if p not in STOPWORDS]
    return " ".join(palavras)


if __name__ == "__main__":
    exemplo = "A embalagem NÃO é excelente! Nota 0/10."
    print("Original:", exemplo)
    print("Mantidas:", preprocessar_texto(exemplo, remover_stopwords=False))
    print("Removidas:", preprocessar_texto(exemplo, remover_stopwords=True))
