import re

# Nossa lista de stopwords (palavras comuns da língua que funcionam apenas como ligação)
# IMPORTANTE: A palavra "não" foi deixada de fora de propósito!
STOPWORDS = [
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para",
    "com", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como"
]


def preprocessar_texto(texto: str, remover_stopwords: bool = False) -> str:
    # 1. Converte todo o texto para letras minúsculas
    texto = texto.lower()

    # 2. Usa regex para pegar apenas palavras com letras e acentos
    # Isso remove pontuação (!, ?, .), números e caracteres especiais
    palavras = re.findall(r"[a-záàâãéêíóôõúçü]+", texto)

    # 3. Remove stopwords se a opção for ativada
    if remover_stopwords:
        palavras = [p for p in palavras if p not in STOPWORDS]

    # 4. Junta as palavras de volta em uma única string separada por espaços
    texto_limpo = " ".join(palavras)
    return texto_limpo


if __name__ == "__main__":
    # Teste rápido para demonstrar o funcionamento
    exemplo = "A embalagem NÃO é excelente! Nota 0/10."

    print("Texto original:")
    print(exemplo)

    print("\n1. Pré-processado (com stopwords MANTIDAS):")
    print(preprocessartexto(exemplo, remover_stopwords=False))

    print("\n2. Pré-processado (com stopwords REMOVIDAS):")
    print(preprocessartexto(exemplo, remover_stopwords=True))
