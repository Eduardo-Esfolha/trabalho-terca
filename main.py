import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.avaliacao import avaliar_modelo_final
from src.casos_dificeis import avaliar_casos_dificeis
from src.exploracao import explorar
from src.modelos import executar_experimentos


def main():
    print("=" * 60)
    print("INICIANDO PIPELINE DE ANÁLISE DE SENTIMENTO")
    print("=" * 60)

    print("\n[1/4] Executando exploração dos dados...")
    explorar()
    print("-> resultados/exploracao.txt gerado com sucesso.")

    print("\n[2/4] Executando validação cruzada das 4 configurações...")
    executar_experimentos()
    print("-> resultados/comparacao.csv gerado com sucesso.")

    print("\n[3/4] Avaliando a melhor configuração (Config A) no teste...")
    avaliar_modelo_final()
    print("-> resultados/matriz_confusao.csv gerado com sucesso.")
    print("-> resultados/relatorio_classificacao.txt gerado com sucesso.")
    print("-> resultados/erros.csv gerado com sucesso.")
    print("-> resultados/palavras_influentes.csv gerado com sucesso.")

    print("\n[4/4] Testando modelo nos casos difíceis (negação, ironia, misto)...")
    avaliar_casos_dificeis()
    print("-> resultados/casos_dificeis_predicoes.csv gerado com sucesso.")

    print("\n" + "=" * 60)
    print("TODAS AS ETAPAS FORAM CONCLUÍDAS COM SUCESSO!")
    print("Todos os arquivos foram gerados na pasta resultados/.")
    print("=" * 60)


if __name__ == "__main__":
    main()
