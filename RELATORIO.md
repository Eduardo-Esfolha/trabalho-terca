# RELATORIO — Análise de Sentimento com PLN

**Nome:** Eduardo Lopes Esfolha
**Repositório:** https://github.com/Eduardo-Esfolha/trabalho-terca
**Data:** 17/11/2026

> Responda cada pergunta com **4 a 12 linhas**, com suas palavras e usando números dos arquivos de `resultados/`. Apague as instruções em itálico ao entregar.

## Tabela-resumo (preencha com os seus resultados)

| Config | Stopwords | Vetorizador | Vocabulário | Acurácia CV (média ± desvio) | F1 CV (média ± desvio) |
|---|---|---|---|---|---|
| A | mantidas | BoW | 420 | 0.8564 ± 0.0926 | 0.8563 ± 0.0926 |
| B | mantidas | TF-IDF | 420 | 0.8462 ± 0.0843 | 0.8459 ± 0.0844 |
| C | removidas | BoW | 405 | 0.8513 ± 0.0750 | 0.8512 ± 0.0750 |
| D | removidas | TF-IDF | 405 | 0.8410 ± 0.0676 | 0.8407 ± 0.0679 |

**Configuração escolhida:** A **Acurácia no teste:** 0.8154 **F1 no teste:** 0.8150

---

## 1. Pré-processamento
*Quais decisões você tomou (minúsculas, pontuação, números, stopwords)? Qual o tamanho do vocabulário com e sem remoção de stopwords? O que você fez com a palavra "não" e por quê?* minusculas para padronizar, numeros para descartarar juntos com pontuaçoes e a acentuaçao por que na nossa querida lingua muda o significado assim como o nao. o nao dependendo do contexto a remoçao mudaria a frase


## 2. BoW vs. TF-IDF
*Compare as quatro configurações levando em conta o desvio padrão. As diferenças são relevantes? Qual você escolheu e com base em quê? Explique a diferença conceitual entre as duas representações.*a bow teve desempenho melhor mais pequena no desvio padrão, e no conceito pondera a raridade do termo fazendo com que palavras mais raras tenham mais peso


## 3. Avaliação no teste
*Mostre a matriz de confusão e interprete precisão, recall e F1 de cada classe. Para uma loja, qual erro é mais grave: classificar uma reclamação como elogio ou o contrário? Por que a acurácia é menos enganosa aqui do que no exemplo de fraude da Aula 8?* positivos (28acertos 4 erros) negativos (23 caertos 8 erros) Métricas: Negativo (Precisão 0.8621, Recall 0.7576, F1 0.8065); Positivo (Precisão 0.7778, Recall 0.8750, F1 0.8235). Classificar uma reclamação (negativo) como elogio (positivo), pois o cliente com problema fica sem atendimento da loja.


## 4. Validação cruzada vs. divisão única
*Por que a validação cruzada é mais confiável do que um único conjunto de teste para comparar configurações? Por que a escolha não pode ser feita olhando o teste? O que seria vazamento de dados neste trabalho e como você o evitou?* A validação cruzada é mais confiável por testar em 5 divisões, trazendo média e desvio reais (0.8564 ± 0.0926 na Config A) sem depender da sorte de uma divisão única. Não se escolhe pelo teste porque ele é o exame cego final; usar ele para decidir causa sobreajuste ao teste. O vazamento ocorreria se o vetorizador aprendesse palavras das 260 frases inteiras; evitamos usando Pipeline, que faz o fit apenas no treino.


## 5. Análise de erros e palavras influentes
*Escolha 3 textos de `erros.csv` e explique por que o modelo pode ter errado. Observe as 10 palavras mais influentes de cada classe: há alguma que não carrega sentimento por si só (ex.: palavra funcional ou de domínio)? O que isso diz sobre o que o modelo realmente aprendeu?*
O modelo errou na frase "Cheguei animado, mas me arrependi, tudo estava sujo" porque focou na palavra animado e ignorou o mas. Errou em "Panela de péssima qualidade, a comida gruda e queima" porque a palavra qualidade puxou para positivo. E errou em "Ficou lindo na sala, combinou perfeitamente com os móveis" por não reconhecer termos positivos fortes no vocabulário.
Palavras como bastante e toda não carregam sentimento sozinhas, mas ganharam pesos altos.
Isso mostra que o modelo apenas associou a frequência das palavras ao invés de entender o sentimento real da frase.


## 6. Casos difíceis e limitações
*Com base em `casos_dificeis_predicoes.csv`, como o modelo se saiu em negação, ironia e frases mistas? Explique por que BoW/TF-IDF têm essa limitação, proponha duas melhorias (cite técnicas vistas na Aula 11) e discuta duas limitações do dataset e um risco de usar este modelo em produção.* nao consegui elaborar essa

(sua resposta)

---

## Declaração de uso de IA

Marque uma opção e complete:

- [ ] Não usei IA generativa em nenhuma parte deste trabalho.
- [ x] Usei IA generativa **somente** para: _corrigir, relembrar conceitos  e me auxiliar no código (claude modo chat sem idea geradora). e usei tambem em um momento para me auxiliar no git pois eu tinha bugado a origem do meu repositorio_____________ (não é permitido gerar código ou relatório).
