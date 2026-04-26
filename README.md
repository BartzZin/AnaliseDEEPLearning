# Atividade 3 - Multilayer Perceptron (MLP)

Projeto simples em Python para avaliar uma rede neural MLP em um dataset real.

## Objetivo atendido

- Implementacao usada: `sklearn.neural_network.MLPClassifier`.
- Dataset: Breast Cancer Wisconsin, disponivel no proprio `scikit-learn`.
- Avaliacao: divisao holdout 70/30 e validacao cruzada 5-fold.
- Experimentos: compara diferentes quantidades de neuronios, taxas de aprendizagem e uso ou nao de normalizacao.
- Saida: tabela na tela, relatorio do melhor modelo e exportacao dos resultados em CSV.