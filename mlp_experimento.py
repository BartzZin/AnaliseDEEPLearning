from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

@dataclass(frozen=True)
class ResultadoMLP:
    neurons: int
    learning_rate: float
    normalized: bool
    holdout_accuracy: float
    cv_mean_accuracy: float
    cv_std_accuracy: float
    train_time: float
    iterations: int


def carregar_dataset():
    dataset = load_breast_cancer()
    return dataset.data, dataset.target, dataset


def criar_modelo(neurons: int, learning_rate: float, normalized: bool) -> Pipeline | MLPClassifier:
    mlp = MLPClassifier(
        hidden_layer_sizes=(neurons,),
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate,
        max_iter=800,
        random_state=RANDOM_STATE,
        early_stopping=True,
        n_iter_no_change=25,
    )

    if normalized:
        return Pipeline(
            steps=[
                ("normalizador", StandardScaler()),
                ("mlp", mlp),
            ]
        )

    return mlp


def _iteracoes(modelo: Pipeline | MLPClassifier) -> int:
    if isinstance(modelo, Pipeline):
        return int(modelo.named_steps["mlp"].n_iter_)
    return int(modelo.n_iter_)


def executar_experimentos(
    neuronios: list[int] | tuple[int, ...],
    taxas_aprendizado: list[float] | tuple[float, ...],
    usar_normalizado: bool = True,
    usar_original: bool = True,
) -> tuple[list[ResultadoMLP], str]:
    X, y, dataset = carregar_dataset()

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    resultados: list[ResultadoMLP] = []
    melhor_modelo = None
    melhor_acuracia = -1.0
    melhor_predicao = None

    normalizacoes = []
    if usar_original:
        normalizacoes.append(False)
    if usar_normalizado:
        normalizacoes.append(True)

    for neurons, learning_rate, normalized in product(neuronios, taxas_aprendizado, normalizacoes):
        modelo = criar_modelo(neurons, learning_rate, normalized)

        inicio = perf_counter()
        modelo.fit(X_treino, y_treino)
        tempo = perf_counter() - inicio

        predicao = modelo.predict(X_teste)
        acuracia_holdout = accuracy_score(y_teste, predicao)

        validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(modelo, X, y, cv=validacao, scoring="accuracy")

        resultado = ResultadoMLP(
            neurons=neurons,
            learning_rate=learning_rate,
            normalized=normalized,
            holdout_accuracy=float(acuracia_holdout),
            cv_mean_accuracy=float(np.mean(cv_scores)),
            cv_std_accuracy=float(np.std(cv_scores)),
            train_time=float(tempo),
            iterations=_iteracoes(modelo),
        )
        resultados.append(resultado)

        if acuracia_holdout > melhor_acuracia:
            melhor_acuracia = acuracia_holdout
            melhor_modelo = modelo
            melhor_predicao = predicao

    resultados.sort(key=lambda item: item.holdout_accuracy, reverse=True)
    relatorio = montar_relatorio(dataset, y_teste, melhor_predicao, resultados[0], melhor_modelo)
    return resultados, relatorio


def montar_relatorio(dataset, y_teste, predicao, melhor: ResultadoMLP, modelo) -> str:
    matriz = confusion_matrix(y_teste, predicao)
    classes = list(dataset.target_names)

    texto = [
        "RELATORIO DO MELHOR MODELO",
        "",
        f"Dataset: Breast Cancer Wisconsin, {dataset.data.shape[0]} amostras e {dataset.data.shape[1]} atributos numericos.",
        "Implementacao usada: sklearn.neural_network.MLPClassifier.",
        f"Classes: {classes[0]} e {classes[1]}.",
        "",
        f"Neuronios na camada oculta: {melhor.neurons}",
        f"Taxa de aprendizagem: {melhor.learning_rate}",
        f"Dados normalizados: {'sim' if melhor.normalized else 'nao'}",
        f"Acuracia holdout 70/30: {melhor.holdout_accuracy:.4f}",
        f"Acuracia media 5-fold: {melhor.cv_mean_accuracy:.4f} (+/- {melhor.cv_std_accuracy:.4f})",
        f"Iteracoes ate parada: {melhor.iterations}",
        "",
        "Matriz de confusao:",
        str(matriz),
        "",
        "Relatorio de classificacao:",
        classification_report(y_teste, predicao, target_names=dataset.target_names),
    ]
    return "\n".join(texto)


def resultados_para_csv(resultados: list[ResultadoMLP]) -> str:
    linhas = [
        "neuronios,taxa_aprendizado,normalizado,acuracia_holdout,acuracia_cv_media,acuracia_cv_desvio,tempo_treino,iteracoes"
    ]
    for item in resultados:
        linhas.append(
            ",".join(
                [
                    str(item.neurons),
                    str(item.learning_rate),
                    "sim" if item.normalized else "nao",
                    f"{item.holdout_accuracy:.6f}",
                    f"{item.cv_mean_accuracy:.6f}",
                    f"{item.cv_std_accuracy:.6f}",
                    f"{item.train_time:.6f}",
                    str(item.iterations),
                ]
            )
        )
    return "\n".join(linhas) + "\n"
