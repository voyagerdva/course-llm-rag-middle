# perceptron_plot.py

from typing import List, Tuple

import matplotlib.pyplot as plt


class Perceptron:
    """
    Простейший перцептрон:
    - вход: вектор x (List[float])
    - выход: 0 или 1
    - веса: self.weights
    - смещение (bias): self.bias
    """

    def __init__(self, inputSize: int, learningRate: float = 0.2) -> None:
        self.weights: List[float] = [0.0] * inputSize
        self.bias: float = 0.0
        self.learningRate: float = learningRate

    def predictRaw(self, x: List[float]) -> float:
        weightedSum = 0.0
        for index in range(len(x)):
            weightedSum += self.weights[index] * x[index]
        weightedSum += self.bias
        return weightedSum

    def predict(self, x: List[float]) -> int:
        rawOutput = self.predictRaw(x)
        return 1 if rawOutput >= 0.0 else 0

    def trainOnExample(self, x: List[float], target: int) -> int:
        """
        Обучение на одном примере по правилу Розенблатта.
        Возвращает error, чтобы можно было смотреть динамику.
        """
        prediction = self.predict(x)
        error = target - prediction

        # <<< Тут удобно ставить breakpoint и смотреть, как меняются веса >>>
        for index in range(len(self.weights)):
            self.weights[index] += self.learningRate * error * x[index]

        self.bias += self.learningRate * error
        return error


def buildTrainingData() -> List[Tuple[List[float], int]]:
    """
    Небольшой линейно-разделимый датасет.
    Класс 1 — "треугольники" (правый верхний кластер),
    Класс 0 — "круги" (левый нижний кластер).
    """
    data: List[Tuple[List[float], int]] = []

    # Класс 1 (метка 1) — треугольники
    data.append(([1.0, 1.0], 1))
    data.append(([2.0, 1.5], 1))
    data.append(([1.5, 2.0], 1))
    data.append(([2.0, 2.5], 1))

    # Класс 0 (метка 0) — круги
    data.append(([-1.0, -1.0], 0))
    data.append(([-2.0, -1.5], 0))
    data.append(([-1.5, -2.0], 0))
    data.append(([-2.0, -2.5], 0))

    return data


def plotDecisionBoundary(perceptron: Perceptron,
                         trainingData: List[Tuple[List[float], int]]) -> None:
    """
    Рисуем:
    - точки класса 0 кружочками
    - точки класса 1 треугольниками
    - прямую решения перцептрона: w1*x + w2*y + b = 0
    """
    xsClass0: List[float] = []
    ysClass0: List[float] = []
    xsClass1: List[float] = []
    ysClass1: List[float] = []

    for x, target in trainingData:
        if target == 0:
            xsClass0.append(x[0])
            ysClass0.append(x[1])
        else:
            xsClass1.append(x[0])
            ysClass1.append(x[1])

    plt.figure()

    # Класс 0 — круги (marker='o')
    plt.scatter(xsClass0, ysClass0, marker='o', label="class 0 (circles)")

    # Класс 1 — треугольники (marker='^')
    plt.scatter(xsClass1, ysClass1, marker='^', label="class 1 (triangles)")

    # Рисуем прямую решения: w1*x + w2*y + b = 0 → y = -(w1*x + b)/w2
    w1, w2 = perceptron.weights
    b = perceptron.bias

    # Берём диапазон X по данным, с небольшим запасом
    allX: List[float] = [x for x, _ in trainingData]
    minX = min(v[0] for v in allX) - 0.5
    maxX = max(v[0] for v in allX) + 0.5

    if abs(w2) > 1e-8:
        xsLine = [minX, maxX]
        ysLine = [-(w1 * x + b) / w2 for x in xsLine]
        plt.plot(xsLine, ysLine, linestyle='-', label="decision boundary")

    plt.axhline(0.0, linewidth=0.5)
    plt.axvline(0.0, linewidth=0.5)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title("Perceptron: circles vs triangles")

    plt.grid(True)
    plt.show()


def main() -> None:
    trainingData = buildTrainingData()
    perceptron = Perceptron(inputSize=2, learningRate=0.2)

    maxEpochs = 20

    for epochIndex in range(maxEpochs):
        totalErrors = 0
        print(f"\n=== ЭПОХА {epochIndex} ===")

        for exampleIndex, (x, target) in enumerate(trainingData):
            error = perceptron.trainOnExample(x, target)
            totalErrors += abs(error)

            print(
                f"[пример {exampleIndex}] x={x}, target={target}, "
                f"error={error}, weights={perceptron.weights}, bias={perceptron.bias:.3f}"
            )

        print(f"Суммарное количество ошибок: {totalErrors}")

        if totalErrors == 0:
            print("Перцептрон сошёлся, дальше можно не учить.")
            break

    # После обучения — рисуем
    plotDecisionBoundary(perceptron, trainingData)


if __name__ == "__main__":
    main()
