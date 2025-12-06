from typing import List, Tuple

import matplotlib.pyplot as plt


class Perceptron:
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
        return 1 if self.predictRaw(x) >= 0.0 else 0

    def trainOnExample(self, x: List[float], target: int) -> int:
        prediction = self.predict(x)
        error = target - prediction

        for index in range(len(self.weights)):
            self.weights[index] += self.learningRate * error * x[index]

        self.bias += self.learningRate * error
        return error


def buildTrainingData() -> List[Tuple[List[float], int]]:
    data: List[Tuple[List[float], int]] = []

    # Класс 1 — треугольники
    data.append(([1.0, 1.0], 1))
    data.append(([2.0, 1.5], 1))
    data.append(([1.5, 2.0], 1))
    data.append(([2.0, 2.5], 1))

    # Класс 0 — круги
    data.append(([-1.0, -1.0], 0))
    data.append(([-2.0, -1.5], 0))
    data.append(([-1.5, -2.0], 0))
    data.append(([-2.0, -2.5], 0))

    return data


def drawState(ax: plt.Axes,
              perceptron: Perceptron,
              trainingData: List[Tuple[List[float], int]],
              title: str) -> None:
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

    ax.clear()

    # точки
    ax.scatter(xsClass0, ysClass0, marker="o", label="class 0 (circles)")
    ax.scatter(xsClass1, ysClass1, marker="^", label="class 1 (triangles)")

    # прямая: w1*x + w2*y + b = 0 → y = -(w1*x + b)/w2
    w1, w2 = perceptron.weights
    b = perceptron.bias

    allX: List[float] = [x for x, _ in trainingData]
    minX = min(v[0] for v in allX) - 0.5
    maxX = max(v[0] for v in allX) + 0.5

    if abs(w2) > 1e-8:
        xsLine = [minX, maxX]
        ysLine = [-(w1 * x + b) / w2 for x in xsLine]
        ax.plot(xsLine, ysLine, linestyle="-", label="decision boundary")

    ax.axhline(0.0, linewidth=0.5)
    ax.axvline(0.0, linewidth=0.5)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def main() -> None:
    trainingData = buildTrainingData()
    perceptron = Perceptron(inputSize=2, learningRate=0.2)

    maxEpochs = 20

    plt.ion()  # интерактивный режим
    figure, axes = plt.subplots()

    for epochIndex in range(maxEpochs):
        totalErrors = 0
        print(f"\n=== ЭПОХА {epochIndex} ===")
