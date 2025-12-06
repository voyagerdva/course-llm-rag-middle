# perceptron_practice.py

from typing import List, Tuple
import random as rnd

import matplotlib.pyplot as plt


class Perceptron:
    """
    Простейший перцептрон с:
    - weights: список весов [w1, w2, ..., wN]
    - bias: смещение (bias)
    - learningRate: шаг обучения η
    """

    def __init__(self, inputSize: int, learningRate: float = 0.2) -> None:
        # Весам задаём начальные случайные значения,
        # чтобы обучение не было "слишком лёгким".
        self.weights: List[float] = [
            rnd.uniform(-0.5, 0.5) for _ in range(inputSize)
        ]
        self.bias: float = rnd.uniform(-0.5, 0.5)
        self.learningRate: float = learningRate

    def predictRaw(self, x: List[float]) -> float:
        """
        Считает "сырую" сумму:
            z = w1*x1 + w2*x2 + ... + wn*xn + b
        """
        weightedSum = 0.0
        for index in range(len(x)):
            weightedSum += self.weights[index] * x[index]
        weightedSum += self.bias
        return weightedSum

    def predict(self, x: List[float]) -> int:
        """
        Пороговая функция по "сырому" выходу:
            если z >= 0 → 1
            иначе → 0
        """
        rawOutput = self.predictRaw(x)
        return 1 if rawOutput >= 0.0 else 0

    def trainOnExample(self, x: List[float], target: int) -> int:
        """
        Обучение на одном примере по правилу Розенблатта:
            error = target - y_pred
            w_i ← w_i + η * error * x_i
            b   ← b   + η * error

        Возвращает error, чтобы можно было смотреть динамику.
        ЗДЕСЬ УДОБНО СТАВИТЬ BREAKPOINT для дебага.
        """
        # 1. Считаем текущий прогноз
        prediction = self.predict(x)

        # 2. Считаем ошибку
        error = target - prediction

        # >>> ТОЧКА НАБЛЮДЕНИЯ "ДО ОБНОВЛЕНИЯ" <<<
        # В этот момент:
        # - self.weights, self.bias — СТАРЫЕ
        # - prediction — текущий ответ
        # - error — +1, 0 или -1
        # Можно поставить сюда брейкпоинт.

        # 3. Обновляем веса и bias
        for index in range(len(self.weights)):
            self.weights[index] += self.learningRate * error * x[index]

        self.bias += self.learningRate * error

        # >>> ТОЧКА НАБЛЮДЕНИЯ "ПОСЛЕ ОБНОВЛЕНИЯ" <<<
        # Здесь уже новые weights и bias.
        return error


def buildTrainingData() -> List[Tuple[List[float], int]]:
    """
    Строим небольшой линейно-разделимый датасет в 2D.
    Класс 1 — "треугольники" (правый верхний кластер),
    Класс 0 — "круги" (левый нижний кластер).
    """
    data: List[Tuple[List[float], int]] = []

    # Класс 1 (метка 1)
    data.append(([1.0, 1.0], 1))
    data.append(([2.0, 1.5], 1))
    data.append(([1.5, 2.0], 1))
    data.append(([2.0, 2.5], 1))

    # Класс 0 (метка 0)
    data.append(([-1.0, -1.0], 0))
    data.append(([-2.0, -1.5], 0))
    data.append(([-1.5, -2.0], 0))
    data.append(([-2.0, -2.5], 0))

    return data


def debugState(prefix: str,
               epochIndex: int,
               exampleIndex: int,
               x: List[float],
               target: int,
               prediction: int,
               error: int,
               weights: List[float],
               bias: float) -> None:
    """
    Удобная вспомогательная функция для логирования состояния.
    Можно отключать или включать выборочно.
    """
    print(
        f"{prefix} [эпоха {epochIndex}, пример {exampleIndex}] "
        f"x={x}, target={target}, pred={prediction}, "
        f"error={error}, weights={weights}, bias={bias:.3f}"
    )


def drawStateLive(ax: plt.Axes,
                  weights: List[float],
                  bias: float,
                  trainingData: List[Tuple[List[float], int]],
                  title: str) -> None:
    """
    Живое обновление графика на уже существующей оси ax.
    Не создаёт новое окно, а:
      - очищает ax
      - рисует точки
      - рисует текущую прямую
      - перерисовывает canvas
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

    ax.clear()

    ax.scatter(xsClass0, ysClass0, marker="o", label="class 0 (circles)")
    ax.scatter(xsClass1, ysClass1, marker="^", label="class 1 (triangles)")

    w1, w2 = weights
    b = bias

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

    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()

def plotDecisionBoundary(weights: List[float],
                         bias: float,
                         trainingData: List[Tuple[List[float], int]],
                         title: str) -> None:
    """
    Рисуем:
    - точки класса 0 кружочками
    - точки класса 1 треугольниками
    - прямую решения: w1*x + w2*y + b = 0 → y = -(w1*x + b)/w2
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

    plt.scatter(xsClass0, ysClass0, marker="o", label="class 0 (circles)")
    plt.scatter(xsClass1, ysClass1, marker="^", label="class 1 (triangles)")

    w1, w2 = weights
    b = bias

    allX: List[float] = [x for x, _ in trainingData]
    minX = min(v[0] for v in allX) - 0.5
    maxX = max(v[0] for v in allX) + 0.5

    if abs(w2) > 1e-8:
        xsLine = [minX, maxX]
        ysLine = [-(w1 * x + b) / w2 for x in xsLine]
        plt.plot(xsLine, ysLine, linestyle="-", label="decision boundary")

    plt.axhline(0.0, linewidth=0.5)
    plt.axvline(0.0, linewidth=0.5)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()


def main() -> None:
    trainingData = buildTrainingData()
    perceptron = Perceptron(inputSize=2, learningRate=0.2)

    maxEpochs = 20

    # включаем интерактивный режим и заводим одно окно
    plt.ion()
    fig, ax = plt.subplots()

    for epochIndex in range(maxEpochs):
        print(f"\n=== ЭПОХА {epochIndex} ===")
        rnd.shuffle(trainingData)

        totalErrors = 0

        for exampleIndex, (x, target) in enumerate(trainingData):
            rawBefore = perceptron.predictRaw(x)
            predBefore = 1 if rawBefore >= 0.0 else 0

            error = perceptron.trainOnExample(x, target)

            rawAfter = perceptron.predictRaw(x)
            predAfter = 1 if rawAfter >= 0.0 else 0

            totalErrors += abs(error)

            print(
                f"[пример {exampleIndex}] x={x}, target={target}, "
                f"pred_before={predBefore}, pred_after={predAfter}, "
                f"error={error}, "
                f"weights={perceptron.weights}, bias={perceptron.bias:.3f}"
            )

            # 👉 живое обновление графика ПОСЛЕ КАЖДОГО ПРИМЕРА
            title = f"Perceptron (epoch {epochIndex}, example {exampleIndex})"
            drawStateLive(
                ax=ax,
                weights=perceptron.weights,
                bias=perceptron.bias,
                trainingData=trainingData,
                title=title,
            )

        print(f"Суммарное количество ошибок в эпохе: {totalErrors}")

        if totalErrors == 0:
            print("Модель сошлась, обучение можно завершать.")
            break

    # После обучения можно выключить интерактив и оставить финальное окно
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
