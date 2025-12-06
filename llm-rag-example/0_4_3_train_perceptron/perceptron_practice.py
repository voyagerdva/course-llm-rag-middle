
# perceptron_practice.py

from typing import List, Tuple
import random
# from sympy.core.random import random


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
        """
        Считает "сырую" сумму:
            z = w1*x1 + w2*x2 + ... + wn*xn + b
        """
        weightedSum = 0.0
        for i in range(len(x)):
            weightedSum += self.weights[i] * x[i]
        weightedSum += self.bias
        return weightedSum

    def predict(self, x: List[float]) -> int:
        """
        Пороговая функция:
            если z >= 0 → 1
            иначе → 0
        """
        rawOutput = self.predictRaw(x)
        return 1 if rawOutput >= 0.0 else 0

    def trainOnExample(self, x: List[float], target: int) -> int:
        """
        Обучение на одном примере по правилу Розенблатта:
            err = target - y_pred
            w_i ← w_i + η * err * x_i
            b   ← b   + η * err

        Возвращает err, чтобы можно было смотреть динамику.
        """
        # 1. Считаем текущий прогноз
        prediction = self.predict(x)

        # 2. Считаем ошибку
        err = target - prediction

        # 3. Обновляем веса и bias
        # >>> ЭТО МЕСТО ИМЕЕТ СМЫСЛ ДЛЯ BREAKPOINT <<<
        for i in range(len(self.weights)):
            self.weights[i] += self.learningRate * err * x[i]

        self.bias += self.learningRate * err

        return err


def buildTrainingData() -> List[Tuple[List[float], int]]:
    """
    Строим небольшой линейно-разделимый датасет в 2D.
    Класс 1 — точки примерно в правой верхней четверти.
    Класс 0 — точки в левой нижней.
    """
    data: List[Tuple[List[float], int]] = []

    # Класс 1 (метка 1)
    data.append(([1.0, 1.0], 1))
    data.append(([2.0, 2.0], 1))
    data.append(([2.0, 1.5], 1))
    data.append(([1.5, 2.0], 1))

    # Класс 0 (метка 0)
    data.append(([-1.0, -1.0], 0))
    data.append(([-2.0, -1.0], 0))
    data.append(([-1.5, -2.0], 0))
    data.append(([-2.0, -2.0], 0))

    return data


def main() -> None:
    # 1. Создаём датасет
    trainingData = buildTrainingData()

    # 2. Создаём перцептрон с 2 входами
    perceptron = Perceptron(inputSize=2, learningRate=0.2)

    maxEpochs = 20

    for epochIndex in range(maxEpochs):
        print(f"\n=== ЭПОХА {epochIndex} ===")
        random.shuffle(trainingData) # перемешали порядок примеров
        totalErrors = 0

        for exampleIndex, (x, target) in enumerate(trainingData):
            # Считаем "сырое" значение и предсказание до обучения
            rawBefore = perceptron.predictRaw(x)
            predictionBefore = 1 if rawBefore >= 0.0 else 0

            # Обучаемся на этом примере
            error = perceptron.trainOnExample(x, target)

            # Считаем "сырое" значение и предсказание после обновления
            rawAfter = perceptron.predictRaw(x)
            predictionAfter = 1 if rawAfter >= 0.0 else 0

            totalErrors += abs(error)

            print(
                f"[пример {exampleIndex}] x={x}, target={target}, "
                f"pred_before={predictionBefore}, pred_after={predictionAfter}, "
                f"error={error}, "
                f"weights={perceptron.weights}, bias={perceptron.bias:.3f}"
            )

        print(f"Суммарное количество ошибок в эпохе: {totalErrors}")

        # Если ошибок нет — можно остановить обучение
        if totalErrors == 0:
            print("Модель сходится, обучение можно завершать.")
            break

    # Финальная проверка
    print("\n=== Финальная проверка ===")
    for x, target in trainingData:
        prediction = perceptron.predict(x)
        print(f"x={x}, target={target}, prediction={prediction}")


if __name__ == "__main__":
    main()
