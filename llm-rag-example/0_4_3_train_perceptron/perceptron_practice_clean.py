from typing import List, Tuple
import random as rnd


class Perceptron:
    """
        Простейший перцептрон:
        - weights: список весов [w1, w2, ..., wN]
        - bias: смещение (bias)
        - learningRate: шаг обучения η
    """

    def __init__(self, inputSize: int, learningRate: float = 0.2) -> None:
        self.weights = [rnd.uniform(-0.5, 0.5) for _ in range(inputSize)]
        self.bias = rnd.uniform(-0.5, 0.5)
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
        Пороговая функция по "сырому" выходу:
            если z >= 0 → 1
            иначе → 0
        """
        return 1 if self.predictRaw(x) >= 0.0 else 0


def buildTrainingData() -> List[Tuple[List[float], int]]:
    """
    Небольшой линейно-разделимый датасет в 2D.
    Класс 1 — «треугольники», класс 0 — «круги».

    Специально чередуем примеры разных классов,
    чтобы при старте с нулевых весов корректировок было несколько,
    а не одна-единственная.
    """
    data: List[Tuple[List[float], int]] = []

    data.append(([-1.0, -1.0], 0))
    data.append(([1.0, 1.0], 1))

    data.append(([-2.0, -1.5], 0))
    data.append(([2.0, 1.5], 1))

    data.append(([-1.5, -2.0], 0))
    data.append(([1.5, 2.0], 1))

    data.append(([-2.0, -2.5], 0))
    data.append(([2.0, 2.5], 1))

    return data

def buildTestData() -> List[Tuple[List[float], int]]:
    data: List[Tuple[List[float], int]] = []

    data.append(([1.1, 1.4], 1))
    data.append(([2.1, 0.8], 1))
    data.append(([1.3, 2.2], 1))
    data.append(([2.5, 2.1], 1))

    data.append(([-1.2, -1.5], 0))
    data.append(([-1.1, -1.1], 0))
    data.append(([-0.5, -2.1], 0))
    data.append(([-1.0, -0.7], 0))

    return data


def main() -> None:
    trainingData = buildTrainingData()

    perceptron = Perceptron(inputSize=2, learningRate=0.2)

    print("== Начальное состояние перцептрона ==")
    print(f"weights={perceptron.weights}, bias={perceptron.bias:.3f}")

    maxEpochs = 20

    for ep in range(maxEpochs):
        print(f"\n=== ЭПОХА {ep} ===")

        # Необязательный шаг, но полезен:
        # каждый раз немного перемешиваем порядок примеров.
        rnd.shuffle(trainingData)

        totalErrors = 0

        for ex, (x, target) in enumerate(trainingData):
            # 1. Считаем предсказание при текущих весах
            rawOutput = perceptron.predictRaw(x)
            prediction = 1 if rawOutput >= 0.0 else 0

            # 2. Считаем ошибку
            err = target - prediction

            # 3. Печатаем "что было до обновления"
            print(
                f"[пример {ex}] x={x}, target={target}, "
                f"prediction={prediction}, err={err}, "
                f"weights_before={perceptron.weights}, "
                f"bias_before={perceptron.bias:.3f}"
            )

            # 4. Обновляем веса и bias по правилу Розенблатта
            #    w_i ← w_i + η * err * x_i
            for i in range(len(perceptron.weights)):
                perceptron.weights[i] += (
                    perceptron.learningRate * err * x[i]
                )

            #    b ← b + η * err
            perceptron.bias += perceptron.learningRate * err

            # 5. Накапливаем модуль ошибки, чтобы понять,
            #    были ли вообще ошибки в этой эпохе
            totalErrors += abs(err)

        print(f"Суммарное количество ошибок в эпохе: {totalErrors}")
        print(
            f"Состояние после эпохи {ep}: "
            f"weights={perceptron.weights}, bias={perceptron.bias:.3f}"
        )

        # Если за эпоху не было ни одной ошибки — модель сошлась
        if totalErrors == 0:
            print("Модель сошлась, обучение можно завершать.")
            break

    # Финальная проверка на всех примерах
    print("\n=== Финальная проверка ===")

    testData = buildTestData()
    rnd.shuffle(testData)

    for x, target in testData:  # берём тестовый набор
        prediction = perceptron.predict(x)
        rawOutput = perceptron.predictRaw(x)
        print(
            f"x={x}, target={target}, prediction={prediction}, "
            f"raw={rawOutput:.3f}"
        )


if __name__ == "__main__":
    main()

###