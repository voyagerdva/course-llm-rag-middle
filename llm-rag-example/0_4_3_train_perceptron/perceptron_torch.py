# perceptron_torch.py

from typing import Tuple

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def build_training_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Строим тот же самый 2D-дата-сет, что и раньше.
    x: тензор формы [N, 2]
    y: тензор формы [N, 1] с метками 0.0 или 1.0
    """
    # Класс 1 (треугольники)
    class1 = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 1.5],
            [1.5, 2.0],
            [2.0, 2.5],
        ],
        dtype=torch.float32,
    )

    # Класс 0 (круги)
    class0 = torch.tensor(
        [
            [-1.0, -1.0],
            [-2.0, -1.5],
            [-1.5, -2.0],
            [-2.0, -2.5],
        ],
        dtype=torch.float32,
    )

    # Склеиваем
    x = torch.cat([class1, class0], dim=0)

    # Метки: первые 4 — 1, вторые 4 — 0
    y = torch.cat(
        [
            torch.ones((class1.shape[0], 1), dtype=torch.float32),
            torch.zeros((class0.shape[0], 1), dtype=torch.float32),
        ],
        dim=0,
    )

    return x, y


class SinglePerceptron(nn.Module):
    """
    Один современный "соло-П" в терминах PyTorch.

    Внутри:
    - self.linear: nn.Linear(2, 1)
      где weight: тензор формы [1, 2]
          bias:   тензор формы [1]

    forward(x):
      1) считает z = x @ W^T + b
      2) через сигмоиду превращает в значение [0..1]
    """

    def __init__(self) -> None:
        super().__init__()
        # 2 входа -> 1 выход, с bias
        self.linear = nn.Linear(in_features=2, out_features=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, 2]
        z = self.linear(x)          # [batch_size, 1]
        y_prob = torch.sigmoid(z)   # [batch_size, 1] значения 0..1
        return y_prob


def main() -> None:
    # 1. Данные
    x, y = build_training_data()

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 2. Модель: соло-перцептрон
    model = SinglePerceptron()

    # 3. Функция потерь:
    #    binary cross entropy по вероятностям (после сигмоиды)
    loss_fn = nn.BCELoss()

    # 4. Оптимизатор SGD:
    #    он хранит ссылку на параметры model (W и b)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    num_epochs = 30

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch_x, batch_y in dataloader:
            # 1) прямой проход
            y_pred = model(batch_x)     # [batch_size, 1]

            # 2) считаем loss
            loss = loss_fn(y_pred, batch_y)

            # 3) обнуляем градиенты перед новым шагом
            optimizer.zero_grad()

            # 4) считаем градиенты dLoss/dW, dLoss/dB
            loss.backward()

            # 5) обновляем W и b:
            #    param <- param - lr * grad
            optimizer.step()

            epoch_loss += loss.item()

        # Смотрим текущее W и b после эпохи
        W = model.linear.weight.data.clone().detach().numpy()
        b = model.linear.bias.data.clone().detach().numpy()
        print(
            f"Эпоха {epoch:02d}, loss={epoch_loss:.4f}, "
            f"W={W}, b={b}"
        )

    # Финальная проверка
    print("\n=== Финальная проверка ===")
    with torch.no_grad():
        y_pred = model(x)
        y_class = (y_pred >= 0.5).float()
        for i in range(x.shape[0]):
            print(
                f"x={x[i].tolist()}, "
                f"target={int(y[i].item())}, "
                f"pred_prob={y_pred[i].item():.3f}, "
                f"pred_class={int(y_class[i].item())}"
            )


if __name__ == "__main__":
    main()
