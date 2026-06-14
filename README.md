# MNIST Digit Recognizer

https://github.com/Pabloomm1/NN_mnist.git

Нейронная сеть для распознавания рукописных цифр. В ее реализации не использовалиь PyTorch/Tensorflow так что результат может быть не идеален.
Архитектура: 784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax), обучение через mini-batch SGD с cross-entropy loss.

## Структура

```
├── NN.ipynb          # обучение
├── nn_test.py        # GUI для рисования и распознавания
├── requirements.txt
├── mnist/            # сюда распаковать датасет
└── parameters/       # сюда сохраняются веса после обучения
```

## Установка

Клонировать репозиторий вместе с submodule-ами:

```bash
git clone https://github.com/Pabloomm1/NN_mnist.git
cd NN_mnist
git submodule init
git submodule update
```

Создать и активировать виртуальное окружение:

```bash
python3 -m venv venv
source venv/bin/activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
pip install jupyterlab
```


## Данные

Датасет лежит в репозитории в архиве. Распаковать в папку `mnist/`:

```bash
unzip mnist/mnist.zip -d mnist/
```

Внутри должны оказаться два файла: `mnist_train.csv` и `mnist_test.csv`.

## Обучение

Запустить JupyterLab:

```bash
jupyter lab
```

Открыть `NN.ipynb` и запустить все ячейки. Обучение занимает несколько минут, в процессе рисуются графики loss и accuracy. После завершения веса автоматически сохранятся в папку `parameters/`.

## Запуск приложения

После обучения (или если веса уже есть в `parameters/`):

```bash
python nn_test.py
```

Откроется окно — рисуй цифру мышью, жми «Предсказать». Слева холст, справа предсказание и вероятности по каждому классу.
