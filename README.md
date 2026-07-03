# Домашняя работа №2
## Линейная и логистическая регрессия на PyTorch

### Описание

В работе выполнены следующие задания:

- модифицирована модель линейной регрессии;
- добавлена L1 и L2 регуляризация;
- реализован механизм Early Stopping;
- реализована многоклассовая логистическая регрессия;
- создан собственный класс Dataset для работы с CSV файлами;
- проведено обучение моделей на реальных датасетах;
- выполнены эксперименты с гиперпараметрами;
- реализован Feature Engineering.

---

## Структура проекта

```
regression_basics/
│
├── data/
│   ├── train.csv
│   └── insurance.csv
│
├── models/
│   ├── titanic_logistic.pth
│   └── insurance_linear.pth
│
├── plots/
│   └── optimizer_comparison.png
│
├── homework_model_modification.py
├── homework_datasets.py
├── homework_experiments.py
├── requirements.txt
└── README.md
```

---

## Используемые датасеты

### Titanic

Используется для задачи бинарной классификации.

Целевой признак:

```
Survived
```

### Insurance

Используется для задачи регрессии.

Целевой признак:

```
charges
```

---

## Запуск

Установить зависимости:

```bash
pip install -r requirements.txt
```

Запуск заданий:

```bash
python homework_model_modification.py
```

```bash
python homework_datasets.py
```

```bash
python homework_experiments.py
```

---

## Результаты

В процессе выполнения:

- обучаются модели;
- сохраняются веса моделей в папку `models`;
- сохраняются графики в папку `plots`.
