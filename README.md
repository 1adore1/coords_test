# Задача отображения координат между камерами

## Описание
Модель отображает координаты точек между камерами:  
src (top / bottom) -> door2

Вход:
- 2 изображения  
- координаты точек на src (в пикселях)  
- source_id (0 - "top", 1 - "bottom")

Выход:
- список координат точек на door2  


## Установка

```
python3 -m venv venv
source venv/bin/activate
cd solution
pip install -r requirements.txt
```

## Обучение

Все обучение находится в train.ipynb. 

После обучения веса best/last.pt, а также отдельные метрики top->door, bottom->door попадут в SAVE_DIR = 'runs/09_04_26'.

- Best val L2: 103.22 px

- top -> door2 L2: 93.67 px

- bottom -> door2 L2: 113.4 px


## Predict

```
cd solution
python3 predict.py```

Внутри predict.py можно указать свои пути до изображений и координаты точек.