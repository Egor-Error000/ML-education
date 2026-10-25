# YOLOv3 и YOLOv8: головы, assignment и декодирование

**Уровень:** основа ветки.

**Предпосылки:** [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>), [FPN и PAN: многомасштабные представления](<cv-pyramids.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Почему YOLOv3 и YOLOv8 нельзя объяснять одной формулой головы?

## Схема вычислений

![YOLOv3 и YOLOv8: головы, assignment и декодирование: вычислительные этапы](<../оформление/course/expanded-cv-yolo.svg>)

YOLOv3: Darknet-53 → 3 масштаба → 3 anchors × (5+K) → Sigmoid center; exp sizes → Objectness×class → NMS. YOLOv8 / v8.3.0: Backbone + PAN features → K logits; 4×16 bins → Softmax → mean distances → Decode points → NMS.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Зафиксировать две конкретные модели

Название YOLO объединяет разные реализации. Здесь YOLOv3 рассматривается по [Redmon и Farhadi](https://arxiv.org/abs/1804.02767), YOLOv8 — по документации Ultralytics и исходному коду тега v8.3.0. Это учебный срез, а не обещание совместимости всех поздних конфигураций. Общая структура — backbone, многомасштабное слияние и dense heads, но targets, выходные каналы и post-processing различаются.

## YOLOv3: anchors и objectness

Darknet-53 извлекает признаки; три масштаба головы возвращают по три anchors. На anchor предсказываются $(t_x,t_y,t_w,t_h)$, objectness и class logits. Для cell $(c_x,c_y)$ центр в координатах сетки $(\sigma(t_x)+c_x,\sigma(t_y)+c_y)$, размеры $p_we^{t_w},p_he^{t_h}$ относительно anchor. Нужно явно привести центры и размеры к одним единицам. Class scores — независимые logistic outputs; они не обязаны суммироваться в единицу.

Assignment использует anchors и GT; objectness отличает заполненный candidate от фона. На inference score связывает objectness и class score, затем применяется filtering и NMS. Потери исходной реализации не следует подменять современным CIoU+DFL только потому, что обе модели называются YOLO.

## YOLOv8: раздельные ветви и distributional regression

В стандартной detection-конфигурации YOLOv8 голова anchor-free относительно размеров рамок, но использует опорные точки сетки. Разделены class и regression branches; отдельного objectness output как у YOLOv3 нет. Каждая сторона рамки задаётся распределением по дискретным bins. Softmax probabilities $p_j$ дают расстояние $d=\sum_jjp_j$, затем расстояния масштабируются stride и декодируются вокруг опорной точки.

Task-aligned assignment связывает candidates с GT по совместной оценке classification и localization. Box loss сочетает IoU-компонент и distribution focal loss, DFL. DFL обучает соседние bins для дробного target: при $d^*=2.3$ веса bins 2 и 3 равны 0.7 и 0.3. В tag v8.3.0 эти детали определены в [loss.py](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py) и [tal.py](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py).

## Разобранный пример: выходы и decoding

На входе $640\times640$ сетки 80,40,20 дают 8400 positions. YOLOv3 с 80 классами выдаёт $3(4+1+80)=255$ channels каждого уровня. Всего $8400\cdot255=2142000$ raw numbers.

YOLOv8 с 16 bins на сторону имеет $4\cdot16+80=144$ raw channels, 1209600 чисел. Для одной стороны probabilities $(0,0.2,0.5,0.3)$ на bins 0..3 дают $d=0.2+1+0.9=2.1$ cells. При stride 8 расстояние 16.8 pixels; для точки $x=100$ левая граница 83.2. Полная голова имеет больше bins; короткое распределение здесь только для арифметики.

## Нюансы решения

Letterbox требует сохранить scale и padding для обратного преобразования boxes. NMS-free свойства других YOLO не относятся автоматически к YOLOv8. Мелкие объекты зависят от resolution и уровней пирамиды; крупные batch и красивые FPS не заменяют измерение нужной latency. Детали другого релиза проверяют по его коду.

## Самопроверка с ответами

**Практика:** для target 2.3 запишите DFL: $-0.7\ln p_2-0.3\ln p_3$. Это CE с мягкой двухточечной целью, не MSE одного среднего.

**Почему anchor-free всё ещё имеет anchor points в коде?** Это опорные координаты, а не набор заранее выбранных размеров рамок.

**Можно ли декодировать YOLOv8 через экспоненту ширины YOLOv3?** Нет. Смысл regression outputs другой.

## Иллюстрации из первоисточников

![yolov3: Figure 2 из авторской статьи](<../оформление/course/papers/yolov3.png>)

**Источник:** [yolov3, Figure 2](https://arxiv.org/abs/1804.02767), PDF, с. 2. Рисунок показывает параметризацию рамки YOLOv3. Формулы этой схемы не относятся к distributional head YOLOv8.

## YOLOv8: подробная схема из авторской публикации

![YOLOv8: backbone, C2f, FPN/PAN и Detect head — RangeKing](<../оформление/course/papers/yolov8-rangeking.jpg>)

**Автор и источник:** [RangeKing, YOLOv8_structure_v1.6](https://github.com/ultralytics/ultralytics/issues/189). Схема опубликована автором RangeKing в репозитории Ultralytics. Показаны backbone, FPN/PAN-путь, C2f, SPPF и раздельная Detect head с CIoU/DFL и BCE. Это community diagram для P5-моделей, а не официальная статья YOLOv8; численные детали главы сверены с кодом v8.3.0.

## Источники и продолжение

- [YOLOv3 — Redmon, Farhadi, 2018](https://arxiv.org/abs/1804.02767)
- [YOLOv8 — Ultralytics documentation](https://docs.ultralytics.com/models/yolov8/)
- [Ultralytics v8.3.0 — loss implementation](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/loss.py)
- [Ultralytics v8.3.0 — task aligned assignment](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/utils/tal.py)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<03-detection/План.md>)
