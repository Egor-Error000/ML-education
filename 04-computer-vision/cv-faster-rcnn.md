# Faster R-CNN: RPN, RoI и обучение двух стадий

**Уровень:** основа ветки.

**Предпосылки:** [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>), [FPN и PAN: многомасштабные представления](<cv-pyramids.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как RPN и RoI-head совместно превращают карту признаков в объекты?

## Схема вычислений

![Faster R-CNN: RPN, RoI и обучение двух стадий: вычислительные этапы](<../оформление/course/expanded-cv-faster-rcnn.svg>)

RPN: Image → backbone / FPN → Anchors → objectness + 4Δ → Decode → proposal NMS → R выбранных областей. RoI-head: Области + общие features → RoIAlign → R×C×7×7 → Class head + box head → Scores → final NMS.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Две стадии с общей картой

Faster R-CNN сначала предлагает области, затем уточняет их классы и координаты. Backbone вычисляет карту всего изображения один раз. RPN, region proposal network, скользит по карте, предсказывая objectness и поправки к anchors. RoI-head извлекает признаки предложенных областей и решает более точную задачу. В [исходной работе](https://arxiv.org/abs/1506.01497) используется RoI pooling; RoIAlign появился позже в Mask R-CNN. Современную комбинацию Faster R-CNN + FPN + RoIAlign нужно явно отличать от исторической архитектуры.

## Targets и RPN

Anchor — рамка заданного масштаба и пропорций, привязанная к позиции карты. Для $A$ anchors на позицию RPN выдаёт $A$ objectness scores и $4A$ координатных offsets. Правило IoU assignment делит anchors на positive, negative и ignored. Positive регрессируют к assigned ground truth, negative участвуют в objectness, ignored не должны создавать обычный background loss. Правило гарантирования кандидата для каждого GT зависит от реализации.

Для anchor с центром $(x_a,y_a)$ и размерами $(w_a,h_a)$ targets задаются $t_x=(x_g-x_a)/w_a$, $t_y=(y_g-y_a)/h_a$, $t_w=\ln(w_g/w_a)$, $t_h=\ln(h_g/h_a)$. Inference обращает это преобразование. Перед RoI-head обычно выполняются clipping, отбор и NMS proposals. Эти операции не заменяют training assignment.

## RoI-признаки, loss и inference

RoIAlign переводит каждую область в сетку фиксированного размера через bilinear sampling, без грубого округления её координат к целым cells. Для $R$ областей получаем $[R,C,h_r,w_r]$, затем classifier выдаёт классы с background, а regressor — offsets. Classification loss применяется к выбранным RoIs, box loss — к positive. Полная цель суммирует RPN classification/regression и RoI classification/regression с указанными нормировками.

При применении меток нет: RPN формирует proposals, RoI-head уточняет, scores фильтруются, рамки проходят final NMS. Плохой proposal recall ограничивает качество второй стадии: она не умеет выбирать область, которой не было среди кандидатов.

## Разобранный пример: обратное преобразование

Anchor имеет центр $(50,40)$ и размеры $(20,10)$, GT — центр $(54,38)$ и размеры $(30,10)$. Targets: $t_x=0.2$, $t_y=-0.2$, $t_w=\ln1.5\approx0.40547$, $t_h=0$. Декодирование: $x=50+0.2\cdot20=54$, $y=40-0.2\cdot10=38$, $w=20e^{0.40547}=30$, $h=10$. Итоговый xyxy — $(39,33,69,43)$.

Для 1000 proposals, $C=256$ и RoI-сетки $7\times7$ получаем $12544000$ активаций, примерно 47.85 MiB float32 до дальнейших слоёв. Уменьшение proposals сокращает стоимость, но может снизить recall мелких и перекрытых объектов.

## Самопроверка с ответами

**Практика:** удалите proposals, перекрывающие маленький GT. Даже идеальный classifier оставшихся RoIs не гарантирует восстановление объекта; сначала измеряют proposal recall.

**Почему background RoI не получает обычный GT box loss?** Для него нет назначенной целевой рамки объекта.

**Равны ли RoI pooling и RoIAlign?** Нет. Различие quantization влияет на пространственную точность; RoIAlign также не означает, что координаты стали безошибочными.

## Иллюстрации из первоисточников

![faster-rcnn: Figure 2 из авторской статьи](<../оформление/course/papers/faster-rcnn.png>)

**Источник:** [faster-rcnn, Figure 2](https://arxiv.org/abs/1506.01497), PDF, с. 3. Историческая схема использует RoI pooling. RoIAlign в современном варианте разобран отдельно; он появился в Mask R-CNN.

## Источники и продолжение

- [Faster R-CNN — Ren et al., NeurIPS 2015](https://arxiv.org/abs/1506.01497)
- [Mask R-CNN / RoIAlign — He et al., ICCV 2017](https://arxiv.org/abs/1703.06870)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<03-detection/План.md>)
