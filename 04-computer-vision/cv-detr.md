# DETR: queries, matching и set prediction

**Уровень:** основа ветки.

**Предпосылки:** [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>), [Vision Transformer: patches, attention и обучение](<cv-vit.md>)

**Используют:** [Deformable DETR и RT-DETR: внимание и скорость](<cv-deformable.md>), [Mask2Former: mask queries и panoptic segmentation](<cv-mask2former.md>)

**Учебный вопрос:** Как обучение множества предсказаний устраняет необходимость обычной NMS?

## Схема вычислений

![DETR: queries, matching и set prediction: вычислительные этапы](<../оформление/course/expanded-cv-detr.svg>)

DETR forward: Image → CNN map + positions → Flatten → encoder memory → Q learned queries → decoder → Q classes + Q boxes. Training only: GT set + predictions → Class / L1 / GIoU costs → Hungarian one-to-one pairs → Matched loss; no-object rest.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Выход как множество

Обычная dense head создаёт много перекрывающихся candidates. DETR обучается выдавать фиксированное число slots, каждому из которых соответствует объект либо no-object. Порядок GT не имеет значения. Backbone превращает изображение в карту $[B,C,H_f,W_f]$, projection приводит каналы к $D$, flatten даёт последовательность $N=H_fW_f$. Positional encodings сохраняют spatial информацию. Transformer encoder связывает позиции; decoder использует $Q$ обучаемых object queries. [Carion et al.](https://arxiv.org/abs/2005.12872) задают этот путь как set prediction.

## Queries, heads и matching

Query — обучаемое состояние слота, не готовая рамка и не класс объекта. После decoder имеем $[B,Q,D]$. Общие heads возвращают $K+1$ class logits и четыре нормированных координаты $(c_x,c_y,w,h)$ для каждого slot. Если истинных объектов $M\le Q$, Hungarian assignment выбирает взаимно однозначное сопоставление GT и slots, минимизирующее сумму pair costs. Стоимость учитывает класс, L1 расстояние boxes и GIoU; её коэффициенты относятся к заданной конфигурации.

Matching дискретен и пересчитывается по текущим predictions. После него matched slots получают classification и box losses, unmatched — target no-object с отдельным весом. Gradient проходит через выбранные predictions, а не через перебор перестановок как обычную гладкую функцию. Auxiliary decoder losses облегчают обучение промежуточных слоёв.

## Почему нельзя смешивать training и inference

На inference GT и Hungarian matching не нужны: модель возвращает class scores и boxes, затем выбираются объектные predictions. Обычная NMS не входит в исходную схему. Обучение one-to-one assignment создаёт стимул одному slot отвечать за объект, но не является математическим доказательством отсутствия всех дубликатов при любых входах. Число queries ограничивает максимальное число объектов, которое можно представить этим выходом.

## Разобранный пример: две перестановки

Пусть два GT, A и B, и два predictions, 1 и 2. Матрица pair costs по строкам predictions: $(1,2)$ и $(1.1,100)$. Greedy берёт пару 1→A с cost 1 и оставляет 2→B с cost 100, всего 101. Глобальный assignment выбирает 1→B и 2→A, всего $2+1.1=3.1$. Именно взаимодействие назначений требует оптимизации всего сопоставления.

При $Q=100$, $K=20$ один кадр имеет $100\cdot21=2100$ class logits и 400 box values. Карта $25\times40$ даёт 1000 encoder tokens; одна полная attention matrix содержит миллион scores на head. Удвоение обеих spatial сторон создаёт 4000 tokens и 16 миллионов scores. Это одна причина стоимости плотного attention.

## Обучение и ограничения

Случайные начальные queries ещё не умеют искать объекты; совместное обучение localization и assignment может быть долгим. Грубая карта затрудняет мелкие объекты. Padding mask должна закрывать искусственные позиции batch. Сравнение с YOLO требует полного времени, включая NMS там, где она используется.

## Самопроверка с ответами

**Практика:** добавьте третий slot без третьего GT. Два slots получат matched targets, оставшийся — no-object; дублировать один GT как positive не нужно.

**Нужен ли Hungarian algorithm при применении?** Нет: он создаёт supervision при обучении.

**Почему нельзя просто сортировать GT по x?** Произвольный порядок создаёт нестабильные цели при близких объектах и не выражает симметрию множества.

## Иллюстрации из первоисточников

![detr: Figure 2 из авторской статьи](<../оформление/course/papers/detr.png>)

**Источник:** [detr, Figure 2](https://arxiv.org/abs/2005.12872), PDF, с. 7. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [DETR — Carion et al., ECCV 2020](https://arxiv.org/abs/2005.12872)

[Следующие углубления: cv-deformable, cv-mask2former](<README.md>) · [Сценарии подраздела](<03-detection/План.md>)
