# DINO, DINOv2 и DINOv3: self-distillation и dense features

**Уровень:** углубление.

**Предпосылки:** [Vision Transformer: patches, attention и обучение](<cv-vit.md>), [Обучение и перенос: frozen features, fine-tuning и аугментации](<cv-transfer.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как self-distillation учит глобальные и пространственные признаки?

## Схема вычислений

![DINO, DINOv2 и DINOv3: self-distillation и dense features: вычислительные этапы](<../оформление/course/expanded-cv-dino.svg>)

Self-distillation: Изображение → разные crops → Student → logits; teacher EMA → Center / sharpen teacher → Cross-entropy → student. Dense readout: Изображение → patch encoder → B×N×D; global / patches → Frozen features → task head → Classes / masks + evaluation.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Teacher без ручных классов

DINO сопоставляет предсказания student и teacher для разных views одного изображения. Их выходные координаты — обучаемые прототипы, не названия классов. Student получает градиент cross-entropy, teacher обновляется EMA весов student. [Исходный DINO](https://arxiv.org/abs/2104.14294) сочетает multi-crop и self-distillation; карты внимания могут показывать структуру объектов, но не заменяют оценённую segmentation head.

## Почему требуется предотвращать collapse

Пусть $u_s,u_t\in\mathbb R^K$ — выходы projection heads. Student probabilities $p_s=\operatorname{softmax}(u_s/\tau_s)$, teacher $p_t=\operatorname{softmax}((u_t-c)/\tau_t)$, где $c$ — обновляемый по batch центр teacher logits. Цель $L=-\sum_kp_{t,k}\log p_{s,k}$ вычисляется для разных views, без градиента через teacher. Sharpening делает распределение teacher более выраженным, centering противодействует доминированию нескольких координат. Их баланс важен: один EMA без прочих условий не доказывает отсутствие collapse.

Teacher обычно обрабатывает global crops, student также local crops. Смысл обучения — согласовать разные доступные наблюдения; крошечный фрагмент не обязан содержать все объекты сцены. Loss не следует интерпретировать как доказательство, что каждый patch получил истинную semantic label.

## От DINO к DINOv2 и DINOv3

[DINOv2](https://arxiv.org/abs/2304.07193) включает масштабирование, curated data и сочетание image-level и patch-level сигналов; это не просто замена backbone в исходном DINO. Качество данных и процедура дистилляции входят в метод. [DINOv3](https://ai.meta.com/research/dinov3/) развивает переносимые dense features; его применение нужно отделять от конкретных heads и протоколов оценки. В этой главе рассматриваются механизмы и использование представлений, без универсального ранжирования семейств.

Для ViT вход $[B,3,H,W]$ после patches даёт $[B,N,D]$. Global vector пригоден для retrieval и classification, patch vectors можно вернуть в $[B,D,H/P,W/P]$ для dense prediction. Frozen head обучается на labels новой задачи; если encoder заморожен, это не новое self-supervised pretraining.

## Разобранный пример: CE и карта признаков

Teacher выдаёт $(0.8,0.2)$, student $(0.6,0.4)$. Cross-entropy равна $-0.8\ln0.6-0.2\ln0.4\approx0.59192$. При student $(0.8,0.2)$ значение $\approx0.50040$, а не ноль: это энтропия мягкой цели. Для сравнения совпадения распределений можно рассматривать KL, вычитая постоянную энтропию teacher.

Изображение $224\times224$ с patch size 14 даёт $16\cdot16=256$ patch vectors. При $D=768$ spatial representation имеет 196608 чисел. Upsampling logits с $16\times16$ до исходной сетки возвращает размер ответа, но не гарантирует точные контуры объектов меньше patch.

## Самопроверка с ответами

**Практика:** сравните linear probe по global vector и dense head по patch vectors. Это разные выходы и задачи; одной accuracy недостаточно для вывода о dense features.

**Почему teacher loss может не стать нулём?** Soft targets имеют собственную ненулевую энтропию.

**Равны ли teacher prototypes классам датасета?** Нет. Их семантика возникает в процессе обучения, а названные классы добавляет downstream-задача.

## Иллюстрации из первоисточников

![dino: Figure 2 из авторской статьи](<../оформление/course/papers/dino.png>)

**Источник:** [dino, Figure 2](https://arxiv.org/abs/2104.14294), PDF, с. 2. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [DINO — Caron et al., ICCV 2021](https://arxiv.org/abs/2104.14294)
- [DINOv2 — Oquab et al., 2023/2024](https://arxiv.org/abs/2304.07193)
- [DINOv3 — Meta, official project](https://ai.meta.com/research/dinov3/)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<02-representations/План.md>)
