# Deformable DETR и RT-DETR: внимание и скорость

**Уровень:** углубление.

**Предпосылки:** [DETR: queries, matching и set prediction](<cv-detr.md>), [FPN и PAN: многомасштабные представления](<cv-pyramids.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как сократить внимание к изображению и ускорить set prediction?

## Схема вычислений

![Deformable DETR и RT-DETR: внимание и скорость: вычислительные этапы](<../оформление/course/expanded-cv-deformable.svg>)

Deformable attention: Query + reference point → Offsets / weights: M×L×K → Bilinear sample multi-scale → Weighted sum → refinement. RT-DETR: Backbone: multi-scale maps → Hybrid encoder: intra / cross → Выбор начальных queries → Decoder → classes / boxes.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Ограничения полного attention

В DETR каждый query может учитывать все spatial tokens. Для высоких разрешений это дорого, а обучение поиска нужных позиций затруднено. Deformable attention связывает query с небольшим числом sampling points около reference point на нескольких масштабах. Разреженность задана числом точек, но их смещения и веса обучаются, а не фиксируются вручную.

## Deformable DETR

Для query $q$, head $m$, уровня $l$ и точки $k$ сеть предсказывает offset $\Delta p_{mlk}$ и attention weight $a_{mlk}$. Вектор извлекается bilinear sampling в позиции $p_q+\Delta p_{mlk}$ соответствующей карты. После нормировки weights суммируются по sampling points и scales. Полученные head outputs смешиваются projection. [Zhu et al.](https://arxiv.org/abs/2010.04159) вводят многомасштабное deformable attention; iterative refinement и two-stage варианты нужно отличать от базового механизма.

Если bilinear sampling попадает между четырьмя cells, выход — взвешенная сумма их vectors. Gradient меняет и features, и offsets через interpolation. Число sampled positions примерно $QMLK$ вместо $QMN$ плотного cross-attention, но сюда не включены все projections, encoder и другие операции. Это оценка выборки features, не доказанное ускорение всей сети в такое же число раз.

## RT-DETR

[RT-DETR](https://arxiv.org/abs/2304.08069) строит efficient hybrid encoder: связывает признаки внутри масштаба и между масштабами разными механизмами. Отбор начальных queries учитывает качество encoder features. Decoder уточняет boxes; обучение сохраняет set prediction, обычная NMS не требуется в стандартном inference. Число decoder layers позволяет менять вычислительный бюджет, но качество конкретного сокращения измеряют, а не считают неизменным по определению.

Оба подхода используют multiscale features и one-to-one supervision, но устройство encoder, initialization queries и training tricks различается. Отсутствие NMS не означает отсутствие score filtering, top-K или стоимости переноса данных.

## Разобранный пример: sampling и interpolation

Пусть $Q=100$, $M=8$, $L=4$, $K=4$. Всего $100\cdot8\cdot4\cdot4=12800$ выборок features. Dense cross-attention при $N=20000$ использует $100\cdot8\cdot20000=16000000$ scores. Разница иллюстрирует структурную экономию, но sampling часто менее эффективно использует память, чем большие матричные умножения.

В одной scalar-карте значения углов cell равны 0,2,4,6. Для координат с долями $u=0.25$, $v=0.5$ weights: 0.375,0.125,0.375,0.125. Результат $0+0.25+1.5+0.75=2.5$. Если перепутать нормированные координаты и pixel indices, sampled signal будет другим даже при правильных tensor shapes.

## Диагностика и выбор

Проверяют padding, scale-normalization offsets и положение reference points. Для edge inference важны поддержка sampling backend и стоимость conversion. Сравнение включает preprocessing, transfer, decoder, filtering и одинаковый batch. Название статьи с заявлением о скорости не заменяет эксперимент на целевой платформе.

## Самопроверка с ответами

**Практика:** удвойте число sampling points. Число выборок удвоится, но latency может измениться нелинейно из-за kernels и памяти.

**Являются ли reference points истинными центрами GT на inference?** Нет: модель получает или предсказывает их без доступа к GT.

**Почему sparse attention не теряет весь глобальный контекст?** Точки выбираются по query на разных уровнях, а информация уже смешана backbone и encoder; разреженность одного шага не означает изолированность всех областей.

## Иллюстрации из первоисточников

![deformable: Figure 2 из авторской статьи](<../оформление/course/papers/deformable.png>)

**Источник:** [deformable, Figure 2](https://arxiv.org/abs/2010.04159), PDF, с. 5. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

![rtdetr: Figure 4 из авторской статьи](<../оформление/course/papers/rtdetr.png>)

**Источник:** [rtdetr, Figure 4](https://arxiv.org/abs/2304.08069), PDF, с. 5. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [Deformable DETR — Zhu et al., ICLR 2021](https://arxiv.org/abs/2010.04159)
- [RT-DETR — Zhao et al., CVPR 2024](https://arxiv.org/abs/2304.08069)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<03-detection/План.md>)
