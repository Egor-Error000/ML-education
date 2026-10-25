# Mask2Former: mask queries и panoptic segmentation

**Уровень:** углубление.

**Предпосылки:** [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>), [DETR: queries, matching и set prediction](<cv-detr.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как один набор mask queries решает разные виды segmentation?

## Схема вычислений

![Mask2Former: mask queries и panoptic segmentation: вычислительные этапы](<../оформление/course/expanded-cv-mask2former.svg>)

Mask2Former: Backbone → pixel decoder → Mask features D×Hm×Wm → Queries → class + embedding → Dot product → Q masks. Masked decoding: Предыдущие query masks → Resize / threshold → support → Masked cross + self-attention → Новые queries → новые masks.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Mask classification вместо отдельного класса каждого пикселя

Mask2Former предсказывает набор masks и class distributions. Thing — счётный экземпляр, например автомобиль; stuff — область вроде дороги. Для panoptic результата каждый pixel получает semantic class и при необходимости instance ID. [Cheng et al.](https://arxiv.org/abs/2112.01527) используют единый mask-classification механизм для semantic, instance и panoptic задач, но разметка и итоговая сборка результата различаются.

## Pixel decoder и Transformer decoder

Backbone формирует multiscale features. Pixel decoder смешивает их и строит mask features $F\in\mathbb R^{D\times H_m\times W_m}$. Transformer decoder обновляет $Q$ queries. Для каждого query linear head даёт class logits, MLP — mask embedding $e_q\in\mathbb R^D$. Mask logits: $s_q(u,v)=e_q^TF(:,u,v)$. Эта общая spatial basis похожа по форме линейной сборки на prototypes, но queries и supervision организованы иначе.

Masked cross-attention ограничивает доступ query к spatial positions по его предсказанной mask предыдущего шага. Маска переводится к размеру используемого feature level; запрещённые positions получают attention mask. Это не GT mask на inference. Self-attention связывает queries, cross-attention читает image features, feed-forward блок обновляет каждое состояние. Многошаговое уточнение позволяет менять поддерживаемую область.

## Training и три вида результата

Hungarian matching сопоставляет predicted masks с GT segments по class и mask costs. Classification loss включает no-object, matched masks получают BCE и Dice. Для уменьшения цены mask supervision используется sampling points; выбранная оценка loss не тождественна обходу всех pixels. Отсутствующие или ignored regions нужно обрабатывать по контракту датасета.

Semantic output может агрегировать class probabilities и mask probabilities по queries. Instance output выбирает объектные masks и их scores. Panoptic assembler разрешает перекрытия и объединяет stuff одного класса, сохраняя раздельные thing IDs. Один forward не делает эти выходные политики взаимозаменяемыми.

## Разобранный пример: semantic aggregation и PQ

Два queries имеют вероятности класса «дорога» 0.8 и 0.1, а mask probabilities в pixel — 0.6 и 0.9. Ненормированный semantic score дороги: $0.8\cdot0.6+0.1\cdot0.9=0.57$. Это сумма contributions, а не готовая откалиброванная вероятность события.

Для panoptic evaluation пусть два matched segments имеют IoU 0.8 и 0.6, ещё есть один FP и один FN. Тогда $PQ=(0.8+0.6)/(2+0.5+0.5)=0.46667$. Segmentation Quality $SQ=1.4/2=0.7$, Recognition Quality $RQ=2/3$, произведение $0.46667$. Matching и правила допустимости классов задаются протоколом, обычно с IoU больше 0.5 для panoptic segments.

## Ограничения

Слишком малое $Q$ ограничивает число segments. Ошибка ранней mask может временно закрыть полезные positions; реализация должна корректно обрабатывать пустую support. Хороший mIoU не доказывает сохранение отдельных объектов. Для сравнения нужны одинаковые task protocol и post-processing.

## Самопроверка с ответами

**Практика:** объедините две соседние машины в одну mask. Semantic class pixels может сохраниться, но instance matching и PQ ухудшатся.

**Используется ли GT mask для masked attention при inference?** Нет, ограничение создаётся моделью.

**Почему сравнивать PQ и mIoU напрямую некорректно?** Первая оценивает также распознавание segments и их matching, вторая — semantic overlap по классам.

## Иллюстрации из первоисточников

![mask2former: Figure 2 из авторской статьи](<../оформление/course/papers/mask2former.png>)

**Источник:** [mask2former, Figure 2](https://arxiv.org/abs/2112.01527), PDF, с. 3. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [Mask2Former — Cheng et al., CVPR 2022](https://arxiv.org/abs/2112.01527)
- [Panoptic segmentation — Kirillov et al., CVPR 2019](https://arxiv.org/abs/1801.00868)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<04-segmentation/План.md>)
