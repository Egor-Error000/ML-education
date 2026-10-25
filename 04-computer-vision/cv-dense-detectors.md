# RetinaNet и FCOS: anchors и dense prediction

**Уровень:** основа ветки.

**Предпосылки:** [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>), [FPN и PAN: многомасштабные представления](<cv-pyramids.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Что меняется при переходе от anchors RetinaNet к расстояниям FCOS?

## Схема вычислений

![RetinaNet и FCOS: anchors и dense prediction: вычислительные этапы](<../оформление/course/expanded-cv-dense-detectors.svg>)

RetinaNet: FPN: H×W×C → A anchors на cell → AK classes + 4A offsets → Focal / box loss → NMS. FCOS: FPN: H×W×C → Позиция внутри assigned GT → K classes + l,t,r,b + c → Decode distances → NMS.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Dense prediction

Одностадийная голова предсказывает объекты во многих positions пирамиды. Большинство positions соответствуют фону, поэтому важны sampling и вес простых negatives. RetinaNet сочетает FPN, anchor-based regression и focal loss. FCOS предсказывает рамку из позиции непосредственно через расстояния до сторон. Anchor-free не означает отсутствие пространственных priors: остаются сетка, stride, scale ranges и правила assignment.

## RetinaNet

На каждом уровне две подсети выдают class scores и четыре offsets для каждого anchor. При $A$ anchors и $K$ классах выходы имеют $AK$ и $4A$ каналов на spatial cell. Classification использует независимые sigmoid и focal loss; regression применяется к положительным anchors. Формы anchors и IoU assignment задают, каким targets обучается голова. [Focal loss](https://arxiv.org/abs/1708.02002) уменьшает вклад хорошо классифицированного фона, но не исправляет неправильную разметку.

## FCOS: точка, четыре расстояния и centerness

Для позиции $(x,y)$ внутри GT box targets $l=x-x_0$, $t=y-y_0$, $r=x_1-x$, $b=y_1-y$. Обратное преобразование: $(x-l,y-t,x+r,y+b)$. Несколько перекрывающихся GT создают неоднозначность: правило выбора GT и scale range должно быть задано. [FCOS](https://arxiv.org/abs/1904.01355) использует centerness для снижения веса низкокачественных boxes, предсказанных далеко от центра:

$$
c^*=\sqrt{\frac{\min(l,r)}{\max(l,r)}\frac{\min(t,b)}{\max(t,b)}}.
$$

Centerness обучается отдельной ветвью. Это геометрический target, не IoU предсказанной рамки с GT. Classification, regression и centerness losses имеют разные targets. На inference combined score и NMS зависят от реализации; нельзя переносить один способ объединения scores на все FCOS-варианты.

## Разобранный пример: положение внутри рамки

GT box $(10,20,50,60)$, позиция $(20,40)$. Расстояния $(10,20,30,20)$, centerness $\sqrt{(10/30)(20/20)}=1/\sqrt3\approx0.57735$. В центре $(30,40)$ расстояния $(20,20,20,20)$ и centerness 1. Для точки вне рамки обычные positive targets не определяются этим правилом.

Пирамида имеет 8400 positions. При девяти anchors RetinaNet и $K=80$ classification выдаёт $8400\cdot9\cdot80=6048000$ scores. FCOS с одним предсказанием на позицию — $8400\cdot80=672000$, плюс regression и centerness. Это сравнение выходных массивов; реальные FLOPs зависят также от ширины и глубины head.

## Диагностика

Если маленький GT не получает positives, сначала проверяют stride, center sampling и scale assignment, а не увеличивают число эпох. При высоком recall до NMS и низком после неё исследуют score ordering и IoU threshold. В crowded scenes соседние истинные объекты могут подавлять друг друга. Повышение resolution увеличивает число candidates и стоимость фильтрации.

## Самопроверка с ответами

**Практика:** сдвиньте точку из центра к левой стороне. Centerness уменьшится из-за дисбаланса $l/r$, хотя истинный класс не меняется.

**Есть ли у anchor-free detector выбор положительных positions?** Да. Его нельзя устранить одним удалением anchor shapes.

**Является ли centerness калиброванной вероятностью корректного объекта?** Нет: target задан геометрической формулой, а итоговую пригодность score проверяют отдельно.

## Иллюстрации из первоисточников

![retinanet: Figure 3 из авторской статьи](<../оформление/course/papers/retinanet.png>)

**Источник:** [retinanet, Figure 3](https://arxiv.org/abs/1708.02002), PDF, с. 5. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

![fcos: Figure 2 из авторской статьи](<../оформление/course/papers/fcos.png>)

**Источник:** [fcos, Figure 2](https://arxiv.org/abs/1904.01355), PDF, с. 3. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [RetinaNet — Lin et al., ICCV 2017](https://arxiv.org/abs/1708.02002)
- [FCOS — Tian et al., ICCV 2019](https://arxiv.org/abs/1904.01355)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<03-detection/План.md>)
