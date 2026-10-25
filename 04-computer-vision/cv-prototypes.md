# YOLACT: prototype masks, коэффициенты и crop

**Уровень:** основа ветки.

**Предпосылки:** [Instance segmentation: Mask R-CNN и prototype masks](<cv-instance.md>)

**Используют:** Самостоятельное применение выбранной ветки.

**Учебный вопрос:** Как получить маску экземпляра без отдельного пространственного decoder на каждый объект?

## Схема вычислений

![YOLACT: prototype masks, коэффициенты и crop: вычислительные этапы](<../оформление/course/expanded-cv-prototypes.svg>)

YOLACT shared branch: Image → backbone / FPN → Protonet → Hp×Wp×K → Общие prototypes P → P используется всеми objects. Instance branch: FPN → classes / boxes / c → Candidate filtering + NMS → P × coefficients → sigmoid → Crop / resize → instance masks.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Две параллельные ветви

YOLACT разделяет instance segmentation на общие prototype maps и коэффициенты конкретных candidates. Backbone/FPN вычисляет shared features. Protonet возвращает $P\in\mathbb R^{H_p\times W_p\times K}$, а detection branch — classes, boxes и $K$ coefficients для каждого candidate. Prototypes не обязаны быть масками известных классов: это обучаемый basis для сборки масок. [Bolya et al.](https://arxiv.org/abs/1904.02689) вводят этот механизм и исследуют его компромиссы.

## Линейная сборка маски

Для объекта $i$ в каждой позиции $(u,v)$ mask logit равен $s_i(u,v)=\sum_kP(u,v,k)c_{ik}$. Затем sigmoid даёт soft mask. Одна и та же карта prototypes используется всеми objects, но коэффициенты разные. Коэффициенты могут быть отрицательными, поэтому комбинация умеет усиливать одни области и подавлять другие. Ограничивать каждый prototype диапазоном вероятностей без основания — изменение модели.

Crop ограничивает маску соответствующей рамкой. Маску приводят к нужной сетке и threshold по объявленному правилу. Порядок resize, sigmoid, crop и threshold зависит от реализации и влияет на границы. Instance ID задаётся выбранным candidate, не номером prototype.

## Targets, loss и выбор candidates

Detection branch обучается по assignment рамок. Для positive candidates собранные masks сравниваются с GT masks через pixel loss, обычно с crop/normalization. Gradient от mask loss идёт одновременно в coefficients и prototypes. Background candidates не должны получать случайную чужую GT mask. Инференс выбирает detection candidates и их coefficients, затем собирает нужные masks; вычислять полную маску каждого фонового candidate обычно избыточно.

YOLACT использует Fast NMS как отдельный компромисс фильтрации. Он не эквивалентен побитно обычной последовательной NMS и не является причиной существования prototype representation. Для crowded scenes проверяют suppression и взаимные перекрытия отдельно от качества mask basis.

## Разобранный пример: смешивание и стоимость

В одной позиции prototypes равны $(2,-1,0.5)$, coefficients экземпляра $(0.5,1,-2)$. Logit $2\cdot0.5-1\cdot1+0.5\cdot(-2)=-1$, вероятность $\sigma(-1)\approx0.26894$. Для другого экземпляра coefficients $(1,0,0)$ дают logit 2 и вероятность 0.88080 в том же пикселе. Shared prototypes не вынуждают все объекты иметь одинаковую маску.

При $H_p=W_p=138$, $K=32$, $n=100$ выбранных экземплярах хранится 609408 prototype values и 3200 coefficients. Сборка всех masks требует около $138^2\cdot32\cdot100=60940800$ MAC и создаёт 1904400 mask values. Экономия shared features не отменяет стоимость матричного произведения и resize большого числа masks.

## Ограничения и диагностика

Грубая prototype grid затрудняет тонкие структуры. Неверный box обрежет верную часть маски; высокий box AP не гарантирует mask AP. Сравнивают мелкие и перекрытые objects, границы и post-processing, сохраняя одинаковую геометрию преобразования исходного изображения.

## Самопроверка с ответами

**Практика:** обнулите coefficients одного объекта. До crop его mask probability станет 0.5 во всех positions, а не нулём: sigmoid от нуля равен 0.5.

**Является ли prototype отдельным экземпляром?** Нет, это общая spatial component.

**Почему правильная маска может исчезнуть после crop?** Crop следует box, и ошибка локализации переносится на итоговую маску.

## Иллюстрации из первоисточников

![yolact: Figure 2 из авторской статьи](<../оформление/course/papers/yolact.png>)

**Источник:** [yolact, Figure 2](https://arxiv.org/abs/1904.02689), PDF, с. 3. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [YOLACT — Bolya et al., ICCV 2019](https://arxiv.org/abs/1904.02689)

[Выбрать следующую ветку](<README.md>) · [Сценарии подраздела](<04-segmentation/План.md>)
