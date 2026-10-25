# CRAFT и DBNet: поиск текста на изображении

**Уровень:** основа ветки.

**Предпосылки:** [Detection: множества объектов, IoU, AP и архитектуры](<cv-detection.md>), [Semantic segmentation: FCN, U-Net и DeepLab](<cv-semantic.md>)

**Используют:** [Document vision: layout, reading order и таблицы](<cv-documents.md>)

**Учебный вопрос:** Как обнаруживать текст разной ориентации и формы?

## Схема вычислений

![CRAFT и DBNet: поиск текста на изображении: вычислительные этапы](<../оформление/course/expanded-cv-text-detection.svg>)

CRAFT: Image → encoder / decoder → Character region scores → Affinity scores → grouping → Components → text polygons. DBNet: Image → multiscale features → Probability P + threshold T → Soft binarization при training → Contours → unclip polygons.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Text detection до распознавания

Detector текста возвращает области, из которых recognizer читает символы. Для повёрнутых строк нужны quadrilaterals, для изогнутых — polygons или иная форма. Axis-aligned box может включить соседнюю строку и испортить OCR даже при неплохом IoU. Сначала задают единицу разметки: character, word или line. Нельзя сравнивать predictions слов с GT строк без правил группировки.

## CRAFT: character regions и affinity

[CRAFT](https://arxiv.org/abs/1904.01941) использует encoder–decoder для двух карт: вероятностных scores областей characters и affinities между соседними characters. Regions помогают локализовать буквы, affinities — объединить их в текстовые последовательности. Gaussian-like targets строятся по character geometry. При наличии только word-level labels метод использует weak supervision и оценённые character targets; качество этих pseudo-labels имеет значение.

На inference карты threshold и объединение components дают word regions. Это не полноценное распознавание: карта не возвращает текст «Москва». Неудачная affinity может соединить соседние слова, а разрыв слабых character regions — разделить одну строку.

## DBNet: differentiable binarization

[DBNet](https://arxiv.org/abs/1911.08947) предсказывает text probability map $P$ и adaptive threshold map $T$. Приближённая binary map $\hat B=1/(1+\exp[-k(P-T)])$ позволяет передавать gradient через soft binarization. Коэффициент $k$ задаёт крутизну перехода. Supervision включает shrunk text regions и threshold boundaries; shrink отделяет соседние instances.

На inference возможно использовать probability branch с фиксированной процедурой contour extraction и polygon expansion. Differentiable training module не означает, что весь polygon post-processing стал differentiable или идентичен train step. Shrink/unclip distances должны соответствовать масштабу и форме polygons.

## Разобранный пример: soft threshold и resize

Для $P=0.6$, $T=0.5$, $k=10$ получаем $\hat B=\sigma(1)\approx0.73106$. При том же $P$ и $T=0.7$ результат $\sigma(-1)=0.26894$. Если $P=T$, ответ 0.5. Поэтому вероятность text и приблизительная binary map — разные выходы.

Исходный кадр $1200\times800$ уменьшен до $600\times400$. Polygon prediction имеет углы $(50,30),(150,20),(160,50),(60,60)$. При простом равномерном resize обратные координаты умножаются на 2. Если затем добавлялся padding, сначала его вычитают. Perspective rectification crop требует четырёхугольника в согласованном порядке; перестановка углов может перевернуть строку.

## Оценка и ошибки

Detection precision/recall измеряют с polygon matching и ignore regions. End-to-end OCR дополнительно требует верного текста. Отдельно исследуют маленькие буквы, blur, low contrast, vertical text и несколько языков. Текстуры решёток и кирпичей могут давать ложные character regions. Увеличение resolution помогает мелкому тексту, но цена растёт с площадью карты.

## Самопроверка с ответами

**Практика:** удалите affinity branch из CRAFT. Наличие character regions ещё не задаёт, какие буквы принадлежат одному слову.

**Почему DBNet обучается на shrunk polygons?** Разделённые внутренние regions упрощают выделение соседних text instances; расширение выполняется отдельным шагом.

**Достаточно ли detector recall для оценки OCR?** Нет, неверная rectification или обрезанные characters могут испортить recognition даже у найденной строки.

## Иллюстрации из первоисточников

![craft: Figure 2 из авторской статьи](<../оформление/course/papers/craft.png>)

**Источник:** [craft, Figure 2](https://arxiv.org/abs/1904.01941), PDF, с. 2. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

![dbnet: Figure 3 из авторской статьи](<../оформление/course/papers/dbnet.png>)

**Источник:** [dbnet, Figure 3](https://arxiv.org/abs/1911.08947), PDF, с. 3. Проследите вход, преобразования и выход по стрелкам. Сопоставьте обозначения статьи с формами и targets, определёнными в тексте главы.

## Источники и продолжение

- [CRAFT — Baek et al., CVPR 2019](https://arxiv.org/abs/1904.01941)
- [DBNet — Liao et al., AAAI 2020](https://arxiv.org/abs/1911.08947)

[Следующие углубления: cv-documents](<README.md>) · [Сценарии подраздела](<07-specialized/План.md>)
