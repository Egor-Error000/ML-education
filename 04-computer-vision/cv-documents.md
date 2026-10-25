# Document vision: layout, reading order и таблицы

**Уровень:** основа ветки.

**Предпосылки:** [CRAFT и DBNet: поиск текста на изображении](<cv-text-detection.md>), [CRNN, CTC и TrOCR: распознавание текста](<cv-text-recognition.md>)

**Используют:** [Сквозные разборы: дефекты, видеопоток и документы](<cv-cases.md>)

**Учебный вопрос:** Как связать геометрию страницы, OCR и структуру документа?

## Схема вычислений

![Document vision: layout, reading order и таблицы: вычислительные этапы](<../оформление/course/expanded-cv-documents.svg>)

Document pipeline: Page image / PDF + page ID → Layout regions + OCR strings → Order graph / table structure → Fields + evidence coordinates. Table understanding: Table region → Rows / columns / spans → Cell assignment + OCR → Structured table → evaluation.

Обучаемые преобразования и фиксированные операции разделены в последующем объяснении; стрелки показывают передачу данных, а направление градиента обсуждается вместе с loss.

## Страница как структурированный объект

OCR возвращает text regions и strings, но документ содержит заголовки, paragraphs, figures, tables, reading order и отношения. Pipeline должен сохранять page index, координаты regions, исходные crops и confidence. Plain concatenation строк теряет колонки и связи key–value. Baseline — layout detection, OCR и геометрические правила, проверенные на реальных типах страниц.

## Layout и reading order

Layout detector классифицирует regions как text, heading, table, figure и другие категории. Они могут вкладываться друг в друга: table содержит cells и text, а не конкурирует с каждой буквой как единственный класс. Reading order — упорядочение blocks с учётом колонок и структуры. Простая сортировка по y, затем x смешивает строки двух колонок. Возможны graph representation с edges «читать после» и constrained ordering.

Multimodal document encoder соединяет text tokens, 2D boxes и image features; [LayoutLMv3](https://arxiv.org/abs/2204.08387) — пример общего pretraining для text/image представлений. Координаты boxes часто нормируются, но конкретный диапазон — контракт модели. Нормировка не должна уничтожать page identity или смешивать разные dimensions.

## Таблицы и поля

Table detection находит region; structure recognition восстанавливает rows, columns, spans и adjacency; OCR заполняет cells. Таблица без видимых границ требует cues alignment и semantic consistency. [PubTables-1M](https://arxiv.org/abs/2110.00061) показывает важность согласованной разметки structure; box IoU всей таблицы недостаточно для оценки её восстановления.

Extraction полей связывает key и value, проверяет типы, единицы и происхождение. Для суммы важно различать «итого», налог и subtotal. Rules могут проверять arithmetic, но совпавшая сумма ещё не доказывает верный assignment полей. Неуверенные critical fields отправляют на review по выбранной политике.

## Разобранный пример: две колонки и сквозная ошибка

Страница шириной 1000 pixels имеет левую колонку с boxes A $(50,100,450,140)$ и B $(50,160,450,200)$, правую — C $(550,110,950,150)$ и D $(550,170,950,210)$. Сортировка только по верхнему y даёт A,C,B,D. Если документ читается колонками, верный порядок A,B,C,D. Сначала определяют колонки, затем порядок внутри каждой.

Пусть из 100 GT fields detector/crop stage сохраняет 90; recognizer верно читает 81 из этих 90; relation assignment правильно связывает 72 из 81. Conditional accuracies: 0.9,0.9, $72/81\approx0.88889$. End-to-end exact match $72/100=0.72$. Усреднение трёх процентов дало бы завышенное представление о результате; ошибки стадий последовательно ограничивают pipeline.

## Диагностика и split

Split по шаблонам и организациям показывает перенос лучше случайного split страниц одной формы. Scanned и digital PDF различают: text layer иногда доступен, но может быть ошибочным. Отдельно оценивают layout, CER, order, table structure и field exact match. Rasterization DPI меняет размер мелких characters и стоимость обработки.

## Самопроверка с ответами

**Практика:** поменяйте смысл задачи на чтение строк поперёк обеих колонок. Тогда прежний порядок A,B,C,D неверен: reading order зависит от документа, а не единой геометрической формулы.

**Почему высокая OCR accuracy не гарантирует правильный счёт?** Верные strings могут быть присвоены неверным полям.

**Нужно ли хранить source boxes после extraction?** Да, для проверки происхождения ответа и локализации ошибок.

## Источники и продолжение

- [LayoutLMv3 — Huang et al., ACM MM 2022](https://arxiv.org/abs/2204.08387)
- [PubTables-1M — Smock et al., CVPR 2022](https://arxiv.org/abs/2110.00061)

[Следующие углубления: cv-cases](<README.md>) · [Сценарии подраздела](<07-specialized/План.md>)
