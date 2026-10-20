# Категориальные признаки: подробные учебные материалы

[Версия для чтения с набранными формулами](<../../Читать конспекты.html#course-3-0>) · работает без интернета.

[Объединённый маршрут данных и признаков](<../../План — данные и признаки.md>) · [Исходный план категорий](<../План — категориальные признаки.md>) · [Общее оглавление](<../../Учебные конспекты.md>)

Девять глав соответствуют девяти пунктам. Это дополнение по плану: отдельного полного чата по категориям нет. Lookup/backprop embeddings связан с основами LLM, но табличные protocols, cross-fitting и native boosting здесь раскрыты самостоятельно. Все формулы написаны в LaTeX.

1. [А — Базовые encoding](<А — Nominal, ordinal, one-hot и dummy encoding.md>): ложный порядок, geometry, dummy trap, intercept, unknown и mapping. Пункт 1.
2. [Б — Высокая cardinality](<Б — Кардинальность, frequency и feature hashing.md>): sparse memory, count/frequency, signed hashing, collisions и границы multi-hash/Bloom аналогии. Пункт 2.
3. [В — Target statistics](<В — Target encoding, cross-fitting, smoothing и WoE.md>): десять строк naive/OOF/smoothed, nested validation, LOO/ordered и WoE. Пункт 3.
4. [Г — Embeddings](<Г — Embeddings, gradients и перенос между задачами.md>): categorical lookup/gradient, geometry, rare/unknown, extraction/transfer и общий протокол learned representations для текста/изображений. Пункт 4 и дополнение F8.
5. [Д — Native boosting](<Д — Нативные категории в LightGBM, CatBoost и XGBoost.md>): partitions, statistics, dtype/mapping, serialization и version-specific edge cases. Пункт 5.
6. [Е — Interactions](<Е — Feature crosses, FM и field-aware взаимодействия.md>): additive limits/XOR, manual crosses, FM identity, FFM и generalization на новые pairs. Пункт 6.
7. [Ж — Multi-valued fields](<Ж — Multi-hot, counts и pooling наборов категорий.md>): set/multiset/sequence, pooling/gradients, empty/PAD и weights. Пункт 7.
8. [З — Missing/unknown/rare](<З — Missing, unknown, rare и стабильный словарь.md>): explicit bucket, mode+indicator, vocabulary contract, round-trip и monitoring. Пункт 8.
9. [И — Время и feedback](<И — Временные статистики и доступность метки.md>): event/availability times, rolling/expanding, delayed labels, online protocol и drift. Пункт 9.

## Порядок и практика исходного плана

Предпосылка — [split и leakage](../../01-data/Учебные%20конспекты/README.md). Ядро А–В, Д, З–И; embeddings/crosses/multi-valued развивают соответствующие задачи. Каждая глава содержит объяснения, разобранный пример, ограничения, вопросы с ответами и practice.

Шесть практических пунктов плана покрыты: one-hot/unknown/missing — А/З; cardinality/memory/hash — Б; десять строк target encoding — В; вложенная валидация — В/И; embeddings/crosses — Г/Е; native boosting и serialization — Д. Итоговая работа собирает один воспроизводимый pipeline и сравнивает representations на одинаковом split, с отдельными rare/unknown slices.

Наличие материала не подтверждает выполненные вами experiments. API-утверждения в Д сверены с official docs; конкретный запуск фиксирует installed library/device/version. Ни один encoding не объявлен universally лучшим.

Дополнения объединённого маршрута данных и признаков размещены рядом с подходящей механикой: числовые transforms и сохранение параметров — в главе Б плана данных §2.4; отбор/AutoFE внутри nested validation — там же в §4.3; общее применение embeddings к тексту и изображениям — в главе Г, §12, и локально определено в первой главе кластеризации.

Все девять глав углублены: А–Б раскрывают геометрию encoding, идентифицируемость, unknown, sparse memory и signed hashing; В–Г — smoothing, OOF/nested validation, WoE и обучение/перенос embeddings; Д–Е — split gain, поиск partitions, native metadata/CTR и FM/FFM; Ж–И — pooling множественных значений, устойчивый словарь, point-in-time статистики и delayed labels. Объединённые дополнения F2/F8/F9 и их точные места перечислены выше и в редакционной карте. Локальные переходы проверены; содержание практических заданий не означает, что читатель уже выполнил их.
