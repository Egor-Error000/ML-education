# Данные и достоверность ML-эксперимента: учебные материалы

[Версия для чтения с набранными формулами](<../../Читать конспекты.html#course-1-0>) · работает без интернета.

[Объединённый маршрут данных и признаков](<../../План — данные и признаки.md>) · [Исходный план данных](<../План — данные и качество ML-эксперимента.md>) · [Общее оглавление](<../../Учебные конспекты.md>)

Девять глав соответствуют девяти пунктам плана. Глава А дополнительно углублена: prediction/group/source units, point-in-time availability, delayed labels, risk/metric и паспорт snapshot. Б–В дополнены split worksheets, pipeline/OOF/nested/time semantics и подробным missingness/imputation/outlier/annotation audit. Г–Д дополнены выводами weighted loss, метриками и калибровкой, геометрией аугментаций, происхождением синтетики и механизмами MinHash/LSH. Е–Ж дополнены механизмами loader/collate, loss normalization, sharding и restart; координатными контрактами, letterbox, RLE, pose flags и геометрией OBB. З–И дополнены Unicode и механизмами tokenizer/template/shift/packing/chunking; drift, importance weighting, Wilson, парным сравнением и group/block bootstrap. Все девять глав получили углубление; окончательная содержательная сверка ещё необходима. Отдельного полного чата по данным среди предоставленных источников нет: это дополнительный учебный курс по плану. Связанные фрагменты LLM/CV указаны в начале прикладных глав; исходные чаты сохранены и не переписывались.

1. [А — Постановка и паспорт](<А — Постановка задачи и паспорт данных.md>): единица обобщения, доступность features, target delay, виды задач, baseline/slices и TabPFN-3.5 как модельный reference с benchmark limits. Пункт 1.
2. [Б — Split и leakage](<Б — Split, cross-validation и утечки.md>): random/stratified/group/time, gap, train-only fit, числовые transforms, selection/AutoFE внутри CV и resampling. Пункт 2 и дополнения F2/F9.
3. [В — Пропуски и labels](<В — Пропуски, выбросы и качество разметки.md>): MCAR/MAR/MNAR, sentinel, imputation, rare cases, disagreement и аудит. Пункт 3.
4. [Г — Дисбаланс и probabilities](<Г — Дисбаланс, метрики, порог и калибровка.md>): PR/ROC/F1, weights/SMOTE, cost threshold, Brier/log loss, calibration и prevalence. Пункт 4.
5. [Д — Augmentation и dedup](<Д — Аугментации, синтетика и дедупликация.md>): инвариантности, synthetic provenance, distillation distinctions, hashes/MinHash/LSH и duplicate groups. Пункт 5.
6. [Е — Загрузка и воспроизводимость](<Е — Dataset, batching, загрузка и воспроизводимость.md>): Dataset/collate, padding/bucketing, bottlenecks, workers/Windows, manifests и streaming restart. Пункт 6.
7. [Ж — CV data](<Ж — CV-данные, координаты и синхронные transforms.md>): formats/IDs, синхронная геометрия, masks/keypoints/OBB, crowd/empty cases, video split и shards. Пункт 7.
8. [З — Текст и LLM data](<З — Текстовые данные, токенизация и chunking.md>): Unicode/tokenizers, templates/masks/truncation, document split/chunks, provenance и task mix. Пункт 8.
9. [И — Drift и надёжность](<И — Drift, интервалы и достоверность вывода.md>): covariate/label/concept shift, uncertainty, group/block bootstrap, seeds и интерпретация сравнения. Пункт 9.

## Как работать с курсом

Начните с паспорта одного dataset и используйте его во всех главах. Материал А–Е и И — общая основа; Ж/З развивают соответствующий домен. У каждой главы есть механика, формулы LaTeX, числовые примеры, ошибки, контрольные вопросы с ответами и практика. Практические результаты нужно измерить отдельно: учебные расчёты не являются вашими benchmark scores.

Итоговый набор artifacts: паспорт/manifest, версии и hashes, split IDs, pipeline с train-only fit, аудит labels/leakage, baseline, одна интервенция и вывод с неопределённостью. Перед следующими курсами убедитесь, что улучшение не объясняется утечкой или сменой оцениваемой популяции.

Дальше: [категориальные признаки](<../../03-categorical-features/План — категориальные признаки.md>), [регуляризация](<../../02-regularization/План — регуляризация.md>), [CV](../../05-computer-vision/Учебные%20конспекты/README.md), [LLM/SFT](../../06-llm/02-adaptation/Учебные%20конспекты/README.md).
