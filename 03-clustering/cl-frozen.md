# Кластеризация замороженных представлений

**Уровень:** основа ветки.

**Предпосылки:** [Качество разбиения: внутренние и внешние метрики](<cl-metrics.md>), [Выбор числа кластеров и устойчивость](<cl-stability.md>)

**Используют:** [Совместное обучение: DEC и IDEC](<cl-deep.md>), [Кластеризация изображений и superpixels](<cl-images.md>), [Текстовые представления, LDA, NMF и Top2Vec](<cl-text.md>)

**Учебный вопрос:** Как разделить качество представления и качество алгоритма группировки?




## Разделить представление и группировку

Embedding — векторное представление объекта, построенное заданной функцией. Frozen encoder означает, что эта функция не обновляется при текущем clustering experiment. Encoder не обязан быть Transformer: это может быть PCA, CNN или табличная модель. Источник обучения и доступная информация всё равно входят в protocol.

Пусть векторы трёх объектов равны (1,0), (0.9,0.1), (0,1). Без normalization squared distances первого до остальных равны 0.02 и 2. После нормировки меняются абсолютные значения и смысл threshold. Измерьте не только clusters, но и соседей в исходном representation space. Визуализация может исказить эти отношения.

Зафиксируйте encoder/version, preprocessing, pooling, normalization, clustering params и seeds. Для нового объекта применяйте тот же encoder и объявленное out-of-sample правило. Если encoder адаптировался на test labels или test population вопреки заявленному inductive split, frozen на последнем шаге не отменяет эту утечку.


## Карта вычисления и границы этапов

![Кластеризация замороженных представлений: входы, преобразования и выходы](<../оформление/course/cl-frozen.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:58cdf320c88a49ac08 -->
<a id="u-58cdf320c88a49ac08"></a>

## Двухэтапный baseline

<!-- /unit:58cdf320c88a49ac08 -->

<!-- unit:6c6097b64db26667ee -->
<a id="u-6c6097b64db26667ee"></a>

Сначала frozen encoder $z_i=f_\theta(x_i)$, затем standard clustering по $z_i$. Это позволяет отдельно диагностировать две части. Если categories не разделимы в representation, смена одного clustering algorithm может не помочь. Если geometry уже хороша, fine-tuning encoder может быть лишней сложностью.

<!-- /unit:6c6097b64db26667ee -->

<!-- unit:9d6b1d9bc1455c46ad -->
<a id="u-9d6b1d9bc1455c46ad"></a>

Сохраняются encoder/tokenizer/preprocessing version, pooling, vector dimensions и normalization. Penultimate image features, CLS token и mean pooled sentence embedding не являются взаимозаменяемыми. Encoder обучался конкретному objective и может группировать nuisance variables вместо целевой semantics.

<!-- /unit:9d6b1d9bc1455c46ad -->

<!-- unit:f58ea4a2f74a70d722 -->
<a id="u-f58ea4a2f74a70d722"></a>

Правильная ablation: fixed algorithm на raw/L2-normalized/PCA embeddings; затем fixed representation с разными algorithms. Одновременная смена encoder, dimension и K не позволяет понять причину improvement. При external labels tuning делается на validation, final test закрыт.

<!-- /unit:f58ea4a2f74a70d722 -->

<!-- unit:a34292204011fc84c1 -->
<a id="u-a34292204011fc84c1"></a>

## Валидация и новые objects

<!-- /unit:a34292204011fc84c1 -->

<!-- unit:66697d1c46a4a12c53 -->
<a id="u-66697d1c46a4a12c53"></a>

При trainable encoder все его fit data принадлежат training protocol. Fit на full corpus допустим для declared transductive study; это не независимый prospective test на ранее unseen documents/images. Предобучение на benchmark test images без labels тоже надо учитывать в protocol.

<!-- /unit:66697d1c46a4a12c53 -->

<!-- unit:8a32c86a659c812de4 -->
<a id="u-8a32c86a659c812de4"></a>

Для новых objects encoder плюс centres дают assignment rule, но probabilities $q$ — не автоматически calibrated class probabilities. Unknown class может быть уверенно притянута к одному известному centre. Нужны distance/novelty policy и evaluation новых regimes, если это deployment requirement.

<!-- /unit:8a32c86a659c812de4 -->

<!-- unit:44554fcefa0a505ed8 -->
<a id="u-44554fcefa0a505ed8"></a>

Сравниваем frozen baseline, DEC и IDEC по same data, K, budget и validation rules; report includes external metrics при наличии labels, internal geometry, stability, coverage, runtime и visual examples. Увеличение compute только одному методу должно быть явно отделено от quality algorithm.

<!-- /unit:44554fcefa0a505ed8 -->

## Самостоятельная практика

**Практика:** Зафиксируйте три вектора из примера и сравните distance до/после unit normalization. Составьте manifest encoder, preprocessing, pooling и clustering. Для нового объекта отдельно опишите transform и assignment; не меняйте encoder после просмотра удержанных labels.

## Результат главы и переход дальше

Ответьте на вопрос: **Как разделить качество представления и качество алгоритма группировки?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Совместное обучение: DEC и IDEC](<cl-deep.md>), [Кластеризация изображений и superpixels](<cl-images.md>), [Текстовые представления, LDA, NMF и Top2Vec](<cl-text.md>).

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
