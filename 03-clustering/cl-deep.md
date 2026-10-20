# Совместное обучение: DEC и IDEC

**Уровень:** углубление.

**Предпосылки:** [Кластеризация замороженных представлений](<cl-frozen.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как self-training изменяет представление и когда усиливает собственные ошибки?




## Когда меняются и признаки, и группы

Deep clustering совместно обновляет представление и структуру группировки. DEC превращает текущие мягкие назначения в усиленное target distribution и минимизирует KL; IDEC добавляет reconstruction для сохранения информации. Self-generated targets не являются независимой разметкой.

Для одного объекта q=(0.6,0.4) и одинаковых нормирующих частот компонент усиление квадратом даёт p=(0.36,0.16)/0.52≈(0.6923,0.3077). Цель стала увереннее, хотя внешнего доказательства правильности не добавилось. Это помогает понять одновременно механизм уточнения и риск закрепления ошибки. Полный DEC учитывает частоты всех объектов; локальный пример намеренно фиксирует их равными.

Сначала сравните frozen baseline, затем совместное обучение при одинаковом protocol. Улучшение внутреннего clustering objective само по себе не доказывает улучшение внешней задачи. Метрики и правила остановки нельзя выбирать после просмотра удержанной разметки.


## Карта вычисления и границы этапов

![Совместное обучение: DEC и IDEC: входы, преобразования и выходы](<../оформление/course/cl-deep.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:5e17c55e3f7d5d4eda -->
<a id="u-5e17c55e3f7d5d4eda"></a>

## DEC: последовательность и objective

<!-- /unit:5e17c55e3f7d5d4eda -->

<!-- unit:ed7c8cd5384a791c38 -->
<a id="u-ed7c8cd5384a791c38"></a>

[DEC](https://proceedings.mlr.press/v48/xieb16.html) сначала инициализирует encoder через autoencoder reconstruction и cluster centres в latent space. Основная clustering phase уточняет encoder и centres по KL к periodically recomputed auxiliary target; decoder/reconstruction term в неё не входит.

<!-- /unit:ed7c8cd5384a791c38 -->

<!-- unit:74dd505432c9488faa -->
<a id="u-74dd505432c9488faa"></a>

Student-t assignments для $z_i$ и centre $\mu_j$:

<!-- /unit:74dd505432c9488faa -->

<!-- unit:4f1893a7f71f4fd9b8 -->
<a id="u-4f1893a7f71f4fd9b8"></a>

$$
q_{ij}=\frac{(1+\|z_i-\mu_j\|^2/\nu)^{-(\nu+1)/2}}
{\sum_l(1+\|z_i-\mu_l\|^2/\nu)^{-(\nu+1)/2}}.
$$

<!-- /unit:4f1893a7f71f4fd9b8 -->

<!-- unit:eec102cb1986eabace -->
<a id="u-eec102cb1986eabace"></a>

Обозначим $f_j=\sum_iq_{ij}$. Target и loss:

<!-- /unit:eec102cb1986eabace -->

<!-- unit:9f4e562a1db2a376ed -->
<a id="u-9f4e562a1db2a376ed"></a>

$$
p_{ij}=\frac{q_{ij}^2/f_j}{\sum_lq_{il}^2/f_l},
\qquad
L_{\mathrm{cluster}}=\sum_{i,j}p_{ij}\log\frac{p_{ij}}{q_{ij}}.
$$

<!-- /unit:9f4e562a1db2a376ed -->

<!-- unit:1bb02a0377e446d194 -->
<a id="u-1bb02a0377e446d194"></a>

В update target фиксируется; затем пересчитывается по новому state. Sharpening усиливает confident assignments, normalization по frequency корректирует массовость clusters. Это self-training signal, не новые human labels. Без хорошей инициализации ошибочный partition тоже может стать увереннее.

<!-- /unit:1bb02a0377e446d194 -->

<!-- unit:b5ead5ef8ba6836ad9 -->
<a id="u-b5ead5ef8ba6836ad9"></a>

## Учебный расчёт и опасная интерпретация

<!-- /unit:b5ead5ef8ba6836ad9 -->

<!-- unit:71b86995b4dc2fe870 -->
<a id="u-71b86995b4dc2fe870"></a>

Для двух points с assignments $(0.8,0.2)$ и $(0.2,0.8)$ frequencies обоих clusters равны 1. Первый sharpened target:

<!-- /unit:71b86995b4dc2fe870 -->

<!-- unit:5cc74e3641361cf5e8 -->
<a id="u-5cc74e3641361cf5e8"></a>

$$
p_1=\left(\frac{0.64}{0.68},\frac{0.04}{0.68}\right)
=\left(\frac{16}{17},\frac1{17}\right)
\approx(0.9412,0.0588).
$$

<!-- /unit:5cc74e3641361cf5e8 -->

<!-- unit:14df28fc7ef3766ba8 -->
<a id="u-14df28fc7ef3766ba8"></a>

Target стал увереннее без добавления independent evidence. Если initial grouping отражал background, objective может усилить background separation. Поэтому decrease KL не равен доказанному улучшению semantic clustering.

<!-- /unit:14df28fc7ef3766ba8 -->

<!-- unit:026d5f025abbd09b08 -->
<a id="u-026d5f025abbd09b08"></a>

Точно так же collapse всех outputs/центров или очень неравные cluster masses требуют диагностики. Нужны size distribution, stability, feature variance и representative examples. Изменение assignments может остановиться потому, что система застряла, а не нашла истинную taxonomy.

<!-- /unit:026d5f025abbd09b08 -->

<!-- unit:0b204a2e32eb18bf1f -->
<a id="u-0b204a2e32eb18bf1f"></a>

## IDEC и сохранение структуры

<!-- /unit:0b204a2e32eb18bf1f -->

<!-- unit:c90987532e47298ff5 -->
<a id="u-c90987532e47298ff5"></a>

[IDEC](https://www.ijcai.org/Proceedings/2017/243) сохраняет reconstruction objective во время clustering refinement. В одной scale convention:

<!-- /unit:c90987532e47298ff5 -->

<!-- unit:d8f266e79c4ceb43ad -->
<a id="u-d8f266e79c4ceb43ad"></a>

$$
L_{\mathrm{IDEC}}=L_{\mathrm{recon}}+\gamma L_{\mathrm{cluster}},
\qquad
L_{\mathrm{recon}}=\frac1n\sum_i\|x_i-g_\phi(f_\theta(x_i))\|^2.
$$

<!-- /unit:d8f266e79c4ceb43ad -->

<!-- unit:571baa68fb9fe1d36a -->
<a id="u-571baa68fb9fe1d36a"></a>

Reconstruction удерживает информацию об исходном input и ограничивает distortion ради cluster objective. Она не гарантирует semantic correctness: background/noise также могут хорошо реконструироваться. Balance зависит от units, dimensionality и normalization losses. Большое число pixels может сделать unnormalized reconstruction dominant.

<!-- /unit:571baa68fb9fe1d36a -->

<!-- unit:b73fd4dc810f349b40 -->
<a id="u-b73fd4dc810f349b40"></a>

Например, две latent mappings дают одинаковый cluster loss, но одна теряет существенный параметр объекта. Reconstruction может различить их, если decoder/data objective действительно требуют этот параметр. Если target semantics никак не отражена в доступных data/prior, дополнительный loss не создаёт её автоматически.

<!-- /unit:b73fd4dc810f349b40 -->

<!-- unit:cb5d443166e60aa638 -->
<a id="u-cb5d443166e60aa638"></a>

## Контроль и практика

<!-- /unit:cb5d443166e60aa638 -->

<!-- unit:7e739379c0a8cc5c15 -->
<a id="u-7e739379c0a8cc5c15"></a>

1. Есть ли reconstruction loss в основной фазе исходного DEC? **Ответ:** нет; сохранение этой части — ключевое отличие IDEC.
2. Создаёт ли sharpening новую truth? **Ответ:** нет; оно преобразует собственные assignments и может усилить ошибку.
3. Равен ли cluster confidence semantic class probability? **Ответ:** нет; это assignment в выбранной model/representation.

<!-- /unit:7e739379c0a8cc5c15 -->

<!-- unit:049f08d5ab6539affb -->
<a id="u-049f08d5ab6539affb"></a>

**Практика:** вычислите $p$ для двух symmetric assignments и затем для сильно неравных frequencies. Опишите collapsed solution, которое нужно обнаружить. Составьте two-stage ablation и отдельный DEC/IDEC experiment без simultaneous change всех components.

<!-- /unit:049f08d5ab6539affb -->

<!-- unit:7fab3def64b7de6b3f -->
<a id="u-7fab3def64b7de6b3f"></a>

**Схема курса:** frozen encoder features or trainable encoder → initial clusters → Student-t assignments and target distribution → KL/reconstruction update → inspect drift/collapse against frozen baseline

<!-- /unit:7fab3def64b7de6b3f -->

## Самостоятельная практика

**Практика:** Вычислите target p для q=(0.6,0.4) при равных частотах компонент. Затем измените частоты и проследите нормировку полного DEC target. Объясните, почему более уверенное p не является новой истинной разметкой.

## Результат главы и переход дальше

Ответьте на вопрос: **Как self-training изменяет представление и когда усиливает собственные ошибки?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<README.md>) · [Происхождение и точные места переноса](<../editorial/coverage.md>)
