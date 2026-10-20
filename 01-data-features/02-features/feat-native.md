# Нативные категории в бустинге

**Уровень:** общее ядро.

**Предпосылки:** [Nominal, ordinal, one-hot и dummy encoding](<feat-categorical.md>), [Target statistics, smoothing, cross-fitting и WoE](<feat-target.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как модель выбирает разбиение категорий и что входит в сохранённый контракт?





## Карта вычисления и границы этапов

![Нативные категории в бустинге: входы, преобразования и выходы](<../../оформление/course/feat-native.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Обучаемые статистики вычисляются на train; validation и новые объекты используют сохранённое состояние. Геометрическая формула с фиксированными константами не превращается в fit только из-за наличия блока на схеме.

<!-- unit:d1e696282a906a369d -->
<a id="u-d1e696282a906a369d"></a>

## Loss и categorical split

<!-- /unit:d1e696282a906a369d -->

<!-- unit:ad90fc58f0f6bf4543 -->
<a id="u-ad90fc58f0f6bf4543"></a>

Numeric split имеет вид $x<t$. Categorical split может иметь вид $c\in S$, где $S$ — subset categories. Пусть A/B/C/D имеют эффекты $(1,-1,1,-1)$, а произвольные codes — 1/2/3/4. Один numeric threshold не отделит A/C от B/D, но subset split сделает это сразу. Поэтому передать codes как обычное число и объявить field categorical — разные операции.

<!-- /unit:ad90fc58f0f6bf4543 -->

<!-- unit:e8e5fc5ee21ad5c3d3 -->
<a id="u-e8e5fc5ee21ad5c3d3"></a>

Полный перебор бинарных partitions $K$ categories требует $2^{K-1}-1$ вариантов без учёта симметрии сторон. Для scalar leaf objectives можно использовать ordering categories по статистике leaf response и проверять contiguous partitions этого порядка. Это порядок, выученный в текущем node по objective, а не порядок исходных IDs.

<!-- /unit:e8e5fc5ee21ad5c3d3 -->

<!-- unit:68e74c146f863c98da -->
<a id="u-68e74c146f863c98da"></a>

При second-order boosting node агрегирует gradients $G$ и Hessians $H$. Типичный scalar leaf weight имеет форму $-G/(H+\lambda)$. При подходящих assumptions categories с близкими optimal responses можно группировать. Для vector outputs универсального scalar ordering нет; алгоритм/approximation зависит от implementation. Нельзя переносить proof scalar optimal partitioning на все objectives без оговорок.

<!-- /unit:68e74c146f863c98da -->

<!-- unit:721ca702592413f80e -->
<a id="u-721ca702592413f80e"></a>

### Дерево исправляет текущий predictor

<!-- /unit:721ca702592413f80e -->

<!-- unit:c16fa6bde4f47b2b52 -->
<a id="u-c16fa6bde4f47b2b52"></a>

На очередном boosting step уже есть predictions $f_i$. Для differentiable loss вычисляются $g_i=\partial\ell(y_i,f_i)/\partial f_i$ и $h_i=\partial^2\ell(y_i,f_i)/\partial f_i^2$. В half-squared loss это $g_i=f_i-y_i$, $h_i=1$; в binary logistic loss по logit — $g_i=p_i-y_i$, $h_i=p_i(1-p_i)$. Поэтому histogram gradients не равна средним исходного target в общем случае: она зависит и от того, что предыдущие trees уже объяснили.

<!-- /unit:c16fa6bde4f47b2b52 -->

<!-- unit:450ec2f5cd69cf28f9 -->
<a id="u-450ec2f5cd69cf28f9"></a>

Если всем объектам leaf назначить scalar increment w, часть second-order objective, зависящая от w, имеет вид

<!-- /unit:450ec2f5cd69cf28f9 -->

<!-- unit:e57d5796e3f25aa499 -->
<a id="u-e57d5796e3f25aa499"></a>

$$
Q(w)=Gw+\frac12(H+\lambda)w^2,
\quad G=\sum_i g_i,\quad H=\sum_i h_i.
$$

<!-- /unit:e57d5796e3f25aa499 -->

<!-- unit:13a5aa8743ebfed5d7 -->
<a id="u-13a5aa8743ebfed5d7"></a>

При $H+\lambda>0$ derivative $G+(H+\lambda)w=0$ даёт $w^*=-G/(H+\lambda)$ и minimum $-G^2/[2(H+\lambda)]$. Здесь lambda — L2 penalty на weight leaf в принятом sum-loss convention. Mean loss, sample weights и другая нормировка objective меняют сопоставимый scale lambda. У nonsmooth objectives или особых Hessian approximations этот вывод нужно сверять с конкретным алгоритмом.

<!-- /unit:13a5aa8743ebfed5d7 -->

<!-- unit:ba455a5501ef9de73a -->
<a id="u-ba455a5501ef9de73a"></a>

Для split с children L/R выигрыш относительно одного parent равен

<!-- /unit:ba455a5501ef9de73a -->

<!-- unit:620b63e2e6ad61026f -->
<a id="u-620b63e2e6ad61026f"></a>

$$
\operatorname{Gain}=\frac12\left[
\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}
-\frac{G_P^2}{H_P+\lambda}\right]-\gamma.
$$

<!-- /unit:620b63e2e6ad61026f -->

<!-- unit:c6a4ff659caa581abd -->
<a id="u-c6a4ff659caa581abd"></a>

Gamma здесь — дополнительная стоимость одного нового leaf при переходе от одного к двум. Это учебная second-order схема, а не утверждение, что все библиотеки хранят одинаковый gain scale или используют одинаковые defaults. Learning rate обычно применяют к найденным leaf increments после построения tree; уменьшить increment и заново пересчитать exact loss — отдельный расчёт, который не обязан совпасть с unshrunk split gain.

<!-- /unit:c6a4ff659caa581abd -->

<!-- unit:80bb0d55991a8362e7 -->
<a id="u-80bb0d55991a8362e7"></a>

### Полный пример из четырёх категорий

<!-- /unit:80bb0d55991a8362e7 -->

<!-- unit:da6323ac280505a23d -->
<a id="u-da6323ac280505a23d"></a>

Пусть в текущем node у A/B/C/D по две строки, Hessian каждой строки 1. В каждой category одинаковые residual gradients: A −2, B 1, C −1, D 2. Category totals G равны (−4,2,−2,4), H — (2,2,2,2). Возьмём lambda=1, gamma=0. Parent G=0,H=8, поэтому parent weight 0.

<!-- /unit:da6323ac280505a23d -->

<!-- unit:88ba9d9d2a36cc9506 -->
<a id="u-88ba9d9d2a36cc9506"></a>

При numeric codes A=1,B=2,C=3,D=4 возможны только три threshold cuts. Для A|BCD gain равен $(16/3+16/7)/2=80/21\approx3.8095$. Для AB|CD — $(4/5+4/5)/2=0.8$. Для ABC|D снова 80/21. Никакой threshold не создаёт AC|BD.

<!-- /unit:88ba9d9d2a36cc9506 -->

<!-- unit:d5b7a3a3db0a2984c0 -->
<a id="u-d5b7a3a3db0a2984c0"></a>

Categorical subset AC имеет G=−6,H=4, дополнение BD — G=6,H=4. Получаются weights 1.2 и −1.2, gain $(36/5+36/5)/2=7.2$. При gamma=0.5 gain уменьшится до 6.7. При learning rate 0.1 добавки к predictor станут +0.12 и −0.12; это не обещание decrease exact loss ровно на 7.2 после shrinkage.

<!-- /unit:d5b7a3a3db0a2984c0 -->

<!-- unit:dd166241125c00304f -->
<a id="u-dd166241125c00304f"></a>

В half-squared setting можно проверить пример без Taylor approximation: положим начальные predictions 0 и targets (2,2,−1,−1,1,1,−2,−2). Начальный half-squared loss 10. После leaf increments ±1.2 data loss 1.36, penalty двух leaf weights 1.44, total 2.8. Разница 7.2 совпадает с derived gain. Это точный учебный objective, не измерение качества LightGBM или XGBoost.

<!-- /unit:dd166241125c00304f -->

<!-- unit:50c2d72deca8bde43e -->
<a id="u-50c2d72deca8bde43e"></a>

One-category-versus-rest split способен выбрать A или D с gain 80/21, но не объединить сразу AC в одном split. Следующие уровни tree могут восстановить нужную функцию, расходуя другую structure/capacity. Поэтому преимущество subset в этом примере относится к одному split и не означает невозможность научиться с one-hot вообще.

<!-- /unit:50c2d72deca8bde43e -->

<!-- unit:4f1b69165f469df874 -->
<a id="u-4f1b69165f469df874"></a>

## Поиск partitions

<!-- /unit:4f1b69165f469df874 -->

<!-- unit:2348cb08912cb2de35 -->
<a id="u-2348cb08912cb2de35"></a>

Для K=4 существует семь unordered nontrivial binary partitions, для K=20 — 524 287. Число $2^{K-1}-1$ получается из всех subsets после исключения пустого/полного и отождествления S с дополнением. Category IDs сами по себе не дают удобного порядка для такого поиска.

<!-- /unit:2348cb08912cb2de35 -->

<!-- unit:82b285c1f95fa51af3 -->
<a id="u-82b285c1f95fa51af3"></a>

Для подходящих scalar objectives категории можно упорядочить по response statistic, затем проверить K−1 contiguous cuts. В нашем примере individual regularized leaf responses равны A=4/3,B=−2/3,C=2/3,D=−4/3. Порядок D,B,C,A приводит к нужному cut DB|CA в середине. Это другой порядок, чем A,B,C,D по исходным кодам.

<!-- /unit:82b285c1f95fa51af3 -->

<!-- unit:efef50da0ddde7e3f2 -->
<a id="u-efef50da0ddde7e3f2"></a>

Наивный algorithm сначала агрегирует category histogram по строкам node, сортирует K statistics, затем проходит prefix sums G/H. После сортировки каждую новую границу можно оценить за постоянное время: перенос одной category изменяет только соответствующие totals. Типичная стоимость такого учебного прохода — O(number of node rows)+O(K log K)+O(K), без exp перебора. Histogram reuse, binning, параллелизм и фактические candidate limits изменяют runtime implementation.

<!-- /unit:efef50da0ddde7e3f2 -->

<!-- unit:06a9f8678a4e206469 -->
<a id="u-06a9f8678a4e206469"></a>

Слово «optimal» относится к определённой задаче, допустимым partitions и предположениям theorem. L1/L2 terms, minimum child constraints, category smoothing, ограниченный список candidates и vector outputs нельзя автоматически включить в доказательство без проверки. Сортировка по raw mean target на всём dataset — также не универсальная замена node-specific gradient ordering.

<!-- /unit:06a9f8678a4e206469 -->

<!-- unit:7ff7073d7abc5e3f6d -->
<a id="u-7ff7073d7abc5e3f6d"></a>

Для vector leaf у category есть вектор response, а не одно число. Оси нельзя упорядочить одновременно одним scalar без дополнительных предположений. Например, vectors (1,2) и (1,−2) имеют одинаковую projection на parent direction (1,0), хотя сильно различаются по второй координате. В актуальном описании [XGBoost categorical partitioning](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html) для vector leaves используется projection на parent Newton direction; это approximation, способная пропустить ортогональные contrasts. Поведение нулевого/слабого parent direction проверяют в конкретной версии, не придумывая универсальный порядок vectors.

<!-- /unit:7ff7073d7abc5e3f6d -->

<!-- unit:267c7dd40ee3568149 -->
<a id="u-267c7dd40ee3568149"></a>

## LightGBM, CatBoost и XGBoost

<!-- /unit:267c7dd40ee3568149 -->

<!-- unit:65b72d1953b67c0ba1 -->
<a id="u-65b72d1953b67c0ba1"></a>

LightGBM поддерживает categorical fields и partitioning. В Python dataframe metadata помогает согласовать categories; при raw integer arrays нужно самостоятельно сохранить codes. Отрицательные category codes рассматриваются как missing, поэтому применять generic unknown=-1 означает выбрать именно missing route, а не выучить новый category effect.

<!-- /unit:65b72d1953b67c0ba1 -->

<!-- unit:596b71f7851d809f0e -->
<a id="u-596b71f7851d809f0e"></a>

Сглаживание и минимальные размеры category groups ограничивают overfit; это hyperparameters, которые подбираются внутри validation. При high cardinality стоит сравнить native mode с manual representations, а не предполагать universal winner. [Официальные categorical/missing conventions](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html).

<!-- /unit:596b71f7851d809f0e -->

<!-- unit:35e2a9da587284a6f6 -->
<a id="u-35e2a9da587284a6f6"></a>

Пример ошибки contract: train mapping A=0,B=1,C=2, inference самостоятельно строит A=0,C=1. Теперь C будет прочитан как B, если metadata не восстановлена и передаются только codes. Совпадение dtype/shape не обнаружит semantic corruption.

<!-- /unit:35e2a9da587284a6f6 -->

<!-- unit:62526842eef602f5b3 -->
<a id="u-62526842eef602f5b3"></a>

CatBoost использует разные numerical statistics категорий и их combinations, а небольшие vocabularies может обрабатывать one-hot в зависимости от mode/config. Ordered target statistics опираются на precedents permutation; это не просто full-data mean. Ordered boosting отдельно связан с уменьшением prediction shift в самом gradient boosting. [Описание алгоритма](https://arxiv.org/abs/1706.09516).

<!-- /unit:62526842eef602f5b3 -->

<!-- unit:aaed573e483b94ea97 -->
<a id="u-aaed573e483b94ea97"></a>

Не нужно предварительно one-hot кодировать все fields, если цель — использовать native mechanism CatBoost: так он потеряет исходные category identities/комбинации. Threshold one-hot и доступные statistics зависят от loss, CPU/GPU и config. [Документация категориальных features](https://catboost.ai/docs/en/features/categorical-features).

<!-- /unit:aaed573e483b94ea97 -->

<!-- unit:5c08c46b198b618cac -->
<a id="u-5c08c46b198b618cac"></a>

Missing marker в string categorical field канонизируется осознанно: строки `"1"` и `"1.0"` не обязаны быть одним значением. Формат train/inference должен совпадать; null handling нельзя механически переносить с numeric features. Автоматические combinations тоже ограничиваются настройками и стоимостью, а не перебирают все возможные interactions без ограничений.

<!-- /unit:5c08c46b198b618cac -->

<!-- unit:9807f1d76fe7e09a28 -->
<a id="u-9807f1d76fe7e09a28"></a>

XGBoost поддерживает one-hot-подобные categorical splits и partitioning; выбор зависит, в частности, от threshold `max_cat_to_onehot`. Передаются categorical dtype/feature types и включается соответствующая поддержка. Для сохранения model используются formats, сохраняющие categorical metadata, например JSON/UBJSON.

<!-- /unit:9807f1d76fe7e09a28 -->

<!-- unit:365650bb6197952da5 -->
<a id="u-365650bb6197952da5"></a>

Python interface начиная с 3.1 умеет auto-recoding для поддержанных dataframe inputs. Это решает согласование codes известных categories, но не делает unseen category статистически известной модели; documentation отдельно указывает ограничения input types и unknown handling. В других interfaces/старых версиях mapping может оставаться ответственностью preprocessing. [Categorical Data](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html).

<!-- /unit:365650bb6197952da5 -->

<!-- unit:2fb21a26d342af0435 -->
<a id="u-2fb21a26d342af0435"></a>

Не следует утверждать исторически, что XGBoost «не умеет native categories», и не следует делать противоположное слишком сильное утверждение «теперь все unknown работают автоматически везде». Round-trip test должен проверять конкретный installed stack.

<!-- /unit:2fb21a26d342af0435 -->

<!-- unit:4f12ebce9724aef4ae -->
<a id="u-4f12ebce9724aef4ae"></a>

### LightGBM: metadata определяет смысл codes

<!-- /unit:4f12ebce9724aef4ae -->

<!-- unit:93bf988764b6791cdb -->
<a id="u-93bf988764b6791cdb"></a>

В [официальном описании LightGBM](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html) categorical integers преобразуются в int32; отрицательные codes означают missing. Dataframe categories помогают согласовать dictionary. При автоматическом определении ordered pandas category может получить numeric semantics; требуемое categorical handling нужно задавать явно и проверять в установленной версии. В описанном dataframe prediction path unseen values направляются как missing.

<!-- /unit:93bf988764b6791cdb -->

<!-- unit:217b0f2051427f73bb -->
<a id="u-217b0f2051427f73bb"></a>

Следовательно, перестановка только raw codes без сохранённого mapping разрушает semantic contract. Две строки могут иметь одинаковое целое 1 и различный исходный meaning. Отдельно проверяется, что scalar 0 — legitimate category, если такое определено schema, а не универсальное «отсутствие значения». Smoothing categories и ограничения групп стабилизируют estimates, но не восстанавливают потерянное значение после ошибочной recoding.

<!-- /unit:217b0f2051427f73bb -->

<!-- unit:938ca56d7c045e463f -->
<a id="u-938ca56d7c045e463f"></a>

### CatBoost: statistics и splits — разные этапы

<!-- /unit:938ca56d7c045e463f -->

<!-- unit:b8bf00ae8f79d2c8bb -->
<a id="u-b8bf00ae8f79d2c8bb"></a>

Упрощённый путь такой: исходный category key или tuple нескольких fields определяет statistic; statistic строится по разрешённой training history/permutation; tree выбирает threshold по полученному numerical feature. Для небольших vocabularies возможен прямой one-hot comparison. Точные CTR types и режимы не сводятся к одному scalar mean из главы [«Target statistics, smoothing, cross-fitting и WoE»](<feat-target.md>).

<!-- /unit:b8bf00ae8f79d2c8bb -->

<!-- unit:102e43bc7061ba7b65 -->
<a id="u-102e43bc7061ba7b65"></a>

Для нового tuple, даже если каждый его компонент известен, reference count combination может быть нулевым. Тогда выбранный statistic использует предусмотренный prior/fallback; наличие известных components не равнозначно наличию наблюдений их joint effect. Число combinations ограничивают budget/config, а не строят полный декартов product всех fields. Строковые representations канонизируются одинаково на fit и prediction.

<!-- /unit:102e43bc7061ba7b65 -->

<!-- unit:2ff3d0d66467638e53 -->
<a id="u-2ff3d0d66467638e53"></a>

Документация [CatBoost categorical features](https://catboost.ai/docs/en/features/categorical-features) предупреждает, что выбор one-hot зависит от training mode; для features, выбранных в этот путь, CTR не вычисляются. Поэтому threshold one_hot_max_size меняет сам representation path, а не только ускоряет одну и ту же функцию. Нельзя подставлять одну memorized default value для всех losses/devices и объявлять воспроизводимость достигнутой.

<!-- /unit:2ff3d0d66467638e53 -->

<!-- unit:e1792f65c3b94ea446 -->
<a id="u-e1792f65c3b94ea446"></a>

### XGBoost: recoding не создаёт statistic unseen значения

<!-- /unit:e1792f65c3b94ea446 -->

<!-- unit:463948d52f22f00ae8 -->
<a id="u-463948d52f22f00ae8"></a>

Auto-recoding поддержанных dataframe inputs сопоставляет сохранённые category identities и коды; raw array interface и старая версия могут требовать собственного dictionary contract. Если validation batch содержит только A/C, локальные codes 0/1 не обязаны совпасть с train A/B/C=0/1/2. Нужен согласованный translation, а не совпадение range codes.

<!-- /unit:463948d52f22f00ae8 -->

<!-- unit:608fd46a1b75832ab8 -->
<a id="u-608fd46a1b75832ab8"></a>

Истинно новая D отличается от отсутствующей в текущем batch известной B. Документация [data consistency](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html) отдельно ограничивает recoding и unseen case. JSON/UBJSON model artifact сохраняет categorical information, но preprocessing schema и разрешённые исходные types тоже нужны. Нельзя обещать, что один enable_categorical автоматически решает любой inference mismatch.

<!-- /unit:608fd46a1b75832ab8 -->

<!-- unit:6a1933bcba55fde390 -->
<a id="u-6a1933bcba55fde390"></a>

## Regularization и новые категории

<!-- /unit:6a1933bcba55fde390 -->

<!-- unit:f52ad174ef3a6075b5 -->
<a id="u-f52ad174ef3a6075b5"></a>

У редкой category случайный gradient может оказаться большим относительно доступного H. При G=1,H=0.001 unregularized scalar response −1000; с lambda=1 — примерно −0.999. Пример показывает стабилизацию denominator, а не правдоподобие таких aggregates для любого loss. Gradient/Hessian statistics должны соответствовать objective и sample weights.

<!-- /unit:f52ad174ef3a6075b5 -->

<!-- unit:b295e5ed50b310f41a -->
<a id="u-b295e5ed50b310f41a"></a>

Minimum child Hessian, число строк и число categories в группе — разные ограничения. В logistic loss один уверенно предсказанный объект имеет h гораздо меньше 0.25, поэтому большое число строк не гарантирует большое H. Объединение rare categories и shrinkage statistics меняют candidate ranking; параметры regularization выбирают внутри честной validation, а не по самому интересному split gain на test.

<!-- /unit:b295e5ed50b310f41a -->

<!-- unit:5112fb665189fd0ec6 -->
<a id="u-5112fb665189fd0ec6"></a>

Для arbitrary unique IDs training tree может найти случайные label patterns. Native categories не создают повторных observations новых IDs и не отменяют memorization. Нужен split, соответствующий deployment: новые entities, известные entities в будущем или новые combinations. Использовать train labels для gradient histogram дерева допустимо: это часть supervised fit. Использовать внешние validation labels для подготовки histogram, CTR, dictionary grouping по target или выбора tree — уже другая операция и утечка.

<!-- /unit:5112fb665189fd0ec6 -->

<!-- unit:34a0ccbb12c3ac2452 -->
<a id="u-34a0ccbb12c3ac2452"></a>

## Сравнение и сериализация

<!-- /unit:34a0ccbb12c3ac2452 -->

<!-- unit:42b3185a29ba536001 -->
<a id="u-42b3185a29ba536001"></a>

Одинаковый split и исходные разрешённые features, comparable tuning budget, одинаковый outcome metric и дополнительные reports memory/latency. Native handling сравнивается с понятным one-hot/frequency/OOF baseline. Одинаковое число trees не всегда означает одинаковый compute или capacity.

<!-- /unit:42b3185a29ba536001 -->

<!-- unit:bef638c5a399191b9f -->
<a id="u-bef638c5a399191b9f"></a>

Обязательно сохраняем pipeline → загружаем → сравниваем predictions на известных values, reordered vocabulary, missing, rare и unseen. Тестируйте также отсутствие category, присутствовавшей в train: это не то же, что появление новой. Статус dictionary и policy fallback фиксируются в artifact.

<!-- /unit:bef638c5a399191b9f -->

<!-- unit:031528404def3267ef -->
<a id="u-031528404def3267ef"></a>

1. Почему codes не равны native handling? **Ответ:** numeric thresholds и subset splits задают разные partitions.
2. Делает ли auto-recoding unseen category выученной? **Ответ:** нет; согласование IDs не создаёт observations её effect.
3. Заменяет ли ordered permutation time split? **Ответ:** нет; chronological availability задаётся отдельно.

<!-- /unit:031528404def3267ef -->

<!-- unit:025fb37668ab46b6e2 -->
<a id="u-025fb37668ab46b6e2"></a>

**Практика:** для A/B/C/D найдите subset с одинаковыми effects и покажите неудачу одного numeric threshold. Составьте serialization cases для трёх libraries с exact versions; выполненные results и project design записывайте раздельно.

<!-- /unit:025fb37668ab46b6e2 -->

<!-- unit:b7d624a305940106ac -->
<a id="u-b7d624a305940106ac"></a>

<!-- source-depth-categorical-Д -->

<!-- /unit:b7d624a305940106ac -->

<!-- unit:db0b944053e29a1952 -->
<a id="u-db0b944053e29a1952"></a>

Сравнивают прежде всего validation predictions на одних разрешённых observations. Одинаковый n_estimators не выравнивает tree growth, depth/leaves, early stopping, hardware или tuning cost. Полезно отдельно фиксировать feature construction, model fit, serialization size и inference path. Overall score дополняют slices: frequent/rare known categories, known components с новым tuple, полностью new category, missing.

<!-- /unit:db0b944053e29a1952 -->

<!-- unit:de629188bb7395141a -->
<a id="u-de629188bb7395141a"></a>

В воспроизводимом artifact нужны category canonicalization, column names/order, declared categorical fields, dictionary/metadata, model format, version/device/config, data cutoff и policy unknown. Для round-trip checks сохраняют маленький набор raw inputs и reference predictions до serialization. После load сравнивают numbers в заявленной tolerance и фактический route error/fallback. Один успешный known input не покрывает unseen/missing cases.

<!-- /unit:de629188bb7395141a -->

<!-- unit:fbb2dc93c86c9adcf4 -->
<a id="u-fbb2dc93c86c9adcf4"></a>

**Дополнительные задания с ответами:** почему H не всегда count? Потому что Hessian определяется loss и текущим prediction. Почему learned order не равен IDs? Он строится по objective в node. Почему native fit не требует OOF для каждого gradient histogram? Он является supervised training model на разрешённом train; внешний evaluation из него исключён. Эти различия нужны, чтобы protocol из главы [«Target statistics, smoothing, cross-fitting и WoE»](<feat-target.md>) не превратился в запрет обычного обучения дерева.

<!-- /unit:fbb2dc93c86c9adcf4 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как модель выбирает разбиение категорий и что входит в сохранённый контракт?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
