# Д. Как дерево делит категории на группы

[Оглавление](README.md). Пункт 5 плана. **Происхождение:** дополнение по плану. API-границы сверены с официальными страницами 17 октября 2026; при реализации нужно фиксировать installed version и device/mode.

## 1. Numeric threshold и categorical subset

Numeric split имеет вид $x<t$. Categorical split может иметь вид $c\in S$, где $S$ — subset categories. Пусть A/B/C/D имеют эффекты $(1,-1,1,-1)$, а произвольные codes — 1/2/3/4. Один numeric threshold не отделит A/C от B/D, но subset split сделает это сразу. Поэтому передать codes как обычное число и объявить field categorical — разные операции.

Полный перебор бинарных partitions $K$ categories требует $2^{K-1}-1$ вариантов без учёта симметрии сторон. Для scalar leaf objectives можно использовать ordering categories по статистике leaf response и проверять contiguous partitions этого порядка. Это порядок, выученный в текущем node по objective, а не порядок исходных IDs.

При second-order boosting node агрегирует gradients $G$ и Hessians $H$. Типичный scalar leaf weight имеет форму $-G/(H+\lambda)$. При подходящих assumptions categories с близкими optimal responses можно группировать. Для vector outputs универсального scalar ordering нет; алгоритм/approximation зависит от implementation. Нельзя переносить proof scalar optimal partitioning на все objectives без оговорок.

## 2. LightGBM

LightGBM поддерживает categorical fields и partitioning. В Python dataframe metadata помогает согласовать categories; при raw integer arrays нужно самостоятельно сохранить codes. Отрицательные category codes рассматриваются как missing, поэтому применять generic unknown=-1 означает выбрать именно missing route, а не выучить новый category effect.

Сглаживание и минимальные размеры category groups ограничивают overfit; это hyperparameters, которые подбираются внутри validation. При high cardinality стоит сравнить native mode с manual representations, а не предполагать universal winner. [Официальные categorical/missing conventions](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html).

Пример ошибки contract: train mapping A=0,B=1,C=2, inference самостоятельно строит A=0,C=1. Теперь C будет прочитан как B, если metadata не восстановлена и передаются только codes. Совпадение dtype/shape не обнаружит semantic corruption.

## 3. CatBoost

CatBoost использует разные numerical statistics категорий и их combinations, а небольшие vocabularies может обрабатывать one-hot в зависимости от mode/config. Ordered target statistics опираются на precedents permutation; это не просто full-data mean. Ordered boosting отдельно связан с уменьшением prediction shift в самом gradient boosting. [Описание алгоритма](https://arxiv.org/abs/1706.09516).

Не нужно предварительно one-hot кодировать все fields, если цель — использовать native mechanism CatBoost: так он потеряет исходные category identities/комбинации. Threshold one-hot и доступные statistics зависят от loss, CPU/GPU и config. [Документация категориальных features](https://catboost.ai/docs/en/features/categorical-features).

Missing marker в string categorical field канонизируется осознанно: строки `"1"` и `"1.0"` не обязаны быть одним значением. Формат train/inference должен совпадать; null handling нельзя механически переносить с numeric features. Автоматические combinations тоже ограничиваются настройками и стоимостью, а не перебирают все возможные interactions без ограничений.

## 4. XGBoost

XGBoost поддерживает one-hot-подобные categorical splits и partitioning; выбор зависит, в частности, от threshold `max_cat_to_onehot`. Передаются categorical dtype/feature types и включается соответствующая поддержка. Для сохранения model используются formats, сохраняющие categorical metadata, например JSON/UBJSON.

Python interface начиная с 3.1 умеет auto-recoding для поддержанных dataframe inputs. Это решает согласование codes известных categories, но не делает unseen category статистически известной модели; documentation отдельно указывает ограничения input types и unknown handling. В других interfaces/старых версиях mapping может оставаться ответственностью preprocessing. [Categorical Data](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html).

Не следует утверждать исторически, что XGBoost «не умеет native categories», и не следует делать противоположное слишком сильное утверждение «теперь все unknown работают автоматически везде». Round-trip test должен проверять конкретный installed stack.

## 5. Как сравнивать библиотеки

Одинаковый split и исходные разрешённые features, comparable tuning budget, одинаковый outcome metric и дополнительные reports memory/latency. Native handling сравнивается с понятным one-hot/frequency/OOF baseline. Одинаковое число trees не всегда означает одинаковый compute или capacity.

Обязательно сохраняем pipeline → загружаем → сравниваем predictions на известных values, reordered vocabulary, missing, rare и unseen. Тестируйте также отсутствие category, присутствовавшей в train: это не то же, что появление новой. Статус dictionary и policy fallback фиксируются в artifact.

## 6. Контроль и практика

1. Почему codes не равны native handling? **Ответ:** numeric thresholds и subset splits задают разные partitions.
2. Делает ли auto-recoding unseen category выученной? **Ответ:** нет; согласование IDs не создаёт observations её effect.
3. Заменяет ли ordered permutation time split? **Ответ:** нет; chronological availability задаётся отдельно.

**Практика:** для A/B/C/D найдите subset с одинаковыми effects и покажите неудачу одного numeric threshold. Составьте serialization cases для трёх libraries с exact versions; выполненные results и project design записывайте раздельно.


<!-- source-depth-categorical-Д -->

## 7. От loss к решению о categorical split

### 7.1. Дерево исправляет текущий predictor

На очередном boosting step уже есть predictions $f_i$. Для differentiable loss вычисляются $g_i=\partial\ell(y_i,f_i)/\partial f_i$ и $h_i=\partial^2\ell(y_i,f_i)/\partial f_i^2$. В half-squared loss это $g_i=f_i-y_i$, $h_i=1$; в binary logistic loss по logit — $g_i=p_i-y_i$, $h_i=p_i(1-p_i)$. Поэтому histogram gradients не равна средним исходного target в общем случае: она зависит и от того, что предыдущие trees уже объяснили.

Если всем объектам leaf назначить scalar increment w, часть second-order objective, зависящая от w, имеет вид

$$
Q(w)=Gw+\frac12(H+\lambda)w^2,
\quad G=\sum_i g_i,\quad H=\sum_i h_i.
$$

При $H+\lambda>0$ derivative $G+(H+\lambda)w=0$ даёт $w^*=-G/(H+\lambda)$ и minimum $-G^2/[2(H+\lambda)]$. Здесь lambda — L2 penalty на weight leaf в принятом sum-loss convention. Mean loss, sample weights и другая нормировка objective меняют сопоставимый scale lambda. У nonsmooth objectives или особых Hessian approximations этот вывод нужно сверять с конкретным алгоритмом.

Для split с children L/R выигрыш относительно одного parent равен

$$
\operatorname{Gain}=\frac12\left[
\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}
-\frac{G_P^2}{H_P+\lambda}\right]-\gamma.
$$

Gamma здесь — дополнительная стоимость одного нового leaf при переходе от одного к двум. Это учебная second-order схема, а не утверждение, что все библиотеки хранят одинаковый gain scale или используют одинаковые defaults. Learning rate обычно применяют к найденным leaf increments после построения tree; уменьшить increment и заново пересчитать exact loss — отдельный расчёт, который не обязан совпасть с unshrunk split gain.

### 7.2. Полный пример из четырёх категорий

Пусть в текущем node у A/B/C/D по две строки, Hessian каждой строки 1. В каждой category одинаковые residual gradients: A −2, B 1, C −1, D 2. Category totals G равны (−4,2,−2,4), H — (2,2,2,2). Возьмём lambda=1, gamma=0. Parent G=0,H=8, поэтому parent weight 0.

При numeric codes A=1,B=2,C=3,D=4 возможны только три threshold cuts. Для A|BCD gain равен $(16/3+16/7)/2=80/21\approx3.8095$. Для AB|CD — $(4/5+4/5)/2=0.8$. Для ABC|D снова 80/21. Никакой threshold не создаёт AC|BD.

Categorical subset AC имеет G=−6,H=4, дополнение BD — G=6,H=4. Получаются weights 1.2 и −1.2, gain $(36/5+36/5)/2=7.2$. При gamma=0.5 gain уменьшится до 6.7. При learning rate 0.1 добавки к predictor станут +0.12 и −0.12; это не обещание decrease exact loss ровно на 7.2 после shrinkage.

В half-squared setting можно проверить пример без Taylor approximation: положим начальные predictions 0 и targets (2,2,−1,−1,1,1,−2,−2). Начальный half-squared loss 10. После leaf increments ±1.2 data loss 1.36, penalty двух leaf weights 1.44, total 2.8. Разница 7.2 совпадает с derived gain. Это точный учебный objective, не измерение качества LightGBM или XGBoost.

One-category-versus-rest split способен выбрать A или D с gain 80/21, но не объединить сразу AC в одном split. Следующие уровни tree могут восстановить нужную функцию, расходуя другую structure/capacity. Поэтому преимущество subset в этом примере относится к одному split и не означает невозможность научиться с one-hot вообще.

## 8. Как заменить экспоненциальный поиск сортировкой

Для K=4 существует семь unordered nontrivial binary partitions, для K=20 — 524 287. Число $2^{K-1}-1$ получается из всех subsets после исключения пустого/полного и отождествления S с дополнением. Category IDs сами по себе не дают удобного порядка для такого поиска.

Для подходящих scalar objectives категории можно упорядочить по response statistic, затем проверить K−1 contiguous cuts. В нашем примере individual regularized leaf responses равны A=4/3,B=−2/3,C=2/3,D=−4/3. Порядок D,B,C,A приводит к нужному cut DB|CA в середине. Это другой порядок, чем A,B,C,D по исходным кодам.

Наивный algorithm сначала агрегирует category histogram по строкам node, сортирует K statistics, затем проходит prefix sums G/H. После сортировки каждую новую границу можно оценить за постоянное время: перенос одной category изменяет только соответствующие totals. Типичная стоимость такого учебного прохода — O(number of node rows)+O(K log K)+O(K), без exp перебора. Histogram reuse, binning, параллелизм и фактические candidate limits изменяют runtime implementation.

Слово «optimal» относится к определённой задаче, допустимым partitions и предположениям theorem. L1/L2 terms, minimum child constraints, category smoothing, ограниченный список candidates и vector outputs нельзя автоматически включить в доказательство без проверки. Сортировка по raw mean target на всём dataset — также не универсальная замена node-specific gradient ordering.

Для vector leaf у category есть вектор response, а не одно число. Оси нельзя упорядочить одновременно одним scalar без дополнительных предположений. Например, vectors (1,2) и (1,−2) имеют одинаковую projection на parent direction (1,0), хотя сильно различаются по второй координате. В актуальном описании [XGBoost categorical partitioning](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html) для vector leaves используется projection на parent Newton direction; это approximation, способная пропустить ортогональные contrasts. Поведение нулевого/слабого parent direction проверяют в конкретной версии, не придумывая универсальный порядок vectors.

## 9. Почему native mode тоже нуждается в regularization

У редкой category случайный gradient может оказаться большим относительно доступного H. При G=1,H=0.001 unregularized scalar response −1000; с lambda=1 — примерно −0.999. Пример показывает стабилизацию denominator, а не правдоподобие таких aggregates для любого loss. Gradient/Hessian statistics должны соответствовать objective и sample weights.

Minimum child Hessian, число строк и число categories в группе — разные ограничения. В logistic loss один уверенно предсказанный объект имеет h гораздо меньше 0.25, поэтому большое число строк не гарантирует большое H. Объединение rare categories и shrinkage statistics меняют candidate ranking; параметры regularization выбирают внутри честной validation, а не по самому интересному split gain на test.

Для arbitrary unique IDs training tree может найти случайные label patterns. Native categories не создают повторных observations новых IDs и не отменяют memorization. Нужен split, соответствующий deployment: новые entities, известные entities в будущем или новые combinations. Использовать train labels для gradient histogram дерева допустимо: это часть supervised fit. Использовать внешние validation labels для подготовки histogram, CTR, dictionary grouping по target или выбора tree — уже другая операция и утечка.

## 10. Три библиотеки: различия маршрута, а не названия метода

### 10.1. LightGBM: metadata определяет смысл codes

В [официальном описании LightGBM](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html) categorical integers преобразуются в int32; отрицательные codes означают missing. Dataframe categories помогают согласовать dictionary. При автоматическом определении ordered pandas category может получить numeric semantics; требуемое categorical handling нужно задавать явно и проверять в установленной версии. В описанном dataframe prediction path unseen values направляются как missing.

Следовательно, перестановка только raw codes без сохранённого mapping разрушает semantic contract. Две строки могут иметь одинаковое целое 1 и различный исходный meaning. Отдельно проверяется, что scalar 0 — legitimate category, если такое определено schema, а не универсальное «отсутствие значения». Smoothing categories и ограничения групп стабилизируют estimates, но не восстанавливают потерянное значение после ошибочной recoding.

### 10.2. CatBoost: statistics и splits — разные этапы

Упрощённый путь такой: исходный category key или tuple нескольких fields определяет statistic; statistic строится по разрешённой training history/permutation; tree выбирает threshold по полученному numerical feature. Для небольших vocabularies возможен прямой one-hot comparison. Точные CTR types и режимы не сводятся к одному scalar mean из главы В.

Для нового tuple, даже если каждый его компонент известен, reference count combination может быть нулевым. Тогда выбранный statistic использует предусмотренный prior/fallback; наличие известных components не равнозначно наличию наблюдений их joint effect. Число combinations ограничивают budget/config, а не строят полный декартов product всех fields. Строковые representations канонизируются одинаково на fit и prediction.

Документация [CatBoost categorical features](https://catboost.ai/docs/en/features/categorical-features) предупреждает, что выбор one-hot зависит от training mode; для features, выбранных в этот путь, CTR не вычисляются. Поэтому threshold one_hot_max_size меняет сам representation path, а не только ускоряет одну и ту же функцию. Нельзя подставлять одну memorized default value для всех losses/devices и объявлять воспроизводимость достигнутой.

### 10.3. XGBoost: recoding не создаёт statistic unseen значения

Auto-recoding поддержанных dataframe inputs сопоставляет сохранённые category identities и коды; raw array interface и старая версия могут требовать собственного dictionary contract. Если validation batch содержит только A/C, локальные codes 0/1 не обязаны совпасть с train A/B/C=0/1/2. Нужен согласованный translation, а не совпадение range codes.

Истинно новая D отличается от отсутствующей в текущем batch известной B. Документация [data consistency](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html) отдельно ограничивает recoding и unseen case. JSON/UBJSON model artifact сохраняет categorical information, но preprocessing schema и разрешённые исходные types тоже нужны. Нельзя обещать, что один enable_categorical автоматически решает любой inference mismatch.

## 11. Что должно доказать сравнение и сохранение pipeline

Сравнивают прежде всего validation predictions на одних разрешённых observations. Одинаковый n_estimators не выравнивает tree growth, depth/leaves, early stopping, hardware или tuning cost. Полезно отдельно фиксировать feature construction, model fit, serialization size и inference path. Overall score дополняют slices: frequent/rare known categories, known components с новым tuple, полностью new category, missing.

В воспроизводимом artifact нужны category canonicalization, column names/order, declared categorical fields, dictionary/metadata, model format, version/device/config, data cutoff и policy unknown. Для round-trip checks сохраняют маленький набор raw inputs и reference predictions до serialization. После load сравнивают numbers в заявленной tolerance и фактический route error/fallback. Один успешный known input не покрывает unseen/missing cases.

**Дополнительные задания с ответами:** почему H не всегда count? Потому что Hessian определяется loss и текущим prediction. Почему learned order не равен IDs? Он строится по objective в node. Почему native fit не требует OOF для каждого gradient histogram? Он является supervised training model на разрешённом train; внешний evaluation из него исключён. Эти различия нужны, чтобы protocol из главы В не превратился в запрет обычного обучения дерева.
