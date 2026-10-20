# Взаимодействия признаков, FM и FFM

**Уровень:** основа ветки.

**Предпосылки:** [Nominal, ordinal, one-hot и dummy encoding](<feat-categorical.md>), [Обучаемые embeddings и перенос представлений](<feat-embeddings.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Когда эффект признака зависит от другого поля и как это представить?





## Карта вычисления и границы этапов

![Взаимодействия признаков, FM и FFM: входы, преобразования и выходы](<../../оформление/course/feat-interactions.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Обучаемые статистики вычисляются на train; validation и новые объекты используют сохранённое состояние. Геометрическая формула с фиксированными константами не превращается в fit только из-за наличия блока на схеме.

<!-- unit:2b81f801e792dc286f -->
<a id="u-2b81f801e792dc286f"></a>

## Пределы additive модели

<!-- /unit:2b81f801e792dc286f -->

<!-- unit:b5e9d32a845dbd2e07 -->
<a id="u-b5e9d32a845dbd2e07"></a>

Пусть вероятность покупки зависит от city и day. В городе A weekend увеличивает спрос, в B уменьшает. Additive effects $b+w_{city}+v_{day}$ навязывают одинаковый day effect во всех cities. Interaction позволяет change effect depending on partner.

<!-- /unit:b5e9d32a845dbd2e07 -->

<!-- unit:9e564f589b856b2096 -->
<a id="u-9e564f589b856b2096"></a>

Для двух бинарных fields XOR — наглядный пример: outcomes $(A,X)=0,(A,Y)=1,(B,X)=1,(B,Y)=0$. Без interaction линейный score не может одновременно реализовать такую threshold pattern. Cross category A×X/A×Y/B×X/B×Y даёт отдельные effects, но число combinations растёт произведением cardinalities.

<!-- /unit:9e564f589b856b2096 -->

<!-- unit:ed145282be342f7a0d -->
<a id="u-ed145282be342f7a0d"></a>

### Difference-in-differences и невозможность XOR

<!-- /unit:ed145282be342f7a0d -->

<!-- unit:879189b7f61f94a223 -->
<a id="u-879189b7f61f94a223"></a>

Для additive score $s_{cd}=b+a_c+d_d$ contrast $s_{AY}-s_{AX}$ равен $d_Y-d_X$ и не зависит от city. Поэтому difference-in-differences $s_{AY}-s_{AX}-s_{BY}+s_{BX}$ равна нулю. Ненулевое значение требует interaction в score. Для logistic model это утверждение о logit, а не о вероятностях: одна и та же прибавка logit может по-разному изменить probability при разных baseline levels.

<!-- /unit:879189b7f61f94a223 -->

<!-- unit:bb0da254bfd516b46e -->
<a id="u-bb0da254bfd516b46e"></a>

Пусть positive означает score>0, negative — score<0. XOR потребует $s_{AX}<0,s_{AY}>0,s_{BX}>0,s_{BY}<0$. Сумма двух positive scores должна быть больше нуля, сумма negative — меньше нуля. Но additive model даёт $s_{AY}+s_{BX}=2b+a_A+a_B+d_X+d_Y=s_{AX}+s_{BY}$. Противоречие доказывает невозможность строгого разделения этого pattern; оно не зависит от удачно выбранного optimizer.

<!-- /unit:bb0da254bfd516b46e -->

<!-- unit:dd43dfa0be77e583b9 -->
<a id="u-dd43dfa0be77e583b9"></a>

При известных categories FM может представить XOR уже с d=1: зададим city factors A=1,B=−1 и day factors X=−1,Y=1. Матрица dot products равна ((−1,1),(1,−1)); threshold 0 даёт нужные labels. Вероятности после sigmoid не станут ровно 0/1 при конечных weights, но classification pattern реализуется. Этот конкретный успех не означает, что любая interaction matrix имеет rank 1.

<!-- /unit:dd43dfa0be77e583b9 -->

<!-- unit:1a27ebfa373035cd7e -->
<a id="u-1a27ebfa373035cd7e"></a>

### Main effects, crosses и идентифицируемость

<!-- /unit:1a27ebfa373035cd7e -->

<!-- unit:e219c6c75ffa28296a -->
<a id="u-e219c6c75ffa28296a"></a>

Если включить intercept, все city indicators, все day indicators и все pair indicators, часть columns линейно зависима. Например, indicator города A равен сумме pair indicators AX и AY при ровно одном day на строку. Predictions могут быть однозначны, а decomposition на main и interaction coefficients — нет. Reference levels, sum-to-zero constraints или regularization задают определённую parametrization.

<!-- /unit:e219c6c75ffa28296a -->

<!-- unit:6f133dbafe84925332 -->
<a id="u-6f133dbafe84925332"></a>

Cross-only model с отдельным coefficient каждой observed pair запоминает её response, но не имеет собственного estimate для совсем новой pair. Main effects и default coefficient позволяют задать fallback, однако его нужно определить явно. Missing cross в sparse design, zero coefficient и статистически доказанное отсутствие взаимодействия — разные понятия.

<!-- /unit:6f133dbafe84925332 -->

<!-- unit:1896f04a152eff42ff -->
<a id="u-1896f04a152eff42ff"></a>

## Ручные crosses

<!-- /unit:1896f04a152eff42ff -->

<!-- unit:5f16a24afda513c365 -->
<a id="u-5f16a24afda513c365"></a>

Cross ключ должен быть однозначным tuple, а не ambiguous concatenation: строки `ab`+`c` и `a`+`bc` иначе совпадут. Namespace/escaping/serialization являются частью contract. Затем cross кодируется one-hot, hash, frequency или supervised statistics по тем же правилам leakage.

<!-- /unit:5f16a24afda513c365 -->

<!-- unit:b1ccbeed59fac034df -->
<a id="u-b1ccbeed59fac034df"></a>

Если city имеет 1000 values, product 10 000, потенциально $10^7$ combinations. Наблюдаемых пар меньше, но многие редки или unseen. Полный отдельный coefficient на каждую пару плохо переносится на новые combinations. Smoothing и shared factors предлагают более компактный prior.

<!-- /unit:b1ccbeed59fac034df -->

<!-- unit:c93d221529d73dc46e -->
<a id="u-c93d221529d73dc46e"></a>

## FM: механизм и forward

<!-- /unit:c93d221529d73dc46e -->

<!-- unit:3fb12e68f6a6897e80 -->
<a id="u-3fb12e68f6a6897e80"></a>

FM второго порядка для sparse input:

<!-- /unit:3fb12e68f6a6897e80 -->

<!-- unit:72e15eb0eca1e97c6e -->
<a id="u-72e15eb0eca1e97c6e"></a>

$$
\widehat y=w_0+\sum_iw_ix_i+
\sum_{i<j}\langle v_i,v_j\rangle x_ix_j,
\qquad v_i\in\mathbb R^d.
$$

<!-- /unit:72e15eb0eca1e97c6e -->

<!-- unit:206c64495384bdfd1f -->
<a id="u-206c64495384bdfd1f"></a>

Вместо независимого coefficient каждой пары используется inner product двух latent vectors. Это parameter sharing, позволяющее оценивать interactions по общим evidence других pairs. Оно ограничивает доступные interaction matrices, но не гарантирует correctness для unseen values.

<!-- /unit:206c64495384bdfd1f -->

<!-- unit:d93b8fef1490f56d1f -->
<a id="u-d93b8fef1490f56d1f"></a>

При one-hot по fields активны несколько features. Для vectors $v_A=(1,0)$ и $v_X=(0.2,0.5)$ pair effect равен 0.2; с $v_Y=(-0.3,0.1)$ — минус 0.3. Изменение одного category vector влияет на все его interactions, что даёт both sharing и coupling ошибок.

<!-- /unit:d93b8fef1490f56d1f -->

<!-- unit:14a2d9d457d1f61178 -->
<a id="u-14a2d9d457d1f61178"></a>

FM sum можно вычислять без явного перебора всех pairs:

<!-- /unit:14a2d9d457d1f61178 -->

<!-- unit:5c00906fc1bb293641 -->
<a id="u-5c00906fc1bb293641"></a>

$$
\sum_{i<j}\langle v_i,v_j\rangle x_ix_j
=\frac12\sum_{r=1}^d\left[
\left(\sum_i v_{i,r}x_i\right)^2-
\sum_i v_{i,r}^2x_i^2\right].
$$

<!-- /unit:5c00906fc1bb293641 -->

<!-- unit:c491f5f8d162f79bcb -->
<a id="u-c491f5f8d162f79bcb"></a>

Это даёт cost порядка числа active features times dimension при подходящей implementation. Формула следует из раскрытия квадрата: diagonal terms вычитаются, каждое pair остаётся дважды до factor 1/2.

<!-- /unit:c491f5f8d162f79bcb -->

<!-- unit:8f70785884d3e10f92 -->
<a id="u-8f70785884d3e10f92"></a>

Для двух fields C и D interaction block имеет entries $M_{cd}=e_c^\top f_d$. В matrix form $M=EF^\top$, значит rank не больше d. Вместо $K_CK_D$ независимых pair coefficients обучается $(K_C+K_D)d$ factors. Например, 1000 cities и 10 000 products дают 10 миллионов explicit pair coefficients против 176 000 factors при d=16. Bias/main effects и optimizer memory добавляются отдельно.

<!-- /unit:8f70785884d3e10f92 -->

<!-- unit:db61aedb0e705432b4 -->
<a id="u-db61aedb0e705432b4"></a>

При d=1 должна выполняться связь $M_{11}M_{22}=M_{12}M_{21}$ для любого 2×2 subblock. Matrix ((1,2),(2,4)) её удовлетворяет, identity matrix ((1,0),(0,1)) — нет. Увеличение d снимает часть ограничений, но повышает parameters и требования к статистике.

<!-- /unit:db61aedb0e705432b4 -->

<!-- unit:a52e7c51090d7889e3 -->
<a id="u-a52e7c51090d7889e3"></a>

Нужно аккуратно формулировать rank statement для FM с общим набором features. Полная Gram matrix VVᵀ имеет rank не больше d и является PSD. Однако FM использует только entries i<j, а не self-interactions. Если удалить diagonal из Gram matrix, rank может вырасти: при трёх scalar factors, равных 1, получаем off-diagonal matrix J−I ранга 3, хотя d=1. Поэтому нельзя доказывать expressivity всей FM фразой «матрица с нулевой диагональю имеет rank d». Ограничение cross-field block выше корректно.

<!-- /unit:a52e7c51090d7889e3 -->

<!-- unit:7aa60637ea27086d74 -->
<a id="u-7aa60637ea27086d74"></a>

Этот prior помогает, если эффекты похожих pairs разделяют скрытые причины. Если каждый pair effect независим и случайный, small-rank approximation будет biased. Factorization не создаёт недостающие labels, а связывает оценки предположением о структуре.

<!-- /unit:7aa60637ea27086d74 -->

<!-- unit:5d9c76aa5984b834dc -->
<a id="u-5d9c76aa5984b834dc"></a>

Возьмём три active features с x=(1,2,−1) и vectors $v_1=(1,2)$, $v_2=(3,-1)$, $v_3=(-2,0.5)$. Здесь input values намеренно не только binary: counts, weights и numerical features тоже могут входить в FM, если scale задан осмысленно.

<!-- /unit:5d9c76aa5984b834dc -->

<!-- unit:f8e21295145cd0d55c -->
<a id="u-f8e21295145cd0d55c"></a>

Dot products равны 1,−1,−6.5 для pairs 12,13,23. С учётом $x_ix_j$ contributions получаются 2,1,13; total interaction 16. Если забыть множители x, получится совсем другая функция. Линейная часть и intercept здесь положены нулевыми только для прозрачности расчёта.

<!-- /unit:f8e21295145cd0d55c -->

<!-- unit:263c2ac61b15fc42b9 -->
<a id="u-263c2ac61b15fc42b9"></a>

Теперь вычислим $s=\sum_i x_i v_i=(9,-0.5)$. Его squared norm 81.25. Diagonal term равен $1^2\cdot5+2^2\cdot10+(-1)^2\cdot4.25=49.25$. Поэтому identity даёт (81.25−49.25)/2=16 — тот же результат без отдельного loop по всем pairs.

<!-- /unit:263c2ac61b15fc42b9 -->

<!-- unit:a5e85394293ba6bbf9 -->
<a id="u-a5e85394293ba6bbf9"></a>

Если nnz активных coordinates равен m, сначала вычисляют d sums по m features, затем diagonal correction. Стоимость O(md) не зависит от полного словаря K в самом forward по sparse observation. Dense optimizer states, regularization update всех rows и data lookup могут иметь другую стоимость; сложность formula не является end-to-end benchmark.

<!-- /unit:a5e85394293ba6bbf9 -->

<!-- unit:4dd8502d17bad7dbd0 -->
<a id="u-4dd8502d17bad7dbd0"></a>

У single-valued one-hot field два разных значения одного field не бывают активны одновременно, поэтому их pair term всегда нулевой в данных. У multi-valued field такие interactions уже могут существовать. Решение включить или исключить within-field pairs задаёт другой model; нельзя молча переносить assumption «в каждом field одна category» на bags интересов.

<!-- /unit:4dd8502d17bad7dbd0 -->

<!-- unit:de414f45af5071773b -->
<a id="u-de414f45af5071773b"></a>

## Градиенты факторов

<!-- /unit:de414f45af5071773b -->

<!-- unit:b3d00f2f2113a428e3 -->
<a id="u-b3d00f2f2113a428e3"></a>

Для FM interaction score и $s=\sum_j v_jx_j$ derivative по vector $v_i$ равна

<!-- /unit:b3d00f2f2113a428e3 -->

<!-- unit:cb0545c965b0507a37 -->
<a id="u-cb0545c965b0507a37"></a>

$$
\frac{\partial\widehat y}{\partial v_i}=x_i(s-v_ix_i).
$$

<!-- /unit:cb0545c965b0507a37 -->

<!-- unit:e8233b5db38fe0bdc1 -->
<a id="u-e8233b5db38fe0bdc1"></a>

Выражение s содержит self contribution $v_ix_i$, который удаляется: self-pair i=i в score нет. На числах предыдущего раздела derivatives по v1,v2,v3 равны (8,−2.5), (6,3), (−7,0). Проверка finite differences должна изменять один factor coordinate и пересчитывать весь score, включая оба terms identity; иначе можно проверить gradient другой функции.

<!-- /unit:e8233b5db38fe0bdc1 -->

<!-- unit:b87433f46328b04907 -->
<a id="u-b87433f46328b04907"></a>

Для binary cross-entropy с logit score error multiplier равен $\delta=\sigma(\widehat y)-y$. При L2 penalty $\lambda\sum_i\|v_i\|^2/2$ gradient имеет вид

<!-- /unit:b87433f46328b04907 -->

<!-- unit:9b5c2526b59afbe8f5 -->
<a id="u-9b5c2526b59afbe8f5"></a>

$$
\nabla_{v_i}L=\delta\,x_i(s-v_ix_i)+\lambda v_i.
$$

<!-- /unit:9b5c2526b59afbe8f5 -->

<!-- unit:f015ce0d8c3df28c10 -->
<a id="u-f015ce0d8c3df28c10"></a>

Если objective усреднён по batch, data gradients усредняют один раз; scale регуляризации объявляют относительно этого loss. Для squared error multiplier другой, а для ranking loss он зависит от пары и sampling. Shared factor обновляется из всех observations/partners, в которых участвует, поэтому улучшение одной interaction может ухудшить другую.

<!-- /unit:f015ce0d8c3df28c10 -->

<!-- unit:d43a62be3fd2d9b651 -->
<a id="u-d43a62be3fd2d9b651"></a>

### Маленький joint step

<!-- /unit:d43a62be3fd2d9b651 -->

<!-- unit:2a13525df038cca9dd -->
<a id="u-2a13525df038cca9dd"></a>

Для одной active pair x1=x2=1, d=1, v1=1,v2=2 и y=1 logit=2, p≈0.880797, delta≈−0.119203. Без regularization gradients факторов равны 2 delta и delta. При LR 0.1 simultaneous update даёт v1≈1.0238406,v2≈2.0119203. Оба gradients берутся по исходным values; последовательное изменение одного factor до вычисления gradient другого — другой алгоритм.

<!-- /unit:2a13525df038cca9dd -->

<!-- unit:e54cc48e22f9920441 -->
<a id="u-e54cc48e22f9920441"></a>

При всех factors, инициализированных нулём, s тоже нулевой и data gradient factors равен нулю. Pure gradient descent с L2 не выйдет из этого состояния interaction branch; main effects могут обучаться, но не создадут gradient factors сами по себе. Ненулевая initialization или иной явный механизм разрушает эту symmetry. Это объясняет engineering choice лучше, чем фраза «random init принято по умолчанию».

<!-- /unit:e54cc48e22f9920441 -->

<!-- unit:cbaa78edb607c72598 -->
<a id="u-cbaa78edb607c72598"></a>

## FFM и поле партнёра

<!-- /unit:cbaa78edb607c72598 -->

<!-- unit:764520222b5bedc927 -->
<a id="u-764520222b5bedc927"></a>

FFM даёт feature отдельный vector в зависимости от field партнёра:

<!-- /unit:764520222b5bedc927 -->

<!-- unit:5197e16f7dd389e8b2 -->
<a id="u-5197e16f7dd389e8b2"></a>

$$
\widehat y=w_0+\sum_iw_ix_i+
\sum_{i<j}\langle v_{i,f_j},v_{j,f_i}\rangle x_ix_j.
$$

<!-- /unit:5197e16f7dd389e8b2 -->

<!-- unit:cc06f6953c195c61cd -->
<a id="u-cc06f6953c195c61cd"></a>

$f_i$ — field feature $i$. City A может иметь разные latent representations для interaction с day и с product. Это гибче FM, но увеличивает parameters примерно с $Kd$ до $KFd$ при полном наборе field-specific vectors и требует большей статистики. Точная стоимость зависит от исключённых interactions и storage.

<!-- /unit:cc06f6953c195c61cd -->

<!-- unit:60971c8029992d97a2 -->
<a id="u-60971c8029992d97a2"></a>

FFM не равен просто concatenation обычных embeddings с linear head: interaction rule задан отдельно. Neural models могут выучить interactions через MLP/attention, но их bias, capacity и optimization отличаются. Автоматические crosses CatBoost — ещё один механизм: combination categories превращаются в statistics/разбиения, а не обязательно factorized dot products. Первичный разбор: [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf).

<!-- /unit:60971c8029992d97a2 -->

<!-- unit:cde15c64e76f36576f -->
<a id="u-cde15c64e76f36576f"></a>

Пусть одновременно активны city A, day D и product P. В FFM у A есть vectors $v_{A,day}$ и $v_{A,product}$. У partner используются встречные vectors $v_{D,city}$ и $v_{P,city}$. Field — это тип признака, а не индекс конкретной category: все products адресуют product-facing representation города A.

<!-- /unit:cde15c64e76f36576f -->

<!-- unit:8b5ae16f9e47d0d953 -->
<a id="u-8b5ae16f9e47d0d953"></a>

Зададим $v_{A,day}=(1,0)$, $v_{D,city}=(3,1)$; $v_{A,product}=(0,2)$, $v_{P,city}=(2,-1)$; $v_{D,product}=(1,1)$, $v_{P,day}=(-1,4)$. Три contributions равны 3,−2,3, total 4. Если заменить только $v_{A,product}$ на (0,3), изменится contribution A–P до −3, а A–D останется 3; total станет 3. В обычной FM у A один общий vector, и его изменение обычно затрагивает все partners.

<!-- /unit:8b5ae16f9e47d0d953 -->

<!-- unit:cc918c59e1c8ada346 -->
<a id="u-cc918c59e1c8ada346"></a>

Для pair i,j data gradient по $v_{i,f_j}$ равен $\delta x_ix_jv_{j,f_i}$; встречный gradient симметричен. При нескольких active categories в field contributions суммируются по соответствующим partners. Значения category ID и field ID должны иметь отдельные стабильные mappings: перестановка field indices без перестановки tensor weights изменит модель даже при неизменных category IDs.

<!-- /unit:cc918c59e1c8ada346 -->

<!-- unit:61315fa8859ccb236d -->
<a id="u-61315fa8859ccb236d"></a>

General FFM теряет простую общую square identity FM: у feature нет одного общего vector во всех pair products. Прямой sparse forward стоит O(m²d) по m active features. Для одного значения на field m≈F; при длинных multi-valued bags m может быть существенно больше F. Специализированная implementation может оптимизировать loops, но linear-in-m cost обычной FM нельзя обещать произвольной FFM.

<!-- /unit:61315fa8859ccb236d -->

<!-- unit:f2a3983eca346af35b -->
<a id="u-f2a3983eca346af35b"></a>

При K=1 000 000,F=20,d=8 полный tensor K×F×d содержит 160 миллионов floats, 640 MB decimal в float32 weights. Если same-field vectors не нужны, можно хранить K(F−1)d: 608 MB. Gradient и optimizer states способны увеличить memory в несколько раз. Sparse storage сокращает неиспользованные entries только если implementation действительно их не выделяет.

<!-- /unit:f2a3983eca346af35b -->

<!-- unit:432c64b03abd8068ab -->
<a id="u-432c64b03abd8068ab"></a>

Первичный источник [Field-aware Factorization Machines](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) различает shared и field-specific latent factors. Примеры, rank caveat и gradient checks здесь развивают механизм отдельно; они не являются воспроизведением авторского CTR benchmark или гарантией преимущества FFM над FM.

<!-- /unit:432c64b03abd8068ab -->

<!-- unit:20603e3753cc3a88bf -->
<a id="u-20603e3753cc3a88bf"></a>

## Новые пары и сравнение

<!-- /unit:20603e3753cc3a88bf -->

<!-- unit:5508c4e255f725c4f7 -->
<a id="u-5508c4e255f725c4f7"></a>

Rare interactions легко запоминаются. Нужны held-out groups/time, penalties и анализ unseen combinations. Если все categories известны, но их pair нова, это другой режим, чем completely new category. Полезно оценивать оба отдельно.

<!-- /unit:5508c4e255f725c4f7 -->

<!-- unit:10049afac69357cf50 -->
<a id="u-10049afac69357cf50"></a>

Сначала additive baseline, затем один cross либо FM/FFM с comparable tuning budget. Проверяется не только overall metric, но память, rare-pair quality и serialization mapping. Если improvement исчезает на group split, interaction мог memorise ID conjunctions.

<!-- /unit:10049afac69357cf50 -->

<!-- unit:35ca5deccbc9fcf433 -->
<a id="u-35ca5deccbc9fcf433"></a>

Известные city A и product P могли встречаться только с другими partners. FM может вычислить score новой пары из уже обученных vectors; качество зависит от подходящего low-rank prior и достаточного evidence. Если A совершенно новая и metadata отсутствует, её individual factor неизвестен. UNK/group fallback даёт определённый output, но не выученный effect A.

<!-- /unit:35ca5deccbc9fcf433 -->

<!-- unit:deb24cff67dc6d5bcb -->
<a id="u-deb24cff67dc6d5bcb"></a>

Полезно заранее разделить режимы evaluation: seen pair; unseen pair из seen categories; pair с одним новым компонентом; pair с двумя новыми компонентами. Group/time split выбирают под задачу, затем внутри его разрешённой training части определяют vocabularies и tuning. Глобальный список «seen» по всем данным может скрыть, что entity ещё не существовала к моменту prediction.

<!-- /unit:deb24cff67dc6d5bcb -->

<!-- unit:85f6d70db8954aa85f -->
<a id="u-85f6d70db8954aa85f"></a>

В sparse feedback, например clicks, отсутствие interaction не всегда является negative label: объект мог вообще не показываться пользователю. Negative sampling, exposure и logging policy меняют training distribution. Поэтому dot product, обученный на sampled negatives, не становится автоматически calibrated real CTR. Сначала задают observed outcome и доступные exposures, затем sampling/weights и evaluation; матричная factorization не решает эту постановку за исследователя.

<!-- /unit:85f6d70db8954aa85f -->

<!-- unit:1f18dcd25b3e89e351 -->
<a id="u-1f18dcd25b3e89e351"></a>

Начальная модель содержит main effects; затем добавляется один заранее обоснованный cross или factor branch. Проверяют consistent loss/link, одинаковые splits, metric, budget и fallback. Для regression score может быть direct prediction; для classification нужен link/loss, а не интерпретация произвольного dot product как probability. Larger d и FFM повышают capacity, но могут ухудшать rare slices и memory/latency.

<!-- /unit:1f18dcd25b3e89e351 -->

<!-- unit:39a293a46933b80c77 -->
<a id="u-39a293a46933b80c77"></a>

Для serialization сохраняют namespaces, normalization, category/field maps, active-value convention binary/count/weight, allowed interaction pairs, factors, main effects, bias, link и unknown rules. Round-trip input должен включать weighted sparse example и multi-valued case, если они поддержаны: ошибка x_i² в diagonal correction не проявится на одних binary values.

<!-- /unit:39a293a46933b80c77 -->

<!-- unit:0e1c5855e3dcf1f6fa -->
<a id="u-0e1c5855e3dcf1f6fa"></a>

**Ответы на расширенную практику:** XOR contradiction следует из равенства сумм additive scores; прямой и быстрый FM forward здесь оба дают 16; нулевые factors не учат interactions обычным gradient step; FFM добавляет field-specific sharing и расход памяти. Ни один из этих аналитических фактов не заменяет реальный benchmark на выбранном deployment split.

<!-- /unit:0e1c5855e3dcf1f6fa -->

<!-- unit:6d6e95a88e9a140a72 -->
<a id="u-6d6e95a88e9a140a72"></a>

## Контроль и практика

<!-- /unit:6d6e95a88e9a140a72 -->

<!-- unit:b842ca76761ea03a5b -->
<a id="u-b842ca76761ea03a5b"></a>

1. Что теряет additive model? **Ответ:** effect одного field не может зависеть от другого без interaction.
2. Почему factorization помогает rare pairs? **Ответ:** pair coefficients делят latent parameters с другими combinations; это prior sharing, а не новая информация.
3. Чем FFM отличается от FM? **Ответ:** vector feature зависит ещё и от field взаимодействующего партнёра.

<!-- /unit:b842ca76761ea03a5b -->

<!-- unit:a058d24ec46e52fe0c -->
<a id="u-a058d24ec46e52fe0c"></a>

**Практика:** покажите невозможность XOR для additive score через четыре inequalities. Посчитайте FM interaction напрямую и через square identity для трёх active features. Спроектируйте test known-pair/new-pair/new-category и сравните manual cross с factorization без mixing splits.

<!-- /unit:a058d24ec46e52fe0c -->

<!-- unit:80fd6550c23537680f -->
<a id="u-80fd6550c23537680f"></a>

**Схема курса:** fields/categories → explicit cross or factorized embedding interactions → pairwise score contributions → prediction → compare capacity/collision/generalization

<!-- /unit:80fd6550c23537680f -->

<!-- unit:177b2b31043107eaad -->
<a id="u-177b2b31043107eaad"></a>

<!-- source-depth-categorical-Е -->

<!-- /unit:177b2b31043107eaad -->

## Результат главы и переход дальше

Ответьте на вопрос: **Когда эффект признака зависит от другого поля и как это представить?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
