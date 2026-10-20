# Кардинальность, частотные кодировки и hashing

**Уровень:** общее ядро.

**Предпосылки:** [Nominal, ordinal, one-hot и dummy encoding](<feat-categorical.md>), [Split, cross-validation и утечки](<../01-reliability/df-split.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как ограничить размер представления и измерить цену столкновений?





## Карта вычисления и границы этапов

![Кардинальность, частотные кодировки и hashing: входы, преобразования и выходы](<../../оформление/course/feat-hashing.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Обучаемые статистики вычисляются на train; validation и новые объекты используют сохранённое состояние. Геометрическая формула с фиксированными константами не превращается в fit только из-за наличия блока на схеме.

<!-- unit:e38cd026d3def95049 -->
<a id="u-e38cd026d3def95049"></a>

## Размерность и статистическая сложность

<!-- /unit:e38cd026d3def95049 -->

<!-- unit:b5217c9a57af7561f3 -->
<a id="u-b5217c9a57af7561f3"></a>

Уникальный ID даёт одной category один training example. Даже если sparse matrix помещается в RAM, оценить её отдельный effect надёжно невозможно. Поэтому high cardinality создаёт одновременно computational и statistical задачи. Редкие/новые categories требуют fallback и regularization, а не только более компактного storage.

<!-- /unit:b5217c9a57af7561f3 -->

<!-- unit:5107f4d4a07246ec2c -->
<a id="u-5107f4d4a07246ec2c"></a>

Для $n=100\,000,K=10\,000$ dense float32 one-hot требует $4nK=4\cdot10^9$ bytes, то есть 4 GB decimal без overhead. Если на строку один nonzero, CSR с float32 values и int32 indices/pointers хранит примерно:

<!-- /unit:5107f4d4a07246ec2c -->

<!-- unit:2ff028e6b126d20036 -->
<a id="u-2ff028e6b126d20036"></a>

$$
4n+4n+4(n+1)\approx1.2\cdot10^6\text{ bytes}.
$$

<!-- /unit:2ff028e6b126d20036 -->

<!-- unit:ccd0d5e0a695007c11 -->
<a id="u-ccd0d5e0a695007c11"></a>

Реальный объект, metadata, несколько fields и преобразования добавляют расходы. Sparse one-hot не обязательно «ломается» при тысячах categories; trade-off зависит от estimator и sparsity. Но большое число coefficients и слабая статистика rare values сохраняются.

<!-- /unit:ccd0d5e0a695007c11 -->

<!-- unit:e20e5d4e97a313c344 -->
<a id="u-e20e5d4e97a313c344"></a>

### Storage и статистическая надёжность

<!-- /unit:e20e5d4e97a313c344 -->

<!-- unit:9fe079dba3c35bffe7 -->
<a id="u-9fe079dba3c35bffe7"></a>

Cardinality K — число различных значений field в объявленном наборе, а не число наблюдений n. Одинаковый K может соответствовать тысячам повторений каждой category либо длинному хвосту значений с одним примером. Память зависит от representation, а оценка индивидуальных effects — от числа и разнообразия observations каждой category.

<!-- /unit:9fe079dba3c35bffe7 -->

<!-- unit:0686cb13def2f61b9e -->
<a id="u-0686cb13def2f61b9e"></a>

Unique ID позволяет запомнить train rows, но его наличие само по себе не доказывает утечку. Для повторного известного пользователя история может быть законным сигналом, если доступна на момент решения. Для переноса на новых пользователей индивидуальный coefficient не переносится на unseen ID. Split должен соответствовать этой цели. Поле, сформированное после target, является отдельной проблемой доступности, которую компактное кодирование не исправляет.

<!-- /unit:0686cb13def2f61b9e -->

<!-- unit:19971d961ae52a914d -->
<a id="u-19971d961ae52a914d"></a>

CSR решает часть storage задачи, но не добавляет данных редкой category. Hashing уменьшает число параметров через sharing, frequency стирает identity, rare grouping объединяет оценки, embeddings задают другое sharing через дальнейшую модель. Ни один из этих механизмов не равен появлению новых независимых observations.

<!-- /unit:19971d961ae52a914d -->

<!-- unit:31aea1173d34a750de -->
<a id="u-31aea1173d34a750de"></a>

### Память sparse и dense по компонентам

<!-- /unit:31aea1173d34a750de -->

<!-- unit:79d8356ff5d0025dd5 -->
<a id="u-79d8356ff5d0025dd5"></a>

Для n=100 000 и K=10 000 dense float32 array содержит миллиард чисел: 4 000 000 000 bytes, то есть 4 GB decimal или около 3.725 GiB. Для single-valued one-hot с одним active element на строку CSR хранит n values, n column indices и n+1 row pointers. При 4 bytes на каждый компонент это 1 200 004 bytes без object overhead и vocabulary.

<!-- /unit:79d8356ff5d0025dd5 -->

<!-- unit:0054df594366b4dba3 -->
<a id="u-0054df594366b4dba3"></a>

Если index/pointer storage использует 8 bytes, а values остаются float32, оценка станет $4n+8n+8(n+1)=2\,000\,008$ bytes. Нельзя сравнивать эти форматы только по dtype values. Vocabulary strings, metadata, optimizer state, model coefficients и временные преобразования считаются отдельно.

<!-- /unit:0054df594366b4dba3 -->

<!-- unit:e86a599cb827bf32bb -->
<a id="u-e86a599cb827bf32bb"></a>

Для 20 fields по 10 000 categories concatenated width равна 200 000, а nnz на строку — 20, если каждое field имеет одно активное значение. Dense float32 займёт 80 GB decimal. CSR с 4-byte values/indices/pointers — $8\cdot2\,000\,000+4\cdot100\,001=16\,400\,004$ bytes. Высокая width сама по себе не означает, что input array обязан быть dense.

<!-- /unit:e86a599cb827bf32bb -->

<!-- unit:59c1e12569ee43a67c -->
<a id="u-59c1e12569ee43a67c"></a>

Но downstream stage может превратить sparse в dense, например при математическом центрировании всех columns. Модель также хранит свои coefficients независимо от sparse input; для multiclass head их число растёт с числом классов. Fit time и predict time зависят от операций estimator, а не только от bytes X. Поэтому сравнивают полный pipeline, включая peak memory и временные buffers.

<!-- /unit:59c1e12569ee43a67c -->

<!-- unit:ee15ace1f69e2147e8 -->
<a id="u-ee15ace1f69e2147e8"></a>

## Count и frequency encoding

<!-- /unit:ee15ace1f69e2147e8 -->

<!-- unit:6e1da13aa04ab58898 -->
<a id="u-6e1da13aa04ab58898"></a>

Count mapping $c\mapsto n_c$ и frequency mapping $c\mapsto n_c/n$ не используют target. Они отражают распространённость category, что может быть meaningful signal. Но A и B с одинаковой частотой получат одно число и потеряют identity. Такие representations не равны полной замене one-hot без потери информации.

<!-- /unit:6e1da13aa04ab58898 -->

<!-- unit:40c9fbf411285ae4f0 -->
<a id="u-40c9fbf411285ae4f0"></a>

Пример train A,A,B,B,C: counts A/B/C равны 2/2/1, frequencies 0.4/0.4/0.2. Новый D может получить 0 либо smoothed prior по объявленному contract. Frequency только по train; будущая popularity недоступна в историческом решении. Если IDs приходят из нескольких namespaces, нужен namespace prefix, иначе одинаковые строки из разных fields сольются.

<!-- /unit:40c9fbf411285ae4f0 -->

<!-- unit:8b0997be280a4e1b92 -->
<a id="u-8b0997be280a4e1b92"></a>

### Какие вопросы отвечает частота

<!-- /unit:8b0997be280a4e1b92 -->

<!-- unit:0d0f02380a9634ef82 -->
<a id="u-0d0f02380a9634ef82"></a>

Для train A,A,B,B,C counts равны 2,2,1, frequencies — 0.4,0.4,0.2. Если model получает только эту числовую feature, A и B неразличимы по ней. Она может выучить зависимость от popularity, но не произвольный отдельный effect A и B. Frequency=0.4 не означает probability положительного target 0.4: статистика вообще не использовала y.

<!-- /unit:0d0f02380a9634ef82 -->

<!-- unit:fd0bfc8f3840e836a3 -->
<a id="u-fd0bfc8f3840e836a3"></a>

Count зависит от размера окна и объёма корпуса. Frequency нормирована на объявленный denominator, но тоже зависит от окна и sampling. Значения по всем историческим событиям, последним семи дням и уникальным пользователям дают разные признаки. Log-count может сжать scale, однако не возвращает identity одинаково частых categories.

<!-- /unit:fd0bfc8f3840e836a3 -->

<!-- unit:015145c47f5c36a267 -->
<a id="u-015145c47f5c36a267"></a>

### Smoothing требует конечного пространства состояний

<!-- /unit:015145c47f5c36a267 -->

<!-- unit:9e277b70ff5771f777 -->
<a id="u-9e277b70ff5771f777"></a>

Введём три известных categories и общий UNKNOWN bucket, всего четыре состояния. При alpha=1 Laplace smoothing для пяти train rows даёт probabilities A=3/9, B=3/9, C=2/9, UNKNOWN=1/9. Сумма равна 1. Новые D и E относятся к одному UNKNOWN состоянию; нельзя каждому из бесконечного множества возможных новых строк независимо обещать ту же отдельную probability 1/9.

<!-- /unit:9e277b70ff5771f777 -->

<!-- unit:54b32804b53056d11c -->
<a id="u-54b32804b53056d11c"></a>

Это smoothing входной частоты, не target encoding. Его prior, denominator и способ обработки missing фиксируются отдельно. Если count=0 используется как fallback без smoothing, модель должна уметь работать в этой области feature space; такая величина может быть вне фактически наблюдавшихся train frequencies. Наличие определённого числа не гарантирует удачного extrapolation.

<!-- /unit:54b32804b53056d11c -->

<!-- unit:9e18d8a30d71398f94 -->
<a id="u-9e18d8a30d71398f94"></a>

### Почему target-free статистика тоже может смотреть в будущее

<!-- /unit:9e18d8a30d71398f94 -->

<!-- unit:58e4c983ae8a3626e0 -->
<a id="u-58e4c983ae8a3626e0"></a>

Если к пяти train rows добавить 100 будущих D, frequency A изменится с 2/5=0.4 до 2/105≈0.01905. Train labels не менялись, но representation уже использовало будущий состав популяции. Для исторического решения такая информация недоступна, если протокол не предусматривает иной transductive режим.

<!-- /unit:58e4c983ae8a3626e0 -->

<!-- unit:b1efb77f6664572001 -->
<a id="u-b1efb77f6664572001"></a>

В CV counts fit на train-fold и применяют к его validation. Финальный mapping для deployment строят на разрешённом training snapshot и сохраняют. Online обновления имеют отдельное правило времени и доступности; это не просто вызов transform, который произвольно пересчитывает frequencies на текущем batch. Более полный temporal protocol разобран в главе [«Временные признаки и доступность метки»](<feat-time.md>).

<!-- /unit:b1efb77f6664572001 -->

<!-- unit:175b19f194a0241159 -->
<a id="u-175b19f194a0241159"></a>

## Signed hashing

<!-- /unit:175b19f194a0241159 -->

<!-- unit:5d2f21e90ef1662502 -->
<a id="u-5d2f21e90ef1662502"></a>

Hashing строит fixed-dimensional vector без хранения полного vocabulary. Category попадает в bucket $h(c)\in\{0,\ldots,m-1\}$. Для набора active features signed hashing использует независимый sign $s(c)\in\{-1,1\}$:

<!-- /unit:5d2f21e90ef1662502 -->

<!-- unit:d32bd9c65a0c71a431 -->
<a id="u-d32bd9c65a0c71a431"></a>

$$
z_j=\sum_{c:h(c)=j}s(c)x_c.
$$

<!-- /unit:d32bd9c65a0c71a431 -->

<!-- unit:b903bff718ea86e116 -->
<a id="u-b903bff718ea86e116"></a>

Новые categories обрабатываются той же функцией, но collision делит их parameters. Signed hash помогает сделать некоторые inner-product distortions unbiased при assumptions hash family; он не устраняет collisions и не гарантирует сохранение каждого индивидуального prediction.

<!-- /unit:b903bff718ea86e116 -->

<!-- unit:b44294988a78355ed5 -->
<a id="u-b44294988a78355ed5"></a>

При независимом uniform hashing $K$ categories ожидаемое число colliding pairs:

<!-- /unit:b44294988a78355ed5 -->

<!-- unit:97c7c2e14c68751a5a -->
<a id="u-97c7c2e14c68751a5a"></a>

$$
\mathbb E[N_{\mathrm{pairs}}]=\frac{K(K-1)}{2m}.
$$

<!-- /unit:97c7c2e14c68751a5a -->

<!-- unit:726fb050cbc4d26695 -->
<a id="u-726fb050cbc4d26695"></a>

Для $K=100,m=1000$ это 4.95 пары. Это не вероятность хотя бы одной collision и не число потерянных categories: несколько pairs могут относиться к одному bucket. Важна также frequency/target конфликтующих values, а не только их count.

<!-- /unit:726fb050cbc4d26695 -->

<!-- unit:702be5359e68245f12 -->
<a id="u-702be5359e68245f12"></a>

Stable hash/config должны совпадать train/inference. Некоторые language built-in hashes зависят от process seed; их нельзя бездумно использовать как сохраняемый feature contract. В отчёте фиксируют implementation, seed, размер buckets и namespace separators.

<!-- /unit:702be5359e68245f12 -->

<!-- unit:f9b21ccd39c17958bd -->
<a id="u-f9b21ccd39c17958bd"></a>

### Key, bucket и sign — три разных соглашения

<!-- /unit:f9b21ccd39c17958bd -->

<!-- unit:58bdefd8bb1ad11ffc -->
<a id="u-58bdefd8bb1ad11ffc"></a>

Сначала из field и значения получают стабильный key, затем bucket index h(key), при signed hashing — ещё sign s(key). Namespaces нужны до hashing: одинаковая строка A в city и device не должна заранее считаться одним исходным признаком. Но после namespacing настоящая hash collision всё ещё возможна.

<!-- /unit:58bdefd8bb1ad11ffc -->

<!-- unit:4d8e61e5571e9af302 -->
<a id="u-4d8e61e5571e9af302"></a>

Конкатенация без разделения неоднозначна: пары (ab,c) и (a,bc) обе дают строку abc. Используют недвусмысленную сериализацию, например escaping или length-prefixed компоненты. Канонизация регистра/Unicode тоже меняет key и потому требует версии. Если spelling variants нужно объединять, это должно быть осознанным правилом, а не случайным совпадением строки.

<!-- /unit:4d8e61e5571e9af302 -->

<!-- unit:91615d8207e130e87d -->
<a id="u-91615d8207e130e87d"></a>

У готового [FeatureHasher](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html) есть конкретные input modes и правила знака. Для собственного contract сохраняют algorithm/seed, размерность, namespaces, сериализацию и summation дубликатов. Произвольно выбранный process-dependent built-in hash не следует считать стабильным словарём между запусками.

<!-- /unit:91615d8207e130e87d -->

<!-- unit:3e1a7fb184429f802a -->
<a id="u-3e1a7fb184429f802a"></a>

### Как collision связывает coefficients

<!-- /unit:3e1a7fb184429f802a -->

<!-- unit:9e1a5ea54e6e2124c6 -->
<a id="u-9e1a5ea54e6e2124c6"></a>

Для исходной sparse feature vector x и hashed z линейный predictor имеет вид $v^\top z=\sum_c s(c)v_{h(c)}x_c$. Следовательно, effective coefficient исходного c равен $s(c)v_{h(c)}$. Два keys одного bucket делят один обучаемый параметр: при одинаковых signs их effects одинаковы, при противоположных — связаны знаком. Это ограничение представления, а не сохранение независимых category parameters.

<!-- /unit:9e1a5ea54e6e2124c6 -->

<!-- unit:cfc8659f018c213936 -->
<a id="u-cfc8659f018c213936"></a>

Для bag из двух активных keys A и B, попавших в один bucket с signs +1 и −1, их вклады могут взаимно уничтожиться. Тогда bag A+B имеет тот же hashed vector, что пустой bag, хотя исходные vectors различаются. Это пример информационной потери даже при signed hashing. Для single-valued field одновременно активна только одна category; pattern потери будет другим.

<!-- /unit:cfc8659f018c213936 -->

<!-- unit:e33e3505534c59a9c1 -->
<a id="u-e33e3505534c59a9c1"></a>

### Что именно сохраняется в ожидании

<!-- /unit:e33e3505534c59a9c1 -->

<!-- unit:cf47fe8dafbb128478 -->
<a id="u-cf47fe8dafbb128478"></a>

Раскроем inner product:

<!-- /unit:cf47fe8dafbb128478 -->

<!-- unit:2e5d10b6b20a0b1e2c -->
<a id="u-2e5d10b6b20a0b1e2c"></a>

$$
\langle\phi(x),\phi(y)\rangle=
\sum_i x_iy_i+
\sum_{i\ne j}\mathbf1[h(i)=h(j)]s(i)s(j)x_iy_j.
$$

<!-- /unit:2e5d10b6b20a0b1e2c -->

<!-- unit:29892f71384f10e9e6 -->
<a id="u-29892f71384f10e9e6"></a>

При подходящей независимой hash/sign family среднее cross terms равно нулю, и expected inner product совпадает с исходным. Для разных basis features x=eA,y=eB истинный inner product=0. При uniform bucket hashing вероятность collision равна 1/m; signed hashed dot тогда равен +1 или −1 с равными вероятностями, иначе 0. Среднее 0, variance=1/m. Без sign hashed dot для этих неотрицательных features имеет среднее 1/m.

<!-- /unit:29892f71384f10e9e6 -->

<!-- unit:29c2e4cafb959f8f64 -->
<a id="u-29c2e4cafb959f8f64"></a>

Выбранное однажды отображение может дать ошибку +1 на конкретной паре. Поэтому unbiased expectation по случайным отображениям не является гарантией каждого prediction фиксированной trained model. Ошибки также зависят от частоты и target-конфликта collided keys. Теоретическая постановка изложена в [Feature Hashing for Large Scale Multitask Learning](https://arxiv.org/abs/0902.2206).

<!-- /unit:29c2e4cafb959f8f64 -->

<!-- unit:128f68ffd282fca089 -->
<a id="u-128f68ffd282fca089"></a>

## Столкновения и multi-hash

<!-- /unit:128f68ffd282fca089 -->

<!-- unit:87d1168413e6e73277 -->
<a id="u-87d1168413e6e73277"></a>

Можно задать несколько независимых bucket mappings и объединить blocks vectors. Вероятность совпадения всех $k$ bucket IDs одной пары при идеальной независимости равна $m^{-k}$. Но dimension/storage увеличиваются, а отдельные collisions остаются. Комбинирование concatenation, sum и learned pooling даёт разные representations.

<!-- /unit:87d1168413e6e73277 -->

<!-- unit:34a518bd14fcbf3300 -->
<a id="u-34a518bd14fcbf3300"></a>

Bloom filter отвечает на membership query с false positives и обычно не хранит обучаемые signed feature values. Это не стандартная «улучшенная one-hot» для любого classifier. Multi-hash encoding использует похожую идею нескольких отображений, но statistical learning и membership — разные задачи. Важен фактический objective representation, а не аналогия названий.

<!-- /unit:34a518bd14fcbf3300 -->

<!-- unit:006a2a237cac74e56a -->
<a id="u-006a2a237cac74e56a"></a>

### Пары, занятые buckets и вероятность события

<!-- /unit:006a2a237cac74e56a -->

<!-- unit:abda97fb44501ead31 -->
<a id="u-abda97fb44501ead31"></a>

Expected число colliding pairs равно K(K−1)/(2m), поскольку каждая из K(K−1)/2 пар сталкивается с вероятностью 1/m. Expected число занятых buckets равно $m[1-(1-1/m)^K]$, а surplus keys сверх первого в каждом bucket — K минус это число. Это разные статистики.

<!-- /unit:abda97fb44501ead31 -->

<!-- unit:69582ef6f82c229380 -->
<a id="u-69582ef6f82c229380"></a>

Для K=3,m=2 collision неизбежна. Expected pairs=1.5, occupied buckets=1.75, surplus keys=1.25. Если все три попали в один bucket, в нём три пары, но лишь два лишних относительно первого key. Поэтому число pairs нельзя читать как число «потерянных категорий».

<!-- /unit:69582ef6f82c229380 -->

<!-- unit:722fccb9b95ea08d21 -->
<a id="u-722fccb9b95ea08d21"></a>

При K≤m вероятность отсутствия collision равна $\prod_{j=0}^{K-1}(1-j/m)$; при K>m она нулевая. Для K=100,m=1000 expected pairs=4.95, но probability хотя бы одной collision близка к 99.4%. Это не означает, что 99.4% всех categories разрушены: событие касается наличия хотя бы одной пары. Значимость отдельных collisions исследуют по их реальному влиянию.

<!-- /unit:722fccb9b95ea08d21 -->

<!-- unit:6dc6af0bd7afb237bf -->
<a id="u-6dc6af0bd7afb237bf"></a>

### Несколько hashes и одинаковый memory budget

<!-- /unit:6dc6af0bd7afb237bf -->

<!-- unit:25d3ce2126240a6108 -->
<a id="u-25d3ce2126240a6108"></a>

Можно concatenation k независимых hash blocks по m buckets. Вероятность совпадения bucket IDs одной пары во всех blocks равна $m^{-k}$ при идеальной независимости. Но partial collisions не исчезают, а общая dimension становится km. Если blocks масштабировать на $1/\sqrt{k}$, inner product объединённого vector станет средним k hashed inner products с сохранением expected norm.

<!-- /unit:25d3ce2126240a6108 -->

<!-- unit:d8d6981b3452e9618d -->
<a id="u-d8d6981b3452e9618d"></a>

Для пары разных basis features с независимыми signed blocks variance такого среднего равна $1/(km)$. При фиксированной общей dimension D=km это 1/D, как у одного идеального signed hash размера D в этом простом примере. Следовательно, меньшая вероятность полного совпадения signature не доказывает автоматически меньшую inner-product variance при той же памяти.

<!-- /unit:d8d6981b3452e9618d -->

<!-- unit:10b426f11727882d80 -->
<a id="u-10b426f11727882d80"></a>

Например, k=2,m=500 даёт D=1000 и probability полного bucket совпадения 1/250000, тогда как отдельные partial collisions остаются. С одним block m=1000 collision probability=1/1000. Эти схемы по-разному распределяют ошибки; выбор concatenation, summation или learned combination дополнительно меняет representation.

<!-- /unit:10b426f11727882d80 -->

<!-- unit:ed324d518075174f31 -->
<a id="u-ed324d518075174f31"></a>

Bloom filter решает membership задачу и имеет собственные правила false positives. Его несколько hashes не являются готовой универсальной заменой one-hot в supervised learning. Аналогия полезна для понимания отображений, но качество classifier определяется другим objective и способом использования collisions.

<!-- /unit:ed324d518075174f31 -->

<!-- unit:7393e3f6b4730df358 -->
<a id="u-7393e3f6b4730df358"></a>

## Контроль и практика

<!-- /unit:7393e3f6b4730df358 -->

<!-- unit:06b6a77cfcd919a60b -->
<a id="u-06b6a77cfcd919a60b"></a>

Сравнить sparse one-hot, frequency и hash нескольких размеров при одном group/time split. Отдельно измерить rare/unknown cases, память, fit/predict time и collision profile. Hash buckets не дают интерпретировать coefficient как effect одной category. Frequency может работать быстро, но потерять полезный distinction между equally frequent classes.

<!-- /unit:06b6a77cfcd919a60b -->

<!-- unit:24e23bc1b0a5319edd -->
<a id="u-24e23bc1b0a5319edd"></a>

1. Решает ли CSR проблему редких coefficients? **Ответ:** он снижает storage, но не добавляет observations для оценки effect.
2. Сохраняет ли frequency identity categories? **Ответ:** нет; одинаковые counts дают одинаковый encoding.
3. Равно ли expected число colliding pairs вероятности collision? **Ответ:** нет; это разные statistics.

<!-- /unit:24e23bc1b0a5319edd -->

<!-- unit:f5e54f18b63234123b -->
<a id="u-f5e54f18b63234123b"></a>

**Практика:** распределите десять categories по четырём buckets двумя hash mappings. Найдите collisions и конфликтующие labels. Посчитайте dense/CSR memory для одного field, затем объясните, какие расходы не вошли в расчёт. Предложите fallback для unknown и проверку стабильности hash после restart.

<!-- /unit:f5e54f18b63234123b -->

<!-- unit:909ba36b8d592cced9 -->
<a id="u-909ba36b8d592cced9"></a>

**Схема курса:** category + train counts → rare/frequency bucket or stable hash → bounded representation + collision/unknown metadata → model input and slice checks

<!-- /unit:909ba36b8d592cced9 -->

<!-- unit:6e5fd48d1f171454af -->
<a id="u-6e5fd48d1f171454af"></a>

<!-- source-depth-categorical-Б -->

<!-- /unit:6e5fd48d1f171454af -->

<!-- unit:649adc6b4dd4f77615 -->
<a id="u-649adc6b4dd4f77615"></a>

Сравнивают sparse one-hot, frequency и несколько hash widths на одном group/time split. В журнале отдельно указывают input storage, vocabulary/model state, peak memory, fit/predict time и качество известных/редких/новых categories. Counterfactual изменение encoding-size не должно одновременно менять split или доступность будущих counts. Для hash нет прямой однозначной обратной ссылки «этот coefficient принадлежит одной category».

<!-- /unit:649adc6b4dd4f77615 -->

<!-- unit:32014de348948b5e49 -->
<a id="u-32014de348948b5e49"></a>

1. **Почему компактный CSR не решает проблему rare levels?** Он экономит хранение тех же наблюдений, но не увеличивает информацию для оценки effects.
2. **Почему frequency может потерять сильный сигнал?** Категории с одинаковой popularity становятся неразличимыми по этому признаку.
3. **Что гарантирует signed hash в простом expectation-расчёте?** Среднее inner product по допустимой случайной family, а не сохранность каждого фиксированного prediction.
4. **Почему вероятность полной multi-hash collision недостаточна для сравнения методов?** Остаются partial collisions, меняются dimension и структура ошибок; фиксированный memory budget задаёт другой вопрос.

<!-- /unit:32014de348948b5e49 -->

<!-- unit:cac11f34a108123128 -->
<a id="u-cac11f34a108123128"></a>

Практика: воспроизвести dense/CSR подсчёт, finite-state smoothed frequencies и небольшой signed hash с взаимным уничтожением features. Полностью перечислить размещения трёх keys в двух buckets и сравнить pairs/occupied/surplus. Затем проверить стабильность mapping после restart. Все mappings, counts и memory numbers главы учебные; benchmark пользовательского pipeline не выполнялся.

<!-- /unit:cac11f34a108123128 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как ограничить размер представления и измерить цену столкновений?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
