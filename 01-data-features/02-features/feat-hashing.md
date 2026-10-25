# Кардинальность, частотные кодировки и hashing

**Уровень:** общее ядро.

**Предпосылки:** [Nominal, ordinal, one-hot и dummy encoding](<feat-categorical.md>), [Split, cross-validation и утечки](<../01-reliability/df-split.md>)

**Используют:** Завершение выбранной ветки: самостоятельный эксперимент по практике этой главы.

**Учебный вопрос:** Как ограничить размер представления и измерить цену столкновений?

## Карта вычисления и границы этапов

![Кардинальность, частотные кодировки и hashing: входы, преобразования и выходы](<../../оформление/course/feat-hashing.svg>)

Хеширование ограничивает размер представления ценой коллизий. Два разных ключа могут обновлять одну координату; размер таблицы выбирают по измеренному компромиссу памяти и качества.

## Размерность и статистическая сложность

Уникальный ID даёт одной category один training example. Даже если sparse matrix помещается в RAM, оценить её отдельный effect надёжно невозможно. Поэтому high cardinality создаёт одновременно computational и statistical задачи. Редкие/новые categories требуют fallback и regularization, а не только более компактного storage.

Для $n=100\,000,K=10\,000$ dense float32 one-hot требует $4nK=4\cdot10^9$ bytes, то есть 4 GB decimal без overhead. Если на строку один nonzero, CSR с float32 values и int32 indices/pointers хранит примерно:

$$
4n+4n+4(n+1)\approx1.2\cdot10^6\text{ bytes}.
$$

Реальный объект, metadata, несколько fields и преобразования добавляют расходы. Sparse one-hot не обязательно «ломается» при тысячах categories; trade-off зависит от estimator и sparsity (разреженность, доля нулевых или исключённых элементов). Но большое число coefficients и слабая статистика rare values сохраняются.

### Storage и статистическая надёжность

Cardinality K — число различных значений field в объявленном наборе, а не число наблюдений n. Одинаковый K может соответствовать тысячам повторений каждой category либо длинному хвосту значений с одним примером. Память зависит от representation, а оценка индивидуальных effects — от числа и разнообразия observations каждой category.

Unique ID позволяет запомнить train rows, но его наличие само по себе не доказывает утечку. Для повторного известного пользователя история может быть законным сигналом, если доступна на момент решения. Для переноса на новых пользователей индивидуальный coefficient не переносится на unseen ID. Split (разбиение данных) должен соответствовать этой цели. Поле, сформированное после target, является отдельной проблемой доступности, которую компактное кодирование не исправляет.

CSR решает часть storage задачи, но не добавляет данных редкой category. Hashing уменьшает число параметров через sharing, frequency стирает identity, rare grouping объединяет оценки, embeddings (векторные представления) задают другое sharing через дальнейшую модель. Ни один из этих механизмов не равен появлению новых независимых observations.

### Память sparse и dense по компонентам

Для n=100 000 и K=10 000 dense float32 array содержит миллиард чисел: 4 000 000 000 bytes, то есть 4 GB decimal или около 3.725 GiB. Для single-valued one-hot с одним active element на строку CSR хранит n values, n column indices и n+1 row pointers. При 4 bytes на каждый компонент это 1 200 004 bytes без object overhead и vocabulary.

Если index/pointer storage использует 8 bytes, а values остаются float32, оценка станет $4n+8n+8(n+1)=2\,000\,008$ bytes. Нельзя сравнивать эти форматы только по dtype values. Vocabulary strings, metadata, optimizer (оптимизатор обновления параметров) state (состояние системы), model coefficients и временные преобразования считаются отдельно.

Для 20 fields по 10 000 categories concatenated width равна 200 000, а nnz на строку — 20, если каждое field имеет одно активное значение. Dense float32 займёт 80 GB decimal. CSR с 4-byte values/indices/pointers — $8\cdot2\,000\,000+4\cdot100\,001=16\,400\,004$ bytes. Высокая width сама по себе не означает, что input array обязан быть dense.

Но downstream stage может превратить sparse в dense, например при математическом центрировании всех columns. Модель также хранит свои coefficients независимо от sparse input; для multiclass head их число растёт с числом классов. Fit time и predict time зависят от операций estimator, а не только от bytes X. Поэтому сравнивают полный pipeline, включая peak memory и временные buffers.

## Count и frequency encoding

Count mapping $c\mapsto n_c$ и frequency mapping $c\mapsto n_c/n$ не используют target. Они отражают распространённость category, что может быть meaningful signal. Но A и B с одинаковой частотой получат одно число и потеряют identity. Такие representations не равны полной замене one-hot без потери информации.

Пример train A,A,B,B,C: counts A/B/C равны 2/2/1, frequencies 0.4/0.4/0.2. Новый D может получить 0 либо smoothed prior по объявленному contract. Frequency только по train; будущая popularity недоступна в историческом решении. Если IDs приходят из нескольких namespaces, нужен namespace prefix, иначе одинаковые строки из разных fields сольются.

### Какие вопросы отвечает частота

Для train A,A,B,B,C counts равны 2,2,1, frequencies — 0.4,0.4,0.2. Если model получает только эту числовую feature (признак), A и B неразличимы по ней. Она может выучить зависимость от popularity, но не произвольный отдельный effect A и B. Frequency=0.4 не означает probability положительного target 0.4: статистика вообще не использовала y.

Count зависит от размера окна и объёма корпуса. Frequency нормирована на объявленный denominator, но тоже зависит от окна и sampling (случайный выбор по распределению). Значения по всем историческим событиям, последним семи дням и уникальным пользователям дают разные признаки. Log-count может сжать scale, однако не возвращает identity одинаково частых categories.

### Smoothing требует конечного пространства состояний

Введём три известных categories и общий UNKNOWN bucket, всего четыре состояния. При alpha=1 Laplace smoothing для пяти train rows даёт probabilities A=3/9, B=3/9, C=2/9, UNKNOWN=1/9. Сумма равна 1. Новые D и E относятся к одному UNKNOWN состоянию; нельзя каждому из бесконечного множества возможных новых строк независимо обещать ту же отдельную probability 1/9.

Это smoothing входной частоты, не target encoding. Его prior, denominator и способ обработки missing фиксируются отдельно. Если count=0 используется как fallback без smoothing, модель должна уметь работать в этой области feature space; такая величина может быть вне фактически наблюдавшихся train frequencies. Наличие определённого числа не гарантирует удачного extrapolation.

### Почему target-free статистика тоже может смотреть в будущее

Если к пяти train rows добавить 100 будущих D, frequency A изменится с 2/5=0.4 до 2/105≈0.01905. Train labels (метки) не менялись, но representation уже использовало будущий состав популяции. Для исторического решения такая информация недоступна, если протокол не предусматривает иной transductive режим.

В CV counts fit на train-fold и применяют к его validation (проверочная часть для выбора решения). Финальный mapping для deployment строят на разрешённом training snapshot и сохраняют. Online обновления имеют отдельное правило времени и доступности; это не просто вызов transform, который произвольно пересчитывает frequencies на текущем batch (пакет примеров). Более полный temporal protocol разобран в главе [«Временные признаки и доступность метки»](<feat-time.md>).

## Signed hashing

Hashing строит fixed-dimensional vector без хранения полного vocabulary. Category попадает в bucket $h(c)\in\{0,\ldots,m-1\}$. Для набора active features (признаки) signed hashing использует независимый sign $s(c)\in\{-1,1\}$:

$$
z_j=\sum_{c:h(c)=j}s(c)x_c.
$$

Новые categories обрабатываются той же функцией, но collision делит их parameters (параметры). Signed hash помогает сделать некоторые inner-product distortions unbiased при assumptions hash family; он не устраняет collisions и не гарантирует сохранение каждого индивидуального prediction.

При независимом uniform hashing $K$ categories ожидаемое число colliding pairs:

$$
\mathbb E[N_{\mathrm{pairs}}]=\frac{K(K-1)}{2m}.
$$

Для $K=100,m=1000$ это 4.95 пары. Это не вероятность хотя бы одной collision и не число потерянных categories: несколько pairs могут относиться к одному bucket. Важна также frequency/target конфликтующих values, а не только их count.

Stable hash/config должны совпадать train/inference (применение обученной модели). Некоторые language built-in hashes зависят от process seed (начальное состояние генератора случайных чисел); их нельзя бездумно использовать как сохраняемый feature contract. В отчёте фиксируют implementation, seed, размер buckets и namespace separators.

### Key, bucket и sign — три разных соглашения

Сначала из field и значения получают стабильный key, затем bucket index h(key), при signed hashing — ещё sign s(key). Namespaces нужны до hashing: одинаковая строка A в city и device не должна заранее считаться одним исходным признаком. Но после namespacing настоящая hash collision всё ещё возможна.

Конкатенация без разделения неоднозначна: пары (ab,c) и (a,bc) обе дают строку abc. Используют недвусмысленную сериализацию, например escaping или length-prefixed компоненты. Канонизация регистра/Unicode тоже меняет key и потому требует версии. Если spelling variants нужно объединять, это должно быть осознанным правилом, а не случайным совпадением строки.

У готового [FeatureHasher](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html) есть конкретные input modes и правила знака. Для собственного contract сохраняют algorithm/seed, размерность, namespaces, сериализацию и summation дубликатов. Произвольно выбранный process-dependent built-in hash не следует считать стабильным словарём между запусками.

### Как collision связывает coefficients

Для исходной sparse feature vector x и hashed z линейный predictor имеет вид $v^\top z=\sum_c s(c)v_{h(c)}x_c$. Следовательно, effective coefficient исходного c равен $s(c)v_{h(c)}$. Два keys одного bucket делят один обучаемый параметр: при одинаковых signs их effects одинаковы, при противоположных — связаны знаком. Это ограничение представления, а не сохранение независимых category parameters.

Для bag из двух активных keys A и B, попавших в один bucket с signs +1 и −1, их вклады могут взаимно уничтожиться. Тогда bag A+B имеет тот же hashed vector, что пустой bag, хотя исходные vectors различаются. Это пример информационной потери даже при signed hashing. Для single-valued field одновременно активна только одна category; pattern потери будет другим.

### Что именно сохраняется в ожидании

Раскроем inner product:

$$
\langle\phi(x),\phi(y)\rangle=
\sum_i x_iy_i+
\sum_{i\ne j}\mathbf1[h(i)=h(j)]s(i)s(j)x_iy_j.
$$

При подходящей независимой hash/sign family среднее cross terms равно нулю, и expected inner product совпадает с исходным. Для разных basis features x=eA,y=eB истинный inner product=0. При uniform bucket hashing вероятность collision равна 1/m; signed hashed dot тогда равен +1 или −1 с равными вероятностями, иначе 0. Среднее 0, variance=1/m. Без sign hashed dot для этих неотрицательных features имеет среднее 1/m.

Выбранное однажды отображение может дать ошибку +1 на конкретной паре. Поэтому unbiased expectation по случайным отображениям не является гарантией каждого prediction фиксированной trained model. Ошибки также зависят от частоты и target-конфликта collided keys. Теоретическая постановка изложена в [Feature Hashing for Large Scale Multitask Learning](https://arxiv.org/abs/0902.2206).

## Столкновения и multi-hash

Можно задать несколько независимых bucket mappings и объединить blocks vectors. Вероятность совпадения всех $k$ bucket IDs одной пары при идеальной независимости равна $m^{-k}$. Но dimension/storage увеличиваются, а отдельные collisions остаются. Комбинирование concatenation, sum и learned pooling (агрегирование состояний) даёт разные representations.

Bloom filter отвечает на membership query (запрос) с false positives и обычно не хранит обучаемые signed feature values. Это не стандартная «улучшенная one-hot» для любого classifier. Multi-hash encoding использует похожую идею нескольких отображений, но statistical learning и membership — разные задачи. Важен фактический objective (оптимизируемая целевая функция) representation, а не аналогия названий.

### Пары, занятые buckets и вероятность события

Expected число colliding pairs равно K(K−1)/(2m), поскольку каждая из K(K−1)/2 пар сталкивается с вероятностью 1/m. Expected число занятых buckets равно $m[1-(1-1/m)^K]$, а surplus keys сверх первого в каждом bucket — K минус это число. Это разные статистики.

Для K=3,m=2 collision неизбежна. Expected pairs=1.5, occupied buckets=1.75, surplus keys=1.25. Если все три попали в один bucket, в нём три пары, но лишь два лишних относительно первого key. Поэтому число pairs нельзя читать как число «потерянных категорий».

При K≤m вероятность отсутствия collision равна $\prod_{j=0}^{K-1}(1-j/m)$; при K>m она нулевая. Для K=100,m=1000 expected pairs=4.95, но probability хотя бы одной collision близка к 99.4%. Это не означает, что 99.4% всех categories разрушены: событие касается наличия хотя бы одной пары. Значимость отдельных collisions исследуют по их реальному влиянию.

### Несколько hashes и одинаковый memory budget

Можно concatenation k независимых hash blocks по m buckets. Вероятность совпадения bucket IDs одной пары во всех blocks равна $m^{-k}$ при идеальной независимости. Но partial collisions не исчезают, а общая dimension становится km. Если blocks масштабировать на $1/\sqrt{k}$, inner product объединённого vector станет средним k hashed inner products с сохранением expected norm.

Для пары разных basis features с независимыми signed blocks variance такого среднего равна $1/(km)$. При фиксированной общей dimension D=km это 1/D, как у одного идеального signed hash размера D в этом простом примере. Следовательно, меньшая вероятность полного совпадения signature не доказывает автоматически меньшую inner-product variance при той же памяти.

Например, k=2,m=500 даёт D=1000 и probability полного bucket совпадения 1/250000, тогда как отдельные partial collisions остаются. С одним block m=1000 collision probability=1/1000. Эти схемы по-разному распределяют ошибки; выбор concatenation, summation или learned combination дополнительно меняет representation.

Bloom filter решает membership задачу и имеет собственные правила false positives. Его несколько hashes не являются готовой универсальной заменой one-hot в supervised learning. Аналогия полезна для понимания отображений, но качество classifier определяется другим objective и способом использования collisions.

## Контроль и практика

Сравнить sparse one-hot, frequency и hash нескольких размеров при одном group/time split. Отдельно измерить rare/unknown cases, память, fit/predict time и collision profile. Hash buckets не дают интерпретировать coefficient как effect одной category. Frequency может работать быстро, но потерять полезный distinction между equally frequent classes.

1. Решает ли CSR проблему редких coefficients? **Ответ:** он снижает storage, но не добавляет observations для оценки effect.
2. Сохраняет ли frequency identity categories? **Ответ:** нет; одинаковые counts дают одинаковый encoding.
3. Равно ли expected число colliding pairs вероятности collision? **Ответ:** нет; это разные statistics.

**Практика:** распределите десять categories по четырём buckets двумя hash mappings. Найдите collisions и конфликтующие labels. Посчитайте dense/CSR memory для одного field, затем объясните, какие расходы не вошли в расчёт. Предложите fallback для unknown и проверку стабильности hash после restart.

**Схема курса:** category + train counts → rare/frequency bucket or stable hash → bounded representation + collision/unknown metadata → model input and slice checks



Сравнивают sparse one-hot, frequency и несколько hash widths на одном group/time split. В журнале отдельно указывают input storage, vocabulary/model state, peak memory, fit/predict time и качество известных/редких/новых categories. Counterfactual изменение encoding-size не должно одновременно менять split или доступность будущих counts. Для hash нет прямой однозначной обратной ссылки «этот coefficient принадлежит одной category».

1. **Почему компактный CSR не решает проблему rare levels?** Он экономит хранение тех же наблюдений, но не увеличивает информацию для оценки effects.
2. **Почему frequency может потерять сильный сигнал?** Категории с одинаковой popularity становятся неразличимыми по этому признаку.
3. **Что гарантирует signed hash в простом expectation-расчёте?** Среднее inner product по допустимой случайной family, а не сохранность каждого фиксированного prediction.
4. **Почему вероятность полной multi-hash collision недостаточна для сравнения методов?** Остаются partial collisions, меняются dimension и структура ошибок; фиксированный memory budget задаёт другой вопрос.

Практика: воспроизвести dense/CSR подсчёт, finite-state smoothed frequencies и небольшой signed hash с взаимным уничтожением features. Полностью перечислить размещения трёх keys в двух buckets и сравнить pairs/occupied/surplus. Затем проверить стабильность mapping после restart. Все mappings, counts и memory numbers главы учебные; benchmark пользовательского pipeline не выполнялся.

## Результат главы и переход дальше

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
