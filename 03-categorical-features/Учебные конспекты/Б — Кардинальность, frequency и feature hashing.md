# Б. Когда категорий слишком много для отдельных столбцов

[Оглавление](README.md). Пункт 2 плана. **Происхождение:** дополнение; технические приёмы не выдаются за уже обсуждённые в чатах.

## 1. Почему cardinality — не только память

Уникальный ID даёт одной category один training example. Даже если sparse matrix помещается в RAM, оценить её отдельный effect надёжно невозможно. Поэтому high cardinality создаёт одновременно computational и statistical задачи. Редкие/новые categories требуют fallback и regularization, а не только более компактного storage.

Для $n=100\,000,K=10\,000$ dense float32 one-hot требует $4nK=4\cdot10^9$ bytes, то есть 4 GB decimal без overhead. Если на строку один nonzero, CSR с float32 values и int32 indices/pointers хранит примерно:

$$
4n+4n+4(n+1)\approx1.2\cdot10^6\text{ bytes}.
$$

Реальный объект, metadata, несколько fields и преобразования добавляют расходы. Sparse one-hot не обязательно «ломается» при тысячах categories; trade-off зависит от estimator и sparsity. Но большое число coefficients и слабая статистика rare values сохраняются.

## 2. Count и frequency encoding

Count mapping $c\mapsto n_c$ и frequency mapping $c\mapsto n_c/n$ не используют target. Они отражают распространённость category, что может быть meaningful signal. Но A и B с одинаковой частотой получат одно число и потеряют identity. Такие representations не равны полной замене one-hot без потери информации.

Пример train A,A,B,B,C: counts A/B/C равны 2/2/1, frequencies 0.4/0.4/0.2. Новый D может получить 0 либо smoothed prior по объявленному contract. Frequency только по train; будущая popularity недоступна в историческом решении. Если IDs приходят из нескольких namespaces, нужен namespace prefix, иначе одинаковые строки из разных fields сольются.

## 3. Hashing trick

Hashing строит fixed-dimensional vector без хранения полного vocabulary. Category попадает в bucket $h(c)\in\{0,\ldots,m-1\}$. Для набора active features signed hashing использует независимый sign $s(c)\in\{-1,1\}$:

$$
z_j=\sum_{c:h(c)=j}s(c)x_c.
$$

Новые categories обрабатываются той же функцией, но collision делит их parameters. Signed hash помогает сделать некоторые inner-product distortions unbiased при assumptions hash family; он не устраняет collisions и не гарантирует сохранение каждого индивидуального prediction.

При независимом uniform hashing $K$ categories ожидаемое число colliding pairs:

$$
\mathbb E[N_{\mathrm{pairs}}]=\frac{K(K-1)}{2m}.
$$

Для $K=100,m=1000$ это 4.95 пары. Это не вероятность хотя бы одной collision и не число потерянных categories: несколько pairs могут относиться к одному bucket. Важна также frequency/target конфликтующих values, а не только их count.

Stable hash/config должны совпадать train/inference. Некоторые language built-in hashes зависят от process seed; их нельзя бездумно использовать как сохраняемый feature contract. В отчёте фиксируют implementation, seed, размер buckets и namespace separators.

## 4. Несколько hashes и границы аналогии с Bloom filter

Можно задать несколько независимых bucket mappings и объединить blocks vectors. Вероятность совпадения всех $k$ bucket IDs одной пары при идеальной независимости равна $m^{-k}$. Но dimension/storage увеличиваются, а отдельные collisions остаются. Комбинирование concatenation, sum и learned pooling даёт разные representations.

Bloom filter отвечает на membership query с false positives и обычно не хранит обучаемые signed feature values. Это не стандартная «улучшенная one-hot» для любого classifier. Multi-hash encoding использует похожую идею нескольких отображений, но statistical learning и membership — разные задачи. Важен фактический objective representation, а не аналогия названий.

## 5. Проверка качества

Сравнить sparse one-hot, frequency и hash нескольких размеров при одном group/time split. Отдельно измерить rare/unknown cases, память, fit/predict time и collision profile. Hash buckets не дают интерпретировать coefficient как effect одной category. Frequency может работать быстро, но потерять полезный distinction между equally frequent classes.

1. Решает ли CSR проблему редких coefficients? **Ответ:** он снижает storage, но не добавляет observations для оценки effect.
2. Сохраняет ли frequency identity categories? **Ответ:** нет; одинаковые counts дают одинаковый encoding.
3. Равно ли expected число colliding pairs вероятности collision? **Ответ:** нет; это разные statistics.

**Практика:** распределите десять categories по четырём buckets двумя hash mappings. Найдите collisions и конфликтующие labels. Посчитайте dense/CSR memory для одного field, затем объясните, какие расходы не вошли в расчёт. Предложите fallback для unknown и проверку стабильности hash после restart.
