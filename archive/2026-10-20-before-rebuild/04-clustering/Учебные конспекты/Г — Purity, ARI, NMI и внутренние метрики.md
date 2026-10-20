# Г. Что именно измеряет качество partition

[Оглавление](README.md). Пункт 4 и практика оценки. **Происхождение:** дополнение по плану; labels можно использовать для независимой проверки, даже если fit был unsupervised.

**Уровень:** общее ядро. **Предпосылки:** C1–C3. **Используют:** C5–C9.

## 1. Не сравнивать номера labels как классы

Предсказанные cluster IDs условны. True labels (A,A,B,B) и clusters (7,7,3,3) дают идеальное совпадение partitions. Literal accuracy после arbitrary mapping IDs может показать ноль. External clustering metrics сравнивают структуру без зависимости от permutation labels. Если используется matched accuracy, mapping обучается/определяется по заявленному protocol; она имеет другой смысл, чем готовая classifier accuracy.

Contingency matrix $n_{kc}$ считает objects cluster $k$ и true class $c$. Row sums $a_k$, column sums $b_c$, total $n$. Эти counts позволяют вычислить several external measures.

## 2. Purity и её ловушка

$$
\operatorname{Purity}=\frac1n\sum_k\max_c n_{kc}.
$$

Если каждому point присвоить отдельный cluster, purity равна 1 независимо от meaningful structure. Поэтому purity не штрафует excessive fragmentation. Её следует сопровождать cluster count/size distribution и complementary measures. Для true A,A,B,B и crossed clusters (1,2,1,2) каждый cluster содержит по A/B, purity равна 0.5.

## 3. ARI: согласие pairs с поправкой на случайность

Пусть $\binom m2=m(m-1)/2$. Adjusted Rand Index:

$$
\operatorname{ARI}=
\frac{\displaystyle\sum_{k,c}\binom{n_{kc}}2
-\dfrac{\left(\sum_k\binom{a_k}2\right)\left(\sum_c\binom{b_c}2\right)}{\binom n2}}
{\displaystyle\frac12\left[\sum_k\binom{a_k}2+\sum_c\binom{b_c}2\right]
-\dfrac{\left(\sum_k\binom{a_k}2\right)\left(\sum_c\binom{b_c}2\right)}{\binom n2}}.
$$

Adjustment относится к определённому random partition model с fixed marginals. ARI=1 для identical partitions, около 0 для chance agreement по этому baseline, возможны negative values. Это не probability и не обязано лежать в [0,1]. Degenerate denominators требуют library convention.

Для crossed four-point example все $n_{kc}=1$, pair sum 0; row/column pair sums по 2, total pairs 6. Expected term $2/3$, denominator $4/3$, поэтому ARI=-0.5. Для четырёх singleton clusters при двух balanced true classes ARI=0, несмотря на purity=1.

## 4. NMI и информация

Mutual information:

$$
I(K;C)=\sum_{k,c}\frac{n_{kc}}n
\log\frac{n\,n_{kc}}{a_kb_c}.
$$

Zero cells дают нулевой вклад. Одна convention NMI — arithmetic normalization:

$$
\operatorname{NMI}=\frac{2I(K;C)}{H(K)+H(C)}.
$$

Другие normalizers дают другие values; convention сообщается. NMI не является chance-adjusted: при большом количестве clusters может завышаться. Для singleton prediction и balanced two-class truth $I=\log2$, $H(K)=\log4$, $H(C)=\log2$, NMI равна $2/3$, а не 1. При необходимости рассматривают AMI как отдельную adjusted measure.

## 5. Internal scores

Silhouette раскрыт в главе В и зависит от pairwise distance. Davies–Bouldin использует within scatter $S_k$ и centre distance $M_{kj}$:

$$
\operatorname{DB}=\frac1K\sum_k\max_{j\ne k}
\frac{S_k+S_j}{M_{kj}}.
$$

Lower лучше по этому criterion. При стандартной Euclidean convention $S_k$ — mean distance до centre. Calinski–Harabasz сравнивает between и within dispersion:

$$
W=\sum_k\sum_{i\in C_k}\|x_i-\mu_k\|^2,
\quad
B=\sum_k n_k\|\mu_k-\mu\|^2,
\quad
\operatorname{CH}=\frac{B/(K-1)}{W/(n-K)}.
$$

Higher лучше при допустимых K и nondegenerate denominator. Для groups {0,1}/{4,5}: $S_1=S_2=0.5$, centre distance 4, DB=0.25. $W=1,B=16,n=4,K=2$, CH=32. Эти measures предпочитают определённую compact geometry и не оценивают universal semantics. Definitions: [DB](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html), [CH](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html).

## 6. Noise и selection bias

У DBSCAN/HDBSCAN noise часто имеет ID -1. Передав его обычной metric function, можно неосознанно считать все unrelated noise points одним cluster. Исключив noise, можно получить прекрасную метрику на 5% лёгких objects. Поэтому report содержит coverage/noise fraction, правило оценки noise и исходное число examples.

External evaluation использует independent labels; если по ним выбирали K/representation, они validation, а не test. Без labels добавляются expert audit типичных/пограничных cases, stability и downstream utility. Не нужно объединять разные internal scales в одно «среднее качество» без осмысленной normalization.

## 7. Контроль и практика

1. Почему purity 1 не доказывает хорошую clustering? **Ответ:** singletons всегда дают purity 1.
2. Почему ARI может быть отрицательным? **Ответ:** agreement хуже ожидаемого по chance baseline.
3. Что сообщать при исключении noise? **Ответ:** долю coverage/noise и правило отбора; score условен на оставшихся points.

**Практика:** вычислите purity/ARI для perfect permutation, crossed groups и singletons на четырёх objects. Затем DB/CH для toy points. Составьте report, где algorithm A покрывает все objects, B — половину; объясните, почему нельзя выбирать B только по silhouette оставшихся.


**Схема курса:** partition + hidden evaluation labels or geometry → external contingency metric / internal metric → noise and coverage report → no test-label tuning

