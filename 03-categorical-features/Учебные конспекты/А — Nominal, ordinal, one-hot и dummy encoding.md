# А. Как превратить категорию в числа без выдуманного порядка

[Оглавление](README.md). Пункт 1 плана. **Происхождение:** самостоятельное дополнение по плану; полного чата по категориальным данным среди источников нет. Принцип embeddings связан с основами LLM, но табличные encoding protocols здесь новые.

## 1. Числовой ID не делает значение числом

Город, цвет и ID товара — nominal categories: между кодами нет арифметического расстояния. Размер одежды или уровень образования могут иметь ordinal structure, но даже естественный порядок не гарантирует равных интервалов. Код «малый=1, средний=2, большой=3» допускает ordering, но линейная модель дополнительно предполагает одинаковый эффект соседнего шага.

При nominal encoding A=1,B=2,C=3 линейный predictor $\widehat y=b+wx$ ставит B ровно посередине A/C. Это искусственное ограничение. Decision tree с numeric thresholds может отделить A от B/C или A/B от C, но не A/C от B одним split. Native categorical split умеет задавать subset; просто integer codes с numeric dtype — другой механизм.

Для target labels integer IDs допустимы как identifiers classes, если loss/evaluator интерпретируют их именно так. Не следует смешивать label encoding target и nominal feature preprocessing.

## 2. One-hot как basis vector

Для $K$ categories значение $c$ кодируется vector $e_c\in\{0,1\}^K$. Например, A/B/C дают $(1,0,0),(0,1,0),(0,0,1)$. Линейный model:

$$
\widehat y=b+w^\top e_c=b+w_c.
$$

Теперь каждый class имеет отдельный effect. Euclidean distance между любыми двумя различными one-hot vectors равна $\sqrt2$. Representation не содержит semantic closeness, но downstream model может выучить разные effects и interactions; это не запрет на learning сходства вообще.

Категории fit на train. New category требует policy: error, explicit unknown bucket, all-zero либо иной предусмотренный fallback. Missing и unknown имеют разный смысл: missing — значение не получено, unknown — получено новое значение вне train vocabulary. Если они кодируются одинаково, это сознательная потеря информации.

## 3. Dummy trap и intercept

При полном one-hot каждая строка удовлетворяет $\sum_{k=1}^K e_{c,k}=1$. Столбец intercept линейно зависим от dummy columns. Для unregularized linear least squares coefficients неидентифицируемы: к intercept можно добавить $a$, ко всем category coefficients вычесть $a$, predictions не изменятся.

Dummy encoding удаляет одну baseline category. Для baseline A:

$$
\widehat y=b+\beta_B\mathbf1[c=B]+\beta_C\mathbf1[c=C].
$$

Тогда $b$ — effect A, $\beta_B$ и $\beta_C$ — differences от A. Если средние outcomes A/B/C равны 10/13/8, coefficients составляют $b=10,\beta_B=3,\beta_C=-2$. Существуют и другие contrast codings; baseline выбран для интерпретации, а не потому, что A важнее.

Удалять column обязательно не для каждого estimator. Regularized model может иметь уникальное penalized solution и с full one-hot; trees не требуют full-rank design matrix. При penalty choice baseline может менять symmetry regularization, поэтому drop-first не универсально нейтрален.

## 4. Unknown и all-zero

Если baseline A закодирован всеми нулями и unseen D тоже получает zero vector, model будет трактовать D как A. Это допустимый fallback только при осознанном решении, а не следствие того, что модель «поняла новый город». Explicit unknown bucket позволяет отдельное поведение, но если он ни разу не встречался в training, его coefficient/embedding не выучен обычными данными.

Rare grouping объединяет малочисленные categories в OTHER по train counts. Это уменьшает variance, но стирает различия; threshold выбирается на validation. Нельзя построить словарь по train+test лишь потому, что labels не используются: для inductive evaluation distribution тоже должна быть доступна только по protocol.

Feature names, category ordering и dtype сохраняются вместе с model. Если при inference B получил index A, prediction ошибочен, хотя shapes совпадают. Serialization round-trip должен включать encoder, а не только final estimator.

## 5. Контроль и практика

1. Почему arbitrary integer codes вредят linear model? **Ответ:** вводят порядок и равные расстояния, которых nominal feature не имеет.
2. Что устраняет drop-first? **Ответ:** зависимость dummy columns с intercept в полной линейной design matrix.
3. Почему unknown нельзя молча приравнивать missing? **Ответ:** это разные процессы; объединение теряет возможный сигнал.

**Практика:** закодируйте пять строк с A/B/C, missing и unseen D. Для full/dummy схем выпишите predictions и fallback. Покажите две разные группы coefficients, дающие одни predictions при intercept+full one-hot. Сохраните vocabulary и проверьте восстановление column order.
