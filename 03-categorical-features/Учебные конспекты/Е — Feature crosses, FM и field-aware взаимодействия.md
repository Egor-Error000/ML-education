# Е. Когда сигнал живёт в сочетании категорий

[Оглавление](README.md). Пункт 6 плана. **Происхождение:** дополнение по плану; FFM — обзор с точной механикой, а не заявленный вами реализованный recommender.

## 1. Почему additive model недостаточна

Пусть вероятность покупки зависит от city и day. В городе A weekend увеличивает спрос, в B уменьшает. Additive effects $b+w_{city}+v_{day}$ навязывают одинаковый day effect во всех cities. Interaction позволяет change effect depending on partner.

Для двух бинарных fields XOR — наглядный пример: outcomes $(A,X)=0,(A,Y)=1,(B,X)=1,(B,Y)=0$. Без interaction линейный score не может одновременно реализовать такую threshold pattern. Cross category A×X/A×Y/B×X/B×Y даёт отдельные effects, но число combinations растёт произведением cardinalities.

## 2. Ручные crosses

Cross ключ должен быть однозначным tuple, а не ambiguous concatenation: строки `ab`+`c` и `a`+`bc` иначе совпадут. Namespace/escaping/serialization являются частью contract. Затем cross кодируется one-hot, hash, frequency или supervised statistics по тем же правилам leakage.

Если city имеет 1000 values, product 10 000, потенциально $10^7$ combinations. Наблюдаемых пар меньше, но многие редки или unseen. Полный отдельный coefficient на каждую пару плохо переносится на новые combinations. Smoothing и shared factors предлагают более компактный prior.

## 3. Factorization Machines

FM второго порядка для sparse input:

$$
\widehat y=w_0+\sum_iw_ix_i+
\sum_{i<j}\langle v_i,v_j\rangle x_ix_j,
\qquad v_i\in\mathbb R^d.
$$

Вместо независимого coefficient каждой пары используется inner product двух latent vectors. Это parameter sharing, позволяющее оценивать interactions по общим evidence других pairs. Оно ограничивает доступные interaction matrices, но не гарантирует correctness для unseen values.

При one-hot по fields активны несколько features. Для vectors $v_A=(1,0)$ и $v_X=(0.2,0.5)$ pair effect равен 0.2; с $v_Y=(-0.3,0.1)$ — минус 0.3. Изменение одного category vector влияет на все его interactions, что даёт both sharing и coupling ошибок.

FM sum можно вычислять без явного перебора всех pairs:

$$
\sum_{i<j}\langle v_i,v_j\rangle x_ix_j
=\frac12\sum_{r=1}^d\left[
\left(\sum_i v_{i,r}x_i\right)^2-
\sum_i v_{i,r}^2x_i^2\right].
$$

Это даёт cost порядка числа active features times dimension при подходящей implementation. Формула следует из раскрытия квадрата: diagonal terms вычитаются, каждое pair остаётся дважды до factor 1/2.

## 4. Field-aware Factorization Machines

FFM даёт feature отдельный vector в зависимости от field партнёра:

$$
\widehat y=w_0+\sum_iw_ix_i+
\sum_{i<j}\langle v_{i,f_j},v_{j,f_i}\rangle x_ix_j.
$$

$f_i$ — field feature $i$. City A может иметь разные latent representations для interaction с day и с product. Это гибче FM, но увеличивает parameters примерно с $Kd$ до $KFd$ при полном наборе field-specific vectors и требует большей статистики. Точная стоимость зависит от исключённых interactions и storage.

FFM не равен просто concatenation обычных embeddings с linear head: interaction rule задан отдельно. Neural models могут выучить interactions через MLP/attention, но их bias, capacity и optimization отличаются. Автоматические crosses CatBoost — ещё один механизм: combination categories превращаются в statistics/разбиения, а не обязательно factorized dot products. Первичный разбор: [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf).

## 5. Regularization и проверка

Rare interactions легко запоминаются. Нужны held-out groups/time, penalties и анализ unseen combinations. Если все categories известны, но их pair нова, это другой режим, чем completely new category. Полезно оценивать оба отдельно.

Сначала additive baseline, затем один cross либо FM/FFM с comparable tuning budget. Проверяется не только overall metric, но память, rare-pair quality и serialization mapping. Если improvement исчезает на group split, interaction мог memorise ID conjunctions.

## 6. Контроль и практика

1. Что теряет additive model? **Ответ:** effect одного field не может зависеть от другого без interaction.
2. Почему factorization помогает rare pairs? **Ответ:** pair coefficients делят latent parameters с другими combinations; это prior sharing, а не новая информация.
3. Чем FFM отличается от FM? **Ответ:** vector feature зависит ещё и от field взаимодействующего партнёра.

**Практика:** покажите невозможность XOR для additive score через четыре inequalities. Посчитайте FM interaction напрямую и через square identity для трёх active features. Спроектируйте test known-pair/new-pair/new-category и сравните manual cross с factorization без mixing splits.
