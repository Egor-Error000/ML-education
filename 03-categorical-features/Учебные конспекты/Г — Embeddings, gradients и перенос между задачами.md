# Г. Когда representations категорий обучаются вместе с задачей

[Оглавление](README.md). Пункт 4 и расчёт параметров из практики. **Основа:** S2:1817–3404 объясняет token embeddings; перенос принципа на табличные entities и transfer между tasks — дополнение.

## 1. Lookup вместо огромного one-hot

Для $K$ categories embedding matrix $E\in\mathbb R^{K\times d}$ возвращает row $e_c$. Математически lookup эквивалентен $E^\top u_c$ для one-hot column $u_c$, но реализация не обязана создавать весь one-hot vector. Обучаются $Kd$ parameters; для $K=10\,000,d=16$ это 160 000, около 0.64 MB float32 weights без optimizer states.

Несколько fields обычно имеют собственные vocabularies и matrices: ID города 7 и ID телефона 7 не один entity. Их vectors concatenated или взаимодействуют по architecture. Dimensions — hyperparameters: большее $d$ повышает capacity, но редкие categories не получают от этого больше observations.

## 2. Как приходит gradient

Пусть scalar predictor $\widehat y=v^\top e_c+b$ и loss $L=\tfrac12(\widehat y-y)^2$. Тогда:

$$
\frac{\partial L}{\partial e_c}=(\widehat y-y)v,
\qquad
\frac{\partial L}{\partial v}=(\widehat y-y)e_c.
$$

В этом example только выбранная row получает data gradient; при batch contributions суммируются по occurrences. Dense regularization/optimizer details могут обновлять и другие parameters, поэтому «невыбранные rows вообще никогда не меняются» слишком сильное утверждение.

Числа: $e_A=(1,2),v=(0.5,-1),b=0,y=0$. Prediction $-1.5$, gradient embedding $(-0.75,1.5)$. При LR 0.1 и обновлении только embedding получаем $(1.075,1.85)$, prediction $-1.3125$, loss уменьшается с $1.125$ до $0.861328125$. Если одновременно обновить $v$, результат другой; в ручном примере это явно зафиксировано.

## 3. Что означает близость

Entity embeddings выучивают полезное для objective, а не универсальную semantic geography. Если target — CTR, города могут сближаться по advertising behaviour, а не по расстоянию на карте. В другой задаче тот же geometry может быть неуместен.

Representation не единственно идентифицируема: для invertible transform $A$ можно заменить embeddings на $Ae_c$, а downstream linear weights на $A^{-\top}v$ и сохранить predictions. Euclidean distances при произвольном transform меняются. Поэтому интерпретировать absolute geometry нужно осторожно, с учетом architecture и regularization.

При embedding+linear output без нелинейных interactions expressive power одного field может быть такой же, как отдельных category coefficients. Польза compact vector проявляется через shared factors, interactions, multitask transfer и data structure, но не гарантируется самим словом «embedding».

## 4. Rare/unknown и regularization

Rare categories получают мало gradient updates. Помогают rare grouping, smaller dimension, weight decay, norm constraints или shared metadata features, если это подтверждается validation. Unknown row должна иметь training signal, например через контролируемое category dropout/rare mapping. Отдельная случайно инициализированная unseen row не знает значение нового entity.

Словарь сохраняется вместе с weights. Перестановка IDs при loading делает lookup wrong без shape error. Missing, PAD и UNK могут требовать разные rows; padding row часто имеет special update rules, которые надо проверить в implementation.

## 5. Transfer и extraction

Embeddings можно freeze и передать другому model, включая boosting. Но если они обучены по labels всего dataset до CV, extracted features уже содержат test information. Внешняя validation должна включать training encoder на своей training части либо использовать genuinely independent pretrained model.

При переносе между tasks нужны стабильные entity IDs, совместимые vocabularies, доступность источников и проверка negative transfer. Source task мог выучить bias старого периода; fine-tuning на новой задаче может забыть полезное. Freeze против update сравниваются на одном split, с отдельным анализом редких и новых entities.

## 6. Контроль и практика

1. Почему lookup не требует dense one-hot? **Ответ:** он сразу выбирает row, хотя математически эквивалентен соответствующему умножению.
2. Является ли embedding distance объективной semantic truth? **Ответ:** нет; она зависит от objective, parametrization и training data.
3. Можно ли сначала обучить embeddings на всех labels, затем CV бустинга? **Ответ:** это утечка; encoder является частью supervised pipeline.

**Практика:** повторите gradient step, затем обновите и embedding, и head одновременно, используя исходные gradients. Посчитайте weights плюс optimizer states для нескольких dimensions. Составьте transfer protocol с независимым source dataset и stable ID mapping.
