# В. Ограничивать обучение можно через его траекторию

[Оглавление](README.md). Пункт 3 и практика early stopping. **Происхождение:** дополнение по плану; обсуждённое в S3 merging не заменяет разбор checkpoints одного training run.

## 1. Почему остановка может менять обобщение

В overparameterized модели продолжение fit иногда начинает подстраивать predictions под шум и особенности training sample. Early stopping выбирает checkpoint по независимому validation criterion. Это ограничение процесса, а не явно добавленный weight penalty.

Для линейной least squares $L(w)=\tfrac12\|Xw-y\|^2$ с $w_0=0$ шаг gradient descent:

$$
w_{t+1}=w_t-\eta X^\top(Xw_t-y).
$$

В SVD $X=U\Sigma V^\top$ коэффициент вдоль $v_j$ после $t$ шагов:

$$
v_j^\top w_t=
\frac{1-(1-\eta\sigma_j^2)^t}{\sigma_j}\,u_j^\top y.
$$

При подходящем LR направления с большими singular values осваиваются быстрее; слабые и потенциально шумные directions дольше подавлены. Это иллюстрирует spectral filtering и связь stopping с regularization в данной модели. В nonlinear network точное равенство ridge из этого не следует.

## 2. Как выбирать checkpoint

Задаются metric direction, frequency evaluation, patience, minimum meaningful improvement и правило возврата к лучшему checkpoint. Patience означает число проверок без improvement, а не обязательно число epochs. Validation curve шумит; частый просмотр небольшого набора может привести к отбору случайного пика.

Пример: validation losses по epochs $(0.60,0.48,0.44,0.45,0.47)$. При patience 2 без min-delta остановка после пятой epoch возвращает третью. Возврат последней вместо лучшей теряет смысл метода. Если metric — F1, выбирать minimum loss не обязательно соответствует нужному operating point; criterion фиксируется заранее.

Test не участвует в patience или выборе epoch. После подбора допустима отдельная final-fit strategy на development data, но её budget/epoch rule также задаётся без подглядывания в test. Training loss не обязан достигать нуля при noise, augmentation и soft targets.

## 3. Warmup и decay — управление шагом

Warmup постепенно увеличивает learning rate от малого значения до целевого. Это противоположно идее «сразу взять большой LR». Дальнейший decay уменьшает шаг, например по cosine schedule:

$$
\eta_t=\eta_{\min}+
\frac{\eta_{\max}-\eta_{\min}}2
\left[1+\cos\left(\pi\frac{t-t_w}{T-t_w}\right)\right],
\quad t\in[t_w,T].
$$

Schedule влияет на стабильность и reachable solutions. Он не гарантирует flat minimum или лучшую generalization. Если training расходится, прежде чем усиливать dropout, проверьте LR, gradients, precision и data. Если обе curves плохи, слишком ранняя остановка может лишь сохранить underfit.

При decoupled decay каждый step даёт factor $(1-\eta_t\lambda)$, поэтому общий shrinkage зависит и от schedule, и от числа steps. Сравнивать lambda без budget и LR недостаточно. Изменение batch size меняет число optimizer steps за epoch и тоже влияет на сравнение.

## 4. SWA и EMA

SWA усредняет выбранные weights одной training trajectory:

$$
\overline w_K=\frac1K\sum_{k=1}^K w_{t_k},
\qquad
\overline w_{K+1}=\frac{K\overline w_K+w_{t_{K+1}}}{K+1}.
$$

В [SWA](https://arxiv.org/abs/1803.05407) специальная поздняя training trajectory с подходящим LR позволяет усреднять соседние решения. Это не универсальный рецепт усреднить произвольные независимые модели. Перестановки hidden units и разные basins могут сделать average weights плохим. Model soup связан по форме, но имеет свой протокол выбора совместимых fine-tunes.

EMA использует экспоненциальное сглаживание $\overline w_t=\alpha\overline w_{t-1}+(1-\alpha)w_t$: поздние checkpoints имеют больший вес. Uniform SWA и EMA различаются временным weighting. В обоих случаях оценивать нужно именно averaged model.

BatchNorm running statistics могут не соответствовать усреднённым weights: иногда требуется их пересчёт на train data без gradient updates. Validation/test для fit этих statistics не используются. Стоимость SWA обычно мала относительно training, но хранение, дополнительный этап и BN recalculation не бесплатны буквально.

## 5. Контроль и практика

1. Почему вернуть last epoch после patience недостаточно? **Ответ:** метод выбирает лучший наблюдённый checkpoint, который мог быть раньше остановки.
2. Гарантирует ли cosine schedule flat minimum? **Ответ:** нет; это правило LR, а не теорема о результате любой модели.
3. Чем SWA отличается от EMA? **Ответ:** способом weighting checkpoints и типовым режимом их сбора.

**Практика:** вручную выберите checkpoint по приведённой curve. Для scalar $X=2,y=4,\eta=0.1,w_0=0$ вычислите первые три GD steps и сравните с SVD formula. Составьте эксперимент baseline/early stopping/SWA при одном split и объявленном compute budget; проверьте BN state.
