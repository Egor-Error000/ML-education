# Ж. Регуляризация через входы, targets и промежуточные representations

[Оглавление](README.md). Объединены близкие пункты 7–8 плана. **Основа:** CV-чат S7 задаёт задачи и annotations; полный вывод data-level regularization — дополнение. [Курс данных](../../01-data/Учебные%20конспекты/README.md) отвечает за provenance, leakage и проверку меток.

## 1. Augmentation как ожидаемый training objective

Обучение с transforms минимизирует риск по выбранному распределению $T$:

$$
\mathcal L(\theta)=
\mathbb E_{(x,y),T}[\ell(f_\theta(Tx),T_y(y))].
$$

Для классификации $T_y$ иногда identity, для detection/segmentation — согласованное геометрическое преобразование. Label-preserving transformations должны быть обоснованы задачей. Усиление augmentation меняет objective и difficulty, поэтому при сравнении фиксируются steps, sampling и остальные настройки.

## 2. Label smoothing: объявить convention

Одна convention смешивает hard label с uniform по всем $K$ classes:

$$
q_k=(1-\epsilon)\mathbf1[k=y]+\frac\epsilon K,
\qquad
L=-\sum_{k=1}^K q_k\log p_k.
$$

При $K=3,\epsilon=0.1$ target для class 1 равен $(0.9333\ldots,0.0333\ldots,0.0333\ldots)$. Другая convention оставляет true class $1-\epsilon$ и делит $\epsilon$ только между остальными: $(0.9,0.05,0.05)$. Одинаковое epsilon не делает их одинаковыми.

Для softmax logits gradient равен $\partial L/\partial z_k=p_k-q_k$. Поэтому smoothing ослабляет стимул бесконечно увеличивать true logit относительно остальных. Это может уменьшить overconfidence, но не гарантирует лучшую calibration или ranking в каждой задаче. Неправильно подобранный smoothing способен мешать learning редких/сложных classes.

Uniform prior тоже не всегда соответствует task structure: соседние classes и заведомо невозможные alternatives различаются. Более сложные soft targets должны иметь собственное обоснование и independent validation.

## 3. MixUp и Manifold Mixup

MixUp строит:

$$
\widetilde x=\lambda x_i+(1-\lambda)x_j,
\qquad
\widetilde y=\lambda y_i+(1-\lambda)y_j,
\qquad
\lambda\sim\operatorname{Beta}(\alpha,\alpha).
$$

При classes A/B и $\lambda=0.3$ target содержит 0.3 для A и 0.7 для B. Cross-entropy с таким target равен weighted sum двух class losses. Смешивать class IDs численно нельзя: label 2 и label 6 не дают semantic class 4. Смешиваются one-hot/distributions.

Linear interpolation задаёт prior о поведении между examples. Не все mixed inputs естественны; польза метода эмпирическая и task-dependent. Для unrelated tasks или incompatible modalities механическая смесь может не иметь смысла.

Manifold Mixup смешивает hidden representations на выбранном layer, затем продолжает forward с mixed target. Простое смешивание двух decoded слов или integer tokens не эквивалентно этому. При frozen encoder нужно понять, на каком уровне trainable model может воспользоваться смешанным representation. Batch pairing, distribution lambda и выбранные layers являются частью experiment config.

## 4. CutMix, Cutout и Random Erasing

CutMix заменяет region image A region из B, а weights targets определяются фактической заменённой площадью:

$$
\lambda_{\mathrm{eff}}=1-\frac{A_{\mathrm{patch}}}{A_{\mathrm{image}}}.
$$

Если в image $100\times100$ вставлен patch $20\times40$, исходному class остаётся weight 0.92, вставленному — 0.08. После clipping patch по границам нужно пересчитать actual area, а не сохранять исходную sampled lambda. При detection/segmentation simple class mixture недостаточна: обновляются spatial labels, видимость и правила для перекрытых objects.

Cutout/Random Erasing удаляют/заменяют region без добавления label другого объекта. Они учат устойчивости к occlusion только пока оставшееся evidence достаточно. Если entire small object исчез, сохранение label может стать noise. DropBlock из главы Г удаляет activations внутри network, а не входные pixels; эффекты и настройки различаются.

AutoAugment ищет политику transformations в заданном search space, RandAugment упрощает управление выбором и strength. Это способы организовать augmentation policy, а не гарантированно безопасные для любой annotation presets. Подбор policy по test недопустим.

## 5. Total variation

Для реконструкции image prior гладкости может задаваться anisotropic TV:

$$
\operatorname{TV}(u)=
\sum_{i,j}\bigl(|u_{i+1,j}-u_{i,j}|+|u_{i,j+1}-u_{i,j}|\bigr),
$$

где границы сумм ограничены существующими neighbours. Isotropic variant использует square root суммы квадратов directional differences. TV предпочитает кусочно-гладкие images, но может стирать texture и мелкие детали. Это prior на восстанавливаемое поле, а не универсальный penalty каждого CV classifier.

## 6. Контроль и практика

1. Почему label smoothing conventions дают разные targets? **Ответ:** epsilon распределяется либо по всем classes, либо только по incorrect classes.
2. Почему CutMix использует actual area? **Ответ:** crop/clipping меняют количество реально заменённых pixels.
3. Можно ли MixUp считать линейной смесью class IDs? **Ответ:** нет; IDs nominal, смешиваются target distributions.

**Практика:** вручную вычислите оба smoothed target при четырёх classes. Сделайте MixUp и CutMix двух examples, проверьте сумму target weights. Для small-object image найдите strength, при которой label уже не оправдан; включите этот случай в ручной аудит до training comparison.
