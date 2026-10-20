# Д. Как обучать представление под clustering и не принять уверенность за истину

[Оглавление](README.md). Пункт 5 плана. **Связь с чатами:** S2 объясняет hidden states, S6 — retrieval embeddings; distinction между representation и clustering раскрывается здесь как дополнение.

**Уровень:** основа ветки. **Предпосылки:** C1–C4, neural training basics. **Используют:** C6/C7 and research route.

## 1. Двухэтапный baseline

Сначала frozen encoder $z_i=f_\theta(x_i)$, затем standard clustering по $z_i$. Это позволяет отдельно диагностировать две части. Если categories не разделимы в representation, смена одного clustering algorithm может не помочь. Если geometry уже хороша, fine-tuning encoder может быть лишней сложностью.

Сохраняются encoder/tokenizer/preprocessing version, pooling, vector dimensions и normalization. Penultimate image features, CLS token и mean pooled sentence embedding не являются взаимозаменяемыми. Encoder обучался конкретному objective и может группировать nuisance variables вместо целевой semantics.

Правильная ablation: fixed algorithm на raw/L2-normalized/PCA embeddings; затем fixed representation с разными algorithms. Одновременная смена encoder, dimension и K не позволяет понять причину improvement. При external labels tuning делается на validation, final test закрыт.

## 2. DEC: последовательность и objective

[DEC](https://proceedings.mlr.press/v48/xieb16.html) сначала инициализирует encoder через autoencoder reconstruction и cluster centres в latent space. Основная clustering phase уточняет encoder и centres по KL к periodically recomputed auxiliary target; decoder/reconstruction term в неё не входит.

Student-t assignments для $z_i$ и centre $\mu_j$:

$$
q_{ij}=\frac{(1+\|z_i-\mu_j\|^2/\nu)^{-(\nu+1)/2}}
{\sum_l(1+\|z_i-\mu_l\|^2/\nu)^{-(\nu+1)/2}}.
$$

Обозначим $f_j=\sum_iq_{ij}$. Target и loss:

$$
p_{ij}=\frac{q_{ij}^2/f_j}{\sum_lq_{il}^2/f_l},
\qquad
L_{\mathrm{cluster}}=\sum_{i,j}p_{ij}\log\frac{p_{ij}}{q_{ij}}.
$$

В update target фиксируется; затем пересчитывается по новому state. Sharpening усиливает confident assignments, normalization по frequency корректирует массовость clusters. Это self-training signal, не новые human labels. Без хорошей инициализации ошибочный partition тоже может стать увереннее.

## 3. Учебный расчёт и опасная интерпретация

Для двух points с assignments $(0.8,0.2)$ и $(0.2,0.8)$ frequencies обоих clusters равны 1. Первый sharpened target:

$$
p_1=\left(\frac{0.64}{0.68},\frac{0.04}{0.68}\right)
=\left(\frac{16}{17},\frac1{17}\right)
\approx(0.9412,0.0588).
$$

Target стал увереннее без добавления independent evidence. Если initial grouping отражал background, objective может усилить background separation. Поэтому decrease KL не равен доказанному улучшению semantic clustering.

Точно так же collapse всех outputs/центров или очень неравные cluster masses требуют диагностики. Нужны size distribution, stability, feature variance и representative examples. Изменение assignments может остановиться потому, что система застряла, а не нашла истинную taxonomy.

## 4. IDEC и сохранение структуры

[IDEC](https://www.ijcai.org/Proceedings/2017/243) сохраняет reconstruction objective во время clustering refinement. В одной scale convention:

$$
L_{\mathrm{IDEC}}=L_{\mathrm{recon}}+\gamma L_{\mathrm{cluster}},
\qquad
L_{\mathrm{recon}}=\frac1n\sum_i\|x_i-g_\phi(f_\theta(x_i))\|^2.
$$

Reconstruction удерживает информацию об исходном input и ограничивает distortion ради cluster objective. Она не гарантирует semantic correctness: background/noise также могут хорошо реконструироваться. Balance зависит от units, dimensionality и normalization losses. Большое число pixels может сделать unnormalized reconstruction dominant.

Например, две latent mappings дают одинаковый cluster loss, но одна теряет существенный параметр объекта. Reconstruction может различить их, если decoder/data objective действительно требуют этот параметр. Если target semantics никак не отражена в доступных data/prior, дополнительный loss не создаёт её автоматически.

## 5. Валидация и новые objects

При trainable encoder все его fit data принадлежат training protocol. Fit на full corpus допустим для declared transductive study; это не независимый prospective test на ранее unseen documents/images. Предобучение на benchmark test images без labels тоже надо учитывать в protocol.

Для новых objects encoder плюс centres дают assignment rule, но probabilities $q$ — не автоматически calibrated class probabilities. Unknown class может быть уверенно притянута к одному известному centre. Нужны distance/novelty policy и evaluation новых regimes, если это deployment requirement.

Сравниваем frozen baseline, DEC и IDEC по same data, K, budget и validation rules; report includes external metrics при наличии labels, internal geometry, stability, coverage, runtime и visual examples. Увеличение compute только одному методу должно быть явно отделено от quality algorithm.

## 6. Контроль и практика

1. Есть ли reconstruction loss в основной фазе исходного DEC? **Ответ:** нет; сохранение этой части — ключевое отличие IDEC.
2. Создаёт ли sharpening новую truth? **Ответ:** нет; оно преобразует собственные assignments и может усилить ошибку.
3. Равен ли cluster confidence semantic class probability? **Ответ:** нет; это assignment в выбранной model/representation.

**Практика:** вычислите $p$ для двух symmetric assignments и затем для сильно неравных frequencies. Опишите collapsed solution, которое нужно обнаружить. Составьте two-stage ablation и отдельный DEC/IDEC experiment без simultaneous change всех components.


**Схема курса:** frozen encoder features or trainable encoder → initial clusters → Student-t assignments and target distribution → KL/reconstruction update → inspect drift/collapse against frozen baseline

