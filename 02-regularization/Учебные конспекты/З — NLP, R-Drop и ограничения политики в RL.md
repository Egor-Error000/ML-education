# З. Регуляризация дискретных последовательностей и policy distributions

[Оглавление](README.md). Объединены пункты 9–10. **Основа:** S2 — embeddings/attention, S3 — SFT/RLHF; остальное отмечено как новое углубление. Полный RL-course находится в [lifecycle](../../06-llm/05-lifecycle/Учебные%20конспекты/README.md).

## 1. Weight tying

Если input embedding matrix $E\in\mathbb R^{V\times d}$ и hidden dimension совпадает, logits могут использовать ту же matrix:

$$
z_t=E h_t+b,
\qquad h_t\in\mathbb R^d,\quad z_t\in\mathbb R^V.
$$

Для row-oriented batch запись эквивалентна $Z=HE^\top+b$. Число независимых parameters уменьшается примерно на $Vd$ относительно отдельного output matrix. Gradient в $E$ приходит и через input lookup, и через output projection. Это constraint на параметры, а не утверждение, что каждый embedding является готовым human-readable «значением слова».

Если dimensions различаются, требуется projection или другая architecture. Не все pretrained models используют tying. Веса нельзя просто связать после training без проверки поведения и contract checkpoint.

## 2. Шум для RNN, tokens и embeddings

Variational dropout для recurrent networks может использовать одну mask на все time steps sequence. Независимые masks каждый step вносят другой temporal noise и могут сильнее разрушать перенос информации. Нужно указать, какие connections маскируются: inputs, recurrent state или outputs.

Word/token dropout удаляет или заменяет tokens по объявленному protocol. `[MASK]` существует в BERT MLM, но не является универсальным meaningful token любой decoder model. MLM corruption — часть pretraining objective, а не произвольный preset на inference.

Noisy embeddings задаёт $\widetilde e=e+\xi$, где $\xi\sim\mathcal N(0,\sigma^2I)$. Scale относительно norm embeddings важен. Такой noise не создаёт дискретно корректную перефразировку и не заменяет проверку semantic invariance. Если target зависит от одной цифры/отрицания, сильный noise может повредить обучению.

## 3. Entropy attention и R-Drop

Для attention distribution $a$ entropy:

$$
H(a)=-\sum_j a_j\log a_j.
$$

Если минимизируемый loss содержит $-\lambda H(a)$, он поощряет более высокую entropy; term $+\lambda H(a)$, наоборот, поощряет концентрацию. Sign принципиален. Masked positions исключаются, $0\log0$ трактуется как 0. У causal tokens разное число доступных positions; сравнение raw entropy требует учитывать эту разницу.

Сосредоточенное attention не обязательно ошибка: точное копирование нужного token может требовать именно этого. Entropy prior не доказывает улучшение reasoning и не является direct measure объяснимости.

R-Drop делает два stochastic forward для одного input с разными dropout masks. Для выходных distributions $p^{(1)},p^{(2)}$:

$$
L=\frac{L_{\mathrm{CE}}^{(1)}+L_{\mathrm{CE}}^{(2)}}2
+\frac\alpha2\left[
D_{\mathrm{KL}}(p^{(1)}\Vert p^{(2)})+
D_{\mathrm{KL}}(p^{(2)}\Vert p^{(1)})\right].
$$

Это consistency penalty между stochastic predictions, не требующий отдельного EMA teacher. В sequence tasks masks и нормировка по active tokens должны совпадать. Два forward увеличивают compute; comparison учитывает budget. [R-Drop](https://arxiv.org/abs/2106.14448).

## 4. Sequence-level smoothing и decoding

Token-level smoothing независимо смягчает target в каждом шаге. Sequence-level objective может распределять массу по альтернативным допустимым целым sequences, например translations. Нужно определить, как генерируются alternatives, чем они взвешены и сохраняют ли смысл. Нельзя назвать любой штраф за длину sequence-level label smoothing.

Length penalty в beam search меняет ranking уже генерируемых кандидатов без gradient update. Training regularization меняет weights. Их effects могут пересекаться по длине outputs, но это разные вмешательства и разные места подбора hyperparameters.

## 5. Entropy, KL и PPO в RL

При maximization expected return entropy bonus имеет sign плюс:

$$
J=\mathbb E[R]+\beta H(\pi)
-\lambda D_{\mathrm{KL}}(\pi\Vert\pi_{\mathrm{ref}}).
$$

Entropy поощряет diversity/exploration относительно самой distribution. KL ограничивает отклонение от заданной reference, которая может быть далеко не uniform. Для $\pi=(0.9,0.1)$ entropy положительна; KL к такой же reference ноль. Они не взаимозаменяемы.

PPO clipping ограничивает incentive больших изменений probability ratios по собранным rollouts, но не является тем же KL penalty. Target network в value-based RL стабилизирует bootstrap targets через медленное обновление; её назначение тоже отличается от entropy regularization. Общий термин «стабилизация» не делает механизмы эквивалентными.

## 6. Контроль и практика

1. Где gradient weight tying? **Ответ:** shared matrix получает суммы вкладов input и output путей.
2. Какой sign повышает attention entropy при minimization? **Ответ:** отрицательный перед $\lambda H$.
3. Почему beam length penalty не training regularizer? **Ответ:** он меняет decoding ranking без изменения parameters.

**Практика:** выпишите shapes tying для $V=1000,d=64$. Вычислите symmetric KL двух distributions $(0.8,0.2)$ и $(0.6,0.4)$. Затем сравните entropy и KL к reference для двух policies; объясните, какой objective поддерживает разнообразие, а какой — близость к прежнему поведению.
