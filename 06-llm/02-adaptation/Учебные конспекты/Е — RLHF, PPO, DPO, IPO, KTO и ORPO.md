# Е. От предпочтений к обновлению policy

[Оглавление](README.md). Объединены пункты 5–6 и протокол 14D: RLHF/PPO и direct preference methods решают связанную задачу, но используют разные objectives и данные.

**Основа:** S3, строки 343–423 и 509–673. Сохранена последовательность reward model → policy optimization → DPO → альтернативы. **Исправления:** роли не равны четырём обязательным копиям модели; KTO не требует пар; reference point KTO не «среднее качество»; никакой метод не гарантирует отсутствие переобучения. **Углубление:** вывод DPO-параметризации, ручной loss и градиент margin.

## 1. Что содержится в preference dataset

Для одного prompt x даны два ответа y_w и y_l, причём первый предпочли по определённой rubric. Такой выбор не обязательно устанавливает объективную истину: один ответ может нравиться из-за длины, стиля или ошибки разметчика. Нужно знать критерий, происхождение примеров и варианты disagreement/ties.

Для fact-checking лучше сначала определить фактическую правильность. Для стиля полезны preferences. Если смешать критерии, reward может научиться предпочитать красивый неверный ответ. Одна скалярная метка сжимает неоднородные требования.

Разделять данные нужно по prompts/семействам задач/источникам. Если ответы на один и тот же вопрос попали в train и validation, оценка переноса на новые вопросы будет завышена. Собранные прежней policy ответы также задают ограниченную область feedback: новая policy может генерировать совсем другие ответы.

## 2. Reward model и классический RLHF

Reward model получает x и готовый ответ y и возвращает scalar r(x,y). Для pairwise comparison часто используется Bradley–Terry:

$$
P(y_w\succ y_l\mid x)=\sigma(r_w-r_l),\qquad\mathcal{L}_{\mathrm{RM}}=-\log\sigma(r_w-r_l).
$$

Если scores 1.5 и 0.3, разность 1.2, predicted preference probability≈0.7685, loss≈0.2633. Если к обоим scores прибавить 100, preference не меняется. Pairwise objective определяет относительные различия, а абсолютная шкала reward требует осторожной трактовки.

Классическая схема: SFT policy даёт старт → собираются preferences и учится RM → policy генерирует новые ответы, получает reward и обновляется RL-алгоритмом. KL к reference ограничивает уход от выбранного базового поведения. Общий sequence-level objective:

$$
\max_\pi\;\mathbb{E}_{y\sim\pi(\cdot\mid x)}[r(x,y)]-\beta\,D_{\mathrm{KL}}\!\left(\pi(\cdot\mid x)\,\Vert\,\pi_{\mathrm{ref}}(\cdot\mid x)\right).
$$

Здесь KL — между распределениями ответов для данного prompt; в практическом rollout используются tokenwise log-prob estimates и другие детали. Значение $\log\pi(y)-\log\pi_{\mathrm{ref}}(y)$ для одного sampled y не обязано быть положительным, хотя точный KL неотрицателен.

В PPO есть **current policy**, **old rollout policy**, **reference**, **reward** и **value estimate** как логические роли. Old policy нужна для importance ratio, reference — для удержания поведения; их нельзя без объяснения считать одним объектом. Некоторые роли используют общие backbones, adapters, precomputed values или другие размещения. Из схемы не следует требование держать четыре полных независимых 7B-модели на одном GPU.

Для token/action ratio $\rho=\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}$ PPO использует clipped surrogate:

$$
\min\!\left(\rho A,\;\operatorname{clip}(\rho,1-\varepsilon,1+\varepsilon)A\right).
$$

При A=2, rho=1.4, $\varepsilon=0.2$ получается min(2.8,2.4)=2.4. Для положительного advantage слишком большое увеличение ratio не даёт дальнейшей выгоды в этом surrogate. Clipping не является жёсткой гарантией bounded KL всей policy и не заменяет reference penalty. Value function снижает variance оценки advantage; это оценка ожидаемого return, а не ещё один независимый «судья вкуса».

**Reward hacking:** policy может повышать RM score за счёт длины или шаблонов, а истинное task quality не растёт. Контроль — независимая оценка, аудит reward correlations, проверка shifts и ограничение update; один KL не доказывает отсутствие проблемы. Исторический пример пайплайна: [InstructGPT](https://arxiv.org/abs/2203.02155).

## 3. От KL-regularized optimum к DPO

Для фиксированного x формальный optimum KL-regularized reward objective имеет вид:

$$
\pi^*(y\mid x)=\frac{\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)}{Z(x)}.
$$

Следовательно, $r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x)$. Подставляя разность rewards в Bradley–Terry, получаем сокращение prompt-dependent $\log Z(x)$ для пары с одним x. Это объясняет, откуда берётся reference-relative margin; DPO не «удаляет обратное распространение» и не становится tuning без данных.

Для trainable pi_theta:

$$
\begin{aligned}\delta&=\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)},\\\mathcal{L}_{\mathrm{DPO}}&=-\log\sigma(\beta\delta).\end{aligned}
$$

Обычная sequence log-prob — сумма token log-probs ответа с правильным conditioning/mask, включая EOS согласно convention. Среднее по длине вместо суммы меняет objective. Prompt tokens нельзя случайно включать как часть ответа. Reference должна оставаться выбранной фиксированной policy, если используется стандартная схема; детали reference-free variants — отдельные варианты.

Ручной пример:

$$
\begin{gathered}\log\pi_\theta(y_w\mid x)=-2,\quad\log\pi_\theta(y_l\mid x)=-3,\\\log\pi_{\mathrm{ref}}(y_w\mid x)=\log\pi_{\mathrm{ref}}(y_l\mid x)=-2.5,\\\delta=0.5-(-0.5)=1,\qquad\beta=0.1,\qquad\mathcal{L}=-\log\sigma(0.1)\approx0.644397.\end{gathered}
$$

Если policy и reference совпадают на обоих ответах, delta=0 и loss=ln2≈0.693147. Производная по delta:

$$
\frac{\partial\mathcal{L}_{\mathrm{DPO}}}{\partial\delta}=-\beta\sigma(-\beta\delta).
$$

При delta=1,$\beta=0.1$ это около -0.047502. Спуск стремится увеличить относительное преимущество chosen. Однако увеличение **margin** не требует обязательно увеличить абсолютную вероятность chosen: можно ещё сильнее уменьшить rejected. Поэтому нужны task evaluation и при необходимости supervised component/другой objective, а не только график preference accuracy.

Beta определяет objective и чувствительность margin. Фраза «больше beta — просто сильнее улучшение» неверна: в исходной KL-regularized постановке beta задаёт trade-off, а в loss также масштабирует logit. Learning rate, данные и текущая delta дополнительно влияют на training dynamics. [DPO](https://arxiv.org/abs/2305.18290).

## 4. IPO, KTO и ORPO: не один и тот же dataset/loss

**IPO** использует другую форму preference objective. В одной распространённой нотации с ненормированной reference-relative разностью h target margin имеет вид $\frac1{2\beta}$ и применяется квадрат $\left(h-\frac{1}{2\beta}\right)^2$. Важно сверять exact convention, включая length normalization, с выбранной работой/trainer. Конечный target меняет давление на margin по сравнению с неограниченным logistic separation; это не гарантирует generalization на шумных preferences.

**KTO** работает с отдельными desirable/undesirable responses, без matched пары для одного prompt. Utility строится через reference-relative log-prob и reference point, связанный с оценкой KL. Для положительных/отрицательных примеров применяются соответствующие ветви utility и weights. Нельзя подменять reference point средним reward по датасету. Асимметрия регулируется objective и weights; «плохое всегда штрафуется сильнее» не универсально. [KTO](https://arxiv.org/abs/2402.01306).

**ORPO** сочетает supervised training preferred ответов с odds-ratio preference term в одной схеме без отдельной reference policy такого же типа, как в стандартном DPO. Это не значит, что исходный checkpoint, данные и базовое качество больше не важны. Pairwise preference component остаётся; точное определение response probability/length normalization берётся из выбранной реализации.

Для выбора метода сначала посмотреть доступную разметку: пары на одном prompt подходят DPO/IPO; независимые бинарные desirability labels естественнее KTO; совместное обучение demonstration/preference требует осознанного objective. Потом проверить compute, calibration feedback и regression suite. Название метода не заменяет качество данных.

## 5. Контроль понимания и эксперимент

1. Чем reference отличается от old PPO policy? **Ответ:** reference задаёт ориентир поведения/KL, old фиксирует sampling distribution rollout для ratio.
2. Обязательно ли обучать RM для DPO? **Ответ:** стандартный DPO использует preferences напрямую без отдельного RM в training loop.
3. Требует ли KTO пары chosen/rejected одного prompt? **Ответ:** нет, отдельные desirable/undesirable labels.
4. Доказывает ли маленький DPO loss правильные факты? **Ответ:** нет; он отражает objective на конкретных preferences.

**Практика:** пересчитать пример с $\beta=0.2$, затем поменять chosen/rejected местами. Проверить, что delta меняет знак и loss растёт. Для небольшого обучения сравнить baseline и DPO на одних prompts, срезах длины и factual tasks; проследить одновременно train margin, validation margin, chosen/rejected log-probs и task success. Все числа выше — ручные примеры, не результаты вашего запуска.

<!-- latex-display-v1 -->
