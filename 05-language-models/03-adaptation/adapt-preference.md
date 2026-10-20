# RLHF, PPO, DPO, IPO, KTO и ORPO

**Уровень:** основа ветки.

**Предпосылки:** [Задача адаптации, baseline и выбор вмешательства](<adapt-choice.md>), [SFT: данные, шаблоны, masks и objective](<adapt-sft.md>)

**Используют:** [AI feedback и continual learning](<adapt-continual.md>)

**Учебный вопрос:** Как предпочтения превращаются в objective и каких данных он требует?





## Карта вычисления и границы этапов

![RLHF, PPO, DPO, IPO, KTO и ORPO: входы, преобразования и выходы](<../../оформление/course/adapt-preference.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:649ef8d7ef33f632df -->
<a id="u-649ef8d7ef33f632df"></a>

## Что содержится в preference dataset

<!-- /unit:649ef8d7ef33f632df -->

<!-- unit:864ee77bddfe3146c1 -->
<a id="u-864ee77bddfe3146c1"></a>

Для одного prompt x даны два ответа y_w и y_l, причём первый предпочли по определённой rubric. Такой выбор не обязательно устанавливает объективную истину: один ответ может нравиться из-за длины, стиля или ошибки разметчика. Нужно знать критерий, происхождение примеров и варианты disagreement/ties.

<!-- /unit:864ee77bddfe3146c1 -->

<!-- unit:7830f4a56f055f4bf3 -->
<a id="u-7830f4a56f055f4bf3"></a>

Для fact-checking лучше сначала определить фактическую правильность. Для стиля полезны preferences. Если смешать критерии, reward может научиться предпочитать красивый неверный ответ. Одна скалярная метка сжимает неоднородные требования.

<!-- /unit:7830f4a56f055f4bf3 -->

<!-- unit:79475ceaefdb68feb9 -->
<a id="u-79475ceaefdb68feb9"></a>

Разделять данные нужно по prompts/семействам задач/источникам. Если ответы на один и тот же вопрос попали в train и validation, оценка переноса на новые вопросы будет завышена. Собранные прежней policy ответы также задают ограниченную область feedback: новая policy может генерировать совсем другие ответы.

<!-- /unit:79475ceaefdb68feb9 -->

<!-- unit:7b9542e9ad7ecf68b0 -->
<a id="u-7b9542e9ad7ecf68b0"></a>

### Разница между demonstration и сравнением

<!-- /unit:7b9542e9ad7ecf68b0 -->

<!-- unit:af86aa44ba8f290e52 -->
<a id="u-af86aa44ba8f290e52"></a>

Один отличный ответ SFT показывает, что воспроизвести. Preference pair показывает, что один из двух ответов лучше по rubric; оба при этом могут быть неверными. Например, chosen кратко и уверенно выдумывает число, rejected выдумывает то же число многословно. Если разметчик предпочёл краткость, gradient научит сравнительному стилю, а не правильному числу. Поэтому «chosen» — метка сравнения, а не сертификат истинности.

<!-- /unit:af86aa44ba8f290e52 -->

<!-- unit:2ff716d6c7a7cb7bb7 -->
<a id="u-2ff716d6c7a7cb7bb7"></a>

В S3 мотивировка звучала как «невозможно написать loss полезности, поэтому нужен RLHF». Точнее: многомерные человеческие требования трудно надёжно выразить одной заранее заданной функцией. Для части задач есть точные verifiers, unit tests или known targets. RM — обучаемый surrogate для выбранного feedback, который тоже может ошибаться и терять нюансы. Роль feedback устанавливают по задаче, а не по обязательной лестнице от SFT к alignment.

<!-- /unit:2ff716d6c7a7cb7bb7 -->

<!-- unit:67cc324871a17a8e66 -->
<a id="u-67cc324871a17a8e66"></a>

Независимые preference labels разных людей могут расходиться. Две группы могут предпочитать разную длину, степень осторожности или терминологию. Складывать их без анализа означает обучать смесь, а не находить единственный объективный «человеческий вкус». Хранят rubric, annotator protocol, sampling policy и причины ties/disagreement. Повторные вопросы одного семейства не делают validation независимой от training.

<!-- /unit:67cc324871a17a8e66 -->

<!-- unit:236552ac2b044cd6d5 -->
<a id="u-236552ac2b044cd6d5"></a>

## Reward model и классический RLHF

<!-- /unit:236552ac2b044cd6d5 -->

<!-- unit:bcb7ad69eda2b66938 -->
<a id="u-bcb7ad69eda2b66938"></a>

Reward model получает x и готовый ответ y и возвращает scalar r(x,y). Для pairwise comparison часто используется Bradley–Terry:

<!-- /unit:bcb7ad69eda2b66938 -->

<!-- unit:b3e6aa07290a119547 -->
<a id="u-b3e6aa07290a119547"></a>

$$
P(y_w\succ y_l\mid x)=\sigma(r_w-r_l),\qquad\mathcal{L}_{\mathrm{RM}}=-\log\sigma(r_w-r_l).
$$

<!-- /unit:b3e6aa07290a119547 -->

<!-- unit:c678a4d18b366e783b -->
<a id="u-c678a4d18b366e783b"></a>

Если scores 1.5 и 0.3, разность 1.2, predicted preference probability≈0.7685, loss≈0.2633. Если к обоим scores прибавить 100, preference не меняется. Pairwise objective определяет относительные различия, а абсолютная шкала reward требует осторожной трактовки.

<!-- /unit:c678a4d18b366e783b -->

<!-- unit:88aa6b088274804fae -->
<a id="u-88aa6b088274804fae"></a>

Классическая схема: SFT policy даёт старт → собираются preferences и учится RM → policy генерирует новые ответы, получает reward и обновляется RL-алгоритмом. KL к reference ограничивает уход от выбранного базового поведения. Общий sequence-level objective:

<!-- /unit:88aa6b088274804fae -->

<!-- unit:c5b2ca9d55c2304acd -->
<a id="u-c5b2ca9d55c2304acd"></a>

$$
\max_\pi\;\mathbb{E}_{y\sim\pi(\cdot\mid x)}[r(x,y)]-\beta\,D_{\mathrm{KL}}\!\left(\pi(\cdot\mid x)\,\Vert\,\pi_{\mathrm{ref}}(\cdot\mid x)\right).
$$

<!-- /unit:c5b2ca9d55c2304acd -->

<!-- unit:14a441241b8e7e1d40 -->
<a id="u-14a441241b8e7e1d40"></a>

Здесь KL — между распределениями ответов для данного prompt; в практическом rollout используются tokenwise log-prob estimates и другие детали. Значение $\log\pi(y)-\log\pi_{\mathrm{ref}}(y)$ для одного sampled y не обязано быть положительным, хотя точный KL неотрицателен.

<!-- /unit:14a441241b8e7e1d40 -->

<!-- unit:486d9847ca01c1da40 -->
<a id="u-486d9847ca01c1da40"></a>

В PPO есть **current policy**, **old rollout policy**, **reference**, **reward** и **value estimate** как логические роли. Old policy нужна для importance ratio, reference — для удержания поведения; их нельзя без объяснения считать одним объектом. Некоторые роли используют общие backbones, adapters, precomputed values или другие размещения. Из схемы не следует требование держать четыре полных независимых 7B-модели на одном GPU.

<!-- /unit:486d9847ca01c1da40 -->

<!-- unit:e7e7be831a652094e0 -->
<a id="u-e7e7be831a652094e0"></a>

Для token/action ratio $\rho=\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{old}}(a\mid s)}$ PPO использует clipped surrogate:

<!-- /unit:e7e7be831a652094e0 -->

<!-- unit:44ecfe8e2b7c016a05 -->
<a id="u-44ecfe8e2b7c016a05"></a>

$$
\min\!\left(\rho A,\;\operatorname{clip}(\rho,1-\varepsilon,1+\varepsilon)A\right).
$$

<!-- /unit:44ecfe8e2b7c016a05 -->

<!-- unit:eabff66f2ce2e202b2 -->
<a id="u-eabff66f2ce2e202b2"></a>

При A=2, rho=1.4, $\varepsilon=0.2$ получается min(2.8,2.4)=2.4. Для положительного advantage слишком большое увеличение ratio не даёт дальнейшей выгоды в этом surrogate. Clipping не является жёсткой гарантией bounded KL всей policy и не заменяет reference penalty. Value function снижает variance оценки advantage; это оценка ожидаемого return, а не ещё один независимый «судья вкуса».

<!-- /unit:eabff66f2ce2e202b2 -->

<!-- unit:24c1866ac77cf97f17 -->
<a id="u-24c1866ac77cf97f17"></a>

**Reward hacking:** policy может повышать RM score за счёт длины или шаблонов, а истинное task quality не растёт. Контроль — независимая оценка, аудит reward correlations, проверка shifts и ограничение update; один KL не доказывает отсутствие проблемы. Исторический пример пайплайна: [InstructGPT](https://arxiv.org/abs/2203.02155).

<!-- /unit:24c1866ac77cf97f17 -->

<!-- unit:875f36ab1e208ab056 -->
<a id="u-875f36ab1e208ab056"></a>

### Почему Bradley–Terry использует разность scores

<!-- /unit:875f36ab1e208ab056 -->

<!-- unit:a2aaf48a4f57bc3a83 -->
<a id="u-a2aaf48a4f57bc3a83"></a>

Для ответа с representation h можно поставить scalar head $r=w^\top h+b$, но architecture RM не обязана быть ровно «отрезали LM head у той же SFT». Важен contract: x,y→scalar. При rewards1.5 и0.3 model comparison depends only on1.2. Derivatives по r_w и r_l равны $-\sigma(-(r_w-r_l))$ и $+\sigma(-(r_w-r_l))$: chosen score поднимается относительно rejected.

<!-- /unit:a2aaf48a4f57bc3a83 -->

<!-- unit:2d944ca4109866d485 -->
<a id="u-2d944ca4109866d485"></a>

Добавление общей константы c(x) ко всем ответам данного prompt ничего не меняет в pairwise likelihood. Следовательно, reward=8 на одном prompt и reward=5 на другом не обязательно означает межзадачное абсолютное качество в общей шкале. Если training data отличаются сложностью или длиной, RM может усвоить нежелательные shortcut features.

<!-- /unit:2d944ca4109866d485 -->

<!-- unit:21a33f5964a94bcaa7 -->
<a id="u-21a33f5964a94bcaa7"></a>

### PPO: от sampled ответа к обучающему сигналу

<!-- /unit:21a33f5964a94bcaa7 -->

<!-- unit:b759534cc34a577a9d -->
<a id="u-b759534cc34a577a9d"></a>

Policy порождает tokens по rollout distribution. RM оценивает завершённый ответ; tokenwise log-prob penalties сравнивают с reference. Value estimate ожидаемого return даёт baseline, advantage измеряет, оказался ли action/trajectory лучше этого ожидания. RM и value отвечают на разные вопросы: первый оценивает полученный текст, второй предсказывает будущий return из состояния.

<!-- /unit:b759534cc34a577a9d -->

<!-- unit:66a59b3fddcd5684cc -->
<a id="u-66a59b3fddcd5684cc"></a>

Old policy фиксирует вероятности, с которыми собран rollout, чтобы при нескольких optimization steps оценивать importance ratios. Reference задаёт желаемый ориентир поведения и может оставаться прежней много rollout cycles. Даже если они в самом начале совпали, их определения различны. Сохранённые old log-probs не требуют отдельной постоянно загруженной old model, а роли RM/reference/value не определяют неизбежно четыре полных независимых weight arrays на одной карте.

<!-- /unit:66a59b3fddcd5684cc -->

<!-- unit:42073e41255b1bea92 -->
<a id="u-42073e41255b1bea92"></a>

Для A=−2 и ρ=0.6 при ε=0.2 clipped surrogate равен min(−1.2,−1.6)=−1.6. Так clipping ограничивает чрезмерное снижение вероятности плохого action в сторону ρ<0.8. Знак advantage важен: нельзя применить рассказ о положительном A ко всем случаям. При этом clipped objective — surrogate, не hard constraint на global KL. [PPO](https://arxiv.org/abs/1707.06347).

<!-- /unit:42073e41255b1bea92 -->

<!-- unit:930aaa83b5773ab878 -->
<a id="u-930aaa83b5773ab878"></a>

Если exact distributions p=(0.8,0.2),q=(0.5,0.5), KL(p||q)=0.8ln1.6+0.2ln0.4≈0.192745. Sampled log-ratio для второго исхода равен ln0.4≈−0.916291. Отрицательное значение одной sampled penalty не опровергает неотрицательность математического ожидания KL. Нужны declared estimator, aggregation и support, а не проверка знака каждого token term.

<!-- /unit:930aaa83b5773ab878 -->

<!-- unit:3a09929264b8e4eae2 -->
<a id="u-3a09929264b8e4eae2"></a>

Reward hacking в S3 иллюстрировали многословием и шаблонными вступлениями. Важно проверить это как гипотезу: при росте RM score меняется длина, factuality, task success или только стиль? Отдельный human/verified holdout и length-matched сравнения помогают отличить реальное улучшение от surrogate exploitation. KL не делает ошибочный reward правильным.

<!-- /unit:3a09929264b8e4eae2 -->

<!-- unit:d25395de8fbd557be1 -->
<a id="u-d25395de8fbd557be1"></a>

## От KL-regularized optimum к DPO

<!-- /unit:d25395de8fbd557be1 -->

<!-- unit:39676747c70175c5b0 -->
<a id="u-39676747c70175c5b0"></a>

Для фиксированного x формальный optimum KL-regularized reward objective имеет вид:

<!-- /unit:39676747c70175c5b0 -->

<!-- unit:bee7f6cfa69ecdae18 -->
<a id="u-bee7f6cfa69ecdae18"></a>

$$
\pi^*(y\mid x)=\frac{\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)}{Z(x)}.
$$

<!-- /unit:bee7f6cfa69ecdae18 -->

<!-- unit:243ed88a0db339e2f1 -->
<a id="u-243ed88a0db339e2f1"></a>

Следовательно, $r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x)$. Подставляя разность rewards в Bradley–Terry, получаем сокращение prompt-dependent $\log Z(x)$ для пары с одним x. Это объясняет, откуда берётся reference-relative margin; DPO не «удаляет обратное распространение» и не становится tuning без данных.

<!-- /unit:243ed88a0db339e2f1 -->

<!-- unit:124a692d5903cfb39c -->
<a id="u-124a692d5903cfb39c"></a>

Для trainable pi_theta:

<!-- /unit:124a692d5903cfb39c -->

<!-- unit:ab3e6e8dfc2555cd2e -->
<a id="u-ab3e6e8dfc2555cd2e"></a>

$$
\begin{aligned}\delta&=\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)},\\\mathcal{L}_{\mathrm{DPO}}&=-\log\sigma(\beta\delta).\end{aligned}
$$

<!-- /unit:ab3e6e8dfc2555cd2e -->

<!-- unit:82aba9364bb0393372 -->
<a id="u-82aba9364bb0393372"></a>

Обычная sequence log-prob — сумма token log-probs ответа с правильным conditioning/mask, включая EOS согласно convention. Среднее по длине вместо суммы меняет objective. Prompt tokens нельзя случайно включать как часть ответа. Reference должна оставаться выбранной фиксированной policy, если используется стандартная схема; детали reference-free variants — отдельные варианты.

<!-- /unit:82aba9364bb0393372 -->

<!-- unit:146414961198b4e26d -->
<a id="u-146414961198b4e26d"></a>

Ручной пример:

<!-- /unit:146414961198b4e26d -->

<!-- unit:4cd7bcd2b9039b5424 -->
<a id="u-4cd7bcd2b9039b5424"></a>

$$
\begin{gathered}\log\pi_\theta(y_w\mid x)=-2,\quad\log\pi_\theta(y_l\mid x)=-3,\\\log\pi_{\mathrm{ref}}(y_w\mid x)=\log\pi_{\mathrm{ref}}(y_l\mid x)=-2.5,\\\delta=0.5-(-0.5)=1,\qquad\beta=0.1,\qquad\mathcal{L}=-\log\sigma(0.1)\approx0.644397.\end{gathered}
$$

<!-- /unit:4cd7bcd2b9039b5424 -->

<!-- unit:a5e623f4ea9115261c -->
<a id="u-a5e623f4ea9115261c"></a>

Если policy и reference совпадают на обоих ответах, delta=0 и loss=ln2≈0.693147. Производная по delta:

<!-- /unit:a5e623f4ea9115261c -->

<!-- unit:6c7bd37ddd5435ef25 -->
<a id="u-6c7bd37ddd5435ef25"></a>

$$
\frac{\partial\mathcal{L}_{\mathrm{DPO}}}{\partial\delta}=-\beta\sigma(-\beta\delta).
$$

<!-- /unit:6c7bd37ddd5435ef25 -->

<!-- unit:83ecfeab5d32a3c208 -->
<a id="u-83ecfeab5d32a3c208"></a>

При delta=1,$\beta=0.1$ это около -0.047502. Спуск стремится увеличить относительное преимущество chosen. Однако увеличение **margin** не требует обязательно увеличить абсолютную вероятность chosen: можно ещё сильнее уменьшить rejected. Поэтому нужны task evaluation и при необходимости supervised component/другой objective, а не только график preference accuracy.

<!-- /unit:83ecfeab5d32a3c208 -->

<!-- unit:f037264a16f825e593 -->
<a id="u-f037264a16f825e593"></a>

Beta определяет objective и чувствительность margin. Фраза «больше beta — просто сильнее улучшение» неверна: в исходной KL-regularized постановке beta задаёт trade-off, а в loss также масштабирует logit. Learning rate, данные и текущая delta дополнительно влияют на training dynamics. [DPO](https://arxiv.org/abs/2305.18290).

<!-- /unit:f037264a16f825e593 -->

<!-- unit:e18297c0a1d22b0b60 -->
<a id="u-e18297c0a1d22b0b60"></a>

### Откуда взялся exponential optimum

<!-- /unit:e18297c0a1d22b0b60 -->

<!-- unit:d8cc329db55a3e8d49 -->
<a id="u-d8cc329db55a3e8d49"></a>

Для фиксированного x максимизируем $\sum_y\pi_y r_y-\beta\sum_y\pi_y\ln(\pi_y/q_y)$ при sumπ=1, где q — reference. С Lagrange multiplier η производная по π_y равна $r_y-\beta[\ln(\pi_y/q_y)+1]+\eta$. Приравнивание нулю даёт $\pi_y=q_y\exp(r_y/\beta)\cdot\text{constant}$, а normalization определяет Z. Предполагаются β>0 и подходящий support: если q_y=0, обычная finite-KL постановка не позволяет безусловно дать ему положительную массу.

<!-- /unit:d8cc329db55a3e8d49 -->

<!-- unit:b6d05fbbf152f256a6 -->
<a id="u-b6d05fbbf152f256a6"></a>

Этот вывод касается formal optimum objective, а не обещания, что finite neural network SGD точно его найдёт. Подстановка reward через log-ratio в preference model объясняет DPO objective. Убрана отдельная learned RM из loop; сами preference assumptions и качество данных никуда не исчезли. DPO всё ещё вычисляет differentiable sequence log-probs и обновляет параметры через backward.

<!-- /unit:b6d05fbbf152f256a6 -->

<!-- unit:4c01cb47d82e97525a -->
<a id="u-4c01cb47d82e97525a"></a>

### Margin может улучшиться при падении chosen probability

<!-- /unit:4c01cb47d82e97525a -->

<!-- unit:74ccd78f43beb188ad -->
<a id="u-74ccd78f43beb188ad"></a>

Возьмём reference, для которой выбранные два ответа имеют одинаковые probability. У current policy сначала p_w=0.4,p_l=0.2, margin=ln2. После update p_w=0.3,p_l=0.05, margin=ln6. Оба значения уменьшаются, но rejected сильнее, поэтому logistic preference loss улучшается. Оставшаяся probability может уйти в другие ответы, качество которых pair не описывает.

<!-- /unit:74ccd78f43beb188ad -->

<!-- unit:097d96ed8da694ffb9 -->
<a id="u-097d96ed8da694ffb9"></a>

Так likelihood displacement нельзя объяснять универсальным буквальным удалением «похожих tokens chosen» из source-метафоры. Наблюдаемый факт — относительный objective допускает уменьшение chosen likelihood. Анализируют chosen/rejected log-probs и task outputs одновременно. Контроль через supervised component или иной objective — отдельное решение, не автоматическое свойство любого DPO-run.

<!-- /unit:097d96ed8da694ffb9 -->

<!-- unit:b96b5687178b1004de -->
<a id="u-b96b5687178b1004de"></a>

Sequence log-prob суммируется по response positions. У длинного ответа больше отрицательных слагаемых; переход к среднему по длине меняет цель, а не просто улучшает numerical stability. Нужно одинаково формировать response boundaries, EOS и truncation для policy/reference. Случайно разный tokenizer делает сравниваемые events вообще другими. [DPO](https://arxiv.org/abs/2305.18290).

<!-- /unit:b96b5687178b1004de -->

<!-- unit:dbd0a2c35d5f37fda5 -->
<a id="u-dbd0a2c35d5f37fda5"></a>

## IPO, KTO и ORPO: не один и тот же dataset/loss

<!-- /unit:dbd0a2c35d5f37fda5 -->

<!-- unit:ae6641ef89d8ad3511 -->
<a id="u-ae6641ef89d8ad3511"></a>

**IPO** использует другую форму preference objective. В одной распространённой нотации с ненормированной reference-relative разностью h target margin имеет вид $\frac1{2\beta}$ и применяется квадрат $\left(h-\frac{1}{2\beta}\right)^2$. Важно сверять exact convention, включая length normalization, с выбранной работой/trainer. Конечный target меняет давление на margin по сравнению с неограниченным logistic separation; это не гарантирует generalization на шумных preferences.

<!-- /unit:ae6641ef89d8ad3511 -->

<!-- unit:b27a3b6637dcf77472 -->
<a id="u-b27a3b6637dcf77472"></a>

**KTO** работает с отдельными desirable/undesirable responses, без matched пары для одного prompt. Utility строится через reference-relative log-prob и reference point, связанный с оценкой KL. Для положительных/отрицательных примеров применяются соответствующие ветви utility и weights. Нельзя подменять reference point средним reward по датасету. Асимметрия регулируется objective и weights; «плохое всегда штрафуется сильнее» не универсально. [KTO](https://arxiv.org/abs/2402.01306).

<!-- /unit:b27a3b6637dcf77472 -->

<!-- unit:dba2fb853667f7e031 -->
<a id="u-dba2fb853667f7e031"></a>

**ORPO** сочетает supervised training preferred ответов с odds-ratio preference term в одной схеме без отдельной reference policy такого же типа, как в стандартном DPO. Это не значит, что исходный checkpoint, данные и базовое качество больше не важны. Pairwise preference component остаётся; точное определение response probability/length normalization берётся из выбранной реализации.

<!-- /unit:dba2fb853667f7e031 -->

<!-- unit:0d43b4a42c30749dc8 -->
<a id="u-0d43b4a42c30749dc8"></a>

Для выбора метода сначала посмотреть доступную разметку: пары на одном prompt подходят DPO/IPO; независимые бинарные desirability labels естественнее KTO; совместное обучение demonstration/preference требует осознанного objective. Потом проверить compute, calibration feedback и regression suite. Название метода не заменяет качество данных.

<!-- /unit:0d43b4a42c30749dc8 -->

<!-- unit:0debd02a00ebe890fa -->
<a id="u-0debd02a00ebe890fa"></a>

### Уточнение пользовательского запроса: «подробнее IPO и KTO»

<!-- /unit:0debd02a00ebe890fa -->

<!-- unit:76037a324b27dba4d7 -->
<a id="u-76037a324b27dba4d7"></a>

В исходном чате IPO описывалась как жёсткий предел margin, после которого пример «перестаёт оптимизироваться». Для quadratic objective это неверно. При форме $L=(h-c)^2$, $c=1/(2\beta)$, derivative $2(h-c)$ равна нулю только при h=c, а при превышении меняет знак и возвращает margin назад к target. Это finite preferred target, а не clipping с плоской областью на всех h>c.

<!-- /unit:76037a324b27dba4d7 -->

<!-- unit:3763623f2456104bb4 -->
<a id="u-3763623f2456104bb4"></a>

При β=0.1 target c=5. Для h=1 loss16, derivative−8; для h=5 —0 и0; для h=7 —4 и+4. Разность reference-relative log-odds target5 также не равна «chosen в пять раз вероятнее» из S3: exp5≈148.413, а reference probabilities и normalization convention входят в смысл отношения. Если выбранный implementation использует average token log-probs, интерпретация будет другой. [IPO](https://arxiv.org/abs/2310.12036).

<!-- /unit:3763623f2456104bb4 -->

<!-- unit:019a34440f18fcb111 -->
<a id="u-019a34440f18fcb111"></a>

У KTO examples могут быть unpaired: (x1,y1,desirable) и (x2,y2,undesirable). Их нельзя превратить в DPO pair, просто сопоставив ответы разных prompts: сравнение должно иметь общий conditioning. Бинарные labels могут быть удобны для logs, но like/dislike часто отражает interface, незавершённый запрос или реакцию пользователя на другой фактор. Разметка не автоматически становится дешевле в конкретное число раз — экономические множители3–10 из разбираемого примера не являются свойствами loss.

<!-- /unit:019a34440f18fcb111 -->

<!-- unit:4a7f3dbab379984d19 -->
<a id="u-4a7f3dbab379984d19"></a>

Один прозрачный вид KTO utility можно описать через $r_\theta=\beta\ln(\pi_\theta(y|x)/\pi_{ref}(y|x))$ и reference point b, связанную с β-scaled KL estimate. Для положительного example loss $\lambda_+[1-\sigma(r_\theta-b)]$, для отрицательного — $\lambda_-\sigma(r_\theta-b)$. В таком обозначении хорошему ответу выгодно расти относительно b, плохому — снижаться. b — не средняя человеческая оценка качества, поскольку человеческих scalar rewards здесь нет.

<!-- /unit:4a7f3dbab379984d19 -->

<!-- unit:d898471eae0022bb41 -->
<a id="u-d898471eae0022bb41"></a>

При r−b=0 и равных λ=1 обе branches loss0.5, gradients по r соответственно−0.25 и+0.25. Если λ_−=2, отрицательный пример имеет loss1 и gradient+0.5 в этой точке. Асимметрию задают weights и objective; название prospect theory само по себе не означает, что плохие всегда штрафуются сильнее. Практическая оценка b, detach policy и sampling входят в exact algorithm. [KTO](https://arxiv.org/abs/2402.01306).

<!-- /unit:d898471eae0022bb41 -->

<!-- unit:265ab704542a22a39e -->
<a id="u-265ab704542a22a39e"></a>

### ORPO и смысл объединённого обучения

<!-- /unit:265ab704542a22a39e -->

<!-- unit:4b6dc384cf7c1f06ff -->
<a id="u-4b6dc384cf7c1f06ff"></a>

ORPO сочетает NLL preferred response с preference term по odds. Для объявленной probability p odds=p/(1−p); разность log-odds chosen/rejected подаётся в logistic loss. При p_w=0.8,p_l=0.2 odds4 и0.25, ratio16, preference loss=−ln(16/17)≈0.060625. NLL component отдельно поощряет preferred output, поэтому objective отличается от одного относительного margin.

<!-- /unit:4b6dc384cf7c1f06ff -->

<!-- unit:dc7ce025f75b00fa7d -->
<a id="u-dc7ce025f75b00fa7d"></a>

В реальной response formulation нужно проверить, как определена p, включая length normalization. Нельзя взять один произвольный token score и объявить его probability всего ответа. Отсутствие отдельной reference model не превращает ORPO в «необязательно иметь начальную модель» и не даёт универсальной экономии50% из S3. Сравнивают полные stages/data/compute, не число названий в pipeline. [ORPO](https://arxiv.org/abs/2403.07691).

<!-- /unit:dc7ce025f75b00fa7d -->

<!-- unit:911849b16ba2098d82 -->
<a id="u-911849b16ba2098d82"></a>

### Как выбирать данные и проверять результат

<!-- /unit:911849b16ba2098d82 -->

<!-- unit:de3d401e1959ba3eac -->
<a id="u-de3d401e1959ba3eac"></a>

Для paired methods сохраняют один prompt, chosen/rejected и provenance, исключают пустые/одинаковые пары и проверяют, не обрезался ли важный ответ. Для KTO отдельно анализируют class balance desirability и rubric. Если выбранный текст больше нравится только из-за длины, алгоритм не обязан сам открыть скрытую factual rubric.

<!-- /unit:de3d401e1959ba3eac -->

<!-- unit:999b053b2dc6208361 -->
<a id="u-999b053b2dc6208361"></a>

Не объявляют стабильность и отсутствие overfitting только из-за отказа от PPO. Следят за validation loss/margin, generation diversity, length, factuality, отказами и целевыми задачами. Простота offline training снижает часть infrastructure, но preferred data distribution может не покрыть новые ответы policy. Сравнение с SFT baseline и сохранённым checkpoint требуется для всех этих методов.

<!-- /unit:999b053b2dc6208361 -->

<!-- unit:76a87667842452e70c -->
<a id="u-76a87667842452e70c"></a>

## Контроль понимания и эксперимент

<!-- /unit:76a87667842452e70c -->

<!-- unit:1fc49a72e6017fbef6 -->
<a id="u-1fc49a72e6017fbef6"></a>

1. Чем reference отличается от old PPO policy? **Ответ:** reference задаёт ориентир поведения/KL, old фиксирует sampling distribution rollout для ratio.
2. Обязательно ли обучать RM для DPO? **Ответ:** стандартный DPO использует preferences напрямую без отдельного RM в training loop.
3. Требует ли KTO пары chosen/rejected одного prompt? **Ответ:** нет, отдельные desirable/undesirable labels.
4. Доказывает ли маленький DPO loss правильные факты? **Ответ:** нет; он отражает objective на конкретных preferences.

<!-- /unit:1fc49a72e6017fbef6 -->

<!-- unit:e4f34c86d3744dd2db -->
<a id="u-e4f34c86d3744dd2db"></a>

**Практика:** пересчитать пример с $\beta=0.2$, затем поменять chosen/rejected местами. Проверить, что delta меняет знак и loss растёт. Для небольшого обучения сравнить baseline и DPO на одних prompts, срезах длины и factual tasks; проследить одновременно train margin, validation margin, chosen/rejected log-probs и task success. Все числа выше — ручные примеры, не результаты вашего запуска.

<!-- /unit:e4f34c86d3744dd2db -->

<!-- unit:052b71fc4588783fe4 -->
<a id="u-052b71fc4588783fe4"></a>

<!-- latex-display-v1 -->

<!-- /unit:052b71fc4588783fe4 -->

<!-- unit:604b29d1d24db41510 -->
<a id="u-604b29d1d24db41510"></a>

<!-- source-depth-adaptation-Е -->

<!-- /unit:604b29d1d24db41510 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как предпочтения превращаются в objective и каких данных он требует?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [AI feedback и continual learning](<adapt-continual.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
