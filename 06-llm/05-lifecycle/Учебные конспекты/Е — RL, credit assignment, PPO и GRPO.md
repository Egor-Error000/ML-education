# Е. Обучение через взаимодействие: от награды к изменению политики

[Оглавление](README.md). Пункт 6 плана. **Основа:** S3:343–423 вводит RLHF/PPO; общая адаптация раскрыта в соседнем разделе. Agentic rollouts, delayed reward и GRPO — новое углубление по плану, а не уже проведённое вами обучение. Первичные источники: [PPO](https://arxiv.org/abs/1707.06347), [DeepSeekMath, GRPO](https://arxiv.org/abs/2402.03300).

## 1. Что оптимизирует политика

В supervised learning мы сообщаем правильное действие. В RL получаем последствия действий и численную награду. Для траектории $\tau$ с действиями $a_t$, историями $h_t$ и наградами $r_t$:

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k,
\qquad
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G_0].
$$

При $\gamma=1$ и конечном горизонте все будущие rewards имеют одинаковый временной вес. Если нужен штраф за каждый tool call, его лучше явно включить в reward или ограничение, а не надеяться, что discount точно выразит стоимость. Например, успешная сессия с тремя вызовами при награде за успех 1 и штрафе $0.05$ за вызов получает $0.85$. Неуспешная без вызовов получает 0. Если штраф чрезмерен, политика может предпочесть ничего не делать; objective тогда расходится с намерением пользователя.

Среда выбирает observations через свой transition mechanism. При фиксированной среде градиент log-probability траектории по $\theta$ содержит только решения политики:

$$
\nabla_\theta J(\theta)=
\mathbb E\left[\sum_t \gamma^tG_t\nabla_\theta\log\pi_\theta(a_t\mid h_t)\right].
$$

Нельзя оптимизировать правдоподобие наблюдений базы данных так, будто их сгенерировала политика. Для LLM log-probability действия — сумма log-probabilities его tokens, обусловленных реальным context. Где закончился action и началось observation, определяет сериализация.


<!-- source-depth-lifecycle-Е -->
### 1.1. Rollout → reward → update: что меняется между итерациями

В начале iteration фиксируют behavior policy, task/environment distribution и reward protocol. Собирают sessions с actual actions/results, считают rewards/advantages и обновляют параметры. Затем новая policy создаёт новое распределение histories. Просто прогнать agent loop без optimizer —inference, а не RL training, даже если после каждого шага записано слово reward.

S3 описывал human-feedback reward model; в нашей toy среде reward может вычисляться проверяемым кодом. Это разные источники feedback для policy optimization. Learned reward model не обязателен всякому RL, а executable reward тоже может быть неполным proxy. Умение среды вернуть число ещё не доказывает, что это число соответствует реальному успеху задачи.

Для действующей policy probability траектории содержит initial-state distribution, policy actions и environment transitions. При фиксированном environment/Reward параметры θ входят через action probabilities. Log-derivative позволяет оценить gradient без дифференцирования внешней database/API. Это не значит, что action consequences игнорируются: они влияют на rewards и будущие histories, на которых вычисляются следующие decisions.

### 1.2. Уточняем discount в policy gradient

Для objective $J=\mathbb E[\sum_t\gamma^t r_t]$ и обычного суммирования по временным позициям trajectory корректный score-function expression содержит $\gamma^tG_t$. Иногда формулы без этого множителя используют γ=1 или иначе взвешенное state distribution; эти conventions нельзя молча смешивать. В этой главе формула выше приведена к explicit discounted-return convention.

Это видно на delayed reward: награда через два шага входит в J с весом γ². Относительный return G_t уже отсчитывает discount от t, поэтому для общего G0 остаётся внешний γ^t. В finite tool tasks часто выбирают γ=1 и задают actual action/token cost отдельно; тогда дополнительный множитель равен 1.

### 1.3. Микромодель gradient с двумя macro actions

Пусть policy выбирает один из двух заранее определённых plans. Успешный plan имеет return0.85, неуспешный−0.05. Вероятность первого $p=\sigma(\theta)$, поэтому $J=0.85p-0.05(1-p)=0.9p-0.05$ и $dJ/d\theta=0.9p(1-p)$. При p0.5 gradient0.225: ascent повышает вероятность хорошего plan.

Тот же результат получается как expectation score-function samples: для успешного choice derivative log p равен 1−p, для неуспешного derivative log(1−p)=−p. Тогда $p(0.85)(1-p)+(1-p)(-0.05)(-p)=0.9p(1-p)$. Пример показывает направление learning signal, а не реализованное обучение LLM или доказательство качества real reward.


## 2. Advantage и отложенный результат

Одинаковая награда 1 может означать большой успех на сложной задаче и обычный исход на простой. Baseline $b(h_t)$ позволяет оценивать преимущество действия относительно ожидаемого результата:

$$
A_t=G_t-b(h_t).
$$

В классическом actor–critic baseline часто представляет value model $V_\phi(h_t)$. Baseline, не зависящий от выбранного текущего действия при данном history, не меняет ожидаемый policy gradient, но способен снизить дисперсию. Плохой critic может дать шумный сигнал; это отдельная обучаемая часть системы.

Credit assignment — вопрос, какие решения ответственны за итог. Если финальный reward пришёл через 20 действий, один и тот же положительный сигнал может усиливать и полезный поиск, и лишние повторы. Return и advantage не доказывают причинный вклад каждого token. Более точная промежуточная обратная связь помогает, если соответствует цели; произвольный бонус «за длинное объяснение» может стимулировать многословие вместо успеха.


### 2.1. Почему baseline не должен зависеть от sampled action

При фиксированном history и action-independent b выполняется $\mathbb E_a[b\nabla\log\pi(a\mid h)]=b\nabla\sum_a\pi(a\mid h)=0$. Поэтому вычитание такого baseline не меняет expected score-function gradient при соответствующем estimator. Оно может уменьшить variance, если baseline полезен. Learned value function предсказывает expected future return, а не «насколько красив текущий текст».

В macro example p0.5 expected return0.4. С baseline0.4 advantages0.45 и−0.45. Оба sampled gradient contributions равны 0.225:0.45·0.5 и(−0.45)·(−0.5). Это особая симметрия toy case, не обещание нулевой variance любого critic. Неверный baseline может увеличить noise; action-dependent subtraction требует отдельного обоснования.

### 2.2. Temporal differences и GAE на трёх шагах

Для rewards[-0.05,−0.05,0.95],γ0.9 и value estimates[0.5,0.6,0.7,0] terminal TD residuals равны−0.01,−0.02,0.25. Последний future value0 используется потому, что это настоящее termination. Внешняя collector truncation не обязательно имеет нулевой future value —различие разобрано в главе Б.

GAE соединяет несколько таких residuals с весами $(\gamma\lambda)^l$. При λ1 первый advantage−0.01+0.9(−0.02)+0.9²·0.25=0.1745, то есть Monte-Carlo return0.6745 минус V0=0.5. При λ0 используется только первая residual−0.01. Это два разных estimates для текущего update; интерполяция управляет trade-off bootstrap bias/variance, а не создаёт идеальный credit assignment.

Если critic оценивает histories, он должен получать нужные observations/remaining budget. Value от полного simulator state и value от ограниченного learner context —разные conditioning assumptions. При shared actor/critic body нужно понимать, какие параметры обновляет value loss; четыре conceptual roles не обязаны быть четырьмя независимыми одинаковыми models.

### 2.3. Terminal reward не доказывает полезность каждого token

Успешная session могла содержать лишний search. При одном session-level positive advantage reinforcement может повысить likelihood и этого action. На других rollouts лишний search тратит budget и приводит к failure; learning должен сравнить последствия. Один успех не даёт causal label «каждый шаг оптимален».

Fine-grained reward/shaping может ускорять feedback, но меняет incentives. Бонус за каждый read поощряет повторы, если нет правильного ограничения. Стоимость tools, длина и task correctness сохраняются как отдельные metrics, даже когда объединены scalar reward. Иначе высокий return может скрыть отказ от сложных задач или эксплуатацию несовершенного проверяющего.


## 3. Зачем PPO ограничивает обновление

Rollouts собраны политикой $\pi_{\mathrm{old}}$, а обновляется $\pi_\theta$. Отношение вероятностей:

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid h_t)}{\pi_{\mathrm{old}}(a_t\mid h_t)}.
$$

PPO использует clipped surrogate, максимизируемый по $\theta$:

$$
\mathcal J_{\mathrm{clip}}=
\mathbb E\left[
\min\left(\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)
\right].
$$

Clipping ограничивает стимул слишком сильно повышать вероятность выгодных actions и слишком сильно понижать вероятность невыгодных по этим данным. Это не жёсткое глобальное ограничение KL и не гарантия монотонного роста task success. Несколько epochs на тех же rollouts требуют следить за устареванием данных и величиной изменения policy.

При token-level objective ratio считается для выбранного token на соответствующем префиксе; произведение вероятностей всей длинной сессии даёт другой estimator и часто большую дисперсию. Нормировка по tokens или trajectories тоже меняет weighting. В реализации эти решения должны быть явными.


### 3.1. Old policy, current policy и reference

Old policy собрала batch; её log probabilities для sampled tokens сохраняют и при обновлении трактуют как fixed denominator. Current policy меняется optimizer steps. Reference policy —отдельный anchor для KL/behavior drift по выбранной постановке; она не обязана совпадать с old после первого раунда. Если denominator пересчитать уже изменённой policy как той же самой current, ratio искусственно станет 1 и objective изменится.

При actual sampling нужно согласовать definition probability с temperature, constraints и action-token mask. Пропущенные protocol differences способны исказить ratios независимо от хорошей reward model. Tool observations относятся к context следующих policy tokens; они не являются новыми actions, которым присваивается policy likelihood как собственному output.

### 3.2. PPO clipping со знаками advantage

При ε0.2,A2 и ratio1.5 raw term3,clipped2.4,min2.4. Дальше увеличивать likelihood этого action по этому positive term уже невыгодно. Но при ratio0.5 positive term остаётся 1, а не 1.6: падение likelihood полезного action не маскируется clipping.

Для A−2,ratio0.5 raw−1,clipped−1.6,min−1.6: улучшение от чрезмерного уменьшения bad-action likelihood ограничено. При ratio1.5 raw−3,clipped−2.4,min−3: увеличение bad-action likelihood всё ещё наказывается. Четыре случая показывают, почему нельзя заменить min просто утверждением «ratio всегда находится между 0.8 и 1.2».

На самом деле parameters общие для многих tokens/states; updates других terms могут двигать ratio дальше. PPO surrogate не проецирует всю policy в жёсткий trust region и не гарантирует monotonic task success. Sign loss также важен: max J реализуют как min(−J) плюс объявленные дополнительные losses, а не минимизацию самого полезного surrogate.

### 3.3. Token ratios и длинная trajectory

Log probability целого action —сумма token log probabilities, probability —их произведение. Если разница log-prob на каждом из 100 tokens равна 0.1, whole-sequence ratio $\exp(10)\approx22026.47$. Множество небольших изменений может дать огромный sequence ratio. Token-level clipped surrogate и whole-session importance sampling —разные estimators.

PPO обычно использует sampled prefixes/states с local ratios как surrogate. Это не означает, что один token ratio точно исправляет всю разницу occupancy distribution между old и current. Несколько optimization epochs на fixed batch повышают data reuse, но stale trajectories и KL и drift ratios требуют мониторинга. Сохранённые old probabilities относятся к тому же serialized input и реально выбранным targets.


## 4. Group-relative advantage в GRPO

В базовой идее GRPO для одного prompt/задания собирается группа из $G$ кандидатов с rewards $R_i$. Отдельный learned critic заменяется относительным сравнением внутри группы:

$$
\overline R=\frac1G\sum_{i=1}^G R_i,
\quad
s_R=\sqrt{\frac1G\sum_{i=1}^G(R_i-\overline R)^2},
\quad
\widehat A_i=\frac{R_i-\overline R}{s_R+\delta}.
$$

Здесь показана population-нормировка и стабилизатор $\delta>0$; конкретная реализация может выбирать другую конвенцию. При rewards $(1,1,0,0)$ имеем $\overline R=0.5$, $s_R=0.5$ и advantages приблизительно $(1,1,-1,-1)$. Если все четыре rewards равны, числители равны нулю: сравнительного reward-сигнала нет. Это бывает и когда всё слишком сложно, и когда всё уже решается. Нулевая дисперсия не доказывает, что модель неспособна обучаться вообще.

Типовая схема objective усредняет clipped token-level terms по ответам и добавляет штраф отклонения от reference policy. Для действий, перемежающихся observations, reward относится к целой rollout-сессии, но обновление применяется только к policy tokens. Перенос GRPO из задач с одним ответом в длинную среду требует отдельно определить группу, stop condition, длиновую нормировку и rollout budget.

Не путайте $\pi_{\mathrm{old}}$ и $\pi_{\mathrm{ref}}$. Первая собрала текущие данные и обновляется между итерациями. Вторая служит якорем поведения, например SFT-моделью. Они могут совпасть в начале, но играют разные роли. Для exact KL на фиксированном history:

$$
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
=\sum_a\pi_\theta(a\mid h)
\log\frac{\pi_\theta(a\mid h)}{\pi_{\mathrm{ref}}(a\mid h)}.
$$

Практические sampled estimators требуют своего обоснования; один log-ratio для отдельного token может быть отрицательным, хотя полный KL неотрицателен. Нельзя объявлять любой tokenwise log-ratio готовой точной KL-метрикой.


### 4.1. Группа должна отвечать одной сопоставимой задаче

В базовой идее GRPO несколько completions/rollouts одного prompt дают относительный reward signal без отдельного learned value critic. Для agentic task нужно ещё зафиксировать initial environment/scope/version и budget. Если одна trajectory получила свежий источник, другая —случайно отсутствующий, reward differences содержат environment noise вместе с action quality.

Нельзя формировать одну group из «лёгкий task все 1» и «трудный task все 0», а затем объявлять advantages сравнением действий при одном state. Это смешает difficulty и decision quality. Reward variance полезна, когда она несёт информацию о разных доступных choices при сопоставимой задаче.

Групповой mean включает reward самого sample; вместе с normalization это не тот же estimator, что произвольный action-independent baseline из предыдущего раздела. Отсутствие critic не доказывает unbiasedness или минимальную variance любого group-relative signal. Конкретный algorithm/normalization нужно определять явно.

### 4.2. Один success в группе и нулевая variance

Для rewards[1,0,0,0] population mean0.25,std√(3/16)≈0.433013. Без существенного δ positive advantage√3≈1.732051, три остальных−1/√3≈−0.577350. При[1,1,0,0] получаем±1. Если использовать sample std вместо population для этой второй группы, advantages будут±√(3/4)≈0.866025. Эти conventions меняют scale update.

При all-zero или all-one group centered numerators0: comparative reward advantage0. Stabilizer предотвращает 0/0, но сам не добавляет новый успешный trajectory. KL/другие terms всё ещё могут давать gradient, и другие tasks в batch могут обучаться. Следовательно, «эта group не даёт reward difference» не равно «всё обучение невозможно».

При IID binary success p и group size4 probability all equal равна $p^4+(1-p)^4$. Для p0.5 это 0.125, для p0.1 —0.6562. Слишком трудные/лёгкие tasks часто не дают comparative signal в маленьких groups. Увеличение group size, curriculum или иной reward могут помочь по ситуации, но стоят дополнительных rollouts и требуют собственной оценки.

### 4.3. Length weighting, masks и варианты реализации

Session-level reward может назначаться всем policy tokens trajectory, но observation spans не становятся самостоятельными sampled actions. Сначала определяют mask, затем reduction: среднее внутри session и по sessions либо общее token-average —разные weights длинных ответов. Длинный tool result может увеличивать context cost без увеличения числа trainable action targets.

[DeepSeekMath](https://arxiv.org/abs/2402.03300) ввёл GRPO в конкретной mathematical setting. Современная [документация TRL](https://huggingface.co/docs/trl/grpo_trainer) содержит несколько loss/reward-scaling вариантов и configurable KL. Поэтому название GRPOTrainer не определяет одну неизменную formula/default; фиксируют version, loss type, normalization, masking и β. Здесь изучается базовая идея и declared examples, без запуска trainer.

Rollout budget считается полностью:8tasks×4trajectories=32episodes. Если каждая имеет 5 environment actions, это 160actions; если из них 3 tool calls, всего 96tool calls. Это не 32calls и не 160model tokens. Reward endpoints, judge и reference/value passes добавляют собственные computation costs.

### 4.4. Exact KL и sampled terms

Для current Bernoulli probabilities[0.75,0.25],reference[0.5,0.5] exact KL≈0.130812. Но log-ratio второго action ln(0.25/0.5)=−ln2<0. Один observed negative log-ratio не противоречит неотрицательности полной expectation KL.

Есть и pointwise nonnegative expression: при x=πref(a)/πcurrent(a) величина x−lnx−1≥0. Если sample действительно из πcurrent и support/normalization подходят, её expectation равна KL(current||ref): expectation x=1, остаётся expectation log(current/ref). На old-policy batch без нужных условий нельзя автоматически объявить любую такую average точной current KL; estimator и gradient treatment требуют отдельного определения.

У реальной системы KL на sampled prefixes не гарантирует отсутствие каждого недопустимого действия и не сохраняет правильные sources. Это regularization/control drift, а не разрешение игнорировать environment oracle. Варианты β0 и β>0 задают разную regularization; отсутствие learned critic не означает отсутствие других memory/computation расходов.


## 5. Reward hacking и toy experiment

Проверяющий код должен наблюдать реальное состояние среды. Если reward выдаётся за фразу «задача выполнена», модель может выучить эту фразу. Если бонус положен за каждый успешный read, можно бесконечно читать один документ. Если оцениваются только видимые unit tests, возможно переобучение на их частные случаи. Training reward и независимый held-out oracle нужны раздельно.

До RL запустите random, простую heuristic и SFT policy. Иначе сложно понять, нужен ли сложный метод задаче, решаемой фиксированным правилом. Для первого опыта допустима маленькая policy в toy environment: цель — понять цикл rollout → reward → advantage → update → новая оценка. Полноразмерное обучение LLM не является обязательным доказательством знания механики.

Логировать следует reward, task success, длины, долю invalid actions, KL, entropy, стоимость и seeds. Сравнивать checkpoints по validation, а итоговый вывод делать на отложенных задачах. Рост reward при падении success — повод исправить objective/верификацию, а не объявлять успешное обучение.

### 5.1. Возвращаемся к RLHF source: четыре роли не обязательные четыре GPU-копии

S3 говорил о policy,reference,reward,value models как одновременно обязательных четырёх models на GPU. Это полезные роли классической схемы, но слишком сильное infrastructural обобщение. Reward может быть executable, critic может иметь shared body/head, reference можно реализовать по supported adapter/compute protocol, а размещение/offload зависит от architecture. Нужно считать actual tensors/activations/optimizer states и rollout compute.

Неверно и обещание, что optimizer неизбежно «найдёт любой взлом». Он оптимизирует доступный proxy и может использовать его flaws, но конкретный результат зависит от exploration/model/data. Counterexamples нужны для проверки reward, а не как универсальное предсказание обязательной дегенерации. SFT не обязан всегда сопровождаться PPO, если задачи/данные этого не требуют.

### 5.2. Что проверять до объявления успеха RL

В toy environment сначала фиксируют reward-independent held-out oracle и random/heuristic/SFT baseline. Затем проверяют rollout identity, action/observation masks, terminal/truncation, returns/advantages и знаки objective. Growth reward сравнивают с task success,recovery,invalid calls,length/cost,KL/entropy и regressions. Смена reward/schema посреди эксперимента требует новой интерпретации результатов.

Маленькая числовая модель или записанные trajectories достаточны для понимания formula, но не доказывают performance trained LLM. Здесь проверены вычисления и incentives; full-scale PPO/GRPO training и external rollouts не заявляются. Следующая глава разберёт memory, которая может менять доступный context при неизменных weights.


## 6. Контроль и практика

1. Почему не дифференцируем внешний database response как действие LLM? **Ответ:** его породила среда; параметры policy управляют выбором actions, а response является условием следующих решений.
2. Чем GRPO отличается от обычного learned critic? **Ответ:** относительный baseline/scale берутся из группы rewards, без отдельной value model в базовой постановке.
3. Является ли clipping гарантией безопасности или роста качества? **Ответ:** нет; это часть оптимизационного surrogate, а качество зависит от данных, reward и оценки.

4. Когда нужен γ^t вне return-to-go? **Ответ:** при objective discounted G0 и обычном суммировании trajectory positions; другие conventions надо объявить.
5. Почему ratio не обязан остаться внутри clip interval? **Ответ:** clipping меняет surrogate incentives, а не жёстко ограничивает все shared parameters.
6. Зачем сохранять old log probabilities? **Ответ:** это fixed denominator behavior policy; current update не должен незаметно менять его.
7. Нужно ли для любой RL среды обучать reward model? **Ответ:** нет, reward может быть executable; его соответствие задаче всё равно проверяется.
8. Что означает all-equal group rewards? **Ответ:** нет comparative centered reward signal в этой group, но не обязательно нет всех training gradients.
9. Неотрицателен ли каждый sampled log-ratio? **Ответ:** нет; full KL —expectation с определённым distribution.
10. Может ли session success усиливать лишний action? **Ответ:** да, грубый shared reward не доказывает causal полезность каждого token.

**Практика:** вычислите advantages для $(0,0,0,0)$, $(1,1,0,0)$ и $(1,1,1,1)$. Предложите две policy, одна из которых получает высокий плохо заданный reward без выполнения задачи. Исправьте oracle и составьте план сравнения, включающий затраты неудачных rollouts.
