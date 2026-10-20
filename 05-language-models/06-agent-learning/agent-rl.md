# Обучение через взаимодействие: PPO и GRPO

**Уровень:** основа ветки.

**Предпосылки:** [Оценка траекторий и воспроизводимость](<agent-evaluation.md>), [Distribution shift, recovery и DAgger](<agent-recovery.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как отложенный reward меняет вероятность последовательности действий?





## Карта вычисления и границы этапов

![Обучение через взаимодействие: PPO и GRPO: входы, преобразования и выходы](<../../оформление/course/agent-rl.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Reward и targets используются в объявленном режиме обучения. При оценке политика фиксирована; success измеряется по всей попытке, включая повторы и бюджет.

<!-- unit:b78ee59978988933f2 -->
<a id="u-b78ee59978988933f2"></a>

## Что оптимизирует политика

<!-- /unit:b78ee59978988933f2 -->

<!-- unit:4aa2e11eb8b88d3c2e -->
<a id="u-4aa2e11eb8b88d3c2e"></a>

В supervised learning мы сообщаем правильное действие. В RL получаем последствия действий и численную награду. Для траектории $\tau$ с действиями $a_t$, историями $h_t$ и наградами $r_t$:

<!-- /unit:4aa2e11eb8b88d3c2e -->

<!-- unit:085ee8484d50a59237 -->
<a id="u-085ee8484d50a59237"></a>

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k,
\qquad
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G_0].
$$

<!-- /unit:085ee8484d50a59237 -->

<!-- unit:aa4a016d83aec04349 -->
<a id="u-aa4a016d83aec04349"></a>

При $\gamma=1$ и конечном горизонте все будущие rewards имеют одинаковый временной вес. Если нужен штраф за каждый tool call, его лучше явно включить в reward или ограничение, а не надеяться, что discount точно выразит стоимость. Например, успешная сессия с тремя вызовами при награде за успех 1 и штрафе $0.05$ за вызов получает $0.85$. Неуспешная без вызовов получает 0. Если штраф чрезмерен, политика может предпочесть ничего не делать; objective тогда расходится с намерением пользователя.

<!-- /unit:aa4a016d83aec04349 -->

<!-- unit:3451f3697f75db429a -->
<a id="u-3451f3697f75db429a"></a>

Среда выбирает observations через свой transition mechanism. При фиксированной среде градиент log-probability траектории по $\theta$ содержит только решения политики:

<!-- /unit:3451f3697f75db429a -->

<!-- unit:5d8171b8addb91fadf -->
<a id="u-5d8171b8addb91fadf"></a>

$$
\nabla_\theta J(\theta)=
\mathbb E\left[\sum_t \gamma^tG_t\nabla_\theta\log\pi_\theta(a_t\mid h_t)\right].
$$

<!-- /unit:5d8171b8addb91fadf -->

<!-- unit:96cd654d2dd7747da2 -->
<a id="u-96cd654d2dd7747da2"></a>

Нельзя оптимизировать правдоподобие наблюдений базы данных так, будто их сгенерировала политика. Для LLM log-probability действия — сумма log-probabilities его tokens, обусловленных реальным context. Где закончился action и началось observation, определяет сериализация.

<!-- /unit:96cd654d2dd7747da2 -->

<!-- unit:831a9cf17ad9decb6c -->
<a id="u-831a9cf17ad9decb6c"></a>

<!-- source-depth-lifecycle-Е -->

<!-- /unit:831a9cf17ad9decb6c -->

<!-- unit:f5329ef93373c2cc6d -->
<a id="u-f5329ef93373c2cc6d"></a>

### Rollout → reward → update: что меняется между итерациями

<!-- /unit:f5329ef93373c2cc6d -->

<!-- unit:a039c7829afa60cdea -->
<a id="u-a039c7829afa60cdea"></a>

В начале iteration фиксируют behavior policy, task/environment distribution и reward protocol. Собирают sessions с actual actions/results, считают rewards/advantages и обновляют параметры. Затем новая policy создаёт новое распределение histories. Просто прогнать agent loop без optimizer —inference, а не RL training, даже если после каждого шага записано слово reward.

<!-- /unit:a039c7829afa60cdea -->

<!-- unit:6f4cc644f384aaa43b -->
<a id="u-6f4cc644f384aaa43b"></a>

S3 описывал human-feedback reward model; в нашей toy среде reward может вычисляться проверяемым кодом. Это разные источники feedback для policy optimization. Learned reward model не обязателен всякому RL, а executable reward тоже может быть неполным proxy. Умение среды вернуть число ещё не доказывает, что это число соответствует реальному успеху задачи.

<!-- /unit:6f4cc644f384aaa43b -->

<!-- unit:86af100abde6f6b22a -->
<a id="u-86af100abde6f6b22a"></a>

Для действующей policy probability траектории содержит initial-state distribution, policy actions и environment transitions. При фиксированном environment/Reward параметры θ входят через action probabilities. Log-derivative позволяет оценить gradient без дифференцирования внешней database/API. Это не значит, что action consequences игнорируются: они влияют на rewards и будущие histories, на которых вычисляются следующие decisions.

<!-- /unit:86af100abde6f6b22a -->

<!-- unit:fd27e6c112de3bbec2 -->
<a id="u-fd27e6c112de3bbec2"></a>

### Уточняем discount в policy gradient

<!-- /unit:fd27e6c112de3bbec2 -->

<!-- unit:08395dd6276316a208 -->
<a id="u-08395dd6276316a208"></a>

Для objective $J=\mathbb E[\sum_t\gamma^t r_t]$ и обычного суммирования по временным позициям trajectory корректный score-function expression содержит $\gamma^tG_t$. Иногда формулы без этого множителя используют γ=1 или иначе взвешенное state distribution; эти conventions нельзя молча смешивать. В этой главе формула выше приведена к explicit discounted-return convention.

<!-- /unit:08395dd6276316a208 -->

<!-- unit:a9145ccb6a5a55ad9b -->
<a id="u-a9145ccb6a5a55ad9b"></a>

Это видно на delayed reward: награда через два шага входит в J с весом γ². Относительный return G_t уже отсчитывает discount от t, поэтому для общего G0 остаётся внешний γ^t. В finite tool tasks часто выбирают γ=1 и задают actual action/token cost отдельно; тогда дополнительный множитель равен 1.

<!-- /unit:a9145ccb6a5a55ad9b -->

<!-- unit:33258824d94b206f81 -->
<a id="u-33258824d94b206f81"></a>

### Микромодель gradient с двумя macro actions

<!-- /unit:33258824d94b206f81 -->

<!-- unit:21bdd6ad0f3a4ce9f6 -->
<a id="u-21bdd6ad0f3a4ce9f6"></a>

Пусть policy выбирает один из двух заранее определённых plans. Успешный plan имеет return0.85, неуспешный−0.05. Вероятность первого $p=\sigma(\theta)$, поэтому $J=0.85p-0.05(1-p)=0.9p-0.05$ и $dJ/d\theta=0.9p(1-p)$. При p0.5 gradient0.225: ascent повышает вероятность хорошего plan.

<!-- /unit:21bdd6ad0f3a4ce9f6 -->

<!-- unit:ee4756ed0dc0b666fe -->
<a id="u-ee4756ed0dc0b666fe"></a>

Тот же результат получается как expectation score-function samples: для успешного choice derivative log p равен 1−p, для неуспешного derivative log(1−p)=−p. Тогда $p(0.85)(1-p)+(1-p)(-0.05)(-p)=0.9p(1-p)$. Пример показывает направление learning signal, а не реализованное обучение LLM или доказательство качества real reward.

<!-- /unit:ee4756ed0dc0b666fe -->

<!-- unit:e4d936b5446e2f8bd5 -->
<a id="u-e4d936b5446e2f8bd5"></a>

## Advantage и отложенный результат

<!-- /unit:e4d936b5446e2f8bd5 -->

<!-- unit:5ec427383bc9349866 -->
<a id="u-5ec427383bc9349866"></a>

Одинаковая награда 1 может означать большой успех на сложной задаче и обычный исход на простой. Baseline $b(h_t)$ позволяет оценивать преимущество действия относительно ожидаемого результата:

<!-- /unit:5ec427383bc9349866 -->

<!-- unit:1a08b8331ad7937306 -->
<a id="u-1a08b8331ad7937306"></a>

$$
A_t=G_t-b(h_t).
$$

<!-- /unit:1a08b8331ad7937306 -->

<!-- unit:55d96ffdce5382934e -->
<a id="u-55d96ffdce5382934e"></a>

В классическом actor–critic baseline часто представляет value model $V_\phi(h_t)$. Baseline, не зависящий от выбранного текущего действия при данном history, не меняет ожидаемый policy gradient, но способен снизить дисперсию. Плохой critic может дать шумный сигнал; это отдельная обучаемая часть системы.

<!-- /unit:55d96ffdce5382934e -->

<!-- unit:eafea9403544c77cf6 -->
<a id="u-eafea9403544c77cf6"></a>

Credit assignment — вопрос, какие решения ответственны за итог. Если финальный reward пришёл через 20 действий, один и тот же положительный сигнал может усиливать и полезный поиск, и лишние повторы. Return и advantage не доказывают причинный вклад каждого token. Более точная промежуточная обратная связь помогает, если соответствует цели; произвольный бонус «за длинное объяснение» может стимулировать многословие вместо успеха.

<!-- /unit:eafea9403544c77cf6 -->

<!-- unit:ed26a81287f49626aa -->
<a id="u-ed26a81287f49626aa"></a>

### Почему baseline не должен зависеть от sampled action

<!-- /unit:ed26a81287f49626aa -->

<!-- unit:fcb1508c84ed7145b5 -->
<a id="u-fcb1508c84ed7145b5"></a>

При фиксированном history и action-independent b выполняется $\mathbb E_a[b\nabla\log\pi(a\mid h)]=b\nabla\sum_a\pi(a\mid h)=0$. Поэтому вычитание такого baseline не меняет expected score-function gradient при соответствующем estimator. Оно может уменьшить variance, если baseline полезен. Learned value function предсказывает expected future return, а не «насколько красив текущий текст».

<!-- /unit:fcb1508c84ed7145b5 -->

<!-- unit:3bf48047682b53a76b -->
<a id="u-3bf48047682b53a76b"></a>

В macro example p0.5 expected return0.4. С baseline0.4 advantages0.45 и−0.45. Оба sampled gradient contributions равны 0.225:0.45·0.5 и(−0.45)·(−0.5). Это особая симметрия toy case, не обещание нулевой variance любого critic. Неверный baseline может увеличить noise; action-dependent subtraction требует отдельного обоснования.

<!-- /unit:3bf48047682b53a76b -->

<!-- unit:5646d3a712a06d2026 -->
<a id="u-5646d3a712a06d2026"></a>

### Temporal differences и GAE на трёх шагах

<!-- /unit:5646d3a712a06d2026 -->

<!-- unit:4c0e424505ec701054 -->
<a id="u-4c0e424505ec701054"></a>

Для rewards[-0.05,−0.05,0.95],γ0.9 и value estimates[0.5,0.6,0.7,0] terminal TD residuals равны−0.01,−0.02,0.25. Последний future value0 используется потому, что это настоящее termination. Внешняя collector truncation не обязательно имеет нулевой future value —различие разобрано в главе [«Среда, observations, actions и horizon»](<agent-environment.md>).

<!-- /unit:4c0e424505ec701054 -->

<!-- unit:987aa7aa92decbbe74 -->
<a id="u-987aa7aa92decbbe74"></a>

GAE соединяет несколько таких residuals с весами $(\gamma\lambda)^l$. При λ1 первый advantage−0.01+0.9(−0.02)+0.9²·0.25=0.1745, то есть Monte-Carlo return0.6745 минус V0=0.5. При λ0 используется только первая residual−0.01. Это два разных estimates для текущего update; интерполяция управляет trade-off bootstrap bias/variance, а не создаёт идеальный credit assignment.

<!-- /unit:987aa7aa92decbbe74 -->

<!-- unit:2e0605cd99551c76ed -->
<a id="u-2e0605cd99551c76ed"></a>

Если critic оценивает histories, он должен получать нужные observations/remaining budget. Value от полного simulator state и value от ограниченного learner context —разные conditioning assumptions. При shared actor/critic body нужно понимать, какие параметры обновляет value loss; четыре conceptual roles не обязаны быть четырьмя независимыми одинаковыми models.

<!-- /unit:2e0605cd99551c76ed -->

<!-- unit:4c53631ab4be5aad04 -->
<a id="u-4c53631ab4be5aad04"></a>

### Terminal reward не доказывает полезность каждого token

<!-- /unit:4c53631ab4be5aad04 -->

<!-- unit:883b4ef4d2324632da -->
<a id="u-883b4ef4d2324632da"></a>

Успешная session могла содержать лишний search. При одном session-level positive advantage reinforcement может повысить likelihood и этого action. На других rollouts лишний search тратит budget и приводит к failure; learning должен сравнить последствия. Один успех не даёт causal label «каждый шаг оптимален».

<!-- /unit:883b4ef4d2324632da -->

<!-- unit:d1cf7c31b879784bf1 -->
<a id="u-d1cf7c31b879784bf1"></a>

Fine-grained reward/shaping может ускорять feedback, но меняет incentives. Бонус за каждый read поощряет повторы, если нет правильного ограничения. Стоимость tools, длина и task correctness сохраняются как отдельные metrics, даже когда объединены scalar reward. Иначе высокий return может скрыть отказ от сложных задач или эксплуатацию несовершенного проверяющего.

<!-- /unit:d1cf7c31b879784bf1 -->

<!-- unit:a0a19de8f06801c631 -->
<a id="u-a0a19de8f06801c631"></a>

## Зачем PPO ограничивает обновление

<!-- /unit:a0a19de8f06801c631 -->

<!-- unit:30c03104f67d597ec2 -->
<a id="u-30c03104f67d597ec2"></a>

Rollouts собраны политикой $\pi_{\mathrm{old}}$, а обновляется $\pi_\theta$. Отношение вероятностей:

<!-- /unit:30c03104f67d597ec2 -->

<!-- unit:7cf0797109ccea1fef -->
<a id="u-7cf0797109ccea1fef"></a>

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid h_t)}{\pi_{\mathrm{old}}(a_t\mid h_t)}.
$$

<!-- /unit:7cf0797109ccea1fef -->

<!-- unit:570972699c37f1a4c0 -->
<a id="u-570972699c37f1a4c0"></a>

PPO использует clipped surrogate, максимизируемый по $\theta$:

<!-- /unit:570972699c37f1a4c0 -->

<!-- unit:8bf8c63d77e5727c1b -->
<a id="u-8bf8c63d77e5727c1b"></a>

$$
\mathcal J_{\mathrm{clip}}=
\mathbb E\left[
\min\left(\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)
\right].
$$

<!-- /unit:8bf8c63d77e5727c1b -->

<!-- unit:a3dd15baeac03d2b29 -->
<a id="u-a3dd15baeac03d2b29"></a>

Clipping ограничивает стимул слишком сильно повышать вероятность выгодных actions и слишком сильно понижать вероятность невыгодных по этим данным. Это не жёсткое глобальное ограничение KL и не гарантия монотонного роста task success. Несколько epochs на тех же rollouts требуют следить за устареванием данных и величиной изменения policy.

<!-- /unit:a3dd15baeac03d2b29 -->

<!-- unit:b2d99dd7b9c74423b9 -->
<a id="u-b2d99dd7b9c74423b9"></a>

При token-level objective ratio считается для выбранного token на соответствующем префиксе; произведение вероятностей всей длинной сессии даёт другой estimator и часто большую дисперсию. Нормировка по tokens или trajectories тоже меняет weighting. В реализации эти решения должны быть явными.

<!-- /unit:b2d99dd7b9c74423b9 -->

<!-- unit:4d0c1390eac3ac87ff -->
<a id="u-4d0c1390eac3ac87ff"></a>

### Old policy, current policy и reference

<!-- /unit:4d0c1390eac3ac87ff -->

<!-- unit:6f38d2a51fb432e576 -->
<a id="u-6f38d2a51fb432e576"></a>

Old policy собрала batch; её log probabilities для sampled tokens сохраняют и при обновлении трактуют как fixed denominator. Current policy меняется optimizer steps. Reference policy —отдельный anchor для KL/behavior drift по выбранной постановке; она не обязана совпадать с old после первого раунда. Если denominator пересчитать уже изменённой policy как той же самой current, ratio искусственно станет 1 и objective изменится.

<!-- /unit:6f38d2a51fb432e576 -->

<!-- unit:60b2aca0d01e910ab6 -->
<a id="u-60b2aca0d01e910ab6"></a>

При actual sampling нужно согласовать definition probability с temperature, constraints и action-token mask. Пропущенные protocol differences способны исказить ratios независимо от хорошей reward model. Tool observations относятся к context следующих policy tokens; они не являются новыми actions, которым присваивается policy likelihood как собственному output.

<!-- /unit:60b2aca0d01e910ab6 -->

<!-- unit:bba11843e57758da43 -->
<a id="u-bba11843e57758da43"></a>

### PPO clipping со знаками advantage

<!-- /unit:bba11843e57758da43 -->

<!-- unit:e05b2fcf57380b3551 -->
<a id="u-e05b2fcf57380b3551"></a>

При ε0.2,A2 и ratio1.5 raw term3,clipped2.4,min2.4. Дальше увеличивать likelihood этого action по этому positive term уже невыгодно. Но при ratio0.5 positive term остаётся 1, а не 1.6: падение likelihood полезного action не маскируется clipping.

<!-- /unit:e05b2fcf57380b3551 -->

<!-- unit:0bf892b9044f252559 -->
<a id="u-0bf892b9044f252559"></a>

Для A−2,ratio0.5 raw−1,clipped−1.6,min−1.6: улучшение от чрезмерного уменьшения bad-action likelihood ограничено. При ratio1.5 raw−3,clipped−2.4,min−3: увеличение bad-action likelihood всё ещё наказывается. Четыре случая показывают, почему нельзя заменить min просто утверждением «ratio всегда находится между 0.8 и 1.2».

<!-- /unit:0bf892b9044f252559 -->

<!-- unit:be3ce19a0da533728a -->
<a id="u-be3ce19a0da533728a"></a>

На самом деле parameters общие для многих tokens/states; updates других terms могут двигать ratio дальше. PPO surrogate не проецирует всю policy в жёсткий trust region и не гарантирует monotonic task success. Sign loss также важен: max J реализуют как min(−J) плюс объявленные дополнительные losses, а не минимизацию самого полезного surrogate.

<!-- /unit:be3ce19a0da533728a -->

<!-- unit:bfd4bdb2b736c2de41 -->
<a id="u-bfd4bdb2b736c2de41"></a>

### Token ratios и длинная trajectory

<!-- /unit:bfd4bdb2b736c2de41 -->

<!-- unit:18728d6d5ff58c263b -->
<a id="u-18728d6d5ff58c263b"></a>

Log probability целого action —сумма token log probabilities, probability —их произведение. Если разница log-prob на каждом из 100 tokens равна 0.1, whole-sequence ratio $\exp(10)\approx22026.47$. Множество небольших изменений может дать огромный sequence ratio. Token-level clipped surrogate и whole-session importance sampling —разные estimators.

<!-- /unit:18728d6d5ff58c263b -->

<!-- unit:46c2ae0a97ef966a63 -->
<a id="u-46c2ae0a97ef966a63"></a>

PPO обычно использует sampled prefixes/states с local ratios как surrogate. Это не означает, что один token ratio точно исправляет всю разницу occupancy distribution между old и current. Несколько optimization epochs на fixed batch повышают data reuse, но stale trajectories и KL и drift ratios требуют мониторинга. Сохранённые old probabilities относятся к тому же serialized input и реально выбранным targets.

<!-- /unit:46c2ae0a97ef966a63 -->

<!-- unit:40d0d6d0554d36e75b -->
<a id="u-40d0d6d0554d36e75b"></a>

## Group-relative advantage в GRPO

<!-- /unit:40d0d6d0554d36e75b -->

<!-- unit:d1d5026b3f65f1a393 -->
<a id="u-d1d5026b3f65f1a393"></a>

В базовой идее GRPO для одного prompt/задания собирается группа из $G$ кандидатов с rewards $R_i$. Отдельный learned critic заменяется относительным сравнением внутри группы:

<!-- /unit:d1d5026b3f65f1a393 -->

<!-- unit:2f201701d230246711 -->
<a id="u-2f201701d230246711"></a>

$$
\overline R=\frac1G\sum_{i=1}^G R_i,
\quad
s_R=\sqrt{\frac1G\sum_{i=1}^G(R_i-\overline R)^2},
\quad
\widehat A_i=\frac{R_i-\overline R}{s_R+\delta}.
$$

<!-- /unit:2f201701d230246711 -->

<!-- unit:5f0b0367de7c0ac13c -->
<a id="u-5f0b0367de7c0ac13c"></a>

Здесь показана population-нормировка и стабилизатор $\delta>0$; конкретная реализация может выбирать другую конвенцию. При rewards $(1,1,0,0)$ имеем $\overline R=0.5$, $s_R=0.5$ и advantages приблизительно $(1,1,-1,-1)$. Если все четыре rewards равны, числители равны нулю: сравнительного reward-сигнала нет. Это бывает и когда всё слишком сложно, и когда всё уже решается. Нулевая дисперсия не доказывает, что модель неспособна обучаться вообще.

<!-- /unit:5f0b0367de7c0ac13c -->

<!-- unit:738f145ad85736f26c -->
<a id="u-738f145ad85736f26c"></a>

Типовая схема objective усредняет clipped token-level terms по ответам и добавляет штраф отклонения от reference policy. Для действий, перемежающихся observations, reward относится к целой rollout-сессии, но обновление применяется только к policy tokens. Перенос GRPO из задач с одним ответом в длинную среду требует отдельно определить группу, stop condition, длиновую нормировку и rollout budget.

<!-- /unit:738f145ad85736f26c -->

<!-- unit:4df378b30ae2663ce7 -->
<a id="u-4df378b30ae2663ce7"></a>

Не путайте $\pi_{\mathrm{old}}$ и $\pi_{\mathrm{ref}}$. Первая собрала текущие данные и обновляется между итерациями. Вторая служит якорем поведения, например SFT-моделью. Они могут совпасть в начале, но играют разные роли. Для exact KL на фиксированном history:

<!-- /unit:4df378b30ae2663ce7 -->

<!-- unit:e7c25c645c1b24000c -->
<a id="u-e7c25c645c1b24000c"></a>

$$
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
=\sum_a\pi_\theta(a\mid h)
\log\frac{\pi_\theta(a\mid h)}{\pi_{\mathrm{ref}}(a\mid h)}.
$$

<!-- /unit:e7c25c645c1b24000c -->

<!-- unit:5c7ecd03fb56effac5 -->
<a id="u-5c7ecd03fb56effac5"></a>

Практические sampled estimators требуют своего обоснования; один log-ratio для отдельного token может быть отрицательным, хотя полный KL неотрицателен. Нельзя объявлять любой tokenwise log-ratio готовой точной KL-метрикой.

<!-- /unit:5c7ecd03fb56effac5 -->

<!-- unit:6eda272b7160558729 -->
<a id="u-6eda272b7160558729"></a>

### Группа должна отвечать одной сопоставимой задаче

<!-- /unit:6eda272b7160558729 -->

<!-- unit:05ce0e8995b7eaaa40 -->
<a id="u-05ce0e8995b7eaaa40"></a>

В базовой идее GRPO несколько completions/rollouts одного prompt дают относительный reward signal без отдельного learned value critic. Для agentic task нужно ещё зафиксировать initial environment/scope/version и budget. Если одна trajectory получила свежий источник, другая —случайно отсутствующий, reward differences содержат environment noise вместе с action quality.

<!-- /unit:05ce0e8995b7eaaa40 -->

<!-- unit:fd81fc58086e43bfbe -->
<a id="u-fd81fc58086e43bfbe"></a>

Нельзя формировать одну group из «лёгкий task все 1» и «трудный task все 0», а затем объявлять advantages сравнением действий при одном state. Это смешает difficulty и decision quality. Reward variance полезна, когда она несёт информацию о разных доступных choices при сопоставимой задаче.

<!-- /unit:fd81fc58086e43bfbe -->

<!-- unit:cc41e2e5c2072888b6 -->
<a id="u-cc41e2e5c2072888b6"></a>

Групповой mean включает reward самого sample; вместе с normalization это не тот же estimator, что произвольный action-independent baseline из предыдущего раздела. Отсутствие critic не доказывает unbiasedness или минимальную variance любого group-relative signal. Конкретный algorithm/normalization нужно определять явно.

<!-- /unit:cc41e2e5c2072888b6 -->

<!-- unit:b9725429ec7a6b60a3 -->
<a id="u-b9725429ec7a6b60a3"></a>

### Один success в группе и нулевая variance

<!-- /unit:b9725429ec7a6b60a3 -->

<!-- unit:194cefb511e8249fb2 -->
<a id="u-194cefb511e8249fb2"></a>

Для rewards[1,0,0,0] population mean0.25,std√(3/16)≈0.433013. Без существенного δ positive advantage√3≈1.732051, три остальных−1/√3≈−0.577350. При[1,1,0,0] получаем±1. Если использовать sample std вместо population для этой второй группы, advantages будут±√(3/4)≈0.866025. Эти conventions меняют scale update.

<!-- /unit:194cefb511e8249fb2 -->

<!-- unit:cf568f5bf31bb3b1e4 -->
<a id="u-cf568f5bf31bb3b1e4"></a>

При all-zero или all-one group centered numerators0: comparative reward advantage0. Stabilizer предотвращает 0/0, но сам не добавляет новый успешный trajectory. KL/другие terms всё ещё могут давать gradient, и другие tasks в batch могут обучаться. Следовательно, «эта group не даёт reward difference» не равно «всё обучение невозможно».

<!-- /unit:cf568f5bf31bb3b1e4 -->

<!-- unit:abbc2962007a5b0674 -->
<a id="u-abbc2962007a5b0674"></a>

При IID binary success p и group size4 probability all equal равна $p^4+(1-p)^4$. Для p0.5 это 0.125, для p0.1 —0.6562. Слишком трудные/лёгкие tasks часто не дают comparative signal в маленьких groups. Увеличение group size, curriculum или иной reward могут помочь по ситуации, но стоят дополнительных rollouts и требуют собственной оценки.

<!-- /unit:abbc2962007a5b0674 -->

<!-- unit:8ef0ce9abb2551b6a5 -->
<a id="u-8ef0ce9abb2551b6a5"></a>

### Length weighting, masks и варианты реализации

<!-- /unit:8ef0ce9abb2551b6a5 -->

<!-- unit:27b9e365f6fae33b90 -->
<a id="u-27b9e365f6fae33b90"></a>

Session-level reward может назначаться всем policy tokens trajectory, но observation spans не становятся самостоятельными sampled actions. Сначала определяют mask, затем reduction: среднее внутри session и по sessions либо общее token-average —разные weights длинных ответов. Длинный tool result может увеличивать context cost без увеличения числа trainable action targets.

<!-- /unit:27b9e365f6fae33b90 -->

<!-- unit:2fe5bbdccc95dd40ea -->
<a id="u-2fe5bbdccc95dd40ea"></a>

[DeepSeekMath](https://arxiv.org/abs/2402.03300) ввёл GRPO в конкретной mathematical setting. Современная [документация TRL](https://huggingface.co/docs/trl/grpo_trainer) содержит несколько loss/reward-scaling вариантов и configurable KL. Поэтому название GRPOTrainer не определяет одну неизменную formula/default; фиксируют version, loss type, normalization, masking и β. Здесь изучается базовая идея и declared examples, без запуска trainer.

<!-- /unit:2fe5bbdccc95dd40ea -->

<!-- unit:b99279dccc9de56575 -->
<a id="u-b99279dccc9de56575"></a>

Rollout budget считается полностью:8tasks×4trajectories=32episodes. Если каждая имеет 5 environment actions, это 160actions; если из них 3 tool calls, всего 96tool calls. Это не 32calls и не 160model tokens. Reward endpoints, judge и reference/value passes добавляют собственные computation costs.

<!-- /unit:b99279dccc9de56575 -->

<!-- unit:3bac7d517884b626c5 -->
<a id="u-3bac7d517884b626c5"></a>

### Exact KL и sampled terms

<!-- /unit:3bac7d517884b626c5 -->

<!-- unit:e4fc9e748f0c1c9622 -->
<a id="u-e4fc9e748f0c1c9622"></a>

Для current Bernoulli probabilities[0.75,0.25],reference[0.5,0.5] exact KL≈0.130812. Но log-ratio второго action ln(0.25/0.5)=−ln2<0. Один observed negative log-ratio не противоречит неотрицательности полной expectation KL.

<!-- /unit:e4fc9e748f0c1c9622 -->

<!-- unit:33c27bdbeaf2ba9a6f -->
<a id="u-33c27bdbeaf2ba9a6f"></a>

Есть и pointwise nonnegative expression: при x=πref(a)/πcurrent(a) величина x−lnx−1≥0. Если sample действительно из πcurrent и support/normalization подходят, её expectation равна KL(current||ref): expectation x=1, остаётся expectation log(current/ref). На old-policy batch без нужных условий нельзя автоматически объявить любую такую average точной current KL; estimator и gradient treatment требуют отдельного определения.

<!-- /unit:33c27bdbeaf2ba9a6f -->

<!-- unit:36fe603c0175bbdd92 -->
<a id="u-36fe603c0175bbdd92"></a>

У реальной системы KL на sampled prefixes не гарантирует отсутствие каждого недопустимого действия и не сохраняет правильные sources. Это regularization/control drift, а не разрешение игнорировать environment oracle. Варианты β0 и β>0 задают разную regularization; отсутствие learned critic не означает отсутствие других memory/computation расходов.

<!-- /unit:36fe603c0175bbdd92 -->

<!-- unit:5b7aea0f62a23c8a02 -->
<a id="u-5b7aea0f62a23c8a02"></a>

## Reward hacking и toy experiment

<!-- /unit:5b7aea0f62a23c8a02 -->

<!-- unit:bd51ba9984078eca7c -->
<a id="u-bd51ba9984078eca7c"></a>

Проверяющий код должен наблюдать реальное состояние среды. Если reward выдаётся за фразу «задача выполнена», модель может выучить эту фразу. Если бонус положен за каждый успешный read, можно бесконечно читать один документ. Если оцениваются только видимые unit tests, возможно переобучение на их частные случаи. Training reward и независимый held-out oracle нужны раздельно.

<!-- /unit:bd51ba9984078eca7c -->

<!-- unit:a41fdfa44b46827a9f -->
<a id="u-a41fdfa44b46827a9f"></a>

До RL запустите random, простую heuristic и SFT policy. Иначе сложно понять, нужен ли сложный метод задаче, решаемой фиксированным правилом. Для первого опыта допустима маленькая policy в toy environment: цель — понять цикл rollout → reward → advantage → update → новая оценка. Полноразмерное обучение LLM не является обязательным доказательством знания механики.

<!-- /unit:a41fdfa44b46827a9f -->

<!-- unit:4a4b57b6896642926e -->
<a id="u-4a4b57b6896642926e"></a>

Логировать следует reward, task success, длины, долю invalid actions, KL, entropy, стоимость и seeds. Сравнивать checkpoints по validation, а итоговый вывод делать на отложенных задачах. Рост reward при падении success — повод исправить objective/верификацию, а не объявлять успешное обучение.

<!-- /unit:4a4b57b6896642926e -->

<!-- unit:5474068812828e8927 -->
<a id="u-5474068812828e8927"></a>

### Возвращаемся к RLHF source: четыре роли не обязательные четыре GPU-копии

<!-- /unit:5474068812828e8927 -->

<!-- unit:670ff2e3663a955b9f -->
<a id="u-670ff2e3663a955b9f"></a>

S3 говорил о policy,reference,reward,value models как одновременно обязательных четырёх models на GPU. Это полезные роли классической схемы, но слишком сильное infrastructural обобщение. Reward может быть executable, critic может иметь shared body/head, reference можно реализовать по supported adapter/compute protocol, а размещение/offload зависит от architecture. Нужно считать actual tensors/activations/optimizer states и rollout compute.

<!-- /unit:670ff2e3663a955b9f -->

<!-- unit:64d346567a02aa3f86 -->
<a id="u-64d346567a02aa3f86"></a>

Неверно и обещание, что optimizer неизбежно «найдёт любой взлом». Он оптимизирует доступный proxy и может использовать его flaws, но конкретный результат зависит от exploration/model/data. Counterexamples нужны для проверки reward, а не как универсальное предсказание обязательной дегенерации. SFT не обязан всегда сопровождаться PPO, если задачи/данные этого не требуют.

<!-- /unit:64d346567a02aa3f86 -->

<!-- unit:32cb85f2cd344ee5be -->
<a id="u-32cb85f2cd344ee5be"></a>

### Что проверять до объявления успеха RL

<!-- /unit:32cb85f2cd344ee5be -->

<!-- unit:7c7600c21e059c15cb -->
<a id="u-7c7600c21e059c15cb"></a>

В toy environment сначала фиксируют reward-independent held-out oracle и random/heuristic/SFT baseline. Затем проверяют rollout identity, action/observation masks, terminal/truncation, returns/advantages и знаки objective. Growth reward сравнивают с task success,recovery,invalid calls,length/cost,KL/entropy и regressions. Смена reward/schema посреди эксперимента требует новой интерпретации результатов.

<!-- /unit:7c7600c21e059c15cb -->

<!-- unit:d0d4b37fa3c662c83a -->
<a id="u-d0d4b37fa3c662c83a"></a>

Маленькая числовая модель или записанные trajectories достаточны для понимания formula, но не доказывают performance trained LLM. Здесь проверены вычисления и incentives; full-scale PPO/GRPO training и external rollouts не заявляются. Следующая глава разберёт memory, которая может менять доступный context при неизменных weights.

<!-- /unit:d0d4b37fa3c662c83a -->

<!-- unit:7909145eca8f297e44 -->
<a id="u-7909145eca8f297e44"></a>

### Исследовательский пример многошагового RL: AgentGym-RL

<!-- /unit:7909145eca8f297e44 -->

<!-- unit:5fcb097b2a587e8bdc -->
<a id="u-5fcb097b2a587e8bdc"></a>

Опубликованный AgentGym-RL отделяет environment, agent и RL training module и поддерживает интерактивные multi-turn rollouts; в статье также предложен ScalingInter-RL с staged interaction horizon. Это показывает, что agent training требует не только policy loss: нужны воспроизводимая среда, protocol действий, rollout store, terminal/reward semantics и ограниченный бюджет взаимодействий.

<!-- /unit:5fcb097b2a587e8bdc -->

<!-- unit:33446bf76101861301 -->
<a id="u-33446bf76101861301"></a>

В статье сообщаются сравнения на web navigation, deep search, games, embodied tasks и science tasks, а также результаты разных открытых/закрытых моделей. Их нельзя переносить как гарантию улучшения на произвольный tool environment: каждый benchmark задаёт API, task distribution, reward/verifier, initialization и computation. Для собственного сравнения задайте эти поля заранее и дополните reward независимым task oracle; измеряйте success и все затраты, включая failed rollouts.

<!-- /unit:33446bf76101861301 -->

<!-- unit:a01f85122c3cd40106 -->
<a id="u-a01f85122c3cd40106"></a>

**Источник и границы:** [AgentGym-RL, ICLR 2026](https://openreview.net/pdf/d8e15fc2cabf98ead8ffbe0d8e6ad1b2764a3d28.pdf), также [версия статьи на arXiv](https://arxiv.org/abs/2509.08755). Утверждения о performance и стабильности относятся к experiment suite авторов; данный учебный материал не содержит пользовательского запуска AgentGym-RL.

<!-- /unit:a01f85122c3cd40106 -->

<!-- unit:64883ccc21dfa82589 -->
<a id="u-64883ccc21dfa82589"></a>

## Контроль и практика

<!-- /unit:64883ccc21dfa82589 -->

<!-- unit:aba233fde28424f256 -->
<a id="u-aba233fde28424f256"></a>

1. Почему не дифференцируем внешний database response как действие LLM? **Ответ:** его породила среда; параметры policy управляют выбором actions, а response является условием следующих решений.
2. Чем GRPO отличается от обычного learned critic? **Ответ:** относительный baseline/scale берутся из группы rewards, без отдельной value model в базовой постановке.
3. Является ли clipping гарантией безопасности или роста качества? **Ответ:** нет; это часть оптимизационного surrogate, а качество зависит от данных, reward и оценки.

<!-- /unit:aba233fde28424f256 -->

<!-- unit:2497642cb4df6cac32 -->
<a id="u-2497642cb4df6cac32"></a>

4. Когда нужен γ^t вне return-to-go? **Ответ:** при objective discounted G0 и обычном суммировании trajectory positions; другие conventions надо объявить.
5. Почему ratio не обязан остаться внутри clip interval? **Ответ:** clipping меняет surrogate incentives, а не жёстко ограничивает все shared parameters.
6. Зачем сохранять old log probabilities? **Ответ:** это fixed denominator behavior policy; current update не должен незаметно менять его.
7. Нужно ли для любой RL среды обучать reward model? **Ответ:** нет, reward может быть executable; его соответствие задаче всё равно проверяется.
8. Что означает all-equal group rewards? **Ответ:** нет comparative centered reward signal в этой group, но не обязательно нет всех training gradients.
9. Неотрицателен ли каждый sampled log-ratio? **Ответ:** нет; full KL —expectation с определённым distribution.
10. Может ли session success усиливать лишний action? **Ответ:** да, грубый shared reward не доказывает causal полезность каждого token.

<!-- /unit:2497642cb4df6cac32 -->

<!-- unit:dc6c84b4706484a91a -->
<a id="u-dc6c84b4706484a91a"></a>

**Практика:** вычислите advantages для $(0,0,0,0)$, $(1,1,0,0)$ и $(1,1,1,1)$. Предложите две policy, одна из которых получает высокий плохо заданный reward без выполнения задачи. Исправьте oracle и составьте план сравнения, включающий затраты неудачных rollouts.

<!-- /unit:dc6c84b4706484a91a -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как отложенный reward меняет вероятность последовательности действий?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
