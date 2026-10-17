# Б. Формальная постановка agentic-задачи

[Оглавление](README.md). Пункт2 плана.

**Основа и дополнение:** S6:4–178 и1377–2197 дают tool loop и coordination. Формальная MDP/POMDP постановка, credit assignment и toy environment ниже — новое углубление по плану, а не восстановленное занятие о training.

## 1. Состояние мира и доступное наблюдение

Environment state $s_t$ описывает существенное состояние задачи: corpus/version, текущие результаты, доступные объекты, время и последствия прошлых действий. Agent получает observation $o_t$, которое может показывать только часть состояния. Action $a_t$ меняет среду или запрашивает новую информацию.

Для MDP задаются states, actions, transition kernel, reward, discount и начальное распределение. В partially observable setting policy обычно опирается на историю:

$$
h_t=(o_1,a_1,o_2,a_2,\ldots,o_t),\qquad
a_t\sim\pi_\theta(\cdot\mid h_t).
$$

Model context — ограниченное представление этой истории, не обязательно полный $h_t$. Truncation/summary могут потерять information. Важно различать hidden state среды и скрытые нейросетевые activations: это разные смыслы слова state.

Transition можно записать:

$$
s_{t+1}\sim P(\cdot\mid s_t,a_t),\quad
o_{t+1}\sim O(\cdot\mid s_{t+1},a_t),\quad
r_t=R(s_t,a_t,s_{t+1}).
$$

Read-only search тоже меняет epistemic/task state: агент получил новые evidence и потратил budget, хотя corpus не изменился. Mutable API меняет внешний мир дополнительно. Для теоретического начала достаточно read-only environment с проверяемым результатом.

## 2. Конкретная маленькая среда

Задача: найти действующий срок хранения отчёта в документах, назвать срок и source/version. Доступны три tools: search corpus, read document, compare numeric dates/версии по заданному правилу. Можно завершить answer или abstain. Corpus содержит устаревший v1 и current v2, один distractor и явные access scopes.

State включает remaining calls/time, returned snippets и выбранные IDs. Success oracle проверяет correct duration, current version, source support и отсутствие forbidden data. Horizon, например6 действий, задаёт maximum interactions, а не длину одного answer в tokens.

Ошибка, невидимая single-turn checker: final text правильного формата, но agent прочитал v1, потом выдумал v2; repeated search зациклился; правильный read observation потерян при state merge; action cap превышен. Эти failures требуют trajectory record.

Reset должен возвращать controlled initial state. Если corpus меняется между candidates, сравнение policy становится неоднозначным. Seeds фиксируют stochastic environment choices, но не заменяют version snapshots.

## 3. Return и delayed credit

Для конечной траектории horizon $H$ return от шага t:

$$
G_t=\sum_{k=t}^{H}\gamma^{k-t}r_k.
$$

Discount $\gamma\in[0,1]$ задаёт относительный вес будущего reward в выбранной постановке. Для finite tasks можно использовать $\gamma=1$ и явную step cost. Reward может приходить только в конце, когда oracle проверил result.

Пример: actions search, read, answer; cost каждого0.05, terminal success bonus1. При $\gamma=1$ total $1-3\cdot0.05=0.85$. Неверная короткая answer из одного шага имеет $-0.05$, если success bonus0. Но если scale penalties слишком велика, policy может предпочесть ранний abstention: reward design меняет incentives.

Credit assignment спрашивает, какие decisions обеспечили outcome. Одна terminal failure не говорит, был ли плохой search query, неверный document selection или answer interpretation. Сохранять intermediate observations и reason-independent action labels важно для диагностики и построения training data.

## 4. Termination и budgets

Environment может terminate при verified completion, permanent error или недопустимом действии; timeout/horizon могут truncать episode без успеха. Эти причины различаются: horizon exhaustion не следует записывать как normal success termination только из-за завершённого текста.

Budget включает tools, tokens, wall-clock и nested work. Policy получает доступное remaining budget или application принудительно ограничивает calls. Если budget скрыт, model может планировать длинный путь, который executor всегда обрезает; train/inference mismatch ухудшает quality.

Частично наблюдаемая environment может требовать уточнения пользователя. Правильный action тогда — запрос missing information, если это предусмотрено task. Нельзя считать failure любую pause только потому, что evaluator ожидает немедленную answer; oracle должен отражать истинную задачу.

## 5. Trajectory data и split

Запись содержит initial task/context, chronological actions, exact observations, versions, errors, rewards/termination и total cost. Tool result хранится отдельно от policy-generated tokens. Пользовательский final goal нельзя заменить более простой подзадачей после неуспеха.

Разделять train/test по задачам, source documents и templates, где нужно. Random split отдельных messages одной trajectory позволяет model видеть продолжение своей будущей test history. Если изучается recovery, holdout должен включать новые tasks/conditions с ошибками, а не те же logs с другой перестановкой строк.

1. Почему observation не равно state? **Ответ:** agent видит только доступную часть мира и может ошибаться о остальном.
2. Что значит reward delayed? **Ответ:** quality ранних actions определяется поздним outcome, а не немедленным label каждого шага.
3. Чем truncation отличается от success? **Ответ:** ограничение остановило episode, но задача могла остаться нерешённой.

**Практика:** записать три toy trajectories: success через current v2, failure через stale v1 и recovery после search timeout. Для каждой указать observations, действия, return, termination и какой feedback доступен для обучения. Начать с этих записей, не с большого RL запуска.
