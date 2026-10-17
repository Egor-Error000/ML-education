# В. SFT на действиях: как учить выбор tool и реакцию на observation

[Оглавление](README.md). Пункт3 плана.

**Связь с источниками:** S3:228–339 объясняет обычный SFT, S6:4–178 — tool execution. Их соединение в trajectory behavior cloning — дополнительный учебный материал. Источники не подтверждают ваш training run агента.

## 1. Что является supervised target

Behavior cloning приближает expert policy: history/observation → допустимое действие. Для LLM action сериализуется как tool name/arguments, answer или другой предусмотренный protocol. Target — не обязательно скрытое рассуждение эксперта; достаточно проверяемых decisions и результатов среды.

Если action разбивается на tokens $a_{t,1:m_t}$, negative log-likelihood:

$$
\mathcal{L}_{\mathrm{BC}}=
-\frac{1}{Z}\sum_{\tau\in\mathcal D}\sum_t\sum_{j=1}^{m_t}
 w_{t,j}\log\pi_\theta(a_{t,j}\mid h_t,a_{t,<j}).
$$

$w_{t,j}$ задаёт mask/weight, $Z$ — объявленный normalizer, например число active tokens. Усреднение по tokens или по trajectories меняет вес длинных sessions. Если длинные failures повторены многократно, они могут доминировать dataset без осознанной weighting policy.

Tool observation входит в context следующего decision, но обычно не является action, которое policy должна имитировать как собственный output. Если оставить loss на внешней lookup response, можно учить model выдумывать database contents вместо вызова tool. Objective должен соответствовать тому, что генерируется на inference.

## 2. Какие demonstrations нужны

Успешные happy paths учат стандартному решению. Но нужны также: отсутствие evidence, invalid argument correction, tool timeout/retry, ненужный tool, conflicting/stale documents, early stop при достаточном result. Без таких data policy может применять один шаблон во всех states.

Проверить truth observations, разрешённость actions и provenance. Teacher-generated trajectory с текстом «tool вернул30дней» не становится реальным recorded execution, если tool не вызывался. Для simulated environment result допустим, но simulator/version и ground truth должны быть объявлены.

Action equivalence важна. Две разные корректные search formulations не должны автоматически считаться неправильными из-за string mismatch. При этом ID/number/version constraints проверяются строго по смыслу. Dataset может содержать несколько допустимых actions при одном history; один hard target лишь задаёт один из путей.

## 3. Template, masks и dependency boundaries

Роли assistant/tool и call IDs сериализуются по contract модели/runtime. Совпадение JSON syntax не гарантирует правильный выбор tool. EOS/turn-end маркеры, loss mask и causal shift проверяются на распечатанном batch, как в обычном SFT.

Пример training sequence: user question → assistant search call → tool results → assistant read call → tool document → assistant final answer. Active loss positions должны отражать выбранные assistant outputs; tool contents сохраняются conditioning. Random opaque call IDs обычно генерируются по protocol, не должны превращаться в memorization target, если реальные IDs создаёт application.

Зависимые actions требуют observations. Если read ID должен прийти из search, demonstration не должна выдавать оба calls параллельно с подсмотренным ID. Независимые reads могут быть parallel, но dataset/runtime должны одинаково представлять их order и result mapping.

ReAct — interleaving decision/action/observation pattern, который можно использовать prompting или data format. Это не отдельная гарантия правильности и не обязательный запрос полного внутреннего chain-of-thought. [ReAct](https://arxiv.org/abs/2210.03629).

## 4. Первый training experiment

Сначала проверить overfit на нескольких correct trajectories: loss уменьшается, outputs соблюдают protocol, observations не генерируются как actions. Затем independent train/validation/test tasks, small learning rate/budget по выбранной model и baseline без tuning.

Измерять selection, schema validity, semantic arguments, call count и final success отдельно. Если validity выросла, а task success нет, возможно data научила format, но не decision sequence. Если training success высокий, test fails после первой ошибки, вероятен distribution shift — следующая глава.

Runtime validators/permissions остаются обязательными даже после высокого held-out score. Learning не заменяет hard constraints. Успешный fine-tuning на read-only toy tools не доказывает пригодность для arbitrary mutating production actions.

## 5. Вопросы и практика

1. Нужно ли обучать model воспроизводить tool result как свой output? **Ответ:** только если это явно соответствует другой задаче; в обычном action cloning observation — context, не policy action.
2. Почему string-exact search query accuracy недостаточна? **Ответ:** несколько queries могут быть semantically equivalent и вести к верному evidence.
3. Доказывает ли schema validity успешный BC? **Ответ:** нет, model может выбрать wrong tool или бессмысленный аргумент в правильном формате.

**Практика:** размечать assistant/tool roles и loss positions в трёх trajectories предыдущей главы. Добавить one recovery path и one no-tool-needed case. Сформулировать oracle для action semantics и final state; затем спроектировать small SFT comparison, не заявляя numerical results до запуска.
