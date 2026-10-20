# Behavior cloning и SFT на демонстрациях

**Уровень:** основа ветки.

**Предпосылки:** [Оценка траекторий и воспроизводимость](<agent-evaluation.md>), [SFT: данные, шаблоны, masks и objective](<../03-adaptation/adapt-sft.md>)

**Используют:** [Distribution shift, recovery и DAgger](<agent-recovery.md>)

**Учебный вопрос:** Как из траектории получить targets для обучения политики?





## Карта вычисления и границы этапов

![Behavior cloning и SFT на демонстрациях: входы, преобразования и выходы](<../../оформление/course/agent-demonstrations.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Reward и targets используются в объявленном режиме обучения. При оценке политика фиксирована; success измеряется по всей попытке, включая повторы и бюджет.

<!-- unit:a9286741d40054f155 -->
<a id="u-a9286741d40054f155"></a>

## Что является supervised target

<!-- /unit:a9286741d40054f155 -->

<!-- unit:816e19ce665d9a5342 -->
<a id="u-816e19ce665d9a5342"></a>

Behavior cloning приближает expert policy: history/observation → допустимое действие. Для LLM action сериализуется как tool name/arguments, answer или другой предусмотренный protocol. Target — не обязательно скрытое рассуждение эксперта; достаточно проверяемых decisions и результатов среды.

<!-- /unit:816e19ce665d9a5342 -->

<!-- unit:8364fcf409f5114d75 -->
<a id="u-8364fcf409f5114d75"></a>

Если action разбивается на tokens $a_{t,1:m_t}$, negative log-likelihood:

<!-- /unit:8364fcf409f5114d75 -->

<!-- unit:517ca943ecbf24f9fa -->
<a id="u-517ca943ecbf24f9fa"></a>

$$
\mathcal{L}_{\mathrm{BC}}=
-\frac{1}{Z}\sum_{\tau\in\mathcal D}\sum_t\sum_{j=1}^{m_t}
 w_{t,j}\log\pi_\theta(a_{t,j}\mid h_t,a_{t,<j}).
$$

<!-- /unit:517ca943ecbf24f9fa -->

<!-- unit:d7696baf4ad169dc68 -->
<a id="u-d7696baf4ad169dc68"></a>

$w_{t,j}$ задаёт mask/weight, $Z$ — объявленный normalizer, например число active tokens. Усреднение по tokens или по trajectories меняет вес длинных sessions. Если длинные failures повторены многократно, они могут доминировать dataset без осознанной weighting policy.

<!-- /unit:d7696baf4ad169dc68 -->

<!-- unit:56b2294eaed293c55a -->
<a id="u-56b2294eaed293c55a"></a>

Tool observation входит в context следующего decision, но обычно не является action, которое policy должна имитировать как собственный output. Если оставить loss на внешней lookup response, можно учить model выдумывать database contents вместо вызова tool. Objective должен соответствовать тому, что генерируется на inference.

<!-- /unit:56b2294eaed293c55a -->

<!-- unit:aa48f9545a4b3652d4 -->
<a id="u-aa48f9545a4b3652d4"></a>

**Схема курса:** expert trajectory: observation/action/result → serialized tokens → role/target masks → token loss on policy action → cloned policy → rollout audit

<!-- /unit:aa48f9545a4b3652d4 -->

<!-- unit:506e63401cd1304b4b -->
<a id="u-506e63401cd1304b4b"></a>

<!-- source-depth-lifecycle-В -->

<!-- /unit:506e63401cd1304b4b -->

<!-- unit:529d036b4cbd4f6585 -->
<a id="u-529d036b4cbd4f6585"></a>

### Из instruction-response в history-action

<!-- /unit:529d036b4cbd4f6585 -->

<!-- unit:b16ad0294142935ca8 -->
<a id="u-b16ad0294142935ca8"></a>

S3 объяснял SFT как demonstrations полезных ответов, а S6 —tool loop с реальными observations. Их соединение даёт training unit: доступная history → очередной допустимый action. Это новое agentic углубление по плану. Нельзя из него заключить, что исходные чаты уже содержали выполненное обучение агента.

<!-- /unit:b16ad0294142935ca8 -->

<!-- unit:077a7ac465f5f9ce09 -->
<a id="u-077a7ac465f5f9ce09"></a>

Для задачи предыдущей главы первый target —search call, следующий после search observation —read(R2), затем после текста R2 —answer90days/R2/v2. Один final answer target не показывает, как достать evidence. Один search-call target не показывает, когда остановиться. Если inference требует именно последовательного поведения, данные должны представлять существенные decision points, включая stop/abstain.

<!-- /unit:077a7ac465f5f9ce09 -->

<!-- unit:b95c08aead60863468 -->
<a id="u-b95c08aead60863468"></a>

В behavior cloning expert демонстрирует допустимое действие на данном history. Обучение максимизирует likelihood таких action tokens. Это не автоматическая оценка environment return: длинный успешный путь и короткий успешный путь могут оба присутствовать, и их относительный вес задают data/normalization. Runtime всё ещё исполняет actions, а θ учится их предлагать.

<!-- /unit:b95c08aead60863468 -->

<!-- unit:990cf620d0601601e6 -->
<a id="u-990cf620d0601601e6"></a>

### Teacher forcing по траектории

<!-- /unit:990cf620d0601601e6 -->

<!-- unit:5a76d1572e53d0c224 -->
<a id="u-5a76d1572e53d0c224"></a>

При training очередной target token предсказывается при известных предыдущих tokens правильной demonstration. Для следующего action в prefix уже включено **настоящее** tool observation этой trajectory. Во время rollout предыдущие actions генерирует сама policy, поэтому history может отличаться. Это источник train/inference mismatch, а не ошибка teacher forcing как арифметической операции.

<!-- /unit:5a76d1572e53d0c224 -->

<!-- unit:6b246a170c59f5518f -->
<a id="u-6b246a170c59f5518f"></a>

Полный transcript можно хранить одной causal sequence: user → assistant call → tool result → assistant call → tool result → assistant answer. Causal mask не даёт раннему search call подсмотреть будущий tool result справа. Если же data preprocessing перенёс gold result в начальный system/user context, causal mask уже не спасёт от leakage: подсказка стала прошлым input.

<!-- /unit:6b246a170c59f5518f -->

<!-- unit:dc33bb20c60a4c55d8 -->
<a id="u-dc33bb20c60a4c55d8"></a>

Отдельные history-action examples тоже допустимы, если history соответствует ровно этому decision point. Нельзя вырезать read target из полной trajectory и оставить у него только исходный вопрос, потеряв search result с ID. И нельзя дать action read(R2) conditioning на observation от другого episode, где R2 обозначает совсем иной document.

<!-- /unit:dc33bb20c60a4c55d8 -->

<!-- unit:0f3168960c7c432b94 -->
<a id="u-0f3168960c7c432b94"></a>

### Три decisions и простая likelihood арифметика

<!-- /unit:0f3168960c7c432b94 -->

<!-- unit:48135e337622d19aef -->
<a id="u-48135e337622d19aef"></a>

Для иллюстрации условимся, что каждый из трёх action targets представлен одним учебным символом, а их условные вероятности 0.5,0.25,0.8. Likelihood всей target sequence0.1. Сумма NLL $-\ln0.5-\ln0.25-\ln0.8=\ln10\approx2.302585$; средний NLL на эти три symbols≈0.767528. Настоящие calls состоят из многих tokens, но принцип произведения условных вероятностей тот же.

<!-- /unit:48135e337622d19aef -->

<!-- unit:51279c5fe444956cf0 -->
<a id="u-51279c5fe444956cf0"></a>

Уменьшить этот loss можно, сделав demonstrated actions вероятнее при demonstrated histories. Из одной величины нельзя вывести успех на ошибочной history, которой в data не было, или корректность runtime idempotency. Low training loss также не означает, что все valid alternative searches должны иметь нулевую вероятность.

<!-- /unit:51279c5fe444956cf0 -->

<!-- unit:47ac35ab506c739379 -->
<a id="u-47ac35ab506c739379"></a>

### Нормировка задаёт относительное влияние sessions

<!-- /unit:47ac35ab506c739379 -->

<!-- unit:208d435e5b90355519 -->
<a id="u-208d435e5b90355519"></a>

Пусть trajectory A имеет 2 active tokens со средним loss0.2, B —8 active tokens со средним loss1.2. Token-average loss=(2·0.2+8·1.2)/10=1.0. Если усреднить сначала внутри каждой trajectory, затем по двум trajectories, получится 0.7. В первом objective B получает 80% token weight, во втором обе sessions равный вес.

<!-- /unit:208d435e5b90355519 -->

<!-- unit:4818f04ad1121fd87e -->
<a id="u-4818f04ad1121fd87e"></a>

Ни одна нормировка не универсально правильна. Long sessions могут содержать больше полезных decisions, а могут быть длинными повторениями final prose. Если финальные ответы на 100tokens доминируют над короткими call decisions, decrease average loss может мало менять sequence behavior. Для data mixture определяют sampling/weights по задачам, calls и outcomes, а не только по количеству строк JSONL.

<!-- /unit:4818f04ad1121fd87e -->

<!-- unit:4c671026a241de5fc4 -->
<a id="u-4c671026a241de5fc4"></a>

## Какие demonstrations нужны

<!-- /unit:4c671026a241de5fc4 -->

<!-- unit:ed6373cb40f1b097a6 -->
<a id="u-ed6373cb40f1b097a6"></a>

Успешные happy paths учат стандартному решению. Но нужны также: отсутствие evidence, invalid argument correction, tool timeout/retry, ненужный tool, conflicting/stale documents, early stop при достаточном result. Без таких data policy может применять один шаблон во всех states.

<!-- /unit:ed6373cb40f1b097a6 -->

<!-- unit:8858150b0f005c60cb -->
<a id="u-8858150b0f005c60cb"></a>

Проверить truth observations, разрешённость actions и provenance. Teacher-generated trajectory с текстом «tool вернул 30 дней» не становится реальным recorded execution, если tool не вызывался. Для simulated environment result допустим, но simulator/version и ground truth должны быть объявлены.

<!-- /unit:8858150b0f005c60cb -->

<!-- unit:55f4268a186f46d7f5 -->
<a id="u-55f4268a186f46d7f5"></a>

Action equivalence важна. Две разные корректные search formulations не должны автоматически считаться неправильными из-за string mismatch. При этом ID/number/version constraints проверяются строго по смыслу. Dataset может содержать несколько допустимых actions при одном history; один hard target лишь задаёт один из путей.

<!-- /unit:55f4268a186f46d7f5 -->

<!-- unit:0ac8abd7bbf5959a70 -->
<a id="u-0ac8abd7bbf5959a70"></a>

### Полная demonstration с наблюдаемыми границами

<!-- /unit:0ac8abd7bbf5959a70 -->

<!-- unit:33e768e8a3d34b1202 -->
<a id="u-33e768e8a3d34b1202"></a>

Учебный transcript использует IDs как placeholders, не как universal special tokens:

<!-- /unit:33e768e8a3d34b1202 -->

<!-- unit:8e69b88b9832b3ef1b -->
<a id="u-8e69b88b9832b3ef1b"></a>

```text
USER: найди действующий срок хранения отчёта проекта P, укажи source/version
ASSISTANT CALL c1: search("report retention P", limit=3)
TOOL c1: R1/v1 obsolete; R2/v2 active; D/v1 unrelated title
ASSISTANT CALL c2: read(document_id="R2")
TOOL c2: R2/v2 active; "Срок хранения отчёта — 90 дней."
ASSISTANT FINAL: 90 дней, R2/v2, с применимой ссылкой на passage
```

<!-- /unit:8e69b88b9832b3ef1b -->

<!-- unit:1335ba38662eb4a711 -->
<a id="u-1335ba38662eb4a711"></a>

Gold data включает реальные либо явно simulated tool results. Teacher text «я нашёл R2» не заменяет execution record. Если simulator задаёт результат, сохраняют simulator version и task state; это честный учебный источник, но не external-world measurement. При synthetic teacher generation проверяют соответствие result действию и исходному corpus.

<!-- /unit:1335ba38662eb4a711 -->

<!-- unit:f20a3df40ea44dc1f5 -->
<a id="u-f20a3df40ea44dc1f5"></a>

Tools descriptions тоже часть conditioning. Если demonstration использует search с одним schema, а inference регистрирует другой limit/type/название, model обучена на несовместимый protocol. Registry revision и template version связываются с dataset. Дубли c1/c2 в разных episodes допустимы только если IDs локальны по protocol и правильно scoped.

<!-- /unit:f20a3df40ea44dc1f5 -->

<!-- unit:c9d9f6fb3feb2fb227 -->
<a id="u-c9d9f6fb3feb2fb227"></a>

### Recovery data должна показывать причину выбора

<!-- /unit:c9d9f6fb3feb2fb227 -->

<!-- unit:3a207fa6a7a79c4303 -->
<a id="u-3a207fa6a7a79c4303"></a>

Для timeout trajectory после первой search call наблюдается transient error, затем повтор при доступном budget. Для no-results наблюдается успешный пустой result; действие может быть rewrite или abstain по задаче. Для denied —доступ не расширяется самовольно. Эти contexts различаются; один шаблон «после любого ERROR повторить то же» учит неправильному поведению.

<!-- /unit:3a207fa6a7a79c4303 -->

<!-- unit:93f743c53aa7128f48 -->
<a id="u-93f743c53aa7128f48"></a>

После stale read(R1) эксперт может перейти к R2 и исправить ответ, если budget позволяет. Он не должен молча переписать старый observation30days в 90: исправляется следующий action/вывод, а фактическая history сохраняется. Успешная recovery trajectory может содержать ошибочный ранний action; чтобы не имитировать его как идеал, training mask/selection labels выбирают по цели.

<!-- /unit:93f743c53aa7128f48 -->

<!-- unit:92e1d6237ee7487135 -->
<a id="u-92e1d6237ee7487135"></a>

Можно сохранить valid prefix, добавить expert correction на реально посещённой history, использовать preference pair или другой подходящий objective. Но простое положительное SFT на всех tokens проваленной session будет учить её действиям, включая bad choice. Negative outcomes сами по себе не делают обычный cross-entropy loss «антиимитацией».

<!-- /unit:92e1d6237ee7487135 -->

<!-- unit:280eaae6d8b4dfd1f4 -->
<a id="u-280eaae6d8b4dfd1f4"></a>

### Нет единственной правильной search строки

<!-- /unit:280eaae6d8b4dfd1f4 -->

<!-- unit:80094ed2a100ad023d -->
<a id="u-80094ed2a100ad023d"></a>

«report retention P» и «срок хранения отчёта проект P» могут вести к одному evidence в данной среде. Exact string-match label отвергнет один valid путь. Для semantic action correctness проверяют запрос/constraints и actual retrieval outcome по protocol; для machine identifiers, numeric units и document revision требования могут быть строгими.

<!-- /unit:80094ed2a100ad023d -->

<!-- unit:028b778c98728f21e9 -->
<a id="u-028b778c98728f21e9"></a>

Если при одном history допустимы несколько actions, dataset может содержать несколько verified demonstrations. Teacher выбор одного пути не доказывает, что остальные неверны. Но indiscriminate augmentation тоже опасна: paraphrase может убрать negation/version, а синтетический synonym —изменить intent. Diversity должна сохранять смысл и наблюдаемую выполнимость.

<!-- /unit:028b778c98728f21e9 -->

<!-- unit:dfb7cf7497da5adcf8 -->
<a id="u-dfb7cf7497da5adcf8"></a>

### Книги и диалоги как обучающие данные

<!-- /unit:dfb7cf7497da5adcf8 -->

<!-- unit:206cb90711ea8da92b -->
<a id="u-206cb90711ea8da92b"></a>

В S3 пользователь спросил, подходят ли книги, если в них встречаются диалоги. Ответ «книги годятся только для roleplay» слишком узок. Важны задача и objective. Сырой literary dialogue не является автоматически instruction→correct action demonstration. Но техническое описание процесса можно преобразовать в проверенные question/evidence/action examples, если действительно восстановим правильный interface и outcome.

<!-- /unit:206cb90711ea8da92b -->

<!-- unit:066d0757e5a2eacb8a -->
<a id="u-066d0757e5a2eacb8a"></a>

Для agentic SFT запись «герой попросил найти документ, другой ответил, что нашёл» не задаёт фактический search call, tool observation и permission context. Одни role tags не превращают это в rollout. Нужны callable tools, известная environment, корректные decisions и проверяемый финальный результат. Иначе модель усвоит имитацию рассказа о действии.

<!-- /unit:066d0757e5a2eacb8a -->

<!-- unit:b99a3711b42d643729 -->
<a id="u-b99a3711b42d643729"></a>

Продолжение language-model training на raw text и assistant-action cloning —разные постановки, даже если обе используют token CE. Тип данных, mask и protocol определяют, что модель учится предсказывать. Подробная связь books/SFT/continued pretraining раскрыта в [главе адаптации](<../03-adaptation/adapt-sft.md>).

<!-- /unit:b99a3711b42d643729 -->

<!-- unit:d692b5cab055562abe -->
<a id="u-d692b5cab055562abe"></a>

## Template, masks и dependency boundaries

<!-- /unit:d692b5cab055562abe -->

<!-- unit:662af49a4faaaa033e -->
<a id="u-662af49a4faaaa033e"></a>

Роли assistant/tool и call IDs сериализуются по contract модели/runtime. Совпадение JSON syntax не гарантирует правильный выбор tool. EOS/turn-end маркеры, loss mask и causal shift проверяются на распечатанном batch, как в обычном SFT.

<!-- /unit:662af49a4faaaa033e -->

<!-- unit:5bd9f255fc16e01b9e -->
<a id="u-5bd9f255fc16e01b9e"></a>

Пример training sequence: user question → assistant search call → tool results → assistant read call → tool document → assistant final answer. Active loss positions должны отражать выбранные assistant outputs; tool contents сохраняются conditioning. Random opaque call IDs обычно генерируются по protocol, не должны превращаться в memorization target, если реальные IDs создаёт application.

<!-- /unit:5bd9f255fc16e01b9e -->

<!-- unit:52212754afa068ea7f -->
<a id="u-52212754afa068ea7f"></a>

Зависимые actions требуют observations. Если read ID должен прийти из search, demonstration не должна выдавать оба calls параллельно с подсмотренным ID. Независимые reads могут быть parallel, но dataset/runtime должны одинаково представлять их order и result mapping.

<!-- /unit:52212754afa068ea7f -->

<!-- unit:ca5a5e6680525ee13e -->
<a id="u-ca5a5e6680525ee13e"></a>

ReAct — interleaving decision/action/observation pattern, который можно использовать prompting или data format. Это не отдельная гарантия правильности и не обязательный запрос полного внутреннего chain-of-thought. [ReAct](https://arxiv.org/abs/2210.03629).

<!-- /unit:ca5a5e6680525ee13e -->

<!-- unit:fbda953d214eaf0d99 -->
<a id="u-fbda953d214eaf0d99"></a>

### Assistant-only loss не означает удаление tool result

<!-- /unit:fbda953d214eaf0d99 -->

<!-- unit:4f3eeea0715e9e12f1 -->
<a id="u-4f3eeea0715e9e12f1"></a>

Возьмём условные block lengths, уже включающие выбранные control tokens: user10,assistant search4,tool20,assistant read4,tool30,assistant final6. Всего 74tokens, active assistant targets14,tool context50,user context10. Удалить 50 tool tokens из input ради «мы не учимся на них» —значит лишить следующие decisions нужных observations. Нужно маскировать target loss, сохраняя conditioning.

<!-- /unit:4f3eeea0715e9e12f1 -->

<!-- unit:2a131bc59c8abda0a0 -->
<a id="u-2a131bc59c8abda0a0"></a>

При causal forward gradient от assistant loss может проходить через representations предыдущих user/tool tokens и attention weights, связывающих их с ответом. Поэтому source фраза «градиенты только на ответе» не означает, что prompt states обязательно detached. Убирается direct token-prediction loss на masked targets, а не вся роль этих inputs в вычислительном графе.

<!-- /unit:2a131bc59c8abda0a0 -->

<!-- unit:9b29d9dabaf2a36a40 -->
<a id="u-9b29d9dabaf2a36a40"></a>

Для конкретного training setup возможны и другие masking objectives. Важно явно выбрать их и проверить соответствие inference policy: модель сама выдаёт actions, а не играет за внешнюю базу данных. Название SFTTrainer или role `tool` не гарантирует корректную маску автоматически; в [TRL SFT documentation](https://huggingface.co/docs/trl/sft_trainer) assistant-only режим выделен как отдельная configuration, зависящая от формата данных/template.

<!-- /unit:9b29d9dabaf2a36a40 -->

<!-- unit:fa8e1ac7241e663e78 -->
<a id="u-fa8e1ac7241e663e78"></a>

### Causal shift на десяти условных tokens

<!-- /unit:fa8e1ac7241e663e78 -->

<!-- unit:0ce1da074dfbdc444a -->
<a id="u-0ce1da074dfbdc444a"></a>

Пусть input sequence имеет positions0..9:

<!-- /unit:0ce1da074dfbdc444a -->

<!-- unit:95b8fa05224c5b55d3 -->
<a id="u-95b8fa05224c5b55d3"></a>

```text
0 BOS | 1 USER | 2 question | 3 ASSIST | 4 CALL |
5 TOOL | 6 observation | 7 ASSIST | 8 ANSWER | 9 EOS

labels = [-100,-100,-100,-100,CALL,-100,-100,-100,ANSWER,EOS]
```

<!-- /unit:95b8fa05224c5b55d3 -->

<!-- unit:238aa561f303fb4c78 -->
<a id="u-238aa561f303fb4c78"></a>

Здесь CALL и ANSWER —учебные symbols вместо многотокенных strings. При обычном causal internal shift logit3 предсказывает label4 CALL;logit7 —label8 ANSWER;logit8 —label9 EOS. На logit3 модель ещё не может видеть observation6. На logit7 observation6 уже в прошлом и влияет на ответ, хотя его собственный label masked.

<!-- /unit:238aa561f303fb4c78 -->

<!-- unit:96e1efbacbac411f7b -->
<a id="u-96e1efbacbac411f7b"></a>

Если вручную сдвинуть labels, а trainer сдвинет их повторно, model будет сопоставлена с неправильными targets. Проверяют не только shape tensors, но и пары decoded prefix→target на active positions. Значение−100 —распространённый ignore_index contract конкретного loss implementation, а не «специальный token словаря»; выбранный trainer должен его действительно интерпретировать как mask.

<!-- /unit:96e1efbacbac411f7b -->

<!-- unit:8066d618f75a8ae4df -->
<a id="u-8066d618f75a8ae4df"></a>

EOS/end-of-turn masks определяются protocol. Если pad token ID совпадает с EOS, безусловно заменить **все** EOS labels наignore по сравнению IDs может удалить обучение остановке настоящего ответа. Padding positions определяются по реальной batch structure/attention mask; stop markers не исчезают только потому, что тот же ID используется для padding.

<!-- /unit:8066d618f75a8ae4df -->

<!-- unit:59b8151284154a5507 -->
<a id="u-59b8151284154a5507"></a>

### Templates, markers и непрозрачные IDs

<!-- /unit:59b8151284154a5507 -->

<!-- unit:e76d8f9946ff6727bd -->
<a id="u-e76d8f9946ff6727bd"></a>

Сначала строят фактическую model-native serialization, затем tokenization/masks. ChatML из S3 —один format family, а не обязательный стандарт всех Transformer models. Role names могут быть обычным текстом внутри control template, tool calls —специально сериализованным span. Если format tools не совпал с training template checkpoint, «валидный JSON» ещё не делает input понятным для tool policy.

<!-- /unit:e76d8f9946ff6727bd -->

<!-- unit:60dd7d01c848dfbfee -->
<a id="u-60dd7d01c848dfbfee"></a>

[Hugging Face chat-template guide](https://huggingface.co/docs/transformers/chat_templating) показывает model-specific control formats. При отдельных шагах formatting/tokenization нужно избегать дублирования уже добавленных special tokens; параметры задаются по фактическому пути preprocessing. Для finished training dialogue не следует бездумно добавлять лишний пустой assistant-generation prefix, предназначенный для нового inference response.

<!-- /unit:60dd7d01c848dfbfee -->

<!-- unit:973a978030836a5cda -->
<a id="u-973a978030836a5cda"></a>

Opaque runtime call IDs не должны становиться задачей memorization случайных значений, если их создаёт application. Если protocol требует model-generated link/reference, обучают именно этот contract и соответствие result mapping. Содержательные document IDs не являются просто шумом: read target должен ссылаться на реально найденный разрешённый документ.

<!-- /unit:973a978030836a5cda -->

<!-- unit:fec2cc34bfb688115e -->
<a id="u-fec2cc34bfb688115e"></a>

### Packing, truncation и границы episodes

<!-- /unit:fec2cc34bfb688115e -->

<!-- unit:bbc0a76119ed70d5e5 -->
<a id="u-bbc0a76119ed70d5e5"></a>

Укороченная последовательность может потерять observation, но оставить answer target. Тогда пример учит утверждать 90/R2 без достаточного input. Или может потеряться final answer, оставив только calls: меняется относительный training weight остановки. Length policy проверяют по semantic boundaries, а не только слева/справа от максимального token count.

<!-- /unit:bbc0a76119ed70d5e5 -->

<!-- unit:43c9fe310f575a8505 -->
<a id="u-43c9fe310f575a8505"></a>

Packing нескольких episodes в одну tensor row экономит padding, но требует понимания attention/reset semantics выбранного backend. Один EOS не является автоматической block-diagonal attention wall во всех implementations. Если task B видит private answer task A, «раздельные JSONL rows» уже не гарантируют изоляцию после packing. Для независимого comparison нужно зафиксировать реальную mask/position policy.

<!-- /unit:43c9fe310f575a8505 -->

<!-- unit:50472820fc2cef302d -->
<a id="u-50472820fc2cef302d"></a>

Parallel calls в demonstration могут следовать одному assistant message, затем multiple results с IDs. Если второй read зависит от search result, его нельзя переносить в тот же initial action с подсмотренным ID. Иначе training history содержит причинно невозможный путь. Независимые reads допустимы, если inference executor умеет такой protocol и budget accounting.

<!-- /unit:50472820fc2cef302d -->

<!-- unit:1623942c942c6c971a -->
<a id="u-1623942c942c6c971a"></a>

## Первый training experiment

<!-- /unit:1623942c942c6c971a -->

<!-- unit:586acedbc36499fb44 -->
<a id="u-586acedbc36499fb44"></a>

Сначала проверить overfit на нескольких correct trajectories: loss уменьшается, outputs соблюдают protocol, observations не генерируются как actions. Затем independent train/validation/test tasks, small learning rate/budget по выбранной model и baseline без tuning.

<!-- /unit:586acedbc36499fb44 -->

<!-- unit:42390fdfba0afe63c3 -->
<a id="u-42390fdfba0afe63c3"></a>

Измерять selection, schema validity, semantic arguments, call count и final success отдельно. Если validity выросла, а task success нет, возможно data научила format, но не decision sequence. Если training success высокий, test fails после первой ошибки, вероятен distribution shift — следующая глава.

<!-- /unit:42390fdfba0afe63c3 -->

<!-- unit:526245babcd2c2b286 -->
<a id="u-526245babcd2c2b286"></a>

Runtime validators/permissions остаются обязательными даже после высокого held-out score. Learning не заменяет hard constraints. Успешный fine-tuning на read-only toy tools не доказывает пригодность для arbitrary mutating production actions.

<!-- /unit:526245babcd2c2b286 -->

<!-- unit:2cc8ea429eadc1dda5 -->
<a id="u-2cc8ea429eadc1dda5"></a>

### До optimizer: inspect, replay и крошечный overfit check

<!-- /unit:2cc8ea429eadc1dda5 -->

<!-- unit:f86b36b394ea0db8f3 -->
<a id="u-f86b36b394ea0db8f3"></a>

Для теоретической проверки сначала распечатывают один serialized example: roles/control tokens, input IDs, labels, active count, causal prediction pairs и tool boundaries. Затем replay actions в declared environment/stub проверяет, что recorded observations им соответствуют. Это разные проверки: правильные tensor masks не доказывают правильные database facts.

<!-- /unit:f86b36b394ea0db8f3 -->

<!-- unit:cb3b07ff0148914a8d -->
<a id="u-cb3b07ff0148914a8d"></a>

Если training эксперимент действительно проводится, маленький overfit sanity check на нескольких verified traces помогает найти ошибку objective/template/labels. Он не является held-out score и не доказывает generalization. В текущем конспекте описывается процедура; полноразмерный training run или измеренный результат пользователю не приписывается.

<!-- /unit:cb3b07ff0148914a8d -->

<!-- unit:2ba6eadd0c421476da -->
<a id="u-2ba6eadd0c421476da"></a>

Перед сравнением фиксируют base/model/adapter/template, dataset revisions, task split, sampling/masks/normalizer, optimizer budget и evaluation protocol. Новая ошибка после inference может быть несовпадением template или среды, а не автоматически «модель не обучилась». Compression/merge/export оцениваются после того, как они действительно изменили tested computation.

<!-- /unit:2ba6eadd0c421476da -->

<!-- unit:99e98a6d9550c2054c -->
<a id="u-99e98a6d9550c2054c"></a>

### Оценка policy отдельно от обучения на красивых messages

<!-- /unit:99e98a6d9550c2054c -->

<!-- unit:6ec3aa3168f86a91dd -->
<a id="u-6ec3aa3168f86a91dd"></a>

На held-out tasks смотрят selection, args, observation use, final success, invalid calls, recovery и costs. При одинаковом task budget baseline и tuned policy получают одинаковые доступные tools/data. Увеличить retries у новой модели и объявить весь success gain следствием SFT нельзя без отдельной ablation.

<!-- /unit:6ec3aa3168f86a91dd -->

<!-- unit:e4f1f5bf1ff0e72e39 -->
<a id="u-e4f1f5bf1ff0e72e39"></a>

Happy-path evaluation не тестирует reaction на ошибки. При injected timeout полезно проверить, корректно ли policy отличает его от empty result, не расширяет ли scope после denied, не подставляет ли выдуманный ID после malformed search output. Новые fault conditions в test должны иметь declared semantics, а не случайно отличаться от описания tool в prompt.

<!-- /unit:e4f1f5bf1ff0e72e39 -->

<!-- unit:c80457ffdda6546e31 -->
<a id="u-c80457ffdda6546e31"></a>

Low supervised loss на expert history и poor rollout success могут сосуществовать: собственная первая ошибка переводит policy на unseen history. Это ведёт к distribution shift/DAgger следующей главы. Но сначала исключают сломанные input masks, data leakage и inconsistent tool contract, потому что dataset aggregation не исправит принципиально неверную serialization сам по себе.

<!-- /unit:c80457ffdda6546e31 -->

<!-- unit:b7bbc90b9d175c63ba -->
<a id="u-b7bbc90b9d175c63ba"></a>

### Почему выученное поведение не заменяет runtime invariant

<!-- /unit:b7bbc90b9d175c63ba -->

<!-- unit:8e26e9b4ada56ae18c -->
<a id="u-8e26e9b4ada56ae18c"></a>

Даже model с высоким held-out score остаётся источником proposals. Scope, registry, argument constraints, idempotency и budget enforcement реализуют по реальному contract приложения. Training data может показывать правильные ограничения и улучшать compliance, но среднее качество не заменяет проверку каждого конкретного call.

<!-- /unit:8e26e9b4ada56ae18c -->

<!-- unit:9b6559a2b2d95d6599 -->
<a id="u-9b6559a2b2d95d6599"></a>

В read-only toy environment некоторые ошибки просто дают failure reward. В mutating environment та же ошибка может иметь реальный effect до следующего token. Перенос требует нового environment/interface/eval анализа, а не только более длинного role prompt. Поэтому learning и execution policy связаны, но их guarantees различаются.

<!-- /unit:9b6559a2b2d95d6599 -->

<!-- unit:d7ad5a0deae4b12b5b -->
<a id="u-d7ad5a0deae4b12b5b"></a>

## Вопросы и практика

<!-- /unit:d7ad5a0deae4b12b5b -->

<!-- unit:944365e22724f9201a -->
<a id="u-944365e22724f9201a"></a>

1. Нужно ли обучать model воспроизводить tool result как свой output? **Ответ:** только если это явно соответствует другой задаче; в обычном action cloning observation — context, не policy action.
2. Почему string-exact search query accuracy недостаточна? **Ответ:** несколько queries могут быть semantically equivalent и вести к верному evidence.
3. Доказывает ли schema validity успешный BC? **Ответ:** нет, model может выбрать wrong tool или бессмысленный аргумент в правильном формате.

<!-- /unit:944365e22724f9201a -->

<!-- unit:9a0be994b9fdc896b0 -->
<a id="u-9a0be994b9fdc896b0"></a>

4. Видит ли ранний call будущий tool result в causal full transcript? **Ответ:** нет, если корректна causal mask и result не перенесён в его prefix preprocessing.
5. Что изменится при token-average вместо trajectory-average? **Ответ:** длинные active sequences получают больший относительный вес.
6. Masked tool target означает detached context? **Ответ:** нет, observation representations могут участвовать в gradient path от следующего assistant loss.
7. Почему нельзя маскировать все EOS по одному ID при pad=EOS? **Ответ:** можно случайно убрать реальные stop targets; padding определяют по batch structure.
8. Что учит обычный positive SFT на failed actions? **Ответ:** имитировать эти targets; negative outcome не превращает CE автоматически в anti-imitation.
9. Требует ли ReAct prompting обязательно fine-tuning? **Ответ:** нет, pattern можно использовать на inference; обучение —отдельное изменение weights.
10. Доказывает ли overfit sanity check успех на новых задачах? **Ответ:** нет, он проверяет возможность обучения/интеграцию на маленьком training subset.

<!-- /unit:9a0be994b9fdc896b0 -->

<!-- unit:9c764a824150fd6d7f -->
<a id="u-9c764a824150fd6d7f"></a>

**Практика:** размечать assistant/tool roles и loss positions в трёх trajectories предыдущей главы. Добавить one recovery path и one no-tool-needed case. Сформулировать oracle для action semantics и final state; затем спроектировать small SFT comparison, не заявляя numerical results до запуска.

<!-- /unit:9c764a824150fd6d7f -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как из траектории получить targets для обучения политики?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Distribution shift, recovery и DAgger](<agent-recovery.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
