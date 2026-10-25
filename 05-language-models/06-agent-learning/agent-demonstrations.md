# Behavior cloning и SFT на демонстрациях

**Уровень:** основа ветки.

**Предпосылки:** [Оценка траекторий и воспроизводимость](<agent-evaluation.md>), [SFT: данные, шаблоны, masks и objective](<../03-adaptation/adapt-sft.md>)

**Используют:** [Distribution shift, recovery и DAgger](<agent-recovery.md>)

**Учебный вопрос:** Как из траектории получить targets для обучения политики?

## Карта вычисления и границы этапов

![Behavior cloning и SFT на демонстрациях: входы, преобразования и выходы](<../../оформление/course/agent-demonstrations.svg>)

Демонстрация связывает действие с доступной в этот момент информацией. Утечка будущего наблюдения в контекст создаёт поведение, которое нельзя воспроизвести при реальном выполнении.

## Что является supervised target

Behavior cloning приближает expert policy: history/observation (доступное наблюдение среды) → допустимое действие. Для LLM action сериализуется как tool (инструмент приложения) name/arguments, answer или другой предусмотренный protocol. Target — не обязательно скрытое рассуждение эксперта; достаточно проверяемых decisions и результатов среды.

Если action разбивается на tokens (токены) $a_{t,1:m_t}$, negative log-likelihood:

$$
\mathcal{L}_{\mathrm{BC}}=
-\frac{1}{Z}\sum_{\tau\in\mathcal D}\sum_t\sum_{j=1}^{m_t}
 w_{t,j}\log\pi_\theta(a_{t,j}\mid h_t,a_{t,<j}).
$$

$w_{t,j}$ задаёт mask/weight, $Z$ — объявленный normalizer, например число active tokens. Усреднение по tokens или по trajectories меняет вес длинных sessions. Если длинные failures повторены многократно, они могут доминировать dataset без осознанной weighting policy.

Tool observation входит в context (доступный модели контекст) следующего decision, но обычно не является action, которое policy должна имитировать как собственный output. Если оставить loss (функция потерь) на внешней lookup response, можно учить model выдумывать database contents вместо вызова tool. Objective (оптимизируемая целевая функция) должен соответствовать тому, что генерируется на inference (применение обученной модели).

**Схема курса:** expert trajectory (траектория действий и наблюдений): observation/action/result → serialized tokens → role/target masks → token (токен) loss on policy action → cloned policy → rollout audit



### Из instruction-response в history-action

Для обучения агента демонстрация полезного действия должна включать реальные доступные наблюдения. Их соединение даёт training unit: доступная history → очередной допустимый action. Это новое agentic углубление по плану. Нельзя из него заключить, что исходные чаты уже содержали выполненное обучение агента.

Для задачи предыдущей главы первый target —search call, следующий после search observation —read(R2), затем после текста R2 —answer90days/R2/v2. Один final answer target не показывает, как достать evidence. Один search-call target не показывает, когда остановиться. Если inference требует именно последовательного поведения, данные должны представлять существенные decision points, включая stop/abstain.

В behavior cloning expert демонстрирует допустимое действие на данном history. Обучение максимизирует likelihood таких action tokens. Это не автоматическая оценка environment (среда взаимодействия) return: длинный успешный путь и короткий успешный путь могут оба присутствовать, и их относительный вес задают data/normalization (нормировка по выбранному правилу). Runtime (среда исполнения модели) всё ещё исполняет actions, а θ учится их предлагать.

### Teacher (модель-учитель) forcing по траектории

При training очередной target token предсказывается при известных предыдущих tokens правильной demonstration. Для следующего action в prefix уже включено **настоящее** tool observation этой trajectory. Во время rollout предыдущие actions генерирует сама policy, поэтому history может отличаться. Это источник train/inference mismatch, а не ошибка teacher forcing как арифметической операции.

Полный transcript можно хранить одной causal sequence: user → assistant call → tool result → assistant call → tool result → assistant answer. Causal mask не даёт раннему search call подсмотреть будущий tool result справа. Если же data preprocessing перенёс gold result в начальный system/user context, causal mask уже не спасёт от leakage (утечка недоступной информации): подсказка стала прошлым input.

Отдельные history-action examples тоже допустимы, если history соответствует ровно этому decision point. Нельзя вырезать read target из полной trajectory и оставить у него только исходный вопрос, потеряв search result с ID. И нельзя дать action read(R2) conditioning на observation от другого episode, где R2 обозначает совсем иной document (документ).

### Три decisions и простая likelihood арифметика

Для иллюстрации условимся, что каждый из трёх action targets (целевые значения) представлен одним учебным символом, а их условные вероятности 0.5,0.25,0.8. Likelihood всей target sequence0.1. Сумма NLL $-\ln0.5-\ln0.25-\ln0.8=\ln10\approx2.302585$; средний NLL на эти три symbols≈0.767528. Настоящие calls состоят из многих tokens, но принцип произведения условных вероятностей тот же.

Уменьшить этот loss можно, сделав demonstrated actions вероятнее при demonstrated histories. Из одной величины нельзя вывести успех на ошибочной history, которой в data не было, или корректность runtime idempotency. Low training loss также не означает, что все valid alternative searches должны иметь нулевую вероятность.

### Нормировка задаёт относительное влияние sessions

Пусть trajectory A имеет 2 active tokens со средним loss0.2, B —8 active tokens со средним loss1.2. Token-average loss=(2·0.2+8·1.2)/10=1.0. Если усреднить сначала внутри каждой trajectory, затем по двум trajectories, получится 0.7. В первом objective B получает 80% token weight, во втором обе sessions равный вес.

Ни одна нормировка не универсально правильна. Long sessions могут содержать больше полезных decisions, а могут быть длинными повторениями final prose. Если финальные ответы на 100tokens доминируют над короткими call decisions, decrease average loss может мало менять sequence behavior. Для data mixture определяют sampling (случайный выбор по распределению)/weights по задачам, calls и outcomes, а не только по количеству строк JSONL.

## Какие demonstrations нужны

Успешные happy paths учат стандартному решению. Но нужны также: отсутствие evidence, invalid argument correction, tool timeout/retry, ненужный tool, conflicting/stale documents, early stop при достаточном result. Без таких data policy может применять один шаблон во всех states.

Проверить truth observations, разрешённость actions и provenance. Teacher-generated trajectory с текстом «tool вернул 30 дней» не становится реальным recorded execution, если tool не вызывался. Для simulated environment result допустим, но simulator/version и ground truth (эталонная разметка) должны быть объявлены.

Action equivalence важна. Две разные корректные search formulations не должны автоматически считаться неправильными из-за string mismatch. При этом ID/number/version constraints проверяются строго по смыслу. Dataset может содержать несколько допустимых actions при одном history; один hard target лишь задаёт один из путей.

### Полная demonstration с наблюдаемыми границами

Учебный transcript использует IDs как placeholders, не как universal special tokens:

```text
USER: найди действующий срок хранения отчёта проекта P, укажи source/version
ASSISTANT CALL c1: search("report retention P", limit=3)
TOOL c1: R1/v1 obsolete; R2/v2 active; D/v1 unrelated title
ASSISTANT CALL c2: read(document_id="R2")
TOOL c2: R2/v2 active; "Срок хранения отчёта — 90 дней."
ASSISTANT FINAL: 90 дней, R2/v2, с применимой ссылкой на passage
```

Gold data включает реальные либо явно simulated tool results. Teacher text «я нашёл R2» не заменяет execution record. Если simulator задаёт результат, сохраняют simulator version и task state (состояние системы); это честный учебный источник, но не external-world measurement. При synthetic teacher generation проверяют соответствие result действию и исходному corpus.

Tools descriptions тоже часть conditioning. Если demonstration использует search с одним schema (схема допустимых данных), а inference регистрирует другой limit/type/название, model обучена на несовместимый protocol. Registry revision и template (шаблон сериализации входа) version связываются с dataset. Дубли c1/c2 в разных episodes допустимы только если IDs локальны по protocol и правильно scoped.

### Recovery data должна показывать причину выбора

Для timeout trajectory после первой search call наблюдается transient error, затем повтор при доступном budget. Для no-results наблюдается успешный пустой result; действие может быть rewrite или abstain по задаче. Для denied —доступ не расширяется самовольно. Эти contexts различаются; один шаблон «после любого ERROR повторить то же» учит неправильному поведению.

После stale read(R1) эксперт может перейти к R2 и исправить ответ, если budget позволяет. Он не должен молча переписать старый observation30days в 90: исправляется следующий action/вывод, а фактическая history сохраняется. Успешная recovery trajectory может содержать ошибочный ранний action; чтобы не имитировать его как идеал, training mask/selection labels (метки) выбирают по цели.

Можно сохранить valid prefix, добавить expert correction на реально посещённой history, использовать preference pair или другой подходящий objective. Но простое положительное SFT на всех tokens проваленной session будет учить её действиям, включая bad choice. Negative outcomes сами по себе не делают обычный cross-entropy loss «антиимитацией».

### Нет единственной правильной search строки

«report retention P» и «срок хранения отчёта проект P» могут вести к одному evidence в данной среде. Exact string-match label (метка) отвергнет один valid путь. Для semantic action correctness проверяют запрос/constraints и actual retrieval (поиск кандидатов) outcome по protocol; для machine identifiers, numeric units и document revision требования могут быть строгими.

Если при одном history допустимы несколько actions, dataset может содержать несколько verified demonstrations. Teacher выбор одного пути не доказывает, что остальные неверны. Но indiscriminate augmentation (преобразование обучающего примера с согласованной целью) тоже опасна: paraphrase может убрать negation/version, а синтетический synonym —изменить intent. Diversity должна сохранять смысл и наблюдаемую выполнимость.

### Книги и диалоги как обучающие данные

Подходят ли для обучения книги с диалогами? Ответ «книги годятся только для roleplay» слишком узок. Важны задача и objective. Сырой literary dialogue не является автоматически instruction→correct action demonstration. Но техническое описание процесса можно преобразовать в проверенные question/evidence/action examples, если действительно восстановим правильный interface и outcome.

Для agentic SFT запись «герой попросил найти документ, другой ответил, что нашёл» не задаёт фактический search call, tool observation и permission context. Одни role tags не превращают это в rollout. Нужны callable tools, известная environment, корректные decisions и проверяемый финальный результат. Иначе модель усвоит имитацию рассказа о действии.

Продолжение language-model training на raw text и assistant-action cloning —разные постановки, даже если обе используют token CE. Тип данных, mask и protocol определяют, что модель учится предсказывать. Подробная связь books/SFT/continued pretraining (предобучение) раскрыта в [главе адаптации](<../03-adaptation/adapt-sft.md>).

## Template, masks и dependency boundaries

Роли assistant/tool и call IDs сериализуются по contract модели/runtime. Совпадение JSON syntax не гарантирует правильный выбор tool. EOS/turn-end маркеры, loss mask (маска позиций, включённых в ошибку) и causal shift проверяются на распечатанном batch (пакет примеров), как в обычном SFT.

Пример training sequence: user question → assistant search call → tool results → assistant read call → tool document → assistant final answer. Active loss positions должны отражать выбранные assistant outputs; tool contents сохраняются conditioning. Random opaque call IDs обычно генерируются по protocol, не должны превращаться в memorization target, если реальные IDs создаёт application.

Зависимые actions требуют observations. Если read ID должен прийти из search, demonstration не должна выдавать оба calls параллельно с подсмотренным ID. Независимые reads могут быть parallel, но dataset/runtime должны одинаково представлять их order и result mapping.

ReAct — interleaving decision/action/observation pattern, который можно использовать prompting или data format. Это не отдельная гарантия правильности и не обязательный запрос полного внутреннего chain-of-thought. [ReAct](https://arxiv.org/abs/2210.03629).

### Assistant-only loss не означает удаление tool result

Возьмём условные block lengths, уже включающие выбранные control tokens: user10,assistant search4,tool20,assistant read4,tool30,assistant final6. Всего 74tokens, active assistant targets14,tool context50,user context10. Удалить 50 tool tokens из input ради «мы не учимся на них» —значит лишить следующие decisions нужных observations. Нужно маскировать target loss, сохраняя conditioning.

При causal forward (прямой проход вычисления) gradient (градиент) от assistant loss может проходить через representations предыдущих user/tool tokens и attention (механизм внимания) weights, связывающих их с ответом. Поэтому source фраза «градиенты только на ответе» не означает, что prompt (входной запрос или инструкция) states обязательно detached. Убирается direct token-prediction loss на masked targets, а не вся роль этих inputs в вычислительном графе.

Для конкретного training setup возможны и другие masking objectives. Важно явно выбрать их и проверить соответствие inference policy: модель сама выдаёт actions, а не играет за внешнюю базу данных. Название SFTTrainer или role `tool` не гарантирует корректную маску автоматически; в [TRL SFT documentation](https://huggingface.co/docs/trl/sft_trainer) assistant-only режим выделен как отдельная configuration, зависящая от формата данных/template.

### Causal shift на десяти условных tokens

Пусть input sequence имеет positions0..9:

```text
0 BOS | 1 USER | 2 question | 3 ASSIST | 4 CALL |
5 TOOL | 6 observation | 7 ASSIST | 8 ANSWER | 9 EOS

labels = [-100,-100,-100,-100,CALL,-100,-100,-100,ANSWER,EOS]
```

Здесь CALL и ANSWER —учебные symbols вместо многотокенных strings. При обычном causal internal shift logit3 предсказывает label4 CALL;logit7 —label8 ANSWER;logit8 —label9 EOS. На logit3 модель ещё не может видеть observation6. На logit7 observation6 уже в прошлом и влияет на ответ, хотя его собственный label masked.

Если вручную сдвинуть labels, а trainer сдвинет их повторно, model будет сопоставлена с неправильными targets. Проверяют не только shape tensors, но и пары decoded prefix→target на active positions. Значение−100 —распространённый ignore_index contract конкретного loss implementation, а не «специальный token словаря»; выбранный trainer должен его действительно интерпретировать как mask.

EOS/end-of-turn masks определяются protocol. Если pad token ID совпадает с EOS, безусловно заменить **все** EOS labels наignore по сравнению IDs может удалить обучение остановке настоящего ответа. Padding (дополнение до общей длины) positions определяются по реальной batch structure/attention mask; stop markers не исчезают только потому, что тот же ID используется для padding.

### Templates, markers и непрозрачные IDs

Сначала строят фактическую model-native serialization, затем tokenization/masks. ChatML  —один format family, а не обязательный стандарт всех Transformer models. Role names могут быть обычным текстом внутри control template, tool calls —специально сериализованным span. Если format tools не совпал с training template checkpoint (сохранённое состояние модели), «валидный JSON» ещё не делает input понятным для tool policy.

[Hugging Face chat-template guide](https://huggingface.co/docs/transformers/chat_templating) показывает model-specific control formats. При отдельных шагах formatting/tokenization нужно избегать дублирования уже добавленных special tokens; параметры задаются по фактическому пути preprocessing. Для finished training dialogue не следует бездумно добавлять лишний пустой assistant-generation prefix, предназначенный для нового inference response.

Opaque runtime call IDs не должны становиться задачей memorization случайных значений, если их создаёт application. Если protocol требует model-generated link/reference, обучают именно этот contract и соответствие result mapping. Содержательные document IDs не являются просто шумом: read target должен ссылаться на реально найденный разрешённый документ.

### Packing, truncation и границы episodes

Укороченная последовательность может потерять observation, но оставить answer target. Тогда пример учит утверждать 90/R2 без достаточного input. Или может потеряться final answer, оставив только calls: меняется относительный training weight остановки. Length policy проверяют по semantic boundaries, а не только слева/справа от максимального token count.

Packing нескольких episodes в одну tensor row экономит padding, но требует понимания attention/reset semantics выбранного backend. Один EOS не является автоматической block-diagonal attention wall во всех implementations. Если task B видит private answer task A, «раздельные JSONL rows» уже не гарантируют изоляцию после packing. Для независимого comparison нужно зафиксировать реальную mask/position policy.

Parallel calls в demonstration могут следовать одному assistant message, затем multiple results с IDs. Если второй read зависит от search result, его нельзя переносить в тот же initial action с подсмотренным ID. Иначе training history содержит причинно невозможный путь. Независимые reads допустимы, если inference executor умеет такой protocol и budget accounting.

## Первый training experiment

Сначала проверить overfit на нескольких correct trajectories: loss уменьшается, outputs соблюдают protocol, observations не генерируются как actions. Затем independent train/validation (проверочная часть для выбора решения)/test tasks, small learning rate (шаг обучения)/budget по выбранной model и baseline (базовый вариант сравнения) без tuning.

Измерять selection, schema validity, semantic arguments, call count и final success отдельно. Если validity выросла, а task success нет, возможно data научила format, но не decision sequence. Если training success высокий, test fails после первой ошибки, вероятен distribution shift — следующая глава.

Runtime validators/permissions остаются обязательными даже после высокого held-out score (оценка). Learning не заменяет hard constraints. Успешный fine-tuning (дообучение модели) на read-only toy tools не доказывает пригодность для arbitrary mutating production actions.

### До optimizer (оптимизатор обновления параметров): inspect, replay и крошечный overfit check

Для теоретической проверки сначала распечатывают один serialized example: roles/control tokens, input IDs, labels, active count, causal prediction pairs и tool boundaries. Затем replay actions в declared environment/stub проверяет, что recorded observations им соответствуют. Это разные проверки: правильные tensor masks не доказывают правильные database facts.

Если training эксперимент действительно проводится, маленький overfit sanity check на нескольких verified traces помогает найти ошибку objective/template/labels. Он не является held-out score и не доказывает generalization. В текущем конспекте описывается процедура; полноразмерный training run или измеренный результат пользователю не приписывается.

Перед сравнением фиксируют base/model/adapter/template, dataset revisions, task split (разбиение данных), sampling/masks/normalizer, optimizer budget и evaluation protocol. Новая ошибка после inference может быть несовпадением template или среды, а не автоматически «модель не обучилась». Compression/merge/export оцениваются после того, как они действительно изменили tested computation.

### Оценка policy отдельно от обучения на красивых messages

На held-out tasks смотрят selection, args, observation use, final success, invalid calls, recovery и costs. При одинаковом task budget baseline и tuned policy получают одинаковые доступные tools/data. Увеличить retries у новой модели и объявить весь success gain следствием SFT нельзя без отдельной ablation (сравнение с удалённой или изменённой частью метода).

Happy-path evaluation не тестирует reaction на ошибки. При injected timeout полезно проверить, корректно ли policy отличает его от empty result, не расширяет ли scope после denied, не подставляет ли выдуманный ID после malformed search output. Новые fault conditions в test должны иметь declared semantics, а не случайно отличаться от описания tool в prompt.

Low supervised loss на expert history и poor rollout success могут сосуществовать: собственная первая ошибка переводит policy на unseen history. Это ведёт к distribution shift/DAgger следующей главы. Но сначала исключают сломанные input masks, data leakage и inconsistent tool contract, потому что dataset aggregation не исправит принципиально неверную serialization сам по себе.

### Почему выученное поведение не заменяет runtime invariant

Даже model с высоким held-out score остаётся источником proposals. Scope, registry, argument constraints, idempotency и budget enforcement реализуют по реальному contract приложения. Training data может показывать правильные ограничения и улучшать compliance, но среднее качество не заменяет проверку каждого конкретного call.

В read-only toy environment некоторые ошибки просто дают failure reward (награда за действие или результат). В mutating environment та же ошибка может иметь реальный effect до следующего token. Перенос требует нового environment/interface/eval анализа, а не только более длинного role prompt. Поэтому learning и execution policy связаны, но их guarantees различаются.

## Вопросы и практика

1. Нужно ли обучать model воспроизводить tool result как свой output? **Ответ:** только если это явно соответствует другой задаче; в обычном action cloning observation — context, не policy action.
2. Почему string-exact search query (запрос) accuracy недостаточна? **Ответ:** несколько queries могут быть semantically equivalent и вести к верному evidence.
3. Доказывает ли schema validity успешный BC? **Ответ:** нет, model может выбрать wrong tool или бессмысленный аргумент в правильном формате.

4. Видит ли ранний call будущий tool result в causal full transcript? **Ответ:** нет, если корректна causal mask и result не перенесён в его prefix preprocessing.
5. Что изменится при token-average вместо trajectory-average? **Ответ:** длинные active sequences получают больший относительный вес.
6. Masked tool target означает detached context? **Ответ:** нет, observation representations могут участвовать в gradient path от следующего assistant loss.
7. Почему нельзя маскировать все EOS по одному ID при pad=EOS? **Ответ:** можно случайно убрать реальные stop targets; padding определяют по batch structure.
8. Что учит обычный positive SFT на failed actions? **Ответ:** имитировать эти targets; negative outcome не превращает CE автоматически в anti-imitation.
9. Требует ли ReAct prompting обязательно fine-tuning? **Ответ:** нет, pattern можно использовать на inference; обучение —отдельное изменение weights.
10. Доказывает ли overfit sanity check успех на новых задачах? **Ответ:** нет, он проверяет возможность обучения/интеграцию на маленьком training subset.

**Практика:** размечать assistant/tool roles и loss positions в трёх trajectories предыдущей главы. Добавить one recovery path и one no-tool-needed case. Сформулировать oracle для action semantics и final state; затем спроектировать small SFT comparison, не заявляя numerical results до запуска.

## Результат главы и переход дальше

Следующие темы: [Distribution shift, recovery и DAgger](<agent-recovery.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
