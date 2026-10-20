# SFT: данные, шаблоны, masks и objective

**Уровень:** основа ветки.

**Предпосылки:** [Задача адаптации, baseline и выбор вмешательства](<adapt-choice.md>)

**Используют:** [Полное обучение или PEFT](<adapt-regimes.md>), [LoRA: low-rank обновление и градиенты](<adapt-lora.md>), [DoRA, AdaLoRA, LoRA+ и QLoRA](<adapt-variants.md>), [Adapters, prefix tuning, soft prompts и BitFit](<adapt-alternatives.md>), [RLHF, PPO, DPO, IPO, KTO и ORPO](<adapt-preference.md>), [Model soup, TIES, DARE и переключение адаптеров](<adapt-merging.md>), [Дистилляция и обучение на траекториях](<adapt-distillation.md>), [AI feedback и continual learning](<adapt-continual.md>), [Оценка, perplexity, judge и экспорт](<adapt-evaluation.md>), [Behavior cloning и SFT на демонстрациях](<../06-agent-learning/agent-demonstrations.md>)

**Учебный вопрос:** Какие позиции становятся целями supervised fine-tuning?

SFT здесь определяется через supervised targets и loss. Метод обновления весов пока не фиксируется: полный fine-tuning и LoRA могут оптимизировать одинаковую цель на одинаковых примерах.



## Карта вычисления и границы этапов

![SFT: данные, шаблоны, masks и objective: входы, преобразования и выходы](<../../оформление/course/adapt-sft.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Сравните формы и смысл каждого перехода с формулами главы. Выход преобразования и доказательство его качества — разные объекты: последняя проверка использует отдельный протокол оценки.

<!-- unit:a61544facb26f84e24 -->
<a id="u-a61544facb26f84e24"></a>

## SFT и pretraining могут использовать похожую CE для разных данных

<!-- /unit:a61544facb26f84e24 -->

<!-- unit:aab56dbb4deaa87abb -->
<a id="u-aab56dbb4deaa87abb"></a>

Base model училась продолжать тексты своего training distribution. Чтобы получить желаемый ответ на инструкцию, supervised fine-tuning показывает примеры нужного поведения в выбранном формате. Для decoder-only модели это часто та же next-token cross-entropy, но на instruction/response данных и с определённой loss mask.

<!-- /unit:aab56dbb4deaa87abb -->

<!-- unit:2da2018da48d3c8934 -->
<a id="u-2da2018da48d3c8934"></a>

Не нужно противопоставлять «pretraining даёт знания, SFT только форму» как строгий закон. Параметры после SFT могут изменять фактические ответы, reasoning и другие способности. Однако маленький набор демонстраций не гарантирует усвоения большого нового домена и не обеспечивает обновляемую базу фактов.

<!-- /unit:2da2018da48d3c8934 -->

<!-- unit:d98f5eb79b38ae0071 -->
<a id="u-d98f5eb79b38ae0071"></a>

Правильная отправная точка — распределение задач. Если нужны краткие ответы по договору, демонстрации должны показывать контекст договора, вопрос, верный ответ и поведение при отсутствии информации. Если обучать только длинным образцовым рассуждениям, модель может усвоить длину/стиль, а не нужный критерий правильности.

<!-- /unit:d98f5eb79b38ae0071 -->

<!-- unit:2e7bb6f7cefd21364b -->
<a id="u-2e7bb6f7cefd21364b"></a>

### Что меняется при превращении base model в instruction model

<!-- /unit:2e7bb6f7cefd21364b -->

<!-- unit:cfb619fac49efb6217 -->
<a id="u-cfb619fac49efb6217"></a>

Задача next-token prediction остаётся: модель оценивает продолжение prefix. Но demonstration задаёт иной prefix и цели. Вместо фрагмента энциклопедии input содержит instruction, optional context и role boundaries, а targets показывают нужный ответ. Модель учится, какая continuation соответствует этому типу общения, а не получает новый отдельный магический механизм «понимать команды».

<!-- /unit:cfb619fac49efb6217 -->

<!-- unit:4939c04103e5558811 -->
<a id="u-4939c04103e5558811"></a>

Например, один raw text «Париж — столица Франции» показывает language continuation. Пример «Вопрос: столица Франции? Ответ: Париж» дополнительно показывает интерфейс запроса. Для полезного QA по переданному документу желательно показать и «В приведённом отрывке этого нет», иначе demonstrations могут подкреплять уверенное дополнение недостающих фактов. Negative/unanswerable examples задают желаемое поведение именно на границе информации.

<!-- /unit:4939c04103e5558811 -->

<!-- unit:42ac2dceccc2377d15 -->
<a id="u-42ac2dceccc2377d15"></a>

SFT демонстрирует выбранные ответы, а не сравнение всех возможных вариантов. Это не означает, что оно способно учить только стиль: target tokens несут и содержание, и последовательность решения. Но исходная формула «SFT недостаточно, после него обязательно RLHF/DPO, чтобы модель стала хорошей» не является универсальным pipeline requirement. Дополнительный preference stage выбирают по проблеме и доступному feedback, а итоговое качество измеряют напрямую.

<!-- /unit:42ac2dceccc2377d15 -->

<!-- unit:5182d4370371c4ce1d -->
<a id="u-5182d4370371c4ce1d"></a>

## Книги: сырьё зависит от задачи

<!-- /unit:5182d4370371c4ce1d -->

<!-- unit:95f659435c1af890e8 -->
<a id="u-95f659435c1af890e8"></a>

Сырой книжный текст естественно использовать для CPT language modeling, если это соответствует цели и допустимым источникам. Но наличие реплик персонажей не делает его готовым набором инструкций для assistant. Роль персонажа, художественная условность и ответы без достаточного контекста могут расходиться с задачей.

<!-- /unit:95f659435c1af890e8 -->

<!-- unit:7db911903aa2d34693 -->
<a id="u-7db911903aa2d34693"></a>

При этом «книжные диалоги всегда вредны» тоже неверно. Для модели художественных диалогов они могут быть подходящими демонстрациями. Для QA по книге можно построить примеры «отрывок → вопрос → подтверждённый ответ», включая unanswerable случаи. Для обучения стилю нужен именно этот стиль и независимая оценка, что factuality/другие задачи не деградируют.

<!-- /unit:7db911903aa2d34693 -->

<!-- unit:f8a5533bfb7b71030b -->
<a id="u-f8a5533bfb7b71030b"></a>

При преобразовании книги в SFT:

<!-- /unit:f8a5533bfb7b71030b -->

<!-- unit:e0eeb233bab337f9fd -->
<a id="u-e0eeb233bab337f9fd"></a>

1. Определить задачу и единицу примера.
2. Отделить источник/главы для holdout до массовой генерации похожих вопросов.
3. Сохранить passage ID и происхождение каждой пары.
4. Проверить ответы по тексту; убрать дубликаты и очевидные ошибки teacher.
5. Проверять перенос на новых вопросах и источниках, а не только те же passages с иной формулировкой.

<!-- /unit:e0eeb233bab337f9fd -->

<!-- unit:dc1f774c80a9d2bd5a -->
<a id="u-dc1f774c80a9d2bd5a"></a>

Teacher-generated data не бесплатны: нужны генерация, фильтрация и проверка. Удачный synthetic training loss не подтверждает качество на реальных запросах.

<!-- /unit:dc1f774c80a9d2bd5a -->

<!-- unit:64eebb7668c3d0e1a2 -->
<a id="u-64eebb7668c3d0e1a2"></a>

**Self-Instruct** расширяет исходный набор задач: из seed-инструкций teacher предлагает новые инструкции и ответы, после чего отсеиваются повторы, непроверяемые задания и ошибки. Если seed охватывает только один тип задачи, увеличение количества само по себе не добавит coverage. **Evol-Instruct** преобразует задачи, добавляя сложность/ограничения; искусственно усложнённый prompt нужно проверять на непротиворечивость и наличие корректного ответа.

<!-- /unit:64eebb7668c3d0e1a2 -->

<!-- unit:528196e43a3ae06628 -->
<a id="u-528196e43a3ae06628"></a>

**Multi-task instruction tuning, FLAN-style:** существующие задачи классификации, QA, перевода и суммаризации переводятся в instruction-response формат и смешиваются. Например, label `positive` превращается в ответ на явную инструкцию классификации. Разные templates и удержанные семейства задач позволяют проверять перенос на новые формулировки. Смесь может улучшить instruction generalization, но не гарантирует решение любой невиданной задачи; пропорции и конфликтующие форматы требуют проверки.

<!-- /unit:528196e43a3ae06628 -->

<!-- unit:735b9b2648a8f3959b -->
<a id="u-735b9b2648a8f3959b"></a>

### Возвращаем вопрос пользователя о книгах с тремя разными objectives

<!-- /unit:735b9b2648a8f3959b -->

<!-- unit:a59568ec53043d4615 -->
<a id="u-a59568ec53043d4615"></a>

Пусть в книге есть реплика «Куда он пошёл?» и ответ «Туда же». Без предыдущей сцены такой pair не определяет полезный ответ. Если обучать assistant на случайно вырезанных репликах, loss можно уменьшать за счёт литературного шаблона, не учась восстанавливать отсутствующее содержание. Проблема в contract examples, а не в самом происхождении текста из книги.

<!-- /unit:a59568ec53043d4615 -->

<!-- unit:519c97c9f2cb57eaea -->
<a id="u-519c97c9f2cb57eaea"></a>

Для CPT сохраняют связные passages и language-model objective; цель — адаптировать распределение языка/домена. Для grounded SFT строят passage+question→supported answer и проверяют, что ответ содержится в доступном passage. Для roleplay/стиля можно намеренно воспроизводить художественные dialogues с контекстом персонажей. Один и тот же источник даёт разные training sets в зависимости от задачи.

<!-- /unit:519c97c9f2cb57eaea -->

<!-- unit:db8ab5016c236ccd1d -->
<a id="u-db8ab5016c236ccd1d"></a>

Разобранный grounded пример: passage «Экспедиция вышла во вторник. Ею руководила Анна». Instruction «Кто руководил экспедицией?» имеет target «Анна». Вопрос «В каком году?» должен получить target об отсутствии года, а не догадку teacher. Если добавить в target год из другой главы, недоступной input, example начнёт обучать другой зависимости, чем заявленное QA по отрывку.

<!-- /unit:db8ab5016c236ccd1d -->

<!-- unit:36fc9ca8046a7260f0 -->
<a id="u-36fc9ca8046a7260f0"></a>

Split делают по источникам/главам/семействам похожих passages до генерации variants. Десять перефразированных вопросов к одному предложению, разнесённые по train/test, не доказывают generalization на новые сведения. Для каждого synthetic pair сохраняют passage ID, teacher prompt и результат проверки. Длинная красивая answer не заменяет подтверждения её фактов.

<!-- /unit:36fc9ca8046a7260f0 -->

<!-- unit:cb383774027b420c9e -->
<a id="u-cb383774027b420c9e"></a>

### Ручные demonstrations, Self-Instruct и FLAN-style mixture

<!-- /unit:cb383774027b420c9e -->

<!-- unit:b74aaa7dee15b10fc7 -->
<a id="u-b74aaa7dee15b10fc7"></a>

Ручная разметка позволяет контролировать quality и policy, но люди тоже ошибаются и расходятся в критериях. Нужны rubric, проверка согласованности и representative task coverage. Число10000–100000 из S3 — пример масштаба, не минимальный допуск SFT и не доказательство невозможности ручного сбора.

<!-- /unit:b74aaa7dee15b10fc7 -->

<!-- unit:6ede5e25843c28d188 -->
<a id="u-6ede5e25843c28d188"></a>

Self-Instruct начинает с seed tasks, генерирует новые instructions/inputs/answers и фильтрует. Если seeds содержат только лёгкий перевод, большой объём похожих examples не создаст автоматически tool use или factual abstention. Evol-Instruct добавляет constraints/сложность, но может сделать prompt противоречивым. Нужно проверять существование корректного ответа и соответствие distribution реальных запросов.

<!-- /unit:6ede5e25843c28d188 -->

<!-- unit:18ae6ef180e5eadf90 -->
<a id="u-18ae6ef180e5eadf90"></a>

FLAN-style преобразует существующие supervised tasks в instructions. Классификация sentiment получает rubric и output label, QA — context/question, перевод — target language. Task templates и пропорции влияют на training signal: большой summarization corpus может подавить редкую structured-output задачу. Проверка на held-out task families сильнее проверки на новой формулировке того же train template, но не гарантирует способность выполнить любую неизвестную инструкцию.

<!-- /unit:18ae6ef180e5eadf90 -->

<!-- unit:58cf8f75c50acacd6d -->
<a id="u-58cf8f75c50acacd6d"></a>

## Роли, template и labels

<!-- /unit:58cf8f75c50acacd6d -->

<!-- unit:9d9af02537986c6dca -->
<a id="u-9d9af02537986c6dca"></a>

Сообщения system/user/assistant превращаются chat template в token sequence. Служебные маркеры разделяют роли и конец сообщений. Если при обучении использовать один template, а при inference другой, вход меняется. Даже правильные по смыслу диалоги могут стать неудачным training set из-за неверных boundaries/EOS.

<!-- /unit:9d9af02537986c6dca -->

<!-- unit:281b284dc2ec524bf4 -->
<a id="u-281b284dc2ec524bf4"></a>

Возьмём условную последовательность:

<!-- /unit:281b284dc2ec524bf4 -->

<!-- unit:b0ae0a98a7ab4ab252 -->
<a id="u-b0ae0a98a7ab4ab252"></a>

```text
pos:       0     1      2       3        4      5
token:    BOS   USER   вопрос   ASSIST   ответ  EOS
```

<!-- /unit:b0ae0a98a7ab4ab252 -->

<!-- unit:b593fe18d9c95e4d58 -->
<a id="u-b593fe18d9c95e4d58"></a>

Предсказания на входных позициях 0–4 имеют targets `[USER,вопрос,ASSIST,ответ,EOS]`. Если нужен loss только на ответе assistant, активны targets `ответ,EOS`; служебные границы включаются или исключаются согласно выбранной явной политике.

<!-- /unit:b593fe18d9c95e4d58 -->

<!-- unit:8edf82b8ef6769060b -->
<a id="u-8edf82b8ef6769060b"></a>

Для probabilities истинных targets $p_{t}$ и mask m:

<!-- /unit:8edf82b8ef6769060b -->

<!-- unit:b2394ec76dfeff35b8 -->
<a id="u-b2394ec76dfeff35b8"></a>

$$
\mathcal{L}=-\frac{\sum_t m_t\log p_t}{\sum_t m_t}.
$$

<!-- /unit:b2394ec76dfeff35b8 -->

<!-- unit:c28a380df702c42a65 -->
<a id="u-c28a380df702c42a65"></a>

Если $p_{\text{ответ}}=0.5$, $p_{\mathrm{EOS}}=0.25$, получим $\mathcal{L}\approx1.03972$. Если считать ошибку по всем токенам, задача и normalizer изменятся. Assistant-only — распространённый выбор для instruction tuning, но all-token training тоже может быть осмысленным при другой цели.

<!-- /unit:c28a380df702c42a65 -->

<!-- unit:87615d528f467e38ff -->
<a id="u-87615d528f467e38ff"></a>

Во многих trainers labels создают копированием IDs, часть заменяют на ignore index (часто -100), а causal shift выполняется внутри модели. Mask в примере выше описывает **целевые токены**; при реализации нужно проверить, к какой оси исходных labels применяется shift. Простого красивого списка role messages недостаточно: распечатать реальные tokens/labels одного batch обязательно.

<!-- /unit:87615d528f467e38ff -->

<!-- unit:f2a2672a552bc45697 -->
<a id="u-f2a2672a552bc45697"></a>

### Прослеживаем masking до конкретного logit

<!-- /unit:f2a2672a552bc45697 -->

<!-- unit:7e8bb90741b29dfd94 -->
<a id="u-7e8bb90741b29dfd94"></a>

В условном примере `[BOS, USER, вопрос, ASSIST, ответ, EOS]` logit позиции3 (`ASSIST`) предсказывает target4 (`ответ`), а logit позиции4 — target5 (`EOS`). Маска целевых positions4/5 не означает «входы0–3 отсоединены». Они участвуют в attention и формировании answer logits, поэтому gradient от answer loss может проходить через representations prompt.

<!-- /unit:7e8bb90741b29dfd94 -->

<!-- unit:a4ab0d45b4dd902cc8 -->
<a id="u-a4ab0d45b4dd902cc8"></a>

Если trainer ожидает labels той же длины, что input IDs, часто masked labels выглядят `[-100,-100,-100,-100,ответ,EOS]`, а internal shift сопоставляет logits[:-1] с labels[1:]. Если сдвинуть labels вручную и оставить ещё internal shift, получится предсказание через два tokens. Mask и shift проверяют на детокенизированном batch, а не только по его tensor shapes.

<!-- /unit:a4ab0d45b4dd902cc8 -->

<!-- unit:848c2101aeb65311ec -->
<a id="u-848c2101aeb65311ec"></a>

У role markers есть собственная политика supervision. Хотим ли учить начало/конец ответа, tool-result boundaries, несколько assistant turns? All-assistant mask и final-turn-only mask дают разные objectives. Для двухходового диалога первый ответ может учиться как target или остаться только context второго. Оба варианта могут быть осмысленны, но это нужно объявить и воспроизвести в collator.

<!-- /unit:848c2101aeb65311ec -->

<!-- unit:37f5875c7b8b360913 -->
<a id="u-37f5875c7b8b360913"></a>

В S3 ChatML назван единым современным стандартом. На практике serialization определяется tokenizer/template checkpoint; не каждый instruct model обучался на этих маркерах. Подмена template меняет токены, boundaries и sometimes positions. Нельзя буквально заменить модельные special tokens человекочитаемыми `USER:`/`ASSISTANT:` и считать, что training input сохранился.

<!-- /unit:37f5875c7b8b360913 -->

<!-- unit:2fa1f9bdf55ea7e657 -->
<a id="u-2fa1f9bdf55ea7e657"></a>

## Padding, truncation и packing

<!-- /unit:2fa1f9bdf55ea7e657 -->

<!-- unit:562ea8fd8cf4d33785 -->
<a id="u-562ea8fd8cf4d33785"></a>

Padding не должен вносить искусственные supervised targets. Не следует автоматически исключать все токены с числом, равным pad_id, если он совпадает с eos_id: можно случайно убрать реальные EOS. Источник истинной длины/attention mask важен.

<!-- /unit:562ea8fd8cf4d33785 -->

<!-- unit:9d7f4ecff7757607d4 -->
<a id="u-9d7f4ecff7757607d4"></a>

Truncation может удалить весь ответ, оставить часть tool call или обрезать EOS. Пример с нулём supervised tokens либо отбрасывают, либо обрабатывают явно. Нужно смотреть распределение длин и долю усечённых ответов, а не только среднюю длину input.

<!-- /unit:9d7f4ecff7757607d4 -->

<!-- unit:1fb0c17bacd0d89177 -->
<a id="u-1fb0c17bacd0d89177"></a>

Packing объединяет короткие примеры для экономии padding. Но `[диалог A, EOS, диалог B]` при обычной causal mask позволяет B видеть A. Если требуется независимость, нужна поддерживаемая политика boundaries/attention isolation или другой способ подготовки. Наличие разделительного EOS само по себе не создаёт hard attention mask.

<!-- /unit:1fb0c17bacd0d89177 -->

<!-- unit:3de0cd2032eb727e90 -->
<a id="u-3de0cd2032eb727e90"></a>

Среднее по всем активным токенам даёт длинным ответам больший вклад, чем коротким. Среднее сначала по каждому ответу, затем по batch задаёт другое взвешивание. При gradient accumulation нормировки должны быть согласованы с effective token count; иначе переменная длина microbatches меняет фактический objective.

<!-- /unit:3de0cd2032eb727e90 -->

<!-- unit:c7eef8cd17e329a736 -->
<a id="u-c7eef8cd17e329a736"></a>

### Почему нормировка по tokens влияет на задачу

<!-- /unit:c7eef8cd17e329a736 -->

<!-- unit:5ba2ce8b8a42cf819b -->
<a id="u-5ba2ce8b8a42cf819b"></a>

Пусть short example содержит один supervised token с CE=2, а long example — девять tokens с CE=0.2 каждый. Среднее по всем десяти tokens=(2+1.8)/10=0.38; среднее двух sequence means=(2+0.2)/2=1.1. В первом objective long example получает девятикратный вес по числу targets. Во втором каждый example весит одинаково. Это различие важно для смеси коротких JSON и длинных объяснений.

<!-- /unit:5ba2ce8b8a42cf819b -->

<!-- unit:c9c046ce6d4f53fc9d -->
<a id="u-c9c046ce6d4f53fc9d"></a>

Gradient accumulation должен воспроизводить выбранное усреднение. Если первый microbatch содержит20 active tokens, второй200, простое среднее их scalar means не равно mean по220 tokens. Менять accumulation и видеть другой результат можно из-за normalization, а не обязательно случайного optimizer behavior.

<!-- /unit:c9c046ce6d4f53fc9d -->

<!-- unit:97fb70b3ec2423cecc -->
<a id="u-97fb70b3ec2423cecc"></a>

Packing экономит PAD, но без block attention isolation второй example может читать первый. EOS — token, а не математическая перегородка. Если нужен independent-example objective, реальные masks/position policy должны это обеспечивать. Если допускается stream-like training, это другая явная постановка. В обоих случаях нельзя включать targets, сформированные случайной склейкой границ.

<!-- /unit:97fb70b3ec2423cecc -->

<!-- unit:07a1f12a13106192b3 -->
<a id="u-07a1f12a13106192b3"></a>

При truncation перед ответом останется zero supervised targets. Деление на их число не определено. Если обрезан только конец, можно потерять EOS и учить модель часто продолжать сверх нужной длины. Анализ доли truncated examples и позиций actual loss помогает обнаружить это раньше большого запуска.

<!-- /unit:07a1f12a13106192b3 -->

<!-- unit:15066a8d78432f9167 -->
<a id="u-15066a8d78432f9167"></a>

## Как выглядит первый осмысленный эксперимент

<!-- /unit:15066a8d78432f9167 -->

<!-- unit:82cafe5bc66d615343 -->
<a id="u-82cafe5bc66d615343"></a>

Зафиксировать baseline, tokenizer/template/revision, train/validation split по prompts/источникам, критерии task quality и regression set. До большого запуска переобучить 10–20 хороших диалогов: цель — проверить, уменьшается ли loss и способна ли модель воспроизвести targets. Это диагностика loop, а не оценка обобщения.

<!-- /unit:82cafe5bc66d615343 -->

<!-- unit:e0826a73c1dded4e14 -->
<a id="u-e0826a73c1dded4e14"></a>

Затем обучить на основном train с отдельным validation и сравнить одну интервенцию: например, all-token против assistant-only при сопоставимом token budget. Не менять одновременно dataset mix, rank, LR и длительность, если нужен причинный вывод.

<!-- /unit:e0826a73c1dded4e14 -->

<!-- unit:438a866d16c4c77de4 -->
<a id="u-438a866d16c4c77de4"></a>

Контролировать train/validation loss, task correctness, длину, формат, срезы и базовые способности. Low validation CE может сочетаться с плохим generated answer из-за decoding, train/inference mismatch или слабой связи objective с задачей. SFT может быть достаточной интервенцией; preference stage не обязателен только потому, что он следующий по списку.

<!-- /unit:438a866d16c4c77de4 -->

<!-- unit:18a50cbe858746b722 -->
<a id="u-18a50cbe858746b722"></a>

**Ошибочные обобщения:** фиксированные «минимум 1000 примеров», гарантированные ускорения trainer, утверждение, что SFT никогда не исправляет hallucinations, и обещание, что alignment после SFT обязательно исправит все оставшиеся проблемы.

<!-- /unit:18a50cbe858746b722 -->

<!-- unit:0360a10d032316c83a -->
<a id="u-0360a10d032316c83a"></a>

## Вопросы с ответами

<!-- /unit:0360a10d032316c83a -->

<!-- unit:fd548c39a11dab135a -->
<a id="u-fd548c39a11dab135a"></a>

1. Подходит ли книга для SFT? **Ответ:** зависит от цели и подготовленных examples; сырой текст естественнее для CPT, но из него можно получить task-specific demonstrations.
2. Что проверить, если loss быстро стал нулевым? **Ответ:** не пусты ли masks, корректны ли labels/shift, не происходит ли утечка; затем качество на независимом наборе.
3. Почему packing может менять задачу? **Ответ:** соседние примеры могут получить общий доступный контекст и другое взвешивание tokens.
4. Чем overfit маленького batch отличается от итоговой оценки? **Ответ:** проверяет способность loop обучаться; не доказывает generalization.

<!-- /unit:fd548c39a11dab135a -->

<!-- unit:155ec5e6da9cf1e485 -->
<a id="u-155ec5e6da9cf1e485"></a>

**Практика:** вручную разметить один двухходовый диалог. Указать input, shifted targets, active loss positions, padding и EOS. Затем обрезать последние три токена и объяснить, остаётся ли полезный supervised signal.

<!-- /unit:155ec5e6da9cf1e485 -->

<!-- unit:79c668e64c6bd73e06 -->
<a id="u-79c668e64c6bd73e06"></a>

**Схема курса:** messages + trusted source → template/tokenization → input tokens и target labels → loss mask → causal CE → optimizer update; неизвестный target исключается, а не маркируется отрицательным

<!-- /unit:79c668e64c6bd73e06 -->

<!-- unit:31e8739a2bd1612c55 -->
<a id="u-31e8739a2bd1612c55"></a>

<!-- latex-display-v1 -->

<!-- /unit:31e8739a2bd1612c55 -->

<!-- unit:a27a95af4bc54d23d7 -->
<a id="u-a27a95af4bc54d23d7"></a>

<!-- source-depth-adaptation-Д -->

<!-- /unit:a27a95af4bc54d23d7 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Какие позиции становятся целями supervised fine-tuning?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Полное обучение или PEFT](<adapt-regimes.md>), [LoRA: low-rank обновление и градиенты](<adapt-lora.md>), [DoRA, AdaLoRA, LoRA+ и QLoRA](<adapt-variants.md>), [Adapters, prefix tuning, soft prompts и BitFit](<adapt-alternatives.md>), [RLHF, PPO, DPO, IPO, KTO и ORPO](<adapt-preference.md>), [Model soup, TIES, DARE и переключение адаптеров](<adapt-merging.md>), [Дистилляция и обучение на траекториях](<adapt-distillation.md>), [AI feedback и continual learning](<adapt-continual.md>), [Оценка, perplexity, judge и экспорт](<adapt-evaluation.md>), [Behavior cloning и SFT на демонстрациях](<../06-agent-learning/agent-demonstrations.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
