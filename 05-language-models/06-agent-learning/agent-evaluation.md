# Оценка траекторий и воспроизводимость

**Уровень:** основа ветки.

**Предпосылки:** [Среда, observations, actions и horizon](<agent-environment.md>)

**Используют:** [Behavior cloning и SFT на демонстрациях](<agent-demonstrations.md>), [Distribution shift, recovery и DAgger](<agent-recovery.md>), [Обучение через взаимодействие: PPO и GRPO](<agent-rl.md>), [Углубление: reasoning и верификация](<agent-reasoning.md>), [Углубление: память и длинный горизонт](<agent-memory.md>), [Углубление: предпочтения траекторий](<agent-preferences.md>), [Углубление: роли и обучение координации](<agent-coordination.md>)

**Учебный вопрос:** Что считать успехом целой попытки и как учитывать retries и стоимость?

Эта глава стоит перед первым training experiment: success, бюджет попытки, split и baseline определяют, какое изменение политики считать улучшением.



## Карта вычисления и границы этапов

![Оценка траекторий и воспроизводимость: входы, преобразования и выходы](<../../оформление/course/agent-evaluation.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Reward и targets используются в объявленном режиме обучения. При оценке политика фиксирована; success измеряется по всей попытке, включая повторы и бюджет.

<!-- unit:dc23eea7b09cf42c36 -->
<a id="u-dc23eea7b09cf42c36"></a>

**Схема курса:** task snapshot → complete run/attempt trace → outcome + failure/recovery + resource costs → grouped uncertainty → reproducible comparison report

<!-- /unit:dc23eea7b09cf42c36 -->

<!-- unit:76dcbcb9c1f57bfc68 -->
<a id="u-76dcbcb9c1f57bfc68"></a>

## Проверяемый успех вместо впечатления

<!-- /unit:76dcbcb9c1f57bfc68 -->

<!-- unit:21d3366a3fb8f28c77 -->
<a id="u-21d3366a3fb8f28c77"></a>

Конечный ответ — лишь один артефакт. Для read-only поиска success может означать: найдено актуальное правило, верно применены исключения, ответ подтверждён существующим passage. Для изменения данных oracle проверяет фактическое состояние, а не фразу «сделано». Если задача требует evidence, угаданное верное число без чтения источника не считается тем же успехом, что подтверждённый ответ.

<!-- /unit:21d3366a3fb8f28c77 -->

<!-- unit:c835e791b810014279 -->
<a id="u-c835e791b810014279"></a>

Для n задач бинарная оценка:

<!-- /unit:c835e791b810014279 -->

<!-- unit:84fc01910b8efeb81b -->
<a id="u-84fc01910b8efeb81b"></a>

$$
\widehat S=\frac1n\sum_{i=1}^{n}\mathbf1[\text{oracle подтвердил успех задачи }i].
$$

<!-- /unit:84fc01910b8efeb81b -->

<!-- unit:00fb8aeca19d242ee1 -->
<a id="u-00fb8aeca19d242ee1"></a>

Среднее можно посчитать и для зависимых записей, но статистическая интерпретация требует учитывать их зависимость. Сначала определяют задачи, источник truth, допустимые действия, deadline и правила partial/unknown outcomes. Потом запускают системы. Если менять критерий успеха после просмотра результатов, можно получить красивый score без реального улучшения.

<!-- /unit:00fb8aeca19d242ee1 -->

<!-- unit:2677221daaf743e8f9 -->
<a id="u-2677221daaf743e8f9"></a>

### Task, run, attempt, action и candidate

<!-- /unit:2677221daaf743e8f9 -->

<!-- unit:f85d6e0e1d69db0276 -->
<a id="u-f85d6e0e1d69db0276"></a>

Task задаёт цель и initial conditions. Run/episode — одно исполнение policy в выбранной среде. Внутри него могут быть tool retries и несколько answer candidates. Повторить целую задачу после reset — другой run; создать ещё один candidate в той же history — не обязательно новый run. Эти единицы нужны в журнале и знаменателях метрик.

<!-- /unit:f85d6e0e1d69db0276 -->

<!-- unit:740302e9c291871fd3 -->
<a id="u-740302e9c291871fd3"></a>

В S3 предлагались JSON validity и task-specific tests. Валидный call проверяет только один слой: syntax/schema могут быть верными при неверном source, аргументе или следующем действии. И наоборот, один invalid call может быть исправлен без потери конечного успеха. Метрики действий помогают объяснить outcome, но не заменяют его.

<!-- /unit:740302e9c291871fd3 -->

<!-- unit:3d398d5ff92498765e -->
<a id="u-3d398d5ff92498765e"></a>

S6 infrastructure metrics отвечают на вопросы о времени и capacity. Высокая throughput не доказывает правильное конечное состояние. Поэтому задача измеряется по двум связанным направлениям: что выполнено и сколько ресурсов на это потребовалось.

<!-- /unit:3d398d5ff92498765e -->

<!-- unit:ebf566685cee9d4e61 -->
<a id="u-ebf566685cee9d4e61"></a>

### Deadline, unknown и partial

<!-- /unit:ebf566685cee9d4e61 -->

<!-- unit:122b07689608d8ffa5 -->
<a id="u-122b07689608d8ffa5"></a>

Если task должен завершиться до deadline, поздний успех не обязательно соответствует требованию. После timeout у mutation возможен unknown: операция произошла, но подтверждение потеряно. У read-only запроса timeout тоже не означает отсутствие документа. В журнале сохраняют наблюдаемый статус, время проверки и outcome по объявленному cutoff.

<!-- /unit:122b07689608d8ffa5 -->

<!-- unit:ddbbc5d73f8ecf09b1 -->
<a id="u-ddbbc5d73f8ecf09b1"></a>

Unknown и partial можно считать неуспехом в конкретной binary metric, одновременно показывая их частоты отдельно. Нельзя удалить все engine errors и показать accuracy только для sessions с полученным ответом как качество всей системы. Нельзя вывести «задача решена» из того, что процесс достиг END.

<!-- /unit:ddbbc5d73f8ecf09b1 -->

<!-- unit:a64dfbb1d9d60cf415 -->
<a id="u-a64dfbb1d9d60cf415"></a>

Иногда некорректен сам evaluation case: сломан fixture или противоречив gold. Независимое правило исключения применяют симметрично ко всем системам и сохраняют причины/counts. Произвольная очистка после просмотра неудач candidate создаёт selection bias.

<!-- /unit:a64dfbb1d9d60cf415 -->

<!-- unit:32e6b06b5886859405 -->
<a id="u-32e6b06b5886859405"></a>

## Recovery, loops и остановка

<!-- /unit:32e6b06b5886859405 -->

<!-- unit:a74c55447a5eddaaef -->
<a id="u-a74c55447a5eddaaef"></a>

Recovery rate требует явного знаменателя:

<!-- /unit:a74c55447a5eddaaef -->

<!-- unit:397e235e9520d4f1c7 -->
<a id="u-397e235e9520d4f1c7"></a>

$$
\widehat R_{\mathrm{recovery}}=
\frac{\text{успешные задачи после заданного восстановимого сбоя}}
{\text{задачи с таким сбоем}}.
$$

<!-- /unit:397e235e9520d4f1c7 -->

<!-- unit:09cf630b3fce6ad823 -->
<a id="u-09cf630b3fce6ad823"></a>

Не сравнивают десять лёгких timeouts с десятью необратимо потерянными данными как одну группу. Можно отдельно оценить conditional recovery для восстановимых случаев, но полная task metric должна сохранять и остальные failures.

<!-- /unit:09cf630b3fce6ad823 -->

<!-- unit:9c0a4b7db35ca71ebb -->
<a id="u-9c0a4b7db35ca71ebb"></a>

### Retry не обязательно loop

<!-- /unit:9c0a4b7db35ca71ebb -->

<!-- unit:1b33bc701e912aafea -->
<a id="u-1b33bc701e912aafea"></a>

Одинаковые arguments ещё не доказывают loop. После transient timeout повтор read может быть корректным; после подтверждённого успеха тот же read без новой цели может быть лишним. Детектор учитывает состояние, evidence, budget и причину повторения. Cycle из нескольких разных calls также может не давать progress, хотя string-exact правило его пропустит.

<!-- /unit:1b33bc701e912aafea -->

<!-- unit:e6c76b771adace0d66 -->
<a id="u-e6c76b771adace0d66"></a>

Premature stop — завершение до обязательной проверки. Это отличается от честного budget-exhausted partial и permanent failure. Диагностические labels могут пересекаться: loop привёл к исчерпанию бюджета. Для таблицы primary causes задают правило приоритетов, иначе сумма категорий превысит число задач.

<!-- /unit:e6c76b771adace0d66 -->

<!-- unit:8e475505f6c5787987 -->
<a id="u-8e475505f6c5787987"></a>

Журнал содержит task/run ID, шаг, доступное observation, action/arguments, result/status, время, usage, stop reason и oracle outcome. Для анализа исполнения достаточно наблюдаемых решений и результатов; скрытое внутреннее рассуждение записывать не требуется. Обработка чувствительных полей следует реальному data contract среды.

<!-- /unit:8e475505f6c5787987 -->

<!-- unit:0b8bf836607f1332ba -->
<a id="u-0b8bf836607f1332ba"></a>

### Полный журнал десяти учебных sessions

<!-- /unit:0b8bf836607f1332ba -->

<!-- unit:88057806f3e10fbd69 -->
<a id="u-88057806f3e10fbd69"></a>

Это заданные примеры, а не измерения модели. Actions включают tool proposals и answer/abstain; tool calls считаются отдельно. Cost — условная полная цена session. Она не означает фиксированную одинаковую цену каждого action или тариф какого-либо API.

<!-- /unit:88057806f3e10fbd69 -->

<!-- unit:2b779e855ed67dc95b -->
<a id="u-2b779e855ed67dc95b"></a>

1. **T1:** clean success; 3 actions, 2 tool calls, 0 invalid; cost 1.
2. **T2:** timeout, разрешённый retry, success; 4 actions, 3 tool calls; cost 2.
3. **T3:** invalid args исправлены, success; 4 actions, 3 tool calls, 1 invalid; cost 2.
4. **T4:** stale source и неправильный final answer; 3 actions, 2 tool calls; cost 1.5.
5. **T5:** повторяющийся no-progress loop, budget failure; 6 actions, 6 tool calls; cost 3.
6. **T6:** permanent unavailability источника, восстановление невозможно по условиям setup; 2 actions, 2 tool calls; cost 1.
7. **T7:** premature answer без нужного evidence; 1 action, 0 tool calls; cost 0.5.
8. **T8:** unanswerable task, корректный abstain; 1 action, 0 tool calls; cost 0.5.
9. **T9:** recoverable timeout, но policy расходует доступные шаги на нерелевантные calls; failure; 6 actions, 6 tool calls; cost 2.
10. **T10:** первый answer candidate отклонён validator, после исправления success; 5 actions, 3 tool calls и 2 answer candidates; cost 2.5.

<!-- /unit:2b779e855ed67dc95b -->

<!-- unit:7b75a8d7d6dfa1ff47 -->
<a id="u-7b75a8d7d6dfa1ff47"></a>

В T9 в момент сбоя ещё были разрешённый recovery path и достаточный budget; policy не использовала их правильно. В T6 восстановление невозможно по заранее заданным условиям. Это различие устанавливают до подсчёта метрик, а не для исключения неудобных failures.

<!-- /unit:7b75a8d7d6dfa1ff47 -->

<!-- unit:c900fa2c2a4d28d129 -->
<a id="u-c900fa2c2a4d28d129"></a>

### Считаем разные outcomes

<!-- /unit:c900fa2c2a4d28d129 -->

<!-- unit:60c2f9b364cc9a44ec -->
<a id="u-60c2f9b364cc9a44ec"></a>

Успешны T1, T2, T3, T8, T10: 5/10=0.5. Здесь успех включает правильный отказ на unanswerable T8. Всего 35 actions, из них 27 tool calls. Единственный schema-invalid call в T3 даёт validity 26/27≈96.30%, хотя end-to-end success только 50%.

<!-- /unit:60c2f9b364cc9a44ec -->

<!-- unit:403697055a0fb5f565 -->
<a id="u-403697055a0fb5f565"></a>

Recovery cases T2, T3, T9 дают 2/3≈66.67%. По timeouts отдельно T2/T9 — 1/2; по invalid args T3 — 1/1. Включение невосстановимого T6 даёт 2/4=0.5: это другой знаменатель. Все десять tasks остаются в общей success metric.

<!-- /unit:403697055a0fb5f565 -->

<!-- unit:319cb97a7f55a93f7a -->
<a id="u-319cb97a7f55a93f7a"></a>

В журнале ровно один run на task. Tool retry T2 и второй answer candidate T10 не являются автоматическим повтором всей задачи с reset. First-attempt success нужно определить до сбора данных, а не восстанавливать по удобному количеству messages.

<!-- /unit:319cb97a7f55a93f7a -->

<!-- unit:f86760fd0b525f4d7f -->
<a id="u-f86760fd0b525f4d7f"></a>

## Сколько стоит успешная задача

<!-- /unit:f86760fd0b525f4d7f -->

<!-- unit:d8211dde174955b586 -->
<a id="u-d8211dde174955b586"></a>

Пусть C_i — полная цена всех попыток task i, а S_i — индикатор её успеха:

<!-- /unit:d8211dde174955b586 -->

<!-- unit:ebb4700d2a08917280 -->
<a id="u-ebb4700d2a08917280"></a>

$$
C_{\mathrm{per\ success}}=\frac{\sum_i C_i}{\sum_i S_i}.
$$

<!-- /unit:ebb4700d2a08917280 -->

<!-- unit:53f73e21d886af5fe1 -->
<a id="u-53f73e21d886af5fe1"></a>

В числителе остаются затраты failures. Например, 100 units при 40 успешных tasks дают 2.5 units/success. При нуле successes показатель не равен нулю: он не определён либо отображается как бесконечный по выбранной convention.

<!-- /unit:53f73e21d886af5fe1 -->

<!-- unit:3cb9c261a938aa9c73 -->
<a id="u-3cb9c261a938aa9c73"></a>

### Цена учебного журнала

<!-- /unit:3cb9c261a938aa9c73 -->

<!-- unit:5eb0d2acb516a65bf4 -->
<a id="u-5eb0d2acb516a65bf4"></a>

Costs T1–T10 суммируются в 16 units. Cost per success =16/5=3.2. Успешные sessions сами обошлись в 8 units. Деление только их цены на 5 дало бы 1.6, но половина расходов исчезла бы из оценки. Average cost per task тоже 16/10=1.6; численное совпадение не делает эти показатели одинаковыми по смыслу.

<!-- /unit:5eb0d2acb516a65bf4 -->

<!-- unit:56a67832483e0eff04 -->
<a id="u-56a67832483e0eff04"></a>

Полный учёт может включать policy/judge/reference вычисления, retrieval/reranking, tools, communication, retries и failed runs. Для выбора trained system дополнительно амортизируют training/data/integration cost на объявленный горизонт. GPU time, token billing и число calls имеют разные единицы и не складываются без заданного преобразования.

<!-- /unit:56a67832483e0eff04 -->

<!-- unit:c9497bbd0801e143e4 -->
<a id="u-c9497bbd0801e143e4"></a>

### Fixed budget и сравнение вмешательств

<!-- /unit:c9497bbd0801e143e4 -->

<!-- unit:58d4efcaf76197c0af -->
<a id="u-58d4efcaf76197c0af"></a>

Исходный пример: A решает 60 из 100 задач за 100 units, B — 75 при большем бюджете за 260. Получается примерно 1.667 и 3.467 units/success. B выше по данному success, A дешевле на успех. Без quality/resource требований нет безусловного победителя.

<!-- /unit:58d4efcaf76197c0af -->

<!-- unit:e9d60b9a8f8d430e7b -->
<a id="u-e9d60b9a8f8d430e7b"></a>

Чтобы проверить эффект обучения или координации, сравнивают также при сопоставимом inference budget. Чтобы выбрать целую систему, задают constraints или cost-quality curve. Прирост от трёх попыток нельзя целиком приписать новым weights, если baseline получила одну.

<!-- /unit:e9d60b9a8f8d430e7b -->

<!-- unit:a7365aeabee8a8722d -->
<a id="u-a7365aeabee8a8722d"></a>

First-run success, успех после N reset attempts и успех после нескольких candidates внутри run различаются. Perfect oracle selection показывает coverage, а не качество реального selector. В cost сохраняют всю серию attempts, даже когда финальный ответ выбран только из одной.

<!-- /unit:a7365aeabee8a8722d -->

<!-- unit:eff50dc0d90de5c76a -->
<a id="u-eff50dc0d90de5c76a"></a>

### Latency и timeouts

<!-- /unit:eff50dc0d90de5c76a -->

<!-- unit:58af1769f4b45774ae -->
<a id="u-58af1769f4b45774ae"></a>

Сохраняют начало/конец session, queues, model/tool stages, time-to-first-visible-output и финальную verification. Первый token не равен выполненной задаче; возвращённый tool proposal не равен исполненной операции. При parallel work wall time и сумма resource time различаются.

<!-- /unit:58af1769f4b45774ae -->

<!-- unit:5d299e3c550e66097f -->
<a id="u-5d299e3c550e66097f"></a>

Latency completed subset — условная метрика. Она может быть полезна вместе с долей timeouts, но её нельзя выдавать за ожидание всех пользователей. Указывают quantile convention, число observations и cohort versions. Средние p95 нескольких pods без raw/cohort данных не превращаются в global p95.

<!-- /unit:5d299e3c550e66097f -->

<!-- unit:2f67f52c5156d79971 -->
<a id="u-2f67f52c5156d79971"></a>

## Split, seeds и статистическая неопределённость

<!-- /unit:2f67f52c5156d79971 -->

<!-- unit:874bd3d5d691794074 -->
<a id="u-874bd3d5d691794074"></a>

Split выбирают по переносу: новые task families, documents или будущие периоды. Случайные paraphrases одной задачи в train/test создают leakage. Messages одной session не являются независимыми задачами. Фиксируют model/tokenizer/template, environment/tools, corpus snapshot, decoding, budget, seed и oracle code.

<!-- /unit:874bd3d5d691794074 -->

<!-- unit:5abfc0c70b3d640b4e -->
<a id="u-5abfc0c70b3d640b4e"></a>

### Две оси variation

<!-- /unit:5abfc0c70b3d640b4e -->

<!-- unit:8377e8895ed900a511 -->
<a id="u-8377e8895ed900a511"></a>

Новые tasks показывают различия между задачами; repeated seeds — variation policy/environment на той же задаче. 10 tasks×10 seeds дают 100 runs, но не 100 независимых новых task families. Resampling/анализ сохраняет groups, если outcomes внутри них зависимы.

<!-- /unit:8377e8895ed900a511 -->

<!-- unit:2e485822f6677073c3 -->
<a id="u-2e485822f6677073c3"></a>

Для paired comparison обе системы выполняют один набор tasks/snapshots. Просматривают disagreements и причины, а не только totals. Разные policies закономерно получают разные observations, вызывая разные tools. Replay не должен подставлять результат одного action под другим именем ради фиктивно одинакового context.

<!-- /unit:2e485822f6677073c3 -->

<!-- unit:d2cb97f168fe8c9d11 -->
<a id="u-d2cb97f168fe8c9d11"></a>

Для грубой интуиции SE доли при независимых Bernoulli trials:

<!-- /unit:d2cb97f168fe8c9d11 -->

<!-- unit:0f1c800c553ac4c942 -->
<a id="u-0f1c800c553ac4c942"></a>

$$
\operatorname{SE}(\widehat S)\approx\sqrt{\frac{\widehat S(1-\widehat S)}{n}}.
$$

<!-- /unit:0f1c800c553ac4c942 -->

<!-- unit:186ff42bde24806354 -->
<a id="u-186ff42bde24806354"></a>

При success 0.8 и n=100 это 0.04. Один дополнительный процентный пункт на таком наборе сам по себе неубедителен. Для малой выборки/крайних долей используют подходящие интервалы, а не слепо интерпретируют normal approximation.

<!-- /unit:186ff42bde24806354 -->

<!-- unit:483836ab5988c34c5a -->
<a id="u-483836ab5988c34c5a"></a>

### Ноль ошибок не доказывает нулевой риск

<!-- /unit:483836ab5988c34c5a -->

<!-- unit:c610d198de76c6da11 -->
<a id="u-c610d198de76c6da11"></a>

Plug-in SE при наблюдаемых 100% success даёт 0, но это не гарантия идеальной policy. При IID binary failure rate q вероятность увидеть 0 failures за n trials равна (1−q)^n. Односторонняя 95% верхняя граница при нуле failures получается из этого выражения:

<!-- /unit:c610d198de76c6da11 -->

<!-- unit:2bbf38c2c9b1607eeb -->
<a id="u-2bbf38c2c9b1607eeb"></a>

$$
(1-q_U)^n=0.05,\qquad q_U=1-0.05^{1/n}.
$$

<!-- /unit:2bbf38c2c9b1607eeb -->

<!-- unit:791fb7e3cbd7680319 -->
<a id="u-791fb7e3cbd7680319"></a>

Для n=10 это≈0.258866, для n=100≈0.029513. Даже десять идеальных runs оставляют существенную uncertainty в такой постановке. Это frequentist confidence construction при объявленных assumptions, а не Bayesian утверждение о случайности q. Методы binomial intervals описаны в [NIST handbook](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

<!-- /unit:791fb7e3cbd7680319 -->

<!-- unit:bef56402a9421cf86c -->
<a id="u-bef56402a9421cf86c"></a>

Отбор удобных cases, ошибки labels и зависимость tasks не устраняются извлечением корня. Интервал описывает sampling uncertainty выбранной модели эксперимента, а не все возможные ошибки процесса evaluation.

<!-- /unit:bef56402a9421cf86c -->

<!-- unit:3d1f40949c2d977fc8 -->
<a id="u-3d1f40949c2d977fc8"></a>

### Подбор по test и независимый вывод

<!-- /unit:3d1f40949c2d977fc8 -->

<!-- unit:dd58d9743213a938f7 -->
<a id="u-dd58d9743213a938f7"></a>

После каждого изменения смотрят development set, а final test не превращают в источник training labels и tuning решений. Выбрать лучшую из ста configs по тому же test и процитировать её score как проверку без подбора — optimistic selection. Для новых failure cases расширяют diagnostic set, сохраняя смысл final holdout.

<!-- /unit:dd58d9743213a938f7 -->

<!-- unit:a3c9dc7f8898090300 -->
<a id="u-a3c9dc7f8898090300"></a>

Checkpoint, oracle code и data revisions должны позволять восстановить проверенные свойства. Маленькая SE не исправляет неверный target. Ranking/citation/judge evaluation подробнее находится в [приложениях](<../05-applications/app-evaluation.md>).

<!-- /unit:a3c9dc7f8898090300 -->

<!-- unit:5ed44767652a4ebf82 -->
<a id="u-5ed44767652a4ebf82"></a>

## Итоговый учебный эксперимент

<!-- /unit:5ed44767652a4ebf82 -->

<!-- unit:7ae85e93cb16528e8d -->
<a id="u-7ae85e93cb16528e8d"></a>

Сначала фиксируют environment specification и oracle, затем собирают baseline logs. Выбирают одну диагностированную проблему, например отсутствие recovery после timeout. Вмешательство — recovery SFT или иной один механизм, а не одновременная смена model, memory и tools. После validation проводят held-out comparison по success, цене, failures и constraints.

<!-- /unit:7ae85e93cb16528e8d -->

<!-- unit:d8e6933a213902fc50 -->
<a id="u-d8e6933a213902fc50"></a>

Отчёт содержит task specification, initial snapshot, action/tool protocol, rubric revision, split groups, policy/template/memory/partner configs, budget, run logs, raw outcomes и costs. Из них восстанавливаются знаменатели, paired changes и uncertainty. Формулировка «в toy read-only среде улучшен recovery на удержанных шаблонах» точнее, чем «агент научился решать любые задачи».

<!-- /unit:d8e6933a213902fc50 -->

<!-- unit:b8fbc9dcde25e76b92 -->
<a id="u-b8fbc9dcde25e76b92"></a>

В этом курсе настоящие PPO/GRPO/DPO/MARL training runs не проводились. Итог lifecycle блока — подробный учебный материал, объявленные toy calculations и проект проверяемых экспериментов. Готовность конспекта не означает наличие обученной пользовательской модели или доказанной production quality.

<!-- /unit:b8fbc9dcde25e76b92 -->

<!-- unit:013d7fe28ce70d88ba -->
<a id="u-013d7fe28ce70d88ba"></a>

### Benchmarks, оценивающие всю траекторию инструмента

<!-- /unit:013d7fe28ce70d88ba -->

<!-- unit:eb61818e128d59ce9f -->
<a id="u-eb61818e128d59ce9f"></a>

Final answer exact match может пропустить неверный выбор инструмента, аргументы, порядок зависимых вызовов и recovery. TRAJECT-Bench предлагает trajectory-aware evaluation с fine-grained metrics для selection/parameterization/ordering. ACEBench разделяет Normal, Special (ambiguous/incomplete requests) и Agent (multi-agent multi-turn) scenarios. Это примеры более подробной рубрики, а не универсальный стандарт для каждой среды.

<!-- /unit:eb61818e128d59ce9f -->

<!-- unit:474248b467fcc8bba3 -->
<a id="u-474248b467fcc8bba3"></a>

Перенесите только те поля, которые существуют в вашем execution trace: choice correctness, argument validity, dependency/order, observed result, recovery, terminal outcome и cost. Сверяйте rubric с oracle: benchmark score не является независимой проверкой неизвестного вам production task. Если API не исполняется, явно укажите, что оценивает симулятор и чего не может подтвердить.

<!-- /unit:474248b467fcc8bba3 -->

<!-- unit:1ec37a198834366b50 -->
<a id="u-1ec37a198834366b50"></a>

**Источники:** [TRAJECT-Bench, ICLR 2026](https://openreview.net/pdf?id=TZWnWvsQ0X) и [ACEBench, Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.697/). Их benchmark outcomes ограничены task schemas, simulator/API, prompts, rubric и test split авторов.

<!-- /unit:1ec37a198834366b50 -->

<!-- unit:545f9d75b6768cce24 -->
<a id="u-545f9d75b6768cce24"></a>

## Контроль и практика

<!-- /unit:545f9d75b6768cce24 -->

<!-- unit:af9f7b7bbcde8dc88b -->
<a id="u-af9f7b7bbcde8dc88b"></a>

1. Нужно ли учитывать стоимость неудачных attempts? **Ответ:** да, иначе цена успеха занижена.
2. Почему split по messages одной session плох? **Ответ:** общая задача и evidence создают зависимость и leakage.
3. Доказывают ли 100% valid calls выполнение task? **Ответ:** нет, нужны semantics и независимый outcome oracle.
4. Равны ли tool retry и новый run? **Ответ:** нет, это разные units, определённые protocol.
5. Почему 96.30% valid calls совместимы с 50% task success? **Ответ:** формат не гарантирует правильные source, meaning и termination.
6. Каков cost per success toy журнала? **Ответ:** 16/5=3.2, включая failures.
7. Почему десять идеальных runs не доказывают q=0? **Ответ:** uncertainty остаётся; IID one-sided 95% upper bound≈0.258866.
8. Что означает late success после deadline? **Ответ:** actual effect и соблюдение task deadline оценивают отдельно.
9. Можно ли отбросить engine failures? **Ответ:** только в явно conditional metric; полный report их учитывает.

<!-- /unit:af9f7b7bbcde8dc88b -->

<!-- unit:a95547a49a00eea2ca -->
<a id="u-a95547a49a00eea2ca"></a>

**Практика:** воспроизвести расчёты десяти sessions, затем изменить один outcome или cost и пересчитать success, validity, recovery и cost per success. Спроектировать comparison двух policies с одинаковым budget; указать, какие выводы десять задач ещё не поддерживают. Реальные metrics заявляются только после запуска соответствующего experiment.

<!-- /unit:a95547a49a00eea2ca -->

## Результат главы и переход дальше

Ответьте на вопрос: **Что считать успехом целой попытки и как учитывать retries и стоимость?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Behavior cloning и SFT на демонстрациях](<agent-demonstrations.md>), [Distribution shift, recovery и DAgger](<agent-recovery.md>), [Обучение через взаимодействие: PPO и GRPO](<agent-rl.md>), [Углубление: reasoning и верификация](<agent-reasoning.md>), [Углубление: память и длинный горизонт](<agent-memory.md>), [Углубление: предпочтения траекторий](<agent-preferences.md>), [Углубление: роли и обучение координации](<agent-coordination.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
