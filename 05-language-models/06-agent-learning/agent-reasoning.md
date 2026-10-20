# Углубление: reasoning и верификация

**Уровень:** углубление.

**Предпосылки:** [Оценка траекторий и воспроизводимость](<agent-evaluation.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Когда дополнительный поиск и проверка улучшают результат?





## Карта вычисления и границы этапов

![Углубление: reasoning и верификация: входы, преобразования и выходы](<../../оформление/course/agent-reasoning.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Reward и targets используются в объявленном режиме обучения. При оценке политика фиксирована; success измеряется по всей попытке, включая повторы и бюджет.

<!-- unit:71de451f7ab3954937 -->
<a id="u-71de451f7ab3954937"></a>

## Три разных механизма улучшения

<!-- /unit:71de451f7ab3954937 -->

<!-- unit:96b4d64be4f67e53e0 -->
<a id="u-96b4d64be4f67e53e0"></a>

CoT-SFT обучает модель на доступных примерах последовательного решения: меняются параметры и вероятности генерируемых последовательностей. При rejection sampling генерируют несколько кандидатов, проверяют и оставляют подходящие для дальнейшего обучения. При best-of-N веса могут вообще не меняться: дополнительные кандидаты и verifier используются только во время решения новой задачи. Эти механизмы можно сочетать, но их затраты и доказательства улучшения различаются.

<!-- /unit:96b4d64be4f67e53e0 -->

<!-- unit:e1203f31cbd6a881f9 -->
<a id="u-e1203f31cbd6a881f9"></a>

Правильный финальный ответ не доказывает правильность всех промежуточных шагов. И наоборот, ошибочный ответ иногда возникает после почти целиком верной процедуры из-за одной арифметической ошибки. Поэтому важно заранее определить, что именно проверяется: итог, отдельные переходы, вызовы инструментов или воспроизводимое доказательство.

<!-- /unit:e1203f31cbd6a881f9 -->

<!-- unit:cff46ef36ba2e45f11 -->
<a id="u-cff46ef36ba2e45f11"></a>

В [STaR](https://arxiv.org/abs/2203.14465) модель итеративно генерирует рассуждения и обучается на отобранных примерах; используется также генерация объяснения с подсказанным правильным ответом для первоначально неуспешных задач. Это полезный способ собирать данные, но знание ответа при генерации нельзя тайно переносить в inference benchmark. На тесте ответ неизвестен.

<!-- /unit:cff46ef36ba2e45f11 -->

<!-- unit:a2eeb97f774c91cf40 -->
<a id="u-a2eeb97f774c91cf40"></a>

**Схема курса:** input → N candidate trajectories under fixed budget → verifier/rubric → select or abstain → measure success, cost and verifier error

<!-- /unit:a2eeb97f774c91cf40 -->

<!-- unit:8fb34bc7a6ef184677 -->
<a id="u-8fb34bc7a6ef184677"></a>

<!-- source-depth-lifecycle-Д -->

<!-- /unit:8fb34bc7a6ef184677 -->

<!-- unit:f5d74c58eebefe651b -->
<a id="u-f5d74c58eebefe651b"></a>

### Prompt, дополнительные попытки и gradient update

<!-- /unit:f5d74c58eebefe651b -->

<!-- unit:971c39f2108f9a66b9 -->
<a id="u-971c39f2108f9a66b9"></a>

Инструкция «объясни решение по шагам» меняет input. Генерация нескольких решений меняет inference compute. SFT на доступных verified demonstrations меняет параметры. Все три способа могут сочетаться, но для отчёта нужно знать, что именно изменилось и где оплачивается работа: при подготовке/training или на каждом новом task.

<!-- /unit:971c39f2108f9a66b9 -->

<!-- unit:57cb2b08a952df09f8 -->
<a id="u-57cb2b08a952df09f8"></a>

В tool task проверяемое решение может выглядеть как план поиска, actual calls, observations и вывод с source. В математике —как явные преобразования и проверка результата. Это внешне доступные артефакты, которые можно оценить по correctness; они не обязаны раскрывать скрытую внутреннюю цепочку рассуждения модели. Наличие длинного prose reasoning само по себе не означает большей истинности.

<!-- /unit:57cb2b08a952df09f8 -->

<!-- unit:55422e8c09bbaec9f5 -->
<a id="u-55422e8c09bbaec9f5"></a>

Final answer может быть верен случайно, по памяти или из-за cancellation двух ошибок. Доказательство требует корректных промежуточных переходов и применимых premises. Например, верное число без source не выполняет задачу «ответь по current документу с цитатой». И наоборот, почти правильная процедура с одной арифметической ошибкой может дать неверный итог. Поэтому заранее выбирают, какое свойство verifier реально оценивает.

<!-- /unit:55422e8c09bbaec9f5 -->

<!-- unit:2e37008c8fc0c68c6e -->
<a id="u-2e37008c8fc0c68c6e"></a>

### Rejection sampling для данных и для текущей задачи

<!-- /unit:2e37008c8fc0c68c6e -->

<!-- unit:56a08c9fd756253cf1 -->
<a id="u-56a08c9fd756253cf1"></a>

Для data collection generator создаёт candidates, filter/oracle проверяет их, а retained examples могут использоваться в следующем training update. Это меняет dataset distribution. Для inference selection те же candidates могут проверяться и выбираться только ради данного ответа, без optimizer step. Похожая внешняя последовательность generate→check не делает эти режимы одним алгоритмом.

<!-- /unit:56a08c9fd756253cf1 -->

<!-- unit:e8667acf3e7a99a4d4 -->
<a id="u-e8667acf3e7a99a4d4"></a>

Filtering только по final answer сохраняет и cases с ошибочными промежуточными объяснениями. Filtering только по syntax сохраняет бессмысленный valid JSON. Поэтому retained data quality зависит от полноты oracle и labels. Дешёвый validator может давать полезный первый фильтр, но его ограниченность должна быть видна в итоговом протоколе.

<!-- /unit:e8667acf3e7a99a4d4 -->

<!-- unit:08e1b6700b0c3bba62 -->
<a id="u-08e1b6700b0c3bba62"></a>

В STaR правильный answer используется при рационализации первоначально неудачных training cases; затем модель обучается на отобранных объяснениях. Это supervision, а не доступный на test магический ресурс. В [STaR paper](https://arxiv.org/abs/2203.14465) этот итеративный цикл заявлен явно. При evaluation нельзя давать generator hidden answer и затем сравнивать с baseline без такой подсказки.

<!-- /unit:08e1b6700b0c3bba62 -->

<!-- unit:517825723f69b78aa9 -->
<a id="u-517825723f69b78aa9"></a>

### Feedback должен относиться к реальной ошибке

<!-- /unit:517825723f69b78aa9 -->

<!-- unit:48be1af1858bf7d7ae -->
<a id="u-48be1af1858bf7d7ae"></a>

«Неверно» сообщает outcome, но не локализует причину. «Прочитана v1, задача требует v2» указывает конкретное нарушение. «Tool timeout, результат неизвестен» не даёт права заключить отсутствие документа. Чем точнее feedback соответствует проверенному факту, тем яснее следующий permitted action. Но выдуманный critique может убедить модель отказаться от первоначально верного ответа.

<!-- /unit:48be1af1858bf7d7ae -->

<!-- unit:f8c5ce483f6e377ef7 -->
<a id="u-f8c5ce483f6e377ef7"></a>

Feedback с подсказкой всех нужных действий меняет information budget. Если critic получает gold source, сравнение показывает пользу этой информации вместе с critic policy. Если critic использует те же неполные observations, нельзя автоматически считать его независимым oracle. Разделяют correctness сигнал, дополнительные evidence и stylistic preference.

<!-- /unit:f8c5ce483f6e377ef7 -->

<!-- unit:32858d9155dd96e857 -->
<a id="u-32858d9155dd96e857"></a>

## Outcome и process supervision

<!-- /unit:32858d9155dd96e857 -->

<!-- unit:cdf5e16008128fd817 -->
<a id="u-cdf5e16008128fd817"></a>

Outcome reward model, ORM, оценивает полный результат. Process reward model, PRM, получает разметку промежуточных шагов или предсказывает их качество. Для бинарных labels простейшая обучающая потеря оценщика:

<!-- /unit:cdf5e16008128fd817 -->

<!-- unit:1c0f64417fa44a6c53 -->
<a id="u-1c0f64417fa44a6c53"></a>

$$
\mathcal L_{\mathrm{verifier}}=
-\frac1M\sum_{i=1}^M
\bigl[y_i\log v_\phi(z_i)+(1-y_i)\log(1-v_\phi(z_i))\bigr].
$$

<!-- /unit:1c0f64417fa44a6c53 -->

<!-- unit:5a79e156d5e1977f71 -->
<a id="u-5a79e156d5e1977f71"></a>

Здесь $z_i$ — полный кандидат для ORM либо доступный префикс решения/шаг для выбранного PRM-протокола. Что является label, задаётся разметкой, а не названием модели. Один и тот же словесный шаг может быть допустим в нескольких формах; согласованность разметчиков и granularity влияют на objective. Пример исследования process supervision — [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050).

<!-- /unit:5a79e156d5e1977f71 -->

<!-- unit:4d152ebbdc6d683d2f -->
<a id="u-4d152ebbdc6d683d2f"></a>

Нельзя без обоснования считать произведение scores шагов точной вероятностью корректности решения: ошибки зависят друг от друга, а scores могут быть некалиброваны. Минимум, сумма логарифмов или обученный агрегатор — разные ranking rules. Их выбирают на validation и проверяют отдельно.

<!-- /unit:4d152ebbdc6d683d2f -->

<!-- unit:e5150d068a0fc71904 -->
<a id="u-e5150d068a0fc71904"></a>

Исполняемый тест часто надёжнее оценки красивого текста, но тест тоже неполон. Программа может пройти известные cases и провалить новые. Агент может получить формально правильное конечное поле, нарушив ограничения действий. Verifier должен охватывать целевое свойство, а его ошибки анализируются независимо от generator.

<!-- /unit:e5150d068a0fc71904 -->

<!-- unit:7a99a0d7d45eed1452 -->
<a id="u-7a99a0d7d45eed1452"></a>

### Что именно предсказывает verifier

<!-- /unit:7a99a0d7d45eed1452 -->

<!-- unit:a4c3f20fe382e618bc -->
<a id="u-a4c3f20fe382e618bc"></a>

ORM label может означать final-answer correctness, task success или preference. PRM label может означать valid transition, полезность шага или вероятность успешного завершения из prefix. Это разные targets. Один step может быть логически допустимым, но не продвигать задачу; другой —полезным запросом информации без готового factual вывода. Названия ORM/PRM не заменяют rubric.

<!-- /unit:a4c3f20fe382e618bc -->

<!-- unit:6e97aa2e19e797d957 -->
<a id="u-6e97aa2e19e797d957"></a>

BCE из основного текста подходит для объявленных binary labels. Если есть uncertain/partial grades, задают соответствующую loss/weighting policy и не превращают ambiguous labels автоматически в 0. Для PRM важно consistent step segmentation: модель не должна получать «верный prefix» с уже вписанным будущим решением там, где реальное приложение оценивает ранний шаг.

<!-- /unit:6e97aa2e19e797d957 -->

<!-- unit:976db20e598e0cf737 -->
<a id="u-976db20e598e0cf737"></a>

Результаты [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) показывают конкретное исследование process supervision в математической постановке. Они не являются доказательством, что любой PRM лучше любого outcome checker в любом tool environment. Task, labels и distribution candidates определяют переносимость.

<!-- /unit:976db20e598e0cf737 -->

<!-- unit:ce585e68035ccc5da2 -->
<a id="u-ce585e68035ccc5da2"></a>

### Произведение step scores: когда это probability, а когда heuristic

<!-- /unit:ce585e68035ccc5da2 -->

<!-- unit:56caba89f4a5e1c125 -->
<a id="u-56caba89f4a5e1c125"></a>

Если q_t —калиброванная conditional probability корректности шага t **при условии корректности всех предыдущих**, chain rule допускает произведение для совместного события. Для этого не требуется отдельное предположение независимости. Но обычный score «этот step выглядит правильным» может быть marginal, некалиброванным или означать helpfulness. Тогда product не является доказанной joint probability.

<!-- /unit:56caba89f4a5e1c125 -->

<!-- unit:cc9de1c074945045b5 -->
<a id="u-cc9de1c074945045b5"></a>

Два решения имеют по 0.95 на каждом step, одно состоит из 2 steps, другое из 10. Products0.9025 и≈0.598737, minima оба 0.95, geometric means оба 0.95. Выбранный aggregator даёт разные rankings/length incentives. Если все step errors полностью связаны одним общим ложным premise, вероятность корректности всех может оставаться 0.95, а не 0.95^10, несмотря на marginal0.95 каждого.

<!-- /unit:cc9de1c074945045b5 -->

<!-- unit:4b8103185d18991e3f -->
<a id="u-4b8103185d18991e3f"></a>

Сумма log scores —монотонное преобразование product при положительных scores; она не устраняет length effect. Нормировка на число steps меняет смысл на geometric mean. Ни один вариант не надо называть истинной probability без calibration и проверки event definition. Для выбора rules используют development data, final holdout оставляют независимым.

<!-- /unit:4b8103185d18991e3f -->

<!-- unit:f7c230d8a84a7cae0e -->
<a id="u-f7c230d8a84a7cae0e"></a>

### Executable oracle и ошибки границ задачи

<!-- /unit:f7c230d8a84a7cae0e -->

<!-- unit:ea851f545daaa69e6f -->
<a id="u-ea851f545daaa69e6f"></a>

Для code candidate запуск tests проверяет именно эти inputs/properties. Он может не видеть скрытые corner cases, performance regressions и нежелательные extra effects. Для arithmetic проверка числа может не видеть неверную единицу. Для agent action финальное поле может быть correct при нарушении разрешённого пути. Полнота oracle —часть постановки, а не автоматический побочный эффект исполнения кода.

<!-- /unit:ea851f545daaa69e6f -->

<!-- unit:a168d4e7465e1617ed -->
<a id="u-a168d4e7465e1617ed"></a>

Если generator видит все tests, он может подобрать частный shortcut вместо общего решения. Если verifier использует source недоступной версии, selection оценён в ином information regime. Если oracle требует exact string, он может отклонять эквивалентную корректную формулировку. Нужно сочетать deterministic checks там, где они точны, с содержательными checks там, где смысл не сводится к строке.

<!-- /unit:a168d4e7465e1617ed -->

<!-- unit:055404b045eda18495 -->
<a id="u-055404b045eda18495"></a>

## Что даёт best-of-N

<!-- /unit:055404b045eda18495 -->

<!-- unit:e91d3fedac45e3ed40 -->
<a id="u-e91d3fedac45e3ed40"></a>

Если каждый кандидат независимо успешен с вероятностью $p$, а совершенный verifier выбирает успешный при его наличии, то:

<!-- /unit:e91d3fedac45e3ed40 -->

<!-- unit:880ff8ce85ca2a549e -->
<a id="u-880ff8ce85ca2a549e"></a>

$$
P(\text{успех хотя бы одного})=1-(1-p)^N.
$$

<!-- /unit:880ff8ce85ca2a549e -->

<!-- unit:d31d473a483ed642cf -->
<a id="u-d31d473a483ed642cf"></a>

При $p=0.4$ и $N=3$ получается $0.784$. Реальные кандидаты коррелируют, а verifier ошибается; поэтому это иллюстрация, а не прогноз. Даже если хороший кандидат существует, ранжировщик может выбрать неправильный. Нужно различать oracle pass@N и успех реально выбранного ответа.

<!-- /unit:d31d473a483ed642cf -->

<!-- unit:a6617a0392024dd602 -->
<a id="u-a6617a0392024dd602"></a>

Если генерация одного кандидата стоит $c_g$, а его проверка $c_v$, простой бюджет равен:

<!-- /unit:a6617a0392024dd602 -->

<!-- unit:2c142ff329786cdea3 -->
<a id="u-2c142ff329786cdea3"></a>

$$
C(N)=N(c_g+c_v)+c_{\mathrm{selection}}.
$$

<!-- /unit:2c142ff329786cdea3 -->

<!-- unit:345e9bbcc27dddc40d -->
<a id="u-345e9bbcc27dddc40d"></a>

При раннем завершении после достоверного успеха средняя стоимость меняется; хвостовые latency и максимальный бюджет сохраняют значение. Три длинные попытки одной системы нельзя сравнивать с одной короткой попыткой другой и весь прирост приписывать архитектуре.

<!-- /unit:345e9bbcc27dddc40d -->

<!-- unit:1e3b86a86a73a93bec -->
<a id="u-1e3b86a86a73a93bec"></a>

### Oracle coverage и реальный selection

<!-- /unit:1e3b86a86a73a93bec -->

<!-- unit:2a4c77dc238fb5abdc -->
<a id="u-2a4c77dc238fb5abdc"></a>

При p=0.4,N3 основная formula даёт 0.784 только при независимых candidates и идеальном обнаружении хотя бы одного correct. Если все три candidates всегда повторяют один и тот же sample, oracle coverage остаётся 0.4. Ещё один похожий длинный ответ не обязан давать independent шанс исправления.

<!-- /unit:2a4c77dc238fb5abdc -->

<!-- unit:8d03ea0f727aa5eac3 -->
<a id="u-8d03ea0f727aa5eac3"></a>

Даже high coverage не гарантирует real selected success. Verifier может отвергнуть correct answer, предпочесть false-positive или вернуть error. Поэтому сохраняют все candidates, oracle labels, scores, selected ID и final outcome. «Правильный кандидат был где-то в списке» —диагностика generator coverage, а не результат пользовательского ответа.

<!-- /unit:8d03ea0f727aa5eac3 -->

<!-- unit:c7f9fba445c8b18eea -->
<a id="u-c7f9fba445c8b18eea"></a>

### Полностью посчитанный noisy-verifier пример

<!-- /unit:c7f9fba445c8b18eea -->

<!-- unit:bdfdfb8714de5f9b0e -->
<a id="u-bdfdfb8714de5f9b0e"></a>

Предположим IID candidates:correct с вероятностью 0.4. Verifier принимает correct с вероятностью 0.9 и incorrect с вероятностью 0.1. Тогда probability correct-and-accepted0.36, incorrect-and-accepted0.06, rejected0.58. Среди accepted precision0.36/0.42=6/7≈85.71%. Это условный toy model, не measured judge accuracy.

<!-- /unit:bdfdfb8714de5f9b0e -->

<!-- unit:5862caed7766f67273 -->
<a id="u-5862caed7766f67273"></a>

Policy «остановиться на первом accepted, максимум 3 попытки» даёт probability correct selection $0.36(1+0.58+0.58^2)=0.689904$. Incorrect accepted —$0.06(1+0.58+0.58^2)=0.114984$, все три rejected —$0.58^3=0.195112$. Сумма 1. Реальный selected success0.689904 ниже oracle pass@3=0.784, потому что filter ошибается и может рано принять неверный candidate.

<!-- /unit:5862caed7766f67273 -->

<!-- unit:38c77aeee4c7051321 -->
<a id="u-38c77aeee4c7051321"></a>

Здесь все rejected —отдельный terminal outcome без selected answer. Если fallback возвращает последний rejected candidate, метрики изменятся и его надо моделировать отдельно. Нельзя прятать rejected episodes из denominator и объявлять conditional precision85.71% общей task success rate.

<!-- /unit:38c77aeee4c7051321 -->

<!-- unit:b766ab5e23964a3ff7 -->
<a id="u-b766ab5e23964a3ff7"></a>

### Цена раннего завершения и fixed-N batch

<!-- /unit:b766ab5e23964a3ff7 -->

<!-- unit:ae7ff5423b40d39df1 -->
<a id="u-ae7ff5423b40d39df1"></a>

Для того же stop-first-accepted правила expected attempts $1+0.58+0.58^2=1.9164$. При generation cost1unit и verification0.2unit expected cost2.29968, максимум 3.6. Если сначала сгенерировать/проверить все 3, cost3.6 независимо от первого accepted. Это разные policies, даже если обе используют «до трёх кандидатов».

<!-- /unit:ae7ff5423b40d39df1 -->

<!-- unit:f449c35c11c99939ee -->
<a id="u-f449c35c11c99939ee"></a>

Parallel generation может уменьшить wall latency до selection, но суммарная работа не исчезает, а ранний stop может уже не сэкономить запущенные запросы. Shared prefixes/KV caching меняют computation cost по backend contract; они не делают outputs независимыми и не гарантируют лучшего semantic выбора. Число candidates, tokens и время рассматриваются отдельно.

<!-- /unit:f449c35c11c99939ee -->

<!-- unit:18f9b246ee45223588 -->
<a id="u-18f9b246ee45223588"></a>

### С ростом N растёт и риск найти verifier blind spot

<!-- /unit:18f9b246ee45223588 -->

<!-- unit:ac0095f94072a13849 -->
<a id="u-ac0095f94072a13849"></a>

Если все candidates неверны, а independent false-accept rate verifier0.05, шанс хотя бы одного false accept за 20 checks $1-0.95^{20}\approx0.641514$. Это conditional all-wrong scenario, показывающий selection pressure на ошибки judge. В реальной системе generator/score correlations могут изменить число, но сам механизм остаётся: больше поиска может находить способы обойти слабый proxy.

<!-- /unit:ac0095f94072a13849 -->

<!-- unit:5ee267e4ffbb37d27d -->
<a id="u-5ee267e4ffbb37d27d"></a>

Нужно оценивать chosen-answer correctness по независимому oracle, а не только maximum verifier score. Если verifier используется ещё и для training selection/reward, его blind spots могут закрепиться в следующих generations. Раздельный held-out evaluation помогает заметить, что proxy улучшился без роста task success.

<!-- /unit:5ee267e4ffbb37d27d -->

<!-- unit:de9a99e53704f727df -->
<a id="u-de9a99e53704f727df"></a>

### Best-of-N, beam и дерево действий

<!-- /unit:de9a99e53704f727df -->

<!-- unit:7b5a7c9026443c9f0c -->
<a id="u-7b5a7c9026443c9f0c"></a>

Best-of-N может генерировать N полных candidates. Beam поддерживает ограниченное множество prefixes и отсекает остальные по score. Tree search над reasoning/actions создаёт branches с промежуточной оценкой; эти branches часто делят общий prefix и observations. Для них нельзя автоматически применять IID formula1−(1−p)^N.

<!-- /unit:7b5a7c9026443c9f0c -->

<!-- unit:3f69b57854ec3e5786 -->
<a id="u-3f69b57854ec3e5786"></a>

Полное дерево branching3,depth4 имеет 81 leaves и 121 nodes, считая root. Число полных решений и число оценённых partial nodes различаются. Pruning уменьшает work, но может удалить правильный путь. Для tool branches нужно обеспечить consistent environment snapshot: imagined result одного action не должен становиться actual observation другой ветки.

<!-- /unit:3f69b57854ec3e5786 -->

<!-- unit:a96b4dd7c6bd410ad1 -->
<a id="u-a96b4dd7c6bd410ad1"></a>

Inference budget расходуется на expansion, scoring, tools и synthesis. Сравнение деревьев одинаковой depth без учёта branching/model calls не задаёт equal compute. Подробная token decoding/beam механика уже разобрана в inference; здесь важно различать search policy и обучение weights.

<!-- /unit:a96b4dd7c6bd410ad1 -->

<!-- unit:80361180fa5c2cc785 -->
<a id="u-80361180fa5c2cc785"></a>

## Самокритика и память об ошибке

<!-- /unit:80361180fa5c2cc785 -->

<!-- unit:dcafbcfac3a9d2052b -->
<a id="u-dcafbcfac3a9d2052b"></a>

Модель может перечитать ответ, найти противоречие и исправить его через контекст без обновления параметров. В [Reflexion](https://arxiv.org/abs/2303.11366) текстовая обратная связь сохраняется и используется в дальнейших попытках. Название «verbal reinforcement learning» не означает, что каждый такой цикл выполняет градиентный RL update весов LLM.

<!-- /unit:dcafbcfac3a9d2052b -->

<!-- unit:4049698073c7e76cc9 -->
<a id="u-4049698073c7e76cc9"></a>

Самокритика полезна при доступном evidence и верифицируемых ошибках, но тот же generator может уверенно подтвердить собственный неправильный ответ. Просьба «проверь ещё раз» не создаёт независимого источника истины. Следует учитывать долю исправленных ошибок и долю испорченных правильных ответов. Если из 20 ошибок исправлены 8, а из 80 правильных испорчены 10, итоговая accuracy падает с $0.80$ до $0.78$.

<!-- /unit:4049698073c7e76cc9 -->

<!-- unit:efd51d2dc2ad25996d -->
<a id="u-efd51d2dc2ad25996d"></a>

Развёрнутые объяснения нужны как проверяемый учебный материал или доступная демонстрация, когда это соответствует задаче. Для обучения tool use достаточно наблюдаемых решений и результатов; требовать скрытую внутреннюю цепочку рассуждения модели не нужно.

<!-- /unit:efd51d2dc2ad25996d -->

<!-- unit:279884ff5dd1b97ff3 -->
<a id="u-279884ff5dd1b97ff3"></a>

### Что хранится в textual reflection

<!-- /unit:279884ff5dd1b97ff3 -->

<!-- unit:7c414319ce75f285ef -->
<a id="u-7c414319ce75f285ef"></a>

В [Reflexion](https://arxiv.org/abs/2303.11366) feedback и текстовая reflection сохраняются в episodic memory и влияют на следующую попытку без обязательного изменения LLM weights. Это изменение context/state. Если позднее эти records используются для SFT, появляется отдельный training этап, который нельзя считать уже выполненным только из-за слова learning в названии.

<!-- /unit:7c414319ce75f285ef -->

<!-- unit:d0214733ff35ca80f4 -->
<a id="u-d0214733ff35ca80f4"></a>

Полезная reflection описывает проверенную ошибку и правило следующего действия: «read timeout не подтвердил отсутствие source; проверить budget и повторить разрешённый read». Плохая —выдумывает причину и фиксирует ложное общее правило. Memory должна сохранять provenance и условия применимости; один сбой сети не доказывает, что вся source collection навсегда пуста.

<!-- /unit:d0214733ff35ca80f4 -->

<!-- unit:6dc285d4d7fe21ffcf -->
<a id="u-6dc285d4d7fe21ffcf"></a>

Слишком длинные reflections вытесняют исходные constraints и evidence. Прежняя попытка может относиться к другой revision/task. Поэтому memory policy и correctness её содержимого —часть общей оценки, а не бесплатный источник истины. Глава Ж подробнее разберёт long-horizon memory.

<!-- /unit:6dc285d4d7fe21ffcf -->

<!-- unit:d840111f43fb087333 -->
<a id="u-d840111f43fb087333"></a>

### Исправления и регрессии считаются вместе

<!-- /unit:d840111f43fb087333 -->

<!-- unit:d96ced8f6ef0938d9a -->
<a id="u-d96ced8f6ef0938d9a"></a>

В исходном примере 80 correct и 20 incorrect. Self-critique исправляет 8 errors, но ломает 10 correct: становится 78 correct. Error-repair rate8/20=0.4, correct-answer regression10/80=0.125; net change−2points. Сказать только «исправляет 40% ошибок» означало бы скрыть вред на другой части набора.

<!-- /unit:d96ced8f6ef0938d9a -->

<!-- unit:f91748f2dca05539f5 -->
<a id="u-f91748f2dca05539f5"></a>

Если дополнительный verifier надёжно обнаруживает класс ошибки, selective critique может быть полезнее blanket «перепроверь всё». Но selector тоже ошибается; его false positives и пропущенные cases входят в protocol. Генератор и critic с разными role prompts могут разделять knowledge gaps, поэтому disagreement не означает, что critic обязательно прав.

<!-- /unit:f91748f2dca05539f5 -->

<!-- unit:20229aeeb893a0020e -->
<a id="u-20229aeeb893a0020e"></a>

Наблюдаемая correction должна быть связана с evidence/test, а не просто новым увереннее написанным ответом. Итоговая accuracy оценивается после всего pipeline, вместе со временем и ценой дополнительных calls. Красота самокритики —отдельный stylistic критерий, если он вообще нужен задаче.

<!-- /unit:20229aeeb893a0020e -->

<!-- unit:33d8eaf711ea34f0e0 -->
<a id="u-33d8eaf711ea34f0e0"></a>

## Эксперимент и ответы

<!-- /unit:33d8eaf711ea34f0e0 -->

<!-- unit:0cb96d9d7d9fa219b8 -->
<a id="u-0cb96d9d7d9fa219b8"></a>

Фиксируем задачи, generator, verifier и общий предел tokens/actions. Сравниваем один ответ, retry с явной обратной связью и best-of-N. Записываем success, oracle pass@N, ошибки выбора, общую стоимость и задержку. Verifier не должен получать скрытый эталон, недоступный реальному приложению; если эталон используется для исследовательского oracle, так и обозначаем.

<!-- /unit:0cb96d9d7d9fa219b8 -->

<!-- unit:e202a572cbe227fd8a -->
<a id="u-e202a572cbe227fd8a"></a>

### Проверяемая ablation sequence

<!-- /unit:e202a572cbe227fd8a -->

<!-- unit:74bb25b2fc33eae167 -->
<a id="u-74bb25b2fc33eae167"></a>

На одном fixed task split сначала оценивают one-shot generator. Затем добавляют inference attempts при том же generator и объявленном verifier. Отдельно оценивают generator+oracle coverage, actual selector, retry feedback и self-critique. Если после этого обучают новую policy на filtered data, её сравнивают и при прежнем inference budget, чтобы увидеть training effect отдельно от compute.

<!-- /unit:74bb25b2fc33eae167 -->

<!-- unit:6e4f0b7198ccce03c8 -->
<a id="u-6e4f0b7198ccce03c8"></a>

Фиксируют версии generator/verifier, rubric, candidate sampling, max tokens/actions, selection/fallback и source/environment snapshots. Ошибки группируют: candidate отсутствовал, candidate был, но проиграл selection, oracle неполон, feedback испортил correct answer, exceeded budget. Эти категории помогают выбрать следующее изменение точнее, чем «нужно больше reasoning».

<!-- /unit:6e4f0b7198ccce03c8 -->

<!-- unit:a3316bed43b5799677 -->
<a id="u-a3316bed43b5799677"></a>

### Вопросы для самопроверки

<!-- /unit:a3316bed43b5799677 -->

<!-- unit:c80ff25faf41f0982d -->
<a id="u-c80ff25faf41f0982d"></a>

1. Повышает ли best-of-N параметры модели? **Ответ:** сам по себе нет; веса меняются только при отдельном training step.
2. Почему pass@N выше реального качества выбора? **Ответ:** наличие верного кандидата не гарантирует, что verifier его распознает.
3. Всегда ли self-critique улучшает ответ? **Ответ:** нет; возможны и полезные исправления, и регрессии первоначально верных ответов.

<!-- /unit:c80ff25faf41f0982d -->

<!-- unit:c547e552b17c31008e -->
<a id="u-c547e552b17c31008e"></a>

4. Когда можно перемножать step probabilities без независимости? **Ответ:** если это согласованные calibrated conditionals для joint correctness по chain rule; arbitrary scores таких гарантий не дают.
5. Почему noisy verifier может остановиться на неверном ответе? **Ответ:** false accept в ранней попытке завершает selection до рассмотрения позднего correct candidate.
6. Одинакова ли цена batch3 и stop-first-accepted3? **Ответ:** нет, при раннем завершении expected work может быть меньше, если ещё не запущенные attempts действительно отменены.
7. Может ли увеличение N ухудшать chosen-answer качество? **Ответ:** возможно при несовершенном selection/proxy, даже если oracle coverage не уменьшается.
8. Почему STaR answer-conditioned rationalization нельзя переносить в test input? **Ответ:** это training supervision, а на реальной test задаче hidden answer неизвестен.
9. Изменяет ли reflection buffer weights? **Ответ:** сам по себе нет; context memory и optimizer update —разные вмешательства.
10. Что скрывает one-number repair rate? **Ответ:** регрессии первоначально правильных решений и дополнительную стоимость.

<!-- /unit:c547e552b17c31008e -->

<!-- unit:1af0a41301f4ba7fde -->
<a id="u-1af0a41301f4ba7fde"></a>

**Практика:** придумайте пять задач с независимым oracle и по три кандидата. Укажите истинный лучший кандидат, выбор вашего verifier и стоимость. Отдельно разберите false positive: почему неверное решение было признано успешным и какой новый тест выявит этот дефект.

<!-- /unit:1af0a41301f4ba7fde -->

## Результат главы и переход дальше

Ответьте на вопрос: **Когда дополнительный поиск и проверка улучшают результат?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
