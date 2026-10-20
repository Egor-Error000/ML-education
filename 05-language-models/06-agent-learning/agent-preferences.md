# Углубление: предпочтения траекторий

**Уровень:** углубление.

**Предпосылки:** [Оценка траекторий и воспроизводимость](<agent-evaluation.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Как сравнивать полные попытки с разным бюджетом и исходом?





## Карта вычисления и границы этапов

![Углубление: предпочтения траекторий: входы, преобразования и выходы](<../../оформление/course/agent-preferences.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Reward и targets используются в объявленном режиме обучения. При оценке политика фиксирована; success измеряется по всей попытке, включая повторы и бюджет.

<!-- unit:dfc93897f36c7a34de -->
<a id="u-dfc93897f36c7a34de"></a>

## Единица предпочтения

<!-- /unit:dfc93897f36c7a34de -->

<!-- unit:d344f41422e5bd3576 -->
<a id="u-d344f41422e5bd3576"></a>

В обычном preference dataset пара ответов относится к одному prompt. Для агента важны все actions, observations и достигнутое состояние. Два одинаковых финальных сообщения могут сопровождаться разной реальностью: один агент действительно прочитал актуальный документ, другой выдумал ссылку. Предпочтение по одному final text не различит их.

<!-- /unit:d344f41422e5bd3576 -->

<!-- unit:7bc2fe4a333c2bcc0f -->
<a id="u-7bc2fe4a333c2bcc0f"></a>

Запись сравнения должна содержать исходную задачу, доступные условия среды, обе траектории и критерии. Условия могут быть стохастическими; тогда разовый выигрыш не обязательно отражает лучшую policy. Для устойчивого вывода нужны повторные запуски либо контролируемый replay с корректно определёнными переходами.

<!-- /unit:7bc2fe4a333c2bcc0f -->

<!-- unit:1c51b0be1ed0a77983 -->
<a id="u-1c51b0be1ed0a77983"></a>

Сначала проверяют обязательные требования: выполнена ли задача, допустимы ли действия, не подменён ли результат. Затем среди допустимых решений сравнивают цену, длину, задержку и понятность ответа. Успешная короткая сессия обычно предпочтительнее успешной бессмысленно длинной, но краткость не должна перевешивать фактический неуспех.

<!-- /unit:1c51b0be1ed0a77983 -->

<!-- unit:9b7c494133267a1207 -->
<a id="u-9b7c494133267a1207"></a>

**Схема курса:** same task + paired trajectories → rubric/constraints → preference or Pareto relation → training signal → evaluator checks success, violations and cost

<!-- /unit:9b7c494133267a1207 -->

<!-- unit:53b9530667f77ff5a1 -->
<a id="u-53b9530667f77ff5a1"></a>

<!-- source-depth-lifecycle-З -->

<!-- /unit:53b9530667f77ff5a1 -->

<!-- unit:750c4c0d77fdd14e63 -->
<a id="u-750c4c0d77fdd14e63"></a>

### Одинаковый final text, разные доказательства

<!-- /unit:750c4c0d77fdd14e63 -->

<!-- unit:2f58c39f85ad7c26f2 -->
<a id="u-2f58c39f85ad7c26f2"></a>

Продолжим вымышленную read-only задачу о report retention. Путь A:search → read(R2/v2) → answer90days с применимой citation. Путь B прочитал только obsolete R1/v1, а затем написал те же 90days/R2/v2 без evidence. По финальной строке ответы выглядят одинаково, но только A подтверждён наблюдаемым source. Предпочтение должно учитывать выполненную задачу с её evidence requirement.

<!-- /unit:2f58c39f85ad7c26f2 -->

<!-- unit:7e830a39834c9a8e21 -->
<a id="u-7e830a39834c9a8e21"></a>

Путь C:search → read(R1/v1) → дополнительный search → read(R2/v2) → answer. Он тоже может быть successful в six-action toy budget, но потратил 5actions вместо 3. Если прочие требования и стоимость steps одинаковы, A предпочтительнее C по расходу. C при этом может быть полезной recovery demonstration, хотя он не оптимальный образец чистого happy path. Обучение «что допустимо» и предпочтение «что эффективнее» —разные labels.

<!-- /unit:7e830a39834c9a8e21 -->

<!-- unit:a61e969839ff38d590 -->
<a id="u-a61e969839ff38d590"></a>

Данные для сравнения содержат original task, initial state/snapshot/scope, обе action/result traces, terminal oracle и cost accounting. Нельзя извлечь две последние assistant фразы из разных tasks и считать их preference pair одного prompt. Даже похожий вопрос может иметь другие report category, актуальную версию и доступные tools.

<!-- /unit:a61e969839ff38d590 -->

<!-- unit:ff896d87d234af345f -->
<a id="u-ff896d87d234af345f"></a>

### Initial conditions и случайность среды

<!-- /unit:ff896d87d234af345f -->

<!-- unit:cbce66309a5e0d7b09 -->
<a id="u-cbce66309a5e0d7b09"></a>

Разные actions закономерно дают разные observations:read(R1) и read(R2) не должны возвращать один и тот же passage. Сопоставимость пары не означает побитно одинаковую trace. Но независимо различающиеся source snapshot или сеть могут дать одной trajectory преимущество, не связанное с качеством policy.

<!-- /unit:cbce66309a5e0d7b09 -->

<!-- unit:3ac746dab94bf86bd5 -->
<a id="u-3ac746dab94bf86bd5"></a>

Один удачный outcome позволяет поставить label конкретному исполнению, но не доказывает, что породившая его policy лучше в ожидании. Для policy-level вывода нужны repeats/controlled task distribution, а иногда сравнение choices при одинаковом доступном history. Переприсвоить teacher result student action ради «равных условий» нельзя: нарушится causality transition.

<!-- /unit:3ac746dab94bf86bd5 -->

<!-- unit:536cb27e5fb2edd0a1 -->
<a id="u-536cb27e5fb2edd0a1"></a>

### Допустимость, успешность и предпочтение

<!-- /unit:536cb27e5fb2edd0a1 -->

<!-- unit:bca6b21ec410cfefac -->
<a id="u-bca6b21ec410cfefac"></a>

Сначала задают обязательные invariants:scope, допустимый action protocol, отсутствие лишних effects, нужная revision и корректный outcome. Затем среди допустимых решений сравнивают ресурсы, задержку и ясность объяснения. Если constraint действительно hard, небольшой utility penalty не должен позволять «купить» его нарушение большим красивым ответом.

<!-- /unit:bca6b21ec410cfefac -->

<!-- unit:26e77d6688c933773e -->
<a id="u-26e77d6688c933773e"></a>

Не каждый denied proposal —уже раскрытие запрещённых данных:executor мог заблокировать его до исполнения, и policy восстановилась. Rubric должна различать invalid proposal, actual effect и final task success, как в предыдущих главах. Severity и обязательность критериев задаются task contract, а не общей догадкой judge.

<!-- /unit:26e77d6688c933773e -->

<!-- unit:d1f4d475c8608cd65e -->
<a id="u-d1f4d475c8608cd65e"></a>

## Многокритериальная rubric

<!-- /unit:d1f4d475c8608cd65e -->

<!-- unit:51bd36ff7baa88d934 -->
<a id="u-51bd36ff7baa88d934"></a>

Можно задать скалярную utility:

<!-- /unit:51bd36ff7baa88d934 -->

<!-- unit:7adfda2c709fe027b1 -->
<a id="u-7adfda2c709fe027b1"></a>

$$
U(\tau)=w_s S(\tau)-w_c C(\tau)-w_e E(\tau),
$$

<!-- /unit:7adfda2c709fe027b1 -->

<!-- unit:6cf005632378fb4190 -->
<a id="u-6cf005632378fb4190"></a>

где $S$ — подтверждённый успех, $C$ — нормированная стоимость, $E$ — число или тяжесть ошибок. Веса задают реальные компромиссы, а не объективный закон качества. Если нарушения недопустимы, лучше constraint $E=0$ и сравнение внутри допустимого множества, чем небольшой штраф за них.

<!-- /unit:6cf005632378fb4190 -->

<!-- unit:1d89680c7b64e057f1 -->
<a id="u-1d89680c7b64e057f1"></a>

Пример: три траектории имеют $(S,C)=(1,3),(1,9),(0,1)$. При $U=S-0.05C$ получаем $0.85,0.55,-0.05$: первая выигрывает. При чрезмерном штрафе за стоимость неуспешная может оказаться выше успешной. Поэтому нужно проверять не только арифметику, но и соответствие ranking цели. Нормировка стоимости тоже существенна: dollars и число calls — разные величины.

<!-- /unit:1d89680c7b64e057f1 -->

<!-- unit:1903cad61227db2edd -->
<a id="u-1903cad61227db2edd"></a>

Pareto-подход сохраняет вектор критериев: одна сессия доминирует другую, если не хуже по всем заданным критериям и строго лучше хотя бы по одному. Если одна быстрее, но дороже, единственного победителя без предпочтений заказчика может не быть. Допустимы ties и неопределённые сравнения; не нужно принудительно назначать winner любой паре.

<!-- /unit:1903cad61227db2edd -->

<!-- unit:0acebef98b480ae7ac -->
<a id="u-0acebef98b480ae7ac"></a>

### Units и веса utility

<!-- /unit:0acebef98b480ae7ac -->

<!-- unit:d920ad3bf4d940734a -->
<a id="u-d920ad3bf4d940734a"></a>

U=S−0.05C имеет смысл только после объявления units C. Если C измерялась в условных dollars, перевод 3dollars→300cents требует coefficient0.0005 вместо 0.05, чтобы penalty остался 0.15. Сохранить 0.05 после смены unit означает изменить preferences, а не просто presentation. Это математический пример scale, не тариф или финансовая рекомендация.

<!-- /unit:d920ad3bf4d940734a -->

<!-- unit:50063f9e7eeb135078 -->
<a id="u-50063f9e7eeb135078"></a>

Параметр β preference loss и weights task utility также не одно и то же. Utility определяет labels/критерий желательности, β задаёт scale log-ratio margin в конкретной loss. Настройка β не исправляет противоречивую rubric или отсутствующий environment oracle.

<!-- /unit:50063f9e7eeb135078 -->

<!-- unit:79cf2a56f4b16e178b -->
<a id="u-79cf2a56f4b16e178b"></a>

### Pareto, ties и недостаток данных

<!-- /unit:79cf2a56f4b16e178b -->

<!-- unit:bea06e39895ef6a5cb -->
<a id="u-bea06e39895ef6a5cb"></a>

Для одинаково успешных допустимых sessions A имеетcost3,latency8;B —cost4,latency4;C —cost5,latency9. A доминирует C по обоим ресурсным критериям, но A и B несравнимы без предпочтения cost/latency. Если задачу ограничили latency≤5, A уже не feasible и B может быть выбран. Порог —часть требований, не универсальное правило.

<!-- /unit:bea06e39895ef6a5cb -->

<!-- unit:a57c3377cfcedf5c3d -->
<a id="u-a57c3377cfcedf5c3d"></a>

Tie означает, что по выбранной rubric различие несущественно/равно. Incomparable означает trade-off без заданного выбора. Insufficient evidence означает, что не хватает фактов даже для применения rubric. Смешивать все три в одну уверенную win/loss label нежелательно. Конкретный training method должен определить, какие labels поддерживаются и как обрабатываются остальные.

<!-- /unit:a57c3377cfcedf5c3d -->

<!-- unit:972dc857286ab21871 -->
<a id="u-972dc857286ab21871"></a>

Если журнал не подтверждает, произошла ли операция, judge не должен считать её выполненной по фразе «успешно». Возможно, нужна дополнительная проверка environment state или исключение пары по заранее заданному правилу. Unknown label не становится победой более красноречивого кандидата.

<!-- /unit:972dc857286ab21871 -->

<!-- unit:58cf99ebc99890dfd7 -->
<a id="u-58cf99ebc99890dfd7"></a>

### Несогласие оценщиков и циклы

<!-- /unit:58cf99ebc99890dfd7 -->

<!-- unit:bcee36097ff16074e8 -->
<a id="u-bcee36097ff16074e8"></a>

Три оценщика ранжируют A>B>C,B>C>A,C>A>B. Тогда majority предпочитает A над B, B над C и C над A:каждая пара 2:1. Цикл может возникнуть даже при transitive ranking каждого человека. Он показывает, что агрегированные preferences не обязаны задавать один безупречный total order.

<!-- /unit:bcee36097ff16074e8 -->

<!-- unit:df26d7de249b61fb16 -->
<a id="u-df26d7de249b61fb16"></a>

Проверяют причины disagreement:разные interpretations constraints, resources или source applicability. Если нужно единое поведение, rubric уточняют, сохраняют annotator/context metadata и применяют adjudication по объявленному протоколу. Простое увеличение числа копий одного judge может воспроизвести тот же blind spot, не разрешив конфликт ценностей.

<!-- /unit:df26d7de249b61fb16 -->

<!-- unit:002c9eca80dd0c0196 -->
<a id="u-002c9eca80dd0c0196"></a>

## Что означает likelihood траектории

<!-- /unit:002c9eca80dd0c0196 -->

<!-- unit:0d58365051c0714416 -->
<a id="u-0d58365051c0714416"></a>

Если обучаем policy по предпочтениям, log-likelihood её решений при наблюдавшейся истории:

<!-- /unit:0d58365051c0714416 -->

<!-- unit:36367a39ed75b80c62 -->
<a id="u-36367a39ed75b80c62"></a>

$$
\ell_\theta(\tau)=
\sum_t\sum_{j=1}^{m_t}
\log\pi_\theta(a_{t,j}\mid h_t,a_{t,<j}).
$$

<!-- /unit:36367a39ed75b80c62 -->

<!-- unit:b90e9595c3c9185d2b -->
<a id="u-b90e9595c3c9185d2b"></a>

Тексты observations являются условиями, а не сгенерированными policy tokens. Две траектории могут иметь разные observations вследствие разных действий. Простое склеивание всей сессии в обычный completion и loss на всех tokens ошибочно учит воспроизводить среду.

<!-- /unit:b90e9595c3c9185d2b -->

<!-- unit:7a747dfba1352773e2 -->
<a id="u-7a747dfba1352773e2"></a>

Для иллюстрации preference loss можно записать DPO-подобную форму:

<!-- /unit:7a747dfba1352773e2 -->

<!-- unit:0e0b5d29cf8b0bfed7 -->
<a id="u-0e0b5d29cf8b0bfed7"></a>

$$
\mathcal L=-\log\sigma\!\left(
\beta\left[
\ell_\theta(\tau^+)-\ell_{\mathrm{ref}}(\tau^+)
-\ell_\theta(\tau^-)+\ell_{\mathrm{ref}}(\tau^-)
\right]\right).
$$

<!-- /unit:0e0b5d29cf8b0bfed7 -->

<!-- unit:8c592876a1874fa971 -->
<a id="u-8c592876a1874fa971"></a>

Это показывает, какие policy log-probabilities участвуют в сравнении. Само наличие формулы не доказывает, что off-policy trajectory dataset удовлетворяет всем предпосылкам стандартной DPO-постановки. Нужно определить common initial task, environment protocol, support, masking и нормировку длины. Полная сумма и среднее по tokens задают разные предпочтения. Для последовательного взаимодействия перенос метода требует отдельного обоснования и проверки.

<!-- /unit:8c592876a1874fa971 -->

<!-- unit:2f5c7fd6c302767567 -->
<a id="u-2f5c7fd6c302767567"></a>

### Action likelihood при разных observations

<!-- /unit:2f5c7fd6c302767567 -->

<!-- unit:65bda54c617732b309 -->
<a id="u-65bda54c617732b309"></a>

Для fixed environment probability path содержит policy terms и transition/observation terms. При отношении Pθ(τ)/Pref(τ) **одного и того же path**, одинакового environment kernel и совместимого support средовые множители сокращаются. Поэтому в policy log-ratio участвуют actions, обусловленные соответствующими histories, а не likelihood выданного database текста как generated answer.

<!-- /unit:65bda54c617732b309 -->

<!-- unit:8739b3abdf2c29ab83 -->
<a id="u-8739b3abdf2c29ab83"></a>

Но это не означает, что можно произвольно заменить observations между positive и negative trajectories. У каждого action свой фактический prefix. И не означает, что любой desired distribution полных paths реализуем causal policy при stochastic transitions:agent не управляет случайным ответом сети. Обоснование preference objective для sequential setup требует определить constraints/среду, а не только скопировать loss из single-completion задачи.

<!-- /unit:8739b3abdf2c29ab83 -->

<!-- unit:40b8fccdeffbe68251 -->
<a id="u-40b8fccdeffbe68251"></a>

Teacher-forced log probability позволяет оценить записанные actions и на offline histories. Сам факт off-policy данных не делает их бесполезными. Вопросы —coverage/support, достоверность transitions, information contract и то, соответствует ли обучаемый ranking целевому rollout поведению. Среда с неподдержанным новым API не становится совместимой от одной token mask.

<!-- /unit:40b8fccdeffbe68251 -->

<!-- unit:ad59c8c8f36b686220 -->
<a id="u-ad59c8c8f36b686220"></a>

### DPO-подобный margin на конкретных числах

<!-- /unit:ad59c8c8f36b686220 -->

<!-- unit:aae8ea5355ba33dad1 -->
<a id="u-aae8ea5355ba33dad1"></a>

Пусть для preferred pathℓθ+=−8,ℓref+=−10,для rejectedℓθ−=−7,ℓref−=−6. Relative improvements2 и−1. Приβ0.2 margin0.2·(2−(−1))=0.6,σ(margin)≈0.645656,loss−lnσ≈0.437488. Preferred path имеет меньшую absolute current likelihood, чем rejected, но больший relative likelihood к reference.

<!-- /unit:aae8ea5355ba33dad1 -->

<!-- unit:feefc801749a63ccd4 -->
<a id="u-feefc801749a63ccd4"></a>

Если current=reference на обоих paths,margin0 иlossln2≈0.693147 независимо от их разных absolute lengths. Это baseline loss, а не доказательство равного реального качества trajectories. Gradient задаёт стимул изменить относительное предпочтение по labels. Значение sigmoid не следует объявлять calibrated probability task success.

<!-- /unit:feefc801749a63ccd4 -->

<!-- unit:2b3d4c4bf0990f2300 -->
<a id="u-2b3d4c4bf0990f2300"></a>

В [DPO paper](https://arxiv.org/abs/2305.18290) preference optimization выводится для определённой KL-regularized постановки и preference model. Здесь formula показана как иллюстрация policy-token comparison; она не является доказанным universal recipe для любой interaction среды. Более подробный стандартный DPO objective и alternatives находятся в [адаптации](<../03-adaptation/adapt-preference.md>).

<!-- /unit:2b3d4c4bf0990f2300 -->

<!-- unit:0ac1be7e069c06ea49 -->
<a id="u-0ac1be7e069c06ea49"></a>

### Длина, masking и scope pair

<!-- /unit:0ac1be7e069c06ea49 -->

<!-- unit:edcb8ea82dc43f131f -->
<a id="u-edcb8ea82dc43f131f"></a>

Сумма action-token log probabilities и их среднее задают разные scores. Нормировка по длине может менять ranking и связь с исходным objective. Длинные observations увеличивают input cost, но не должны увеличивать число policy output targets, если они external. Stop/end-of-turn markers и unsupported actions обрабатываются по реально выбранному protocol.

<!-- /unit:edcb8ea82dc43f131f -->

<!-- unit:ce982e2f864e8a3986 -->
<a id="u-ce982e2f864e8a3986"></a>

Pairs должны сохранять общий task group и environment/template revisions. Несколько comparisons одной session не становятся независимыми tasks для split. Если judge уже использован для выбора training pairs, финальную оценку полезно вести по независимому oracle/human subset и held-out tasks, иначе возможна специализация на данном proxy.

<!-- /unit:ce982e2f864e8a3986 -->

<!-- unit:28244db71593b00955 -->
<a id="u-28244db71593b00955"></a>

Source обещания «DPO автоматически решает alignment на одной GPU» не переносятся в этот chapter как resource guarantee. Размер models, reference computation, adapters, lengths и batch дают разные расходы. Сокращение отдельных training roles не заменяет cost/quality measurement всей системы.

<!-- /unit:28244db71593b00955 -->

<!-- unit:5ba8d17c039c32286c -->
<a id="u-5ba8d17c039c32286c"></a>

## Ошибки judge и разметки

<!-- /unit:5ba8d17c039c32286c -->

<!-- unit:7efcb64e4195fb21d3 -->
<a id="u-7efcb64e4195fb21d3"></a>

Judge может предпочитать уверенный стиль, длинные ответы, определённый порядок кандидатов или знакомый формат. Покажите обе перестановки пары, скройте имена моделей и оцените согласие людей. Для task success используйте oracle среды, а не впечатление judge. При конфликте «красиво объяснил» и «не выполнил» rubric должна давать явный ответ.

<!-- /unit:7efcb64e4195fb21d3 -->

<!-- unit:ccd5f828305b968532 -->
<a id="u-ccd5f828305b968532"></a>

Не все различия доступны из transcript. Например, финальный response сообщает обновление файла, но журнал не содержит проверяемого состояния. Тогда правильная разметка может быть «недостаточно данных», а не guessed winner. Сохранение проверяемых artifacts — часть качества dataset.

<!-- /unit:ccd5f828305b968532 -->

<!-- unit:4584e093ea916a6648 -->
<a id="u-4584e093ea916a6648"></a>

После preference tuning повторно проверяются неизменённые tasks, recovery, длина и стоимость. Улучшение agreement с judge недостаточно, если фактический success падает. Не следует обучать и окончательно оценивать по одному и тому же набору judge-предпочтений: возможна специализация на слабостях оценщика.

<!-- /unit:4584e093ea916a6648 -->

<!-- unit:b32232e2dff3e677f2 -->
<a id="u-b32232e2dff3e677f2"></a>

### Разметка пары как проверяемая процедура

<!-- /unit:b32232e2dff3e677f2 -->

<!-- unit:b70e16a100793a54c0 -->
<a id="u-b70e16a100793a54c0"></a>

Сначала одинаково скрывают names policies и показывают task/constraints, evidence и actual outcome обеих sessions. Отдельно отмечают feasibility, correctness, recovery и resource costs. Лишь затем присваивают preference по rubric. Перестановка A/B позволяет обнаружить часть position bias, но не устраняет общие ошибки интерпретации source.

<!-- /unit:b70e16a100793a54c0 -->

<!-- unit:4c7e1117e3dc73fd77 -->
<a id="u-4c7e1117e3dc73fd77"></a>

Для pairs с почти одинаковым final text оценивают trace. Для pairs с разным style, но одинаковым state outcome, не переоценивают красоту как доказательство actions. При unknown external status не угадывают winner. Разметка должна иметь reason codes и evidence, чтобы следующий review мог понять основание выбора.

<!-- /unit:4c7e1117e3dc73fd77 -->

<!-- unit:e96441a245de3f4a4e -->
<a id="u-e96441a245de3f4a4e"></a>

### Проверка после preference training

<!-- /unit:e96441a245de3f4a4e -->

<!-- unit:08cecfab8c6cb75217 -->
<a id="u-08cecfab8c6cb75217"></a>

Повторяют held-out rollouts при одном budget и environment protocol, включая clean, recovery, no-evidence и stochastic cases. Agreement с judge сравнивают с фактическим task success, invalid calls, длиной и стоимостью. Если win-rate растёт, а correct state outcome падает, это failure proxy, а не достаточное доказательство улучшения.

<!-- /unit:08cecfab8c6cb75217 -->

<!-- unit:cd6efe60d74ea0a49e -->
<a id="u-cd6efe60d74ea0a49e"></a>

В этом конспекте дан учебный протокол и арифметика; настоящий trajectory DPO training, judge campaign и пользовательский deployment не проводились.

<!-- /unit:cd6efe60d74ea0a49e -->

<!-- unit:47531f191de7328fee -->
<a id="u-47531f191de7328fee"></a>

## Контроль и практика

<!-- /unit:47531f191de7328fee -->

<!-- unit:8d7f7798acbae2230e -->
<a id="u-8d7f7798acbae2230e"></a>

1. Почему одинаковые final answers не означают одинаковую trajectory quality? **Ответ:** различаются выполненные actions, evidence, стоимость и достигнутое состояние.
2. Почему observation tokens обычно исключаются из policy likelihood? **Ответ:** их генерирует среда, а политика управляет действиями.
3. Нужно ли принудительно выбирать winner при разных компромиссах? **Ответ:** нет; допустимы ties, частичный порядок и запрос недостающего критерия.

<!-- /unit:8d7f7798acbae2230e -->

<!-- unit:4f85b6499296022b21 -->
<a id="u-4f85b6499296022b21"></a>

4. Почему единицы cost влияют на utility? **Ответ:** coefficient должен согласованно измениться при rescaling, иначе меняются preferences.
5. Может ли majority preferences образовать цикл? **Ответ:** да, даже при transitive ranking каждого оценщика.
6. Почему unknown не равно tie? **Ответ:** неизвестен outcome/evidence, а не установлено равенство качества.
7. Что сокращается в fixed-path policy ratio? **Ответ:** одинаковые environment factors при нужных support/contract assumptions; policy targets остаются actions.
8. Нужно ли preferred path всегда иметь большую absolute likelihood? **Ответ:** DPO-подобная loss использует relative log-ratios к reference.
9. Доказывает ли снижение preference loss успех environment задачи? **Ответ:** нет, нужны held-out rollout oracle и проверки ресурсов/ограничений.

<!-- /unit:4f85b6499296022b21 -->

<!-- unit:43d7a778bbb379fbd5 -->
<a id="u-43d7a778bbb379fbd5"></a>

**Практика:** составьте rubric из успеха, корректности действий, цены и ясности. Разметьте шесть пар, включая короткий неправильный ответ, длинный правильный и две одинаково успешные сессии. Для спорных пар запишите, какие факты нужны оценщику. Проверьте, меняется ли решение при перестановке кандидатов.

<!-- /unit:43d7a778bbb379fbd5 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как сравнивать полные попытки с разным бюджетом и исходом?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
