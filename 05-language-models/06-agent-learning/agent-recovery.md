# Distribution shift, recovery и DAgger

**Уровень:** основа ветки.

**Предпосылки:** [Оценка траекторий и воспроизводимость](<agent-evaluation.md>), [Behavior cloning и SFT на демонстрациях](<agent-demonstrations.md>)

**Используют:** [Обучение через взаимодействие: PPO и GRPO](<agent-rl.md>)

**Учебный вопрос:** Как научиться действовать в состояниях, порождённых собственными ошибками?





## Карта вычисления и границы этапов

![Distribution shift, recovery и DAgger: входы, преобразования и выходы](<../../оформление/course/agent-recovery.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Reward и targets используются в объявленном режиме обучения. При оценке политика фиксирована; success измеряется по всей попытке, включая повторы и бюджет.

<!-- unit:a9b3c2f7c124b67837 -->
<a id="u-a9b3c2f7c124b67837"></a>

## У данных есть распределение состояний

<!-- /unit:a9b3c2f7c124b67837 -->

<!-- unit:ed6912ace1bdd3ce3a -->
<a id="u-ed6912ace1bdd3ce3a"></a>

В behavior cloning примеры обычно собраны экспертом. Эксперт получает корректный результат поиска, читает нужный документ и отвечает. Модель после ошибочного запроса получает другой набор документов. Теперь ей нужно выбирать действие в истории, которой не было при обучении. Даже небольшая ошибка меняет последующие observations и тем самым входы самой модели.

<!-- /unit:ed6912ace1bdd3ce3a -->

<!-- unit:fc8950deaea1472326 -->
<a id="u-fc8950deaea1472326"></a>

Обозначим распределение историй на шаге $t$ при политике $\pi$ через $d_\pi^t$. Supervised objective на экспертных демонстрациях минимизирует ошибку под $d_{\pi_E}^t$, а развёрнутая система работает под $d_{\pi_\theta}^t$. Равенство этих распределений ничем не гарантировано:

<!-- /unit:fc8950deaea1472326 -->

<!-- unit:f4e97a82e6717b330e -->
<a id="u-f4e97a82e6717b330e"></a>

$$
\mathbb E_{h\sim d_{\pi_E}}[\ell(\pi_\theta(h),\pi_E(h))]
\quad\text{и}\quad
\mathbb E_{h\sim d_{\pi_\theta}}[\ell(\pi_\theta(h),\pi_E(h))]
$$

<!-- /unit:f4e97a82e6717b330e -->

<!-- unit:4e038dc029ebe0519d -->
<a id="u-4e038dc029ebe0519d"></a>

могут существенно различаться. Это объясняет, почему высокая accuracy на отдельных экспертных шагах совместима с плохим успехом полных сессий.

<!-- /unit:4e038dc029ebe0519d -->

<!-- unit:c311033fffbbe99056 -->
<a id="u-c311033fffbbe99056"></a>

Для интуиции предположим независимую вероятность ошибки $\varepsilon$ на каждом из $H$ шагов и отсутствие восстановления. Тогда вероятность хотя бы одной ошибки:

<!-- /unit:c311033fffbbe99056 -->

<!-- unit:c1348811aa70ee084f -->
<a id="u-c1348811aa70ee084f"></a>

$$
P(\text{ошибка в сессии})=1-(1-\varepsilon)^H.
$$

<!-- /unit:c1348811aa70ee084f -->

<!-- unit:a98e8841159bd5f2a5 -->
<a id="u-a98e8841159bd5f2a5"></a>

При $\varepsilon=0.02$ и $H=20$ получаем примерно $0.3324$. Это иллюстрация накопления ошибок при сильных допущениях, а не универсальная оценка реального агента. Ошибки могут зависеть друг от друга, быть безвредными или исправляться.

<!-- /unit:a98e8841159bd5f2a5 -->

<!-- unit:b4e430f6fbe4589c66 -->
<a id="u-b4e430f6fbe4589c66"></a>

**Схема курса:** current policy rollout → visited states and errors → expert correction/recovery labels → aggregate data → retrain → held-out recovery comparison

<!-- /unit:b4e430f6fbe4589c66 -->

<!-- unit:b974b88febbffca734 -->
<a id="u-b974b88febbffca734"></a>

<!-- source-depth-lifecycle-Г -->

<!-- /unit:b974b88febbffca734 -->

<!-- unit:36ed62086e8274cb1f -->
<a id="u-36ed62086e8274cb1f"></a>

### Почему iid minibatch не делает rollout iid задачей

<!-- /unit:36ed62086e8274cb1f -->

<!-- unit:5d424a70bcad59b200 -->
<a id="u-5d424a70bcad59b200"></a>

В SFT optimizer может независимо выбирать batches из фиксированного dataset. Но при исполнении action меняет следующий input. Неверный search query выдаёт другой набор IDs, read выбирает другой passage, поздний answer опирается уже на иной context. Успех следующего decision нельзя оценивать так, будто previous history всегда создана экспертом.

<!-- /unit:5d424a70bcad59b200 -->

<!-- unit:379c2ceac72fbefc90 -->
<a id="u-379c2ceac72fbefc90"></a>

Распределение $d_\pi^t$ зависит от initial task distribution, environment transitions, policy и времени t. Среднее по шагам можно определить как $d_\pi=H^{-1}\sum_{t=1}^{H}d_\pi^t$ для фиксированного horizon. Если episodes разной длины, способ усреднения отдельно задаёт вес длинных сессий. «Ошибка 1% на dataset» без указания того, откуда взяты histories, слишком мало говорит о rollout.

<!-- /unit:379c2ceac72fbefc90 -->

<!-- unit:df916a232edcf9213e -->
<a id="u-df916a232edcf9213e"></a>

В supervised loss target выбирается на уже данном history. Во время deployment эти histories созданы прежними решениями learner. Поэтому даже без изменения corpus и tools возникает policy-induced distribution shift. Это отличается от внешнего drift, когда изменились документы/пользователи, хотя оба эффекта могут происходить одновременно.

<!-- /unit:df916a232edcf9213e -->

<!-- unit:640317451a6bf63a35 -->
<a id="u-640317451a6bf63a35"></a>

### Как накапливаются ошибки: допущения и границы вывода

<!-- /unit:640317451a6bf63a35 -->

<!-- unit:62ea5536047f0342aa -->
<a id="u-62ea5536047f0342aa"></a>

Основная формула выше использует одинаковую независимую вероятность ошибки 0.02 на каждом из 20 шагов и отсутствие recovery. Вероятность чистого пути $0.98^{20}\approx0.667608$, хотя каждый отдельный шаг имеет 98% accuracy. Union bound для события «есть ошибка» при соответствующих marginal bounds даёт не более 20·0.02=0.4; это более грубая граница, чем IID пример≈0.332392.

<!-- /unit:62ea5536047f0342aa -->

<!-- unit:be8b34908ffdc36942 -->
<a id="u-be8b34908ffdc36942"></a>

Число ошибочных decisions и число последствий ошибки —разные величины. Если первый неверный action переводит в плохую область, остальные шаги могут стать бесполезными. Сумма возможных downstream effects в грубой coupling-интуиции масштабируется как $\varepsilon\sum_{t=1}^{H}(H-t+1)=\varepsilon H(H+1)/2$. При ε0.02,H20 это 4.2. Это иллюстрация квадратичного накопления влияния при определённых bounds, а не обещание 4.2 фактических ошибок у любого агента.

<!-- /unit:be8b34908ffdc36942 -->

<!-- unit:4f749859d5b579b529 -->
<a id="u-4f749859d5b579b529"></a>

Иногда ошибка безвредна, иногда recovery возвращает на правильный путь, иногда один пропуск делает задачу необратимой. Поэтому toy bound без восстановления не заменяет измерение real recovery. Важны task success, error severity, remaining horizon и то, как следующий action зависит от error observation.

<!-- /unit:4f749859d5b579b529 -->

<!-- unit:368c31aa924508de38 -->
<a id="u-368c31aa924508de38"></a>

### Маленький distribution-shift пример

<!-- /unit:368c31aa924508de38 -->

<!-- unit:cf2a332a444821e004 -->
<a id="u-cf2a332a444821e004"></a>

Пусть expert dataset содержит только normal histories. На них learner выбирает correct action в 95% случаев. В rollout20% decision points —timeout histories, на которых learner никогда не даёт правильной реакции; остальные 80% —normal. Тогда one-step accuracy на visited mixture0.8·0.95+0.2·0=0.76, несмотря на 95% validation accuracy по expert-only set.

<!-- /unit:cf2a332a444821e004 -->

<!-- unit:2a79ce5c41ea452f2d -->
<a id="u-2a79ce5c41ea452f2d"></a>

После targeted demonstrations предположим 95% на normal и 80% на timeout. На **той же** mixture ожидается 0.8·0.95+0.2·0.8=0.92. Это учебное сравнение conditional accuracies, не результат model training. В настоящем rollout mixture после улучшения policy тоже может измениться, так что фиксированная weighted sum служит лишь controlled diagnostic.

<!-- /unit:2a79ce5c41ea452f2d -->

<!-- unit:6382f8b9fd0fe6065b -->
<a id="u-6382f8b9fd0fe6065b"></a>

Если recovery улучшилась ценой ухудшения normal quality до 90%, average был бы 0.88. Поэтому добавление error states проверяют вместе с обычными задачами. Один recovery score без регрессий на clean histories не описывает весь эффект.

<!-- /unit:6382f8b9fd0fe6065b -->

<!-- unit:8abb8579a9cc3554c4 -->
<a id="u-8abb8579a9cc3554c4"></a>

## Как работает агрегирование данных

<!-- /unit:8abb8579a9cc3554c4 -->

<!-- unit:67a1a7452b8d183b26 -->
<a id="u-67a1a7452b8d183b26"></a>

DAgger — Dataset Aggregation. Важное изменение: спрашивать эксперта о правильном действии в состояниях, которые посещает обучаемая политика. На итерации $i$ запускаем текущую политику, сохраняем наблюдаемые истории и запрашиваем экспертную разметку. Новые примеры объединяем с накопленным набором:

<!-- /unit:67a1a7452b8d183b26 -->

<!-- unit:db15c7d53d110da0da -->
<a id="u-db15c7d53d110da0da"></a>

$$
\mathcal D_{i+1}=\mathcal D_i\cup
\{(h,\pi_E(h)):h\text{ посещена на итерации }i\}.
$$

<!-- /unit:db15c7d53d110da0da -->

<!-- unit:aa000ade3749de7ec7 -->
<a id="u-aa000ade3749de7ec7"></a>

Затем обучаем следующую политику на агрегированном наборе. В исходной постановке возможна смешанная политика, которая с вероятностью $\beta_i$ выполняет экспертное действие, а иначе действие ученика. Постепенное уменьшение $\beta_i$ позволяет смещаться к распределению ученика. Для LLM это требует явного протокола: фиксируют доступную learner history и информацию teacher, проверяют реализуемость expert labels и не добавляют student недоступный инструмент или скрытый answer в evaluation input.

<!-- /unit:aa000ade3749de7ec7 -->

<!-- unit:180062c716ee6df2cd -->
<a id="u-180062c716ee6df2cd"></a>

Экспертная разметка на ошибочной истории бывает сложнее разметки обычной задачи. Если уже прочитан устаревший документ, хорошее действие — проверить версию, а не сразу повторить финальный ответ. Иногда восстановление невозможно: нужно завершить сессию с корректным сообщением о неуспехе. Нельзя размечать невозможное действие лишь потому, что оно красиво выглядит в демонстрации.

<!-- /unit:180062c716ee6df2cd -->

<!-- unit:b06916069082af8c92 -->
<a id="u-b06916069082af8c92"></a>

### Исполненное действие и экспертный label —два отдельных объекта

<!-- /unit:b06916069082af8c92 -->

<!-- unit:2d92a99651db40888d -->
<a id="u-2d92a99651db40888d"></a>

Ключ DAgger —получить экспертное действие на history, реально посещённой learner/смешанной policy, добавить эти пары в dataset и снова обучить supervised policy. Это отличие сбора data, а не замена action loss на environment reward. [Исходная DAgger работа](https://arxiv.org/abs/1011.0686) связывает induced observation distributions с online no-regret reduction при дополнительных assumptions; название алгоритма само по себе не доказывает их выполнение для конкретной LLM.

<!-- /unit:2d92a99651db40888d -->

<!-- unit:25159dd116806d0423 -->
<a id="u-25159dd116806d0423"></a>

На visited history h learner мог выполнить read(R1), а expert label для этого же момента —read(R2). В record сохраняют executed action, observed result и отдельный expert target. Если expert действие не выполнялось, его предполагаемый result не следует записывать как уже полученное observation. Новая supervised пара говорит «на этом history следовало сделать иначе», а не переписывает прошлую среду.

<!-- /unit:25159dd116806d0423 -->

<!-- unit:bd321b385404a4abf0 -->
<a id="u-bd321b385404a4abf0"></a>

При дальнейшем rollout можно либо продолжить learner action, либо выполнить expert intervention по выбранной mixture. Тогда последующая history зависит от **действительно выполненного** action. Смешать result экспертного read(R2) с реально выполненным learner read(R1) означает создать несуществующую transition.

<!-- /unit:bd321b385404a4abf0 -->

<!-- unit:f93b28bbe32144dac8 -->
<a id="u-f93b28bbe32144dac8"></a>

### Один раунд сбора по шагам

<!-- /unit:f93b28bbe32144dac8 -->

<!-- unit:5e8471810816f5a8ce -->
<a id="u-5e8471810816f5a8ce"></a>

1. Фиксируют task split, environment snapshot, current learner и collection budget.
2. Запускают learner или объявленную mixed behavior policy.
3. Сохраняют model-visible histories, фактические actions/results и remaining budget.
4. На выбранных histories получают экспертные labels с provenance и проверяют допустимость.
5. Добавляют verified pairs в накопленные data с определённой weighting/sampling policy.
6. Обучают candidate на этих data и оценивают learner-only rollouts на независимых tasks.

<!-- /unit:5e8471810816f5a8ce -->

<!-- unit:cda235a76be41db917 -->
<a id="u-cda235a76be41db917"></a>

Это schematic algorithm, а не заявленный запуск framework. У teacher может быть человек, rule-based oracle или проверяемая сильная policy; каждое средство имеет область применимости. Сильная LLM без доступа к relevant evidence может ошибиться так же, как student. Label validity отдельно проверяют по environment contract.

<!-- /unit:cda235a76be41db917 -->

<!-- unit:d564f45167596232b3 -->
<a id="u-d564f45167596232b3"></a>

### Mixture β и почему assisted rollout не равен student evaluation

<!-- /unit:d564f45167596232b3 -->

<!-- unit:61914b9beaec367c74 -->
<a id="u-61914b9beaec367c74"></a>

При step-wise mixture β=0.1 каждый шаг с вероятностью 0.1 исполняет expert action, иначе learner action. На 20 шагах expected expert interventions2, но вероятность хотя бы одной intervention $1-0.9^{20}\approx0.878423$. Значит, «90% learner actions» не означает, что 90% episodes прошли совсем без помощи.

<!-- /unit:61914b9beaec367c74 -->

<!-- unit:9565ecb475da9e92d1 -->
<a id="u-9565ecb475da9e92d1"></a>

Episode-wise mixture —выбрать целиком teacher или student на весь episode —создаёт другое распределение histories. Нужно указать, какой механизм использован. Teacher может предотвратить ранний сбой и тем самым не дать learner попасть в recovery region. Schedule β и collection strategy выбирают вместе, а final success candidate измеряют без скрытой teacher помощи, если задача deployment её не предполагает.

<!-- /unit:9565ecb475da9e92d1 -->

<!-- unit:3ac4031b669b391373 -->
<a id="u-3ac4031b669b391373"></a>

Expert-query count также не равен числу interventions. Можно спросить label на каждом из 20 visited states, но выполнить экспертное действие только 2 раза в среднем. Цена annotation и цена environment execution —разные компоненты. Снижение β уменьшает вмешательство, но не обязательно число запросов labels.

<!-- /unit:3ac4031b669b391373 -->

<!-- unit:4f3d66fe25e4fe42b9 -->
<a id="u-4f3d66fe25e4fe42b9"></a>

### Агрегирование, sampling и устаревшие labels

<!-- /unit:4f3d66fe25e4fe42b9 -->

<!-- unit:b5a0d4e848ad9eb595 -->
<a id="u-b5a0d4e848ad9eb595"></a>

Обозначение union в формуле означает накопление информации. Implementation может хранить multiset с sample weights, deduplicated examples или replay buffer. Эти варианты дают разный objective. После 1000 initial examples,200 первого раунда и 300 второго всего 1500; при uniform sampling initial data занимают 2/3, все новые 1/3, последний раунд 1/5.

<!-- /unit:b5a0d4e848ad9eb595 -->

<!-- unit:c435077c7f239bcccc -->
<a id="u-c435077c7f239bcccc"></a>

Если oversample rare recovery cases, их training weight будет больше доли в deployment. Это может быть полезной интервенцией, но quality надо оценить на объявленном deployment mixture и отдельно по slices. Многократное повторение одной error trace не создаёт столько же independent tasks.

<!-- /unit:c435077c7f239bcccc -->

<!-- unit:e5c2c7fb9242eeb882 -->
<a id="u-e5c2c7fb9242eeb882"></a>

Env/tool updates могут сделать прежний label невалидным. Старый call с уже удалённым аргументом нельзя без проверки продолжать имитировать в новом protocol. Хранят версии data/environment/template и решают, какие records совместимы. Иначе aggregation одновременно борется с одним shift и создаёт другой.

<!-- /unit:e5c2c7fb9242eeb882 -->

<!-- unit:18b6536f71f0a534ed -->
<a id="u-18b6536f71f0a534ed"></a>

### Привилегированный expert и реализуемость policy

<!-- /unit:18b6536f71f0a534ed -->

<!-- unit:3d345cb3bcc8e6946a -->
<a id="u-3d345cb3bcc8e6946a"></a>

Expert иногда видит больше, чем student: hidden simulator state, полный corpus или gold outcome. Это может быть законным способом supervision, но label должен быть реализуемой целью для student information. Если одинаковый observable history соответствует двум скрытым states, требующим разных terminal answers, детерминированный student не сможет всегда угадать правильный label.

<!-- /unit:3d345cb3bcc8e6946a -->

<!-- unit:69d762494726d908d0 -->
<a id="u-69d762494726d908d0"></a>

При равновероятных скрытых мирах с одним и тем же h и несовместимыми answers максимум accuracy по terminal guess0.5. Если доступен read/check action, expert может обучить запросить различающее evidence. Если такого действия нет, DAgger не создаёт информацию из ничего. В dataset обозначают teacher information и не переносят hidden answer в student input при evaluation.

<!-- /unit:69d762494726d908d0 -->

<!-- unit:5d6e39b8894639d4e8 -->
<a id="u-5d6e39b8894639d4e8"></a>

Это уточняет фразу выше про «не подглядывать»: задача не запрещает teacher знать правильный ответ для labels; запрещено незаметно менять information contract learner или считать невозможный для него action-path корректной demonstration. Истинность target и его выводимость из доступной history —разные требования.

<!-- /unit:5d6e39b8894639d4e8 -->

<!-- unit:cb1f3398bc0d7db1a6 -->
<a id="u-cb1f3398bc0d7db1a6"></a>

## Разобранная recovery-траектория

<!-- /unit:cb1f3398bc0d7db1a6 -->

<!-- unit:4b436682f4dc4f4db5 -->
<a id="u-4b436682f4dc4f4db5"></a>

Задача: найти актуальный срок возврата товара. У ученика в истории уже есть документ v1 со сроком 14 дней, но metadata указывает, что существует v2. Happy-path dataset показывал сразу v2. Новая разметка должна сохранить реально полученный v1 и добавить корректирующее действие: найти v2, прочитать её, сверить дату действия и ответить со ссылкой. Подмена observation v1 на v2 уничтожит именно тот случай восстановления, ради которого собирались данные.

<!-- /unit:4b436682f4dc4f4db5 -->

<!-- unit:e3aaaecf32a7201327 -->
<a id="u-e3aaaecf32a7201327"></a>

Добавим второй сбой: read вернул timeout. Коррекция зависит от контракта среды: ограниченный retry, другой доступный источник либо остановка. Бесконечно повторять read нельзя. Нужны положительные примеры восстановления и отрицательные случаи, когда дополнительных действий уже недостаточно.

<!-- /unit:e3aaaecf32a7201327 -->

<!-- unit:71070b9cdf61be5658 -->
<a id="u-71070b9cdf61be5658"></a>

DAgger-подобная процедура не означает автоматически обучение RL. Разметка по-прежнему supervised; отличие — способ сбора входных состояний. Если эксперт не может разметить все состояния, выбирают поднабор по неопределённости или типу сбоя, но это меняет покрытие и требует учитывать смещение отбора.

<!-- /unit:71070b9cdf61be5658 -->

<!-- unit:cba1fdae3f84fe4584 -->
<a id="u-cba1fdae3f84fe4584"></a>

### Два fictional policy примера и общий read-only corpus

<!-- /unit:cba1fdae3f84fe4584 -->

<!-- unit:342ef92d339320819c -->
<a id="u-342ef92d339320819c"></a>

Исходный короткий пример этой главы использует условную policy возврата товара v1/14days. Это вымышленное правило в toy task, не утверждение о законе или реальном магазине. Для единой среды из главы [«Среда, observations, actions и horizon»](<agent-environment.md>) тот же pattern можно записать как report retention:v1/30days уже прочитана, current v2 ещё не прочитана.

<!-- /unit:342ef92d339320819c -->

<!-- unit:a104ad482085986af7 -->
<a id="u-a104ad482085986af7"></a>

Expert на stale history получает реально видимый v1 content и metadata о v2. Correct continuation —найти/прочитать доступную active version и затем ответить. Нельзя просто заменить число 30 на 90 в прежнем observation: model потеряет training example того, как распознать и исправить wrong-source choice.

<!-- /unit:a104ad482085986af7 -->

<!-- unit:924b3c16fb2616da6d -->
<a id="u-924b3c16fb2616da6d"></a>

Если в remaining budget помещаются read+answer, recovery возможна. Если остался один slot и policy должна ещё прочитать source, корректный target может быть partial/abstain по task contract. Один универсальный label «всегда ещё ищи v2» игнорирует horizon. Budget должен присутствовать в history или доступном trusted state.

<!-- /unit:924b3c16fb2616da6d -->

<!-- unit:e3e7d00539db540250 -->
<a id="u-e3e7d00539db540250"></a>

### После timeout, denied и необратимого effect

<!-- /unit:e3e7d00539db540250 -->

<!-- unit:12e942e46146343476 -->
<a id="u-12e942e46146343476"></a>

Read timeout допускает bounded retry, если contract/budget позволяют. No-results может требовать rewrite; denied —разрешённого альтернативного path или завершения. Teacher correction должна учитывать категорию ошибки и actual state, а не демонстрировать обход прав через выдуманный tool.

<!-- /unit:12e942e46146343476 -->

<!-- unit:98c24a823e019e50fa -->
<a id="u-98c24a823e019e50fa"></a>

В mutating environment missed observation может означать уже выполненный effect. Коррекция «повтори всё с новым request ID» способна создать duplicate. Expert должен использовать status/idempotency protocol, если он доступен. Если recovery не гарантирована или задача уже необратимо нарушена, корректно показать failure/partial, а не fabricated success trace.

<!-- /unit:98c24a823e019e50fa -->

<!-- unit:d31169276e246a6f37 -->
<a id="u-d31169276e246a6f37"></a>

Для изучения достаточно контролируемой read-only simulation. Нельзя переносить DAgger collection в настоящую environment с effects только на том основании, что «нужны ошибочные states». Возможности сбора определяются реально разрешённой средой; algorithmic интерес не меняет её contract.

<!-- /unit:d31169276e246a6f37 -->

<!-- unit:dad9dc8b221aac573a -->
<a id="u-dad9dc8b221aac573a"></a>

### Когда добавление coverage не помогает

<!-- /unit:dad9dc8b221aac573a -->

<!-- unit:5591bde436b61a58e9 -->
<a id="u-5591bde436b61a58e9"></a>

Если read result теряется из state при каждом переходе, новая demonstration сама по себе не исправит merge implementation. Если tokenizer/template ломает call serialization, target learner фактически не тот, который предполагал annotator. Если corpus не содержит active rule, teacher label answer90 может быть невоспроизводим для student. Сначала проверяют interface и environment invariants.

<!-- /unit:5591bde436b61a58e9 -->

<!-- unit:2ee5c65396823cdd8d -->
<a id="u-2ee5c65396823cdd8d"></a>

Если model capacity/optimization не позволяют аппроксимировать expert на достаточной history, больше labels не гарантирует успех. DAgger меняет distribution данных; это не доказательство, что конкретный optimizer достиг нужного no-regret поведения. Held-out rollout и независимый oracle остаются необходимыми.

<!-- /unit:2ee5c65396823cdd8d -->

<!-- unit:781dc227efccbf2e25 -->
<a id="u-781dc227efccbf2e25"></a>

## Как сравнивать честно

<!-- /unit:781dc227efccbf2e25 -->

<!-- unit:9aa23b8c454dbcaa7f -->
<a id="u-9aa23b8c454dbcaa7f"></a>

Сравним expert-only SFT и SFT с recovery-примерами при одинаковом числе размеченных decisions, одинаковой модели и бюджете обучения. Иначе преимущество может объясняться просто большим dataset. Отдельно нужны обычные задачи и задачи с контролируемой инъекцией сбоя. В обеих группах измеряем конечный успех, долю восстановления, лишние вызовы и преждевременную остановку.

<!-- /unit:9aa23b8c454dbcaa7f -->

<!-- unit:e03d6ddc506c5bb5f4 -->
<a id="u-e03d6ddc506c5bb5f4"></a>

Held-out задачи не должны попадать в разметку итераций. Если после каждого теста добавлять правильное решение этой же задачи в training set, измеряется запоминание теста. Разделять следует семейства задач и версии документов, когда перенос на новые версии — часть цели.

<!-- /unit:e03d6ddc506c5bb5f4 -->

<!-- unit:1edb7d6e98bf33a8ee -->
<a id="u-1edb7d6e98bf33a8ee"></a>

Улучшение recovery не должно скрывать регрессию обычного поведения. Модель может начать искать ошибку даже при достаточном корректном evidence, делая лишние вызовы. Поэтому в отчёте нужны и чистые сессии, и сбои.

<!-- /unit:1edb7d6e98bf33a8ee -->

<!-- unit:f94e4b08bfb5a09a67 -->
<a id="u-f94e4b08bfb5a09a67"></a>

### Controlled comparison и независимый final test

<!-- /unit:f94e4b08bfb5a09a67 -->

<!-- unit:92484176e13d3aa118 -->
<a id="u-92484176e13d3aa118"></a>

Сравнивают expert-only1500 labels с initial1000+500 visited-state labels, если хотят отделить coverage от количества данных. Training steps/tokens и annotation cost тоже указывают: одинаковое число records не означает одинаковые длины/action complexity. Candidate с большим overall budget может быть полезен как система, но его прирост не следует целиком приписывать именно DAgger data collection.

<!-- /unit:92484176e13d3aa118 -->

<!-- unit:6d2060b596ad6076f3 -->
<a id="u-6d2060b596ad6076f3"></a>

Recovery rate определяют на tasks, где действительно возник конкретный recoverable failure. Исключить необратимые cases допустимо для отдельной conditional metric, но full task denominator должен показывать их долю. Вести clean success, recovery success, invalid actions, loop/premature-stop rate и total cost полезнее одного supervised accuracy.

<!-- /unit:6d2060b596ad6076f3 -->

<!-- unit:83c86bb2e5dbd21a66 -->
<a id="u-83c86bb2e5dbd21a66"></a>

После каждого раунда current policy проверяют на development tasks, а final holdout не используют как источник новых expert labels. Если fixed final test постоянно входит в aggregation, рост score может быть memorization. Во всех данных сохраняют task/session group, чтобы prefixes одного rollout не попали по разные стороны split.

<!-- /unit:83c86bb2e5dbd21a66 -->

<!-- unit:88b1ed08ad99fce119 -->
<a id="u-88b1ed08ad99fce119"></a>

## Контроль и практика

<!-- /unit:88b1ed08ad99fce119 -->

<!-- unit:79b3b528ed63ea3256 -->
<a id="u-79b3b528ed63ea3256"></a>

1. Почему случайное перемешивание экспертных шагов не устраняет distribution shift? **Ответ:** оно меняет порядок примеров, но не добавляет истории, возникающие после ошибок ученика.
2. Можно ли заменить ошибочное observation правильным перед разметкой? **Ответ:** тогда исчезает задача восстановления; размечать надо действие в действительно посещённой истории.
3. Гарантирует ли DAgger безошибочность? **Ответ:** нет; нужны доступный эксперт, покрытие, обучаемая модель и предпосылки анализа. Частичная наблюдаемость и непоправимые действия сохраняются.

<!-- /unit:79b3b528ed63ea3256 -->

<!-- unit:d478e12aa2c91dd0cc -->
<a id="u-d478e12aa2c91dd0cc"></a>

4. Чем expert query отличается от intervention? **Ответ:** label можно получить, не исполняя экспертное действие; только реально исполненный action меняет следующий state.
5. Что означает β=0.1 на каждом шаге? **Ответ:**10% probability expert action per step, а не 10% episodes с помощью.
6. Может ли teacher знать hidden state? **Ответ:** может по setup, но targets должны учитывать student information и не создавать evaluation leakage.
7. Исправляет ли aggregation отсутствие информации? **Ответ:** нет, одинаковые observations могут оставаться неразличимыми без нового доступного action/evidence.
8. Почему сравнивают equal label/compute budget? **Ответ:** иначе эффект coverage смешивается с количеством data/обучения.
9. Является ли DAgger автоматическим RL? **Ответ:** нет, action labels могут оставаться supervised; меняется способ получения histories.

<!-- /unit:d478e12aa2c91dd0cc -->

<!-- unit:816d1ee3c2f594ede4 -->
<a id="u-816d1ee3c2f594ede4"></a>

**Практика:** для среды из главы [«Среда, observations, actions и horizon»](<agent-environment.md>) запишите четыре истории: правильный документ, устаревшая версия, timeout и пустой поиск. Разметьте следующие действия и stop condition. Составьте две равные по размеру обучающие выборки и независимый тест; заранее определите знаменатель recovery rate и случаи, которые вообще восстановимы.

<!-- /unit:816d1ee3c2f594ede4 -->

## Результат главы и переход дальше

Ответьте на вопрос: **Как научиться действовать в состояниях, порождённых собственными ошибками?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Обучение через взаимодействие: PPO и GRPO](<agent-rl.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
