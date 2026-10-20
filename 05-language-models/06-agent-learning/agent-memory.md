# Углубление: память и длинный горизонт

**Уровень:** углубление.

**Предпосылки:** [Оценка траекторий и воспроизводимость](<agent-evaluation.md>)

**Используют:** Выбранная прикладная задача и итоговая проверка результата.

**Учебный вопрос:** Какую информацию сохранить и как измерить её полезность?





## Карта вычисления и границы этапов

![Углубление: память и длинный горизонт: входы, преобразования и выходы](<../../оформление/course/agent-memory.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Reward и targets используются в объявленном режиме обучения. При оценке политика фиксирована; success измеряется по всей попытке, включая повторы и бюджет.

<!-- unit:86c35fc8b6a3c4c31f -->
<a id="u-86c35fc8b6a3c4c31f"></a>

## История не равна полезной памяти

<!-- /unit:86c35fc8b6a3c4c31f -->

<!-- unit:f15375416618a7ec72 -->
<a id="u-f15375416618a7ec72"></a>

Агент видит только часть состояния среды. Полная история помогает восстановить, что уже сделано, но растёт с каждым вызовом. Повторно передавать десятки документов дорого, а нужное условие может теряться среди подробностей. Ограниченный context заставляет выбирать, что оставить доступным следующему decision.

<!-- /unit:f15375416618a7ec72 -->

<!-- unit:ebcd789006def7f850 -->
<a id="u-ebcd789006def7f850"></a>

Обозначим память $m_t$ и её обновление:

<!-- /unit:ebcd789006def7f850 -->

<!-- unit:686c73f26c336b32e7 -->
<a id="u-686c73f26c336b32e7"></a>

$$
m_{t+1}=f(m_t,o_{t+1},a_t),
\qquad
a_{t+1}\sim\pi_\theta(\cdot\mid o_{t+1},m_{t+1},g).
$$

<!-- /unit:686c73f26c336b32e7 -->

<!-- unit:4d93cc42c900c73b8d -->
<a id="u-4d93cc42c900c73b8d"></a>

$g$ — цель задачи. Функция $f$ может быть обычным кодом, LLM-summary, retrieval-механизмом или обучаемой частью системы. Наличие памяти само по себе не означает обучение параметров модели. Задача памяти — сохранять информацию, меняющую будущие решения, с допустимой стоимостью и ошибкой.

<!-- /unit:4d93cc42c900c73b8d -->

<!-- unit:e9491152e04b1f7f1a -->
<a id="u-e9491152e04b1f7f1a"></a>

Если $m_t$ было бы достаточной статистикой истории для будущих rewards/transitions, вся история не требовалась бы. В реальной системе summary редко доказывается достаточным. Поэтому сжатие нужно оценивать по последствиям для task success, а не по тому, насколько красиво выглядит пересказ.

<!-- /unit:e9491152e04b1f7f1a -->

<!-- unit:41c9392c620c94e274 -->
<a id="u-41c9392c620c94e274"></a>

<!-- source-depth-lifecycle-Ж -->

<!-- /unit:41c9392c620c94e274 -->

<!-- unit:3c57687d5036efdff6 -->
<a id="u-3c57687d5036efdff6"></a>

### Память —часть input decision, а не обязательно изменение weights

<!-- /unit:3c57687d5036efdff6 -->

<!-- unit:3c2b9f284e60db9c6d -->
<a id="u-3c2b9f284e60db9c6d"></a>

В S5/S6 обсуждались история, summary, state и локальные agent loops. Их общий смысл: следующий model call получает определённые данные о прошлом. Если weights θ не обновлялись, это context/runtime memory. Если затем episodes использованы для SFT, появляется отдельный training процесс. Сохранить checkpoint или текст «помни об ошибке» не означает выполнить gradient update.

<!-- /unit:3c2b9f284e60db9c6d -->

<!-- unit:5948af027ec9411b91 -->
<a id="u-5948af027ec9411b91"></a>

У long-horizon задачи прошлый факт может менять следующее действие через десятки шагов. Ограничение пользователя «только проект P», observation «v1 obsolete», уже выполненный call и оставшийся budget имеют разные роли. Сжать всё в один сюжетный пересказ удобно, но model может потерять именно тот discriminating detail, ради которого память нужна.

<!-- /unit:5948af027ec9411b91 -->

<!-- unit:885ae23210fefa5f08 -->
<a id="u-885ae23210fefa5f08"></a>

### Достаточное представление и необратимая потеря информации

<!-- /unit:885ae23210fefa5f08 -->

<!-- unit:6bf8f956e9cbc5550a -->
<a id="u-6bf8f956e9cbc5550a"></a>

Если два histories требуют разных действий, но f сжимает их в одно и то же m, policy по этому m не сможет гарантированно различить случаи без нового evidence. Например, h1 содержит «30days —current», h2 —«30days —obsolete». Summary «срок 30days» одинаков, но в одном случае можно отвечать, в другом требуется current source.

<!-- /unit:6bf8f956e9cbc5550a -->

<!-- unit:9fb1eb3f5ecea81d4e -->
<a id="u-9fb1eb3f5ecea81d4e"></a>

В равновероятной toy задаче с двумя несовместимыми ответами и одинаковым observation после такого сжатия maximum guessing accuracy0.5. Более сильная LLM не восстановит удалённый бит достоверно. Возможны refetch источника или иная memory representation; именно их стоимость/доступность определяет, насколько потеря поправима.

<!-- /unit:9fb1eb3f5ecea81d4e -->

<!-- unit:aa56af8bf390b42175 -->
<a id="u-aa56af8bf390b42175"></a>

Sufficient statistic означает достаточность для выбранного будущего решения/transition/reward, а не сохранение всех слов исходника. В реальных summaries её обычно не доказывают. Поэтому memory оценивают по invariants и task outcomes: сохранены ли negation, version, unresolved uncertainty, scope, operation identity и stop condition.

<!-- /unit:aa56af8bf390b42175 -->

<!-- unit:e7f646d29fb3b6f888 -->
<a id="u-e7f646d29fb3b6f888"></a>

## Несколько видов памяти

<!-- /unit:e7f646d29fb3b6f888 -->

<!-- unit:93ec2c2253261e7978 -->
<a id="u-93ec2c2253261e7978"></a>

**Структурное состояние** хранит поля с ясной семантикой: выбранный document ID/version, завершённые подзадачи, число retries, ограничения пользователя. Его удобно проверять кодом. Например, `current_document_version` не должен незаметно меняться после чтения устаревшего источника.

<!-- /unit:93ec2c2253261e7978 -->

<!-- unit:24e025fca30cba0a42 -->
<a id="u-24e025fca30cba0a42"></a>

**Краткое текстовое summary** переносит смысл между окнами контекста. Оно компактно и гибко, но может потерять отрицание, срок действия или степень уверенности. «Проверили срок 14 дней» опасно, если исходное observation говорило «14 дней в устаревшей версии».

<!-- /unit:24e025fca30cba0a42 -->

<!-- unit:c100a4f4fac63f3898 -->
<a id="u-c100a4f4fac63f3898"></a>

**Внешняя эпизодическая память** хранит записи событий и evidence с ID, timestamp, provenance. По запросу возвращаются релевантные фрагменты. Это близко к RAG, но отбор может зависеть от текущей подзадачи и уже выполненных действий. Непроверенная догадка агента должна иметь другой статус, чем observation инструмента.

<!-- /unit:c100a4f4fac63f3898 -->

<!-- unit:4a3bac720be4c94955 -->
<a id="u-4a3bac720be4c94955"></a>

**Checkpoint** сохраняет состояние исполняющей системы, чтобы продолжить после сбоя. Он не гарантирует, что внешний мир остался прежним: документ мог обновиться, а операция — уже завершиться. После восстановления нужно сверять versions/status и избегать повторного необратимого действия. В read-only учебной среде достаточно воспроизводимо восстановить историю, IDs и счётчики.

<!-- /unit:4a3bac720be4c94955 -->

<!-- unit:97a50f05f7124b6a7d -->
<a id="u-97a50f05f7124b6a7d"></a>

### Структурное state и memory of claims

<!-- /unit:97a50f05f7124b6a7d -->

<!-- unit:892ade2debfb548144 -->
<a id="u-892ade2debfb548144"></a>

Trusted constraints задаются task/application, а не свободным model-generated summary. Scope P, запрет определённого tool и remaining budget имеют authoritative источник. Evidence claims хранят content/value, source ID/revision/span и статус проверки. Hypothesis хранится как hypothesis; план —как pending work. Эти поля не должны сливаться в одно «мы уже выяснили».

<!-- /unit:892ade2debfb548144 -->

<!-- unit:46f538e715d38c37b9 -->
<a id="u-46f538e715d38c37b9"></a>

Полезный claim record:subject,predicate,value/unit,applicability,source revision,observation ID,observed_at,status. Это учебная схема, не универсальный mandatory database design. Для retention нужны service/report category и currentness; для code tool —artifact/version и test outcome. Confidence score сам по себе не заменяет источник или calibration.

<!-- /unit:46f538e715d38c37b9 -->

<!-- unit:20064d0ba3f6344021 -->
<a id="u-20064d0ba3f6344021"></a>

Status observed означает «получено такое observation», а не автоматически «это вечная истина». Verified может означать отдельную проверку по конкретному oracle. Superseded помечает прежнюю применимую версию; record остаётся историческим, но не выбирается как current. Summary не вправе поднять hypothesis до verified только ради краткости.

<!-- /unit:20064d0ba3f6344021 -->

<!-- unit:cb029414c305d84464 -->
<a id="u-cb029414c305d84464"></a>

### Журнал событий и полученное состояние

<!-- /unit:cb029414c305d84464 -->

<!-- unit:a882af1059dff7c2c8 -->
<a id="u-a882af1059dff7c2c8"></a>

Event log хранит chronology: proposed call,validated execution,result/error,state update. Materialized state даёт удобный текущий view. Если один observation replayed дважды, naive append/sum может продублировать факт или расход. Stable event IDs и explicit reducer позволяют учитывать repeated delivery по policy; одно natural-language «не повторять» этого не обеспечивает.

<!-- /unit:a882af1059dff7c2c8 -->

<!-- unit:1c6dada8f326eaec2b -->
<a id="u-1c6dada8f326eaec2b"></a>

У события полезно различать observation time, source effective time и время публикации revision. Поздно пришедший result не обязательно самый свежий source. `last write wins` по arrival не заменяет правило currentness. Для параллельных calls version/scope/effect identity особенно важны, потому что completion order не равен request order.

<!-- /unit:1c6dada8f326eaec2b -->

<!-- unit:59fc87ceb8abbbafa1 -->
<a id="u-59fc87ceb8abbbafa1"></a>

### Summary, buffer и episodic retrieval

<!-- /unit:59fc87ceb8abbbafa1 -->

<!-- unit:936b0aa4e3fda1f735 -->
<a id="u-936b0aa4e3fda1f735"></a>

Buffer сохраняет подробный недавний context; summary —выбранное сжатие старого; episodic retrieval достаёт старые records по задаче. Можно использовать несколько уровней, но каждый retrieval stage имеет собственные failures: запись не найдена, выбрана нерелевантная, потеряна applicability, generator неправильно использовал найденное. Наличие memory store не гарантирует, что нужный факт был в actual prompt.

<!-- /unit:936b0aa4e3fda1f735 -->

<!-- unit:86dfe016b1f3576766 -->
<a id="u-86dfe016b1f3576766"></a>

При memory retrieval фильтруют task/user/scope/revision и сохраняют provenance. Большая semantic similarity между двумя reports не делает их facts взаимозаменяемыми. Если память содержит model conjectures, их нельзя индексировать как confirmed tool observations без статуса. Иначе последующая retrieval создаст ложное независимое «подтверждение» собственной догадки.

<!-- /unit:86dfe016b1f3576766 -->

<!-- unit:87ed51032541c4073e -->
<a id="u-87ed51032541c4073e"></a>

Summary тоже является transformation с source references. При конфликте summary с original evidence можно восстановить passage и проверить. Если original удалён и остался только пересказ, verification возможности сузились; это должно быть явно известно. Запись trust level важнее уверенного стиля summary.

<!-- /unit:87ed51032541c4073e -->

<!-- unit:0036a0f36818814f33 -->
<a id="u-0036a0f36818814f33"></a>

### Checkpoint, KV reuse и learned memory различаются

<!-- /unit:0036a0f36818814f33 -->

<!-- unit:f0335e04b2b7908063 -->
<a id="u-f0335e04b2b7908063"></a>

Checkpoint сохраняет workflow state по contract runtime. KV/prefix cache переиспользует совместимые computations input prefix. External memory store хранит события/знания. Model weights могут хранить обученные закономерности. Один механизм не выполняет все эти функции одновременно.

<!-- /unit:f0335e04b2b7908063 -->

<!-- unit:5f91b92202996a378d -->
<a id="u-5f91b92202996a378d"></a>

KV cache не выбирает, какие факты истинны и актуальны. Увеличить context window не гарантирует внимание к нужному раннему условию; сжать prompt не означает сжать weights. Checkpoint не откатывает внешний мир и не доказывает, что pending mutation не произошла. После restart нужен recovery/status protocol, а не только восстановление текста последней мысли.

<!-- /unit:5f91b92202996a378d -->

<!-- unit:af7608b2b688ef2132 -->
<a id="u-af7608b2b688ef2132"></a>

Подробные cache/long-context mechanisms находятся в [инференсе](<../04-inference/README.md>), state/reducers —в [приложениях](<../05-applications/app-state.md>). Здесь изучается, как выбранная memory влияет на длинную последовательность решений.

<!-- /unit:af7608b2b688ef2132 -->

<!-- unit:8f6aac8306bedfc4fd -->
<a id="u-8f6aac8306bedfc4fd"></a>

## Иерархический план и обновление evidence

<!-- /unit:8f6aac8306bedfc4fd -->

<!-- unit:35875a6af946a22d69 -->
<a id="u-35875a6af946a22d69"></a>

В длинной задаче верхний уровень задаёт подцели, а нижний выбирает действия. Например: найти актуальное правило → проверить исключения → собрать ответ. План полезен как рабочее представление, но его нужно пересматривать при новых observations. Если rule не существует, продолжать заранее записанный happy path неверно.

<!-- /unit:35875a6af946a22d69 -->

<!-- unit:15c4b29524a37c91e3 -->
<a id="u-15c4b29524a37c91e3"></a>

Хранить следует различие между целью, гипотезой и фактом. Поле «проверить v2» означает pending action; поле «v2 прочитана, hash такой-то» означает подтверждённое событие. Смешение этих статусов создаёт ложное ощущение выполненной работы. После summary нельзя превращать предположение «вероятно срок 30 дней» в установленное «срок 30 дней».

<!-- /unit:15c4b29524a37c91e3 -->

<!-- unit:d9d8cecedc7f238024 -->
<a id="u-d9d8cecedc7f238024"></a>

Практический memory contract включает обязательные поля, правила обновления и ссылки на evidence. Для срока возврата важны дата начала действия правила и категория товара; текст рекламного описания можно повторно извлечь при необходимости. Точный выбор зависит от задачи, а не от универсального списка полей.

<!-- /unit:d9d8cecedc7f238024 -->

<!-- unit:53f39d909d845288c4 -->
<a id="u-53f39d909d845288c4"></a>

### План —ожидание будущего, evidence —наблюдаемое прошлое

<!-- /unit:53f39d909d845288c4 -->

<!-- unit:b76c10f24fe62ff0c7 -->
<a id="u-b76c10f24fe62ff0c7"></a>

План «найти current rule → проверить исключения → ответить» задаёт подцели, но не доказывает, что они выполнены. У каждой есть status, входные constraints и критерий completion. Новый observation может сделать прежний план неприменимым: current source отсутствует, доступ запрещён или обнаружен конфликт versions.

<!-- /unit:b76c10f24fe62ff0c7 -->

<!-- unit:ec63c69879b8af6750 -->
<a id="u-ec63c69879b8af6750"></a>

Parent planner может получать от subroutine verified result,partial,error и usage. Простая строка «готово» без source/state не заменяет контракт результата. При handoff передаются незавершённые вопросы и global budget, иначе каждый child начинает собственный неограниченный happy path. Это modular control flow при fixed weights, пока отдельно не обучена policy.

<!-- /unit:ec63c69879b8af6750 -->

<!-- unit:0bc2875fd30fe4ffde -->
<a id="u-0bc2875fd30fe4ffde"></a>

Условия исходной задачи сохраняют отдельно от изменяемого плана. Если требовалось правило и исключения, отсутствие исключений в memory не позволяет переопределить цель как «назвать одно число». Можно честно завершиться partial при ограничении, но status и missing aspects должны остаться видимыми. Это помогает оценивать long-horizon success без подмены задачи.

<!-- /unit:0bc2875fd30fe4ffde -->

<!-- unit:cd6593bc69215de25f -->
<a id="u-cd6593bc69215de25f"></a>

### Полный учебный журнал из 12 событий

<!-- /unit:cd6593bc69215de25f -->

<!-- unit:33c6464f50400166bd -->
<a id="u-33c6464f50400166bd"></a>

Все значения ниже относятся к вымышленной report-retention среде главы [«Среда, observations, actions и horizon»](<agent-environment.md>).

<!-- /unit:33c6464f50400166bd -->

<!-- unit:65123400d275e2d6dc -->
<a id="u-65123400d275e2d6dc"></a>

1. Task:найти current retention для standard reports проекта P, назвать source/version.
2. Trusted scope:только P;global cap6environment actions.
3. Search c1 вернул R1/v1 obsolete,R2/v2 active и distractor D.
4. Read c2:R1/v1 содержит 30days,но metadata obsolete.
5. Model hypothesis:«возможно 30days»;это ещё не current answer.
6. Read c3:R2 завершился timeout,content не получен.
7. Policy разрешает один retry;записывается, что next retry ещё pending.
8. Read c4:R2 успешно вернул 90days,active v2.
9. В R2 условие:правило относится к standard reports;temporary reports здесь не покрыты.
10. Evidence record:R2/v2,span retention,hash H2,observation c4;R1 остаётся stale evidence.
11. Answer candidate ещё не выдан;нужно сформулировать вывод и проверить citation.
12. Checkpoint сохраняет log/state;из 6actions израсходованы 4,осталось 2;retry уже использован.

<!-- /unit:65123400d275e2d6dc -->

<!-- unit:4b1c680d9c39c5d5d1 -->
<a id="u-4b1c680d9c39c5d5d1"></a>

Событие «retry planned» не равняется событию «retry выполнен». Timeout c3 не имеет content;success c4 не превращает c3 в успешный result задним числом. Исторический 30days из R1 не удаляется, как будто его не было;меняется его применимость для current question.

<!-- /unit:4b1c680d9c39c5d5d1 -->

<!-- unit:6a5f12de1baa33c3f6 -->
<a id="u-6a5f12de1baa33c3f6"></a>

### Сравниваем плохое summary и проверяемое state

<!-- /unit:6a5f12de1baa33c3f6 -->

<!-- unit:74359b9870c646f38a -->
<a id="u-74359b9870c646f38a"></a>

Плохое summary:«проверили срок 30/90 дней,источники есть,можно завершать». Оно потеряло version,currentness,report category и pending answer status. Поле source support нельзя восстановить из фразы «источники есть»; slash30/90 не определяет правильный ответ.

<!-- /unit:74359b9870c646f38a -->

<!-- unit:4f725312f03e08b2c7 -->
<a id="u-4f725312f03e08b2c7"></a>

Хорошее компактное представление сохраняет:task standard/P;current evidence R2/v2=90days,c4/H2;R1/v1=30days obsolete;temporary reports не покрыты;answer pending;remaining_actions2;retry_used1. Полный passage доступен по source reference для повторной проверки. В этом record нет утверждения, что итог уже сообщён пользователю.

<!-- /unit:4f725312f03e08b2c7 -->

<!-- unit:3d083dc19a55ed39f4 -->
<a id="u-3d083dc19a55ed39f4"></a>

Проверяемый следующий путь — answer с 90days/R2/v2 и ограничением applicability, затем validation по выбранному protocol. Если evaluator считает answer и validation разными environment actions, это ровно два remaining slots; если validation internal, accounting другое и должно быть объявлено. Журнал сам не определяет единицы runtime автоматически.

<!-- /unit:3d083dc19a55ed39f4 -->

<!-- unit:1c2e993c9868ba6786 -->
<a id="u-1c2e993c9868ba6786"></a>

### Новая revision и восстановление после паузы

<!-- /unit:1c2e993c9868ba6786 -->

<!-- unit:6aa2f8b68b76e2d65d -->
<a id="u-6aa2f8b68b76e2d65d"></a>

После checkpoint source может обновиться. Fact «в observation c4 R2/v2 было 90days» остаётся исторически верным, но ответ на «какое правило сейчас» требует currentness policy. Если task просит as-of snapshot v2, новый live source не должен молча подменить snapshot. Если task требует текущую версию, нужна проверка/retrieval свежего evidence при доступном budget.

<!-- /unit:6aa2f8b68b76e2d65d -->

<!-- unit:5491ec413eaf0abf33 -->
<a id="u-5491ec413eaf0abf33"></a>

После external mutation дополнительно проверяют status/intent identity. Pending operation в checkpoint могла успеть завершиться после последней записи. Restore не даёт права бесконечно повторять действие; запись «попробовать снова» в summary не заменяет idempotency contract. В read-only toy task последствия мягче, но версионная неоднозначность всё равно важна.

<!-- /unit:5491ec413eaf0abf33 -->

<!-- unit:db473e74fc5c97afaf -->
<a id="u-db473e74fc5c97afaf"></a>

## Цена и разобранный пример

<!-- /unit:db473e74fc5c97afaf -->

<!-- unit:c47b8f1c26074ac1fc -->
<a id="u-c47b8f1c26074ac1fc"></a>

Пусть каждый из $H=20$ шагов добавляет $b=500$ tokens history. Если каждый новый запрос заново обрабатывает всю накопленную историю без переиспользования вычислений, только добавленная часть даёт:

<!-- /unit:c47b8f1c26074ac1fc -->

<!-- unit:b25a65dba9d612b83b -->
<a id="u-b25a65dba9d612b83b"></a>

$$
\sum_{t=1}^{H}bt=b\frac{H(H+1)}2=105\,000
$$

<!-- /unit:b25a65dba9d612b83b -->

<!-- unit:4ce478549a51a95fe8 -->
<a id="u-4ce478549a51a95fe8"></a>

обработанных input tokens. Это упрощённый учёт; начальный prompt, ответы модели и реальный cache protocol добавляют свои затраты. Prefix/KV cache может снизить повторные вычисления, но не делает контекст семантически компактным и не отменяет лимит context window.

<!-- /unit:4ce478549a51a95fe8 -->

<!-- unit:fcc449e5968587c624 -->
<a id="u-fcc449e5968587c624"></a>

Если вместо всей истории на каждом шаге передавать summary на 1000 tokens, input-составляющая составит около 20 000 tokens плюс текущие observations и стоимость создания summaries. Выигрыш может исчезнуть, если summary генерируется часто, длинно или приводит к лишним вызовам из-за потери facts. Сравнивать надо полную стоимость.

<!-- /unit:fcc449e5968587c624 -->

<!-- unit:0e17d37f6bc402ec8a -->
<a id="u-0e17d37f6bc402ec8a"></a>

Критический тест: на 15-м шаге спросить об условии из первого observation, которое меняет финальное действие. Если summary сохранило общий сюжет, но потеряло исключение, память не выполнила свою функцию. Второй тест — обновившаяся версия источника: нельзя продолжать использовать старый fact без проверки актуальности.

<!-- /unit:0e17d37f6bc402ec8a -->

<!-- unit:68c875fded93e5e144 -->
<a id="u-68c875fded93e5e144"></a>

### Откуда появляются 105000 и 95000 tokens

<!-- /unit:68c875fded93e5e144 -->

<!-- unit:ab839ef1d1dc0259d9 -->
<a id="u-ab839ef1d1dc0259d9"></a>

Сумма $500\sum_{t=1}^{20}t=105000$ выше предполагает, что на каждом из 20 измеряемых calls уже доступны очередные 500 tokens, включая первый. Если 500 tokens добавляются **после** каждого call, input этих 20 calls содержит 0, 500, ..., 9500 tokens новой history: всего 95000. Разница — граница учёта, а не два противоречащих закона стоимости.

<!-- /unit:ab839ef1d1dc0259d9 -->

<!-- unit:f3cada17a8c2959a44 -->
<a id="u-f3cada17a8c2959a44"></a>

Начальные инструкции, ответы модели, tool content и обслуживание summary считают дополнительно. Повторная обработка prefix без cache может иметь другую compute стоимость, чем token billing; конкретный backend/provider задаёт contract. При prefix reuse сохраняется длина context, поэтому cache не делает бесконечную history помещающейся в конечное окно.

<!-- /unit:f3cada17a8c2959a44 -->

<!-- unit:e1851f3b5cc3936ab9 -->
<a id="u-e1851f3b5cc3936ab9"></a>

### Сжатие оплачивается и может требовать refetch

<!-- /unit:e1851f3b5cc3936ab9 -->

<!-- unit:0cae9f6c4223008822 -->
<a id="u-0cae9f6c4223008822"></a>

Если каждый из 20 calls содержит summary на 1000 tokens, получается 20000 input tokens этой части. Допустим, summary обновляется 4 раза, каждое обновление читает 3000 tokens и генерирует 1000. Тогда обслуживание summary добавит 12000 input и 4000 output tokens. Это разные counts; цены/compute input и output нельзя молча приравнять.

<!-- /unit:0cae9f6c4223008822 -->

<!-- unit:23890e29bb96b963ab -->
<a id="u-23890e29bb96b963ab"></a>

Итого для указанной упрощённой части — 32000 input и 4000 output вместо 105000 input полной history, но ещё не учтены текущие observations, генерация основной policy и initial prompt. Два дополнительных refetch по 2000 input tokens добавят 4000 input. Полную экономию оценивают вместе с изменением success и retry costs, а не только сокращением следующего prompt.

<!-- /unit:23890e29bb96b963ab -->

<!-- unit:0b7dc41189e338829e -->
<a id="u-0b7dc41189e338829e"></a>

### Token budget памяти должен учитывать упаковку

<!-- /unit:0b7dc41189e338829e -->

<!-- unit:84813efcfc0aafc92d -->
<a id="u-84813efcfc0aafc92d"></a>

При context 8192 tokens резерв instructions/question 1200, output 1200 и current observations 1800 оставляет для остальной memory 3992 tokens. Если summary занимает 1000 и structured state 400, на извлечённые event snippets остаётся 2592. При размере 700 tokens, включая metadata, помещаются 3 snippets — 2100 tokens; четыре требуют 2800 и уже не помещаются.

<!-- /unit:84813efcfc0aafc92d -->

<!-- unit:420a8e8164649cbde4 -->
<a id="u-420a8e8164649cbde4"></a>

Считается фактическая serialization с нужным tokenizer: model/template headers, separators, tool definitions и citations могут добавить tokens. Если memory cap определён по characters, он не обязательно соблюдает token limit. Если нужное исключение осталось четвёртым snippet за cutoff, его отсутствие в final prompt — memory retrieval/packing failure, а не только reasoning failure.

<!-- /unit:420a8e8164649cbde4 -->

<!-- unit:95618c4b61b1d1db0a -->
<a id="u-95618c4b61b1d1db0a"></a>

## Ablation и ответы

<!-- /unit:95618c4b61b1d1db0a -->

<!-- unit:1998bbb2eec82f4132 -->
<a id="u-1998bbb2eec82f4132"></a>

Сравниваем при одном budget: полную доступную историю, структурное state, summary, retrieval из журнала. Меняем один компонент за раз. Измеряем success, потерю ограничений, противоречия памяти, повторные calls, latency и цену. Желательно отдельно разметить ошибки retrieval и ошибки использования правильно извлечённого evidence.

<!-- /unit:1998bbb2eec82f4132 -->

<!-- unit:15f9bbda9709e63332 -->
<a id="u-15f9bbda9709e63332"></a>

### Ablation с одинаковым information contract

<!-- /unit:15f9bbda9709e63332 -->

<!-- unit:c12802fd06a1104a60 -->
<a id="u-c12802fd06a1104a60"></a>

Сравнивают отсутствие памяти, полную доступную history, structured state, summary + buffer и retrieval из event log на одних tasks/snapshots. Baseline без памяти не должен случайно получать gold answer через initial prompt. Режим полной history имеет свои context limits: если она обрезается, сохраняют actual input, а не объявляют этот режим идеальной бесконечной памятью.

<!-- /unit:c12802fd06a1104a60 -->

<!-- unit:6aa4caa732c70286e7 -->
<a id="u-6aa4caa732c70286e7"></a>

Помимо final success измеряют сохранность constraints/versions, смешение planned и done, ложные facts, повторённые actions, recovery после паузы, cost/latency. Отдельно размечают: record отсутствует в store, не извлечён, отрезан packing или неверно использован после передачи. Так видно, что исправлять: запись памяти, retrieval, serializer или generator.

<!-- /unit:6aa4caa732c70286e7 -->

<!-- unit:5ab1546a6422b80668 -->
<a id="u-5ab1546a6422b80668"></a>

Для обучения policy используют тот же memory contract, который доступен на inference, либо migration оценивают явно. Одновременная смена model, context window, tools и memory оценивает новую систему целиком, но не изолированный memory effect. Проверки выше — учебный протокол; реальный memory benchmark или пользовательский deployment не заявляются.

<!-- /unit:5ab1546a6422b80668 -->

<!-- unit:0b2f3101745f614fd7 -->
<a id="u-0b2f3101745f614fd7"></a>

### Вопросы для самопроверки

<!-- /unit:0b2f3101745f614fd7 -->

<!-- unit:44c98450b7bfd58d15 -->
<a id="u-44c98450b7bfd58d15"></a>

1. Заменяет ли prefix cache смысловую память? **Ответ:** он переиспользует вычисления совпадающего префикса; выбор и достоверность facts остаются отдельной задачей.
2. Почему summary не считается ground truth? **Ответ:** это преобразование исходных observations, которое может внести ошибку или потерять условие.
3. Достаточен ли checkpoint для безопасного повтора действия? **Ответ:** нет; состояние внешней системы может измениться и требует сверки.
4. Гарантирует ли сильная модель восстановление удалённой детали? **Ответ:** нет, одинаковая memory для несовместимых histories может не содержать нужной информации.
5. Почему observed не равно current verified truth? **Ответ:** observation имеет источник, версию и время; применимость может измениться.
6. Planned retry значит уже выполненный retry? **Ответ:** нет, действие и его result должны быть отдельно подтверждены.
7. Что остаётся верным после обновления source? **Ответ:** история прежнего observation, но не обязательно его current applicability.
8. Отчего 105000 input превращаются в 95000 при 20 шагах по 500? **Ответ:** отличается учёт до или после добавления history на вызове.
9. Бесплатно ли summary? **Ответ:** нет, есть input/output для обслуживания, возможны refetch и ошибки сжатия.
10. Доказывает ли checkpoint отсутствие external effect? **Ответ:** нет, внешняя система могла продолжить работу после последнего сохранения.

<!-- /unit:44c98450b7bfd58d15 -->

<!-- unit:e53cc091195bd7f09d -->
<a id="u-e53cc091195bd7f09d"></a>

**Практика:** возьмите историю из 12 событий с двумя версиями документа, timeout и исправленной гипотезой. Составьте summary и структурное state. Затем восстановите действия только по памяти; перечислите facts, которые пришлось бы заново извлечь, и проверьте сохранность отрицаний, versions и stop condition.

<!-- /unit:e53cc091195bd7f09d -->

## Результат главы и переход дальше

Ответьте на вопрос: **Какую информацию сохранить и как измерить её полезность?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
