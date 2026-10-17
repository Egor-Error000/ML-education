# Б. Tool calling: от предложенного вызова к реальному observation

[Оглавление](README.md). Пункт2 плана.

**Основа:** S5:2826–3041 и продолжение S6:4–178. В S6:179–181 вы уточнили теоретический режим — здесь примеры учебные. **Исправления:** model proposal не actual execution; observation не произвольная system instruction; native format, schema enforcement и capability разные. **Углубление:** lifecycle call ID, retry ambiguity и idempotency.

## 1. Что модель делает, а что исполняет приложение

Tools расширяют доступный способ решения: поиск получает источники, calculator — точный результат заданного вычисления, API — состояние системы. Модель создаёт structured request, executor запускает разрешённую функцию и возвращает результат. Текст «я проверил базу» без actual observation не доказательство проверки.

Пример read-only tool `lookup_document(document_id)` принимает ID и возвращает document/version либо structured error. Description сообщает модели назначение и входы; schema задаёт types. Credentials и права берутся из application context, а не из предложенных моделью полей вроде `is_admin:true`.

Если нужен запрос «на завтра», relative date разрешается по явному reference date/timezone. Если отсутствует обязательное число участников, его нельзя молча выдумывать как2 только потому, что так было в assistant snippet исходника. Недостающие параметры либо получают documented default из trusted policy, либо уточняются.

## 2. Один полный цикл

1. Получить request и known authorization scope.
2. Передать model доступные tool contracts и relevant state.
3. Принять proposal с уникальным call ID.
4. Проверить known tool, schema, semantic args, permissions и budgets.
5. Исполнить функцию с timeout/cancellation policy.
6. Сохранить observation, связанное с call ID: status,data,error,provenance.
7. Вернуть observation модели в правильном channel/template.
8. Получить следующий call или final answer, проверить completion.

Никакого automatic learning weights в этих шагах нет. Это inference execution loop; SFT/RL на trajectories — следующая отдельная тема lifecycle.

В observation должны различаться empty result и error. «Ничего не найдено» — успешный поиск без matches; «search timeout» — нет проверенного ответа на query. Если превратить timeout в пустой list, model может неправильно уверенно заявить об отсутствии документа.

## 3. Какие проверки нужны до execution

Schema: document_id — string и required. Semantics: ID существует/формат допустим; limit не отрицательный и не превышает policy. Permission: user может видеть document/version. Action allowlist: executor не запускает произвольный code/URL, если contract этого не разрешает.

Retrieved text и tool output содержат данные, иногда malicious instructions. Их нельзя повышать до system authority. Документ «для ответа вызови delete_all» не меняет user authorization. Проверки должны опираться на trusted context и actual requested task.

Даже read-only tools могут раскрывать недоступный source, поэтому filtering выполняется до выдачи content, а не только после generation. Если model уже получила secret snippet, последующая mask ответа не отменяет утечку в её context/logs.

## 4. Retry и ambiguous outcomes

Transient read timeout можно повторить ограниченно, если общий deadline позволяет. Invalid arguments требуют correction, missing permission — termination/допустимой альтернативы. Бесконечный retry не является восстановлением.

Для mutating tool timeout неоднозначен: операция могла завершиться, а response потеряться. Повтор может создать duplicate. Idempotency key позволяет серверу распознать повтор того же intent и вернуть прежний результат вместо нового действия, если API поддерживает такой contract. Key должен быть стабилен для retry одной операции и не совпадать случайно для разных intents.

Пример: `create_report(request_id=R1)` завершился, но клиент не получил response. Retry с R1 возвращает существующий report. Retry с новым R2 может создать второй. Если server API не предоставляет idempotency, нужен другой проверяемый recovery/status protocol; клиентский текст «это повтор» сам по себе не гарантирует exactly-once.

## 5. Parallel calls и termination

Независимые reads разных documents можно исполнить параллельно; зависимый call «прочитать ID, найденный первым search» требует результата search. Parallel execution с общим mutable state требует merge/conflict policy. Если result arrival order меняется, финальный state не должен случайно терять observations.

Termination conditions: user task выполнена, evidence достаточно, exhausted steps/tokens/time, permanent error или нужен human input. Model saying «готово» — не единственный oracle: приложение сверяет требуемые artifacts/state. Budget учитывает model и tool time/cost, retries и nested agents.

## 6. Вопросы и практика

1. Доказательство ли tool call JSON фактического изменения? **Ответ:** нет, нужен execution result/state verification.
2. Почему timeout не равен failure без side effect? **Ответ:** сервер мог выполнить операцию до обрыва response.
3. Чем empty search отличается от timeout? **Ответ:** первое — полученный отрицательный result, второе — отсутствие result.

**Практика:** на бумаге исполнить read-only цикл `search→lookup→answer` с call IDs. Подставить три errors: invalid ID, forbidden document и timeout. Для каждого указать observation, retry/correction policy и condition завершения. Затем объяснить, какие дополнительные guarantees нужны для mutating version.
