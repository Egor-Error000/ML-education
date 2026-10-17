# А. Structured output: формат, значения и проверяемый контракт

[Оглавление](README.md). Пункт1 плана.

**Основа:** S5:120–242, S6:4–178. Сохранено сопоставление parser и constrained generation; исправлено смешение tool-tuned model с grammar constraint. **Углубление:** end-to-end data contract, ограничения схемы и четыре валидных, но неправильных результата.

## 1. Почему свободный текст неудобен приложению

Пользователь может прочитать «примерно три дня», а программа ожидает integer $days=3$ или явное unknown. Structured output задаёт contract fields/types, который можно parse, validate и использовать дальше. Это полезно для extraction, classification, routing и аргументов tools.

Contract начинается со смысла: что означает days, calendar или business days, от какой даты отсчитывается, допускается ли unknown, есть ли source evidence. Если эти вопросы не определены, syntactically perfect JSON лишь аккуратно упакует ambiguity.

Пример extraction из документа: `status∈{found,not_found}`, `value:number|null`, `source_ids:list`. Нужно определить invariant: при not_found value=null; при found value подтверждается указанным source. Некоторые cross-field constraints можно выразить supported schema, другие проверяет application.

## 2. Три способа получить структуру

Prompt просит нужный формат; model может выполнить, но enforcement нет. Parser после generation разбирает строку и выявляет нарушение; он не мешал model выбирать лишние tokens. Constrained decoder строит допустимые token continuations grammar/schema во время generation.

Tool-tuned model обучена воспроизводить соответствующие patterns; это training свойство. Grammar masking — runtime механизм. Они могут сочетаться, но не являются одинаковыми гарантиями: tuned model может нарушить формат, constrained weak model может создать семантически плохой, но валидный output.

В supported constraint decoder запрещённые logits маскируются, allowed renormalized. Tokenizer pieces могут покрывать несколько symbols; validator должен учитывать весь возможный token continuation, escaping, числа и nested state. Реальный backend может поддерживать только subset JSON Schema, поэтому schema acceptance/validation проверяют заранее.

## 3. Final validation остаётся нужна

Generation может быть прервана max tokens, timeout, cancellation, tool transition или engine error. Stream prefix `{"value":` не является final valid object. Не следует выполнять действие по неполному streaming fragment. Сначала terminal state и полная validation выбранного output.

Различать:

1. **Parse:** строка разбирается как JSON.
2. **Schema:** поля/types/enums/required constraints соблюдены.
3. **Semantics:** данные существуют, units и cross-field invariants верны.
4. **Task correctness:** извлечён правильный факт/выбрано нужное действие.

`{"status":"found","value":10,"source_ids":["D1"]}` может быть parser/schema-valid, но D1 не существует, содержит другое число, недоступен этому user или число относится к другой currency. Это четыре разных ошибки. «JSON validity100%» не равно extraction accuracy100%.

Schema должна представлять неизвестность, если task допускает отсутствие ответа. Если forced enum содержит только yes/no при недостатке evidence, модель вынуждена выбрать один неверно определённый вариант. Принудительная структура не должна принуждать придумывать факт.

## 4. Исправления, retries и data provenance

При parse error можно bounded retry или запросить repair, но сохранять original output/error для diagnosis. «Починка JSON» не должна незаметно менять value. После repair заново применяются все semantic checks.

Для numeric extraction сохранять source span/ID и units. Для dates — declared timezone и reference date, не автоматически текущее время runtime. Для entity IDs — existence lookup; пользовательский display name не всегда уникальный ID.

При schema evolution old cached outputs могут стать несовместимыми. Версионировать schema и parsing rules, зафиксировать migration/defaults. Если добавили required field, старый output не становится корректным только от удачного JSON parse.

## 5. Учебный разбор одного запроса

Документ D1: «Срок обработки —3 рабочих дня». Question: «Сколько календарных дней занимает обработка?» Простое extraction days3 с unit=calendar неправильно. Корректный output должен сохранить business_days и/или указать, что calendar duration зависит от calendar/start date. Формат не снимает смысловое различие.

Пусть contract требует `duration_days` без unit и unknown запрещён. Проблема в schema/task design, а не только в model quality. Добавление более сильной model не исправляет недоопределённый output specification.

## 6. Вопросы и практика

1. Что гарантирует parser? **Ответ:** разбор структуры по syntax, не truth/authorization.
2. Устраняет ли tool fine-tuning необходимость validation? **Ответ:** нет, training не равен строгому runtime invariant.
3. Можно ли исполнять partial JSON из stream? **Ответ:** только по специально определённому incremental protocol; обычный незавершённый object нельзя считать готовым действием.

**Практика:** задать contract для «сумма из счёта»: value,currency,source,unknown. Написать три schema-valid semantic failures и rules, обнаруживающие их. Механика logits/sampling подробно в [инференсе](<../../03-inference/Учебные конспекты/Е — Greedy, temperature, top-p, beam и constrained decoding.md>).
