# От prompt к обычному циклу генерации

**Уровень:** общее ядро.

**Предпосылки:** [Обычная генерация по одному токену](<../01-foundations/lm-generation.md>)

**Используют:** [Prefill, decode, KV-cache, MQA и GQA](<infer-kv.md>), [Контекст, sliding window и prefix caching](<infer-context.md>), [Greedy, temperature, top-p, beam и ограничения](<infer-decoding.md>), [Serving, batching и PagedAttention](<infer-serving.md>), [Speculative decoding и rejection sampling](<infer-speculative.md>), [Квантизация: PTQ, QAT, GPTQ и AWQ](<infer-quantization.md>), [Pruning и границы сжатия](<infer-pruning.md>), [Deployment, cold start, очереди и масштабирование](<infer-deployment.md>), [Кэш ответов, monitoring и качество](<infer-monitoring.md>)

**Учебный вопрос:** Какие состояния меняются при генерации и какие веса остаются фиксированными?




## Контракт runtime до оптимизаций

В основах определён autoregressive цикл. Здесь рассматриваем его как вычислительную процедуру с измеряемыми затратами. На вход поступают ID prompt и параметры остановки; модель возвращает scores, приложение выбирает новый ID. Веса фиксированы, меняются prefix, число оставшихся шагов и служебное состояние запроса.

```text
ids = tokenizer(prompt)
repeat until new_tokens == limit:
    logits = forward(ids, attention_mask)
    next_id = policy(logits[last_real_position])
    ids.append(next_id)
    if next_id == EOS: stop
```

Это baseline без KV-cache. При prompt длины 4 и двух новых токенах первая обработка читает 4 позиции, следующая — 5. Нельзя оценить стоимость только по числу новых токенов: prompt тоже обрабатывается. С cache первый проход обрабатывает prompt, а следующие используют сохранённые K/V; точные shapes и память разобраны в следующей главе.

## Что измеряется

Отдельно фиксируют время до первого токена, задержку между следующими токенами, полную latency, prompt/new token counts и peak memory. Очередь сервиса добавляет ещё один компонент. Быстрый отдельный forward не доказывает низкую задержку многопользовательского приложения.

Выбор правильной последней позиции зависит от padding layout. Режим eval и отключение записи autograd решают разные задачи. EOS и token limit — технические критерии остановки, а задача приложения может требовать ещё валидацию структуры и результата.

## Контроль и практика

Пройдите два шага без cache и выпишите реально обработанные prefix lengths. Затем спроектируйте сравнение с cache на одинаковых prompt, policy и weights: проверяется не только скорость, но и совпадение logits в объявленной численной tolerance. Ошибка — сравнить разные precision, sampling policies и batches и приписать всю разницу одному cache.


## Карта вычисления и границы этапов

![От prompt к обычному циклу генерации: входы, преобразования и выходы](<../../оформление/course/infer-generation.svg>)

Читайте схему по направлению стрелок: первый блок задаёт вход, средние — преобразования и их состояние, последний — результат и способ проверки. Параметры модели при обычном выполнении фиксированы. Состояние запроса, cache и состояние внешнего инструмента имеют разные жизненные циклы; их нельзя смешивать с optimizer states.

<!-- unit:cc9cd133c2a6fc1e0d -->
<a id="u-cc9cd133c2a6fc1e0d"></a>

## От prompt к следующему токену

<!-- /unit:cc9cd133c2a6fc1e0d -->

<!-- unit:96410b097652509590 -->
<a id="u-96410b097652509590"></a>

Модель получает контекст и выдаёт logits для следующей позиции. Для игрушечных logits $z=(2,1,0)$ softmax даёт примерно $(0.665,0.245,0.090)$: greedy выбирает первый токен, а sampling может выбрать любой токен с ненулевой вероятностью. Выбранный токен добавляется к контексту; модель вычисляет следующее распределение, пока не встретится stop token или заданное приложением условие остановки. Это autoregressive decode. Здесь мы пока считаем один такой шаг; повторное вычисление состояний и KV-cache разбираются в главе [«Prefill, decode, KV-cache, MQA и GQA»](<infer-kv.md>).

<!-- /unit:96410b097652509590 -->

<!-- unit:24901d03e66280c51b -->
<a id="u-24901d03e66280c51b"></a>

Схема: `контекст → logits → decoding policy → следующий токен → обновлённый контекст → остановка или следующий шаг`. Практика: для logits выше назвать greedy token, вероятность первого токена и что меняется при sampling. Ограничение: logits и вероятности отражают модельное распределение, а не проверенную истинность ответа.

<!-- /unit:24901d03e66280c51b -->

## Результат главы и переход дальше

Ответьте на вопрос: **Какие состояния меняются при генерации и какие веса остаются фиксированными?** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.

Следующие темы: [Prefill, decode, KV-cache, MQA и GQA](<infer-kv.md>), [Контекст, sliding window и prefix caching](<infer-context.md>), [Greedy, temperature, top-p, beam и ограничения](<infer-decoding.md>), [Serving, batching и PagedAttention](<infer-serving.md>), [Speculative decoding и rejection sampling](<infer-speculative.md>), [Квантизация: PTQ, QAT, GPTQ и AWQ](<infer-quantization.md>), [Pruning и границы сжатия](<infer-pruning.md>), [Deployment, cold start, очереди и масштабирование](<infer-deployment.md>), [Кэш ответов, monitoring и качество](<infer-monitoring.md>).

[Маршрут модуля](<../README.md>) · [Происхождение и точные места переноса](<../../editorial/coverage.md>)
