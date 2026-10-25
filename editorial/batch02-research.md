# Первичные источники пакета 02

Проверка 25 октября 2026. Ссылки ниже фиксируют основание написания, а не установленную версию библиотек или проведённый эксперимент. В учебных главах источники стоят рядом с утверждениями; даты проверки остаются в редакционном реестре. Новые иллюстрации авторские, внешние рисунки в этом пакете не копировались.

## Данные и параметризация адаптации

- [Self-Instruct](https://arxiv.org/abs/2212.10560v2), [FLAN](https://arxiv.org/abs/2210.11416v5), [Evol-Instruct](https://arxiv.org/abs/2304.12244v3): происхождение и расширение инструкций, смесь задач; не универсальное минимальное число демонстраций.
- [LoRA](https://arxiv.org/abs/2106.09685v2), [DoRA](https://arxiv.org/abs/2402.09353v6), [AdaLoRA](https://arxiv.org/abs/2303.10512v2), [LoRA+](https://arxiv.org/abs/2402.12354v2), [QLoRA](https://arxiv.org/abs/2305.14314v1): разные изменяемые объекты. Числовые примеры курса рассчитаны самостоятельно; они не результаты обучения авторов.
- [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora), [PEFT AdaLoRA](https://huggingface.co/docs/peft/en/package_reference/adalora): формы настройки, адресация модулей, расписание. Страница LoRA при проверке перенаправляла на документацию v0.21.0; `tfinal` относится к числу завершающих шагов. Не заявлена поддержка любого checkpoint или любое сочетание вариантов.
- [Адаптеры Houlsby](https://arxiv.org/abs/1902.00751v2), [prefix tuning](https://arxiv.org/abs/2101.00190), [prompt tuning](https://arxiv.org/abs/2104.08691v2), [P-Tuning v2](https://arxiv.org/abs/2110.07602v3), [BitFit](https://arxiv.org/abs/2106.10199v5): нелинейные модули, входные векторы, глубокие префиксы и исходные bias имеют разные графы.

## Предпочтения, объединение и сохранение навыков

- [DPO](https://arxiv.org/abs/2305.18290v3), [IPO](https://arxiv.org/abs/2310.12036v2), [KTO](https://arxiv.org/abs/2402.01306v5), [ORPO](https://arxiv.org/abs/2403.07691v2): перепроверены тип сигнала, опорная политика и форма цели. KTO v5 опубликована 8 сентября 2026. У KTO в тексте показана объявленная форма loss; детали оценки опорной точки не подменяются средним человеческим баллом.
- [Model soups](https://arxiv.org/abs/2203.05482v3), [TIES](https://arxiv.org/abs/2306.01708v2), [DARE](https://arxiv.org/abs/2311.03099v3): согласование координат, выбор знака и случайное прореживание дельт. В курсе p у DARE явно означает вероятность сохранения, что отличает запись от конвенции вероятности удаления.
- [DistilBERT](https://arxiv.org/abs/1910.01108v4): BERT-base как сравнительная основа, отдельные цели переноса. В тексте не приписана собственная проверка качества/скорости или рецепт закрытой модели.
- [Constitutional AI](https://arxiv.org/abs/2212.08073), [EWC](https://arxiv.org/abs/1612.00796v2), [оценка моделью-судьёй](https://arxiv.org/abs/2306.05685v4): источник принципов, локальное удержание прежнего поведения и ограничения автоматической оценки. Одобрение критика не считается доказательством факта.

## Кеш, контекст и выполнение

- [GQA](https://arxiv.org/abs/2305.13245v3), [StreamingLLM](https://arxiv.org/abs/2309.17453v4), [H₂O](https://arxiv.org/abs/2306.14048v3), [KIVI](https://arxiv.org/abs/2402.02750v2): совместное использование голов, удаление источников и числовая ошибка кеша — разные механизмы.
- [Position Interpolation](https://arxiv.org/abs/2306.15595v2), [YaRN](https://arxiv.org/abs/2309.00071v3), [FlashAttention](https://arxiv.org/abs/2205.14135v2), [Ring Attention](https://arxiv.org/abs/2310.01889v4): позиции, IO и распределённый доступ не отождествляются с удалением логических связей. YaRN v3 датирована 6 февраля 2026.
- [Mamba](https://arxiv.org/abs/2312.00752v2), [Jamba](https://arxiv.org/abs/2403.19887v2): рекуррентное состояние и обученный гибрид AI21; присутствующее полное внимание сохраняет свои затраты.
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu), [PagedAttention](https://arxiv.org/abs/2309.06180), [prefix caching vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching/), [vLLM serve](https://docs.vllm.ai/en/stable/cli/serve/): итерационное расписание, адресация KV, идентичность префикса и отдельные бюджеты. Возможности и скорость конкретного установленного сервера не измерялись.
- [JSON Schema if/then](https://json-schema.org/understanding-json-schema/reference/conditionals): условное обязательное поле выразимо стандартом; поддержка генерационным backend проверяется отдельно.

## Спекулятивное декодирование

- [Speculative decoding](https://arxiv.org/abs/2211.17192v2), [speculative sampling](https://arxiv.org/abs/2302.01318): доказательство через принятую массу min(p,q) и нормированный положительный остаток. Температура, маски и отбор входят в фактические p/q.
- [LayerSkip](https://arxiv.org/abs/2404.16710v4), [Medusa](https://arxiv.org/abs/2401.10774v3), [EAGLE](https://arxiv.org/abs/2401.15077v3), [DeepSeek-V3 MTP](https://arxiv.org/abs/2412.19437v2): место предложений и обусловливание. Их наличие в статье не подтверждает произвольную поддержку движком.
- [DSpark](https://arxiv.org/abs/2607.05147v1), 6 июля 2026: полуауторегрессионная подготовка, лёгкая последовательная часть, длина по survival и профилю движка. Диапазон роста пользовательской скорости 60–85% относительно MTP-1 при сопоставимой общей пропускной способности приведён с атрибуцией авторам, а не как опыт курса.
- [Tree adaptation DeepSeek-V4](https://arxiv.org/abs/2609.24698v1), 21 сентября 2026: изоляция временного сжатого состояния и обновление принятого пути. Не сделан вывод, что каждый рабочий запрос сервиса использует это дерево.

## Сжатие и эксплуатация

- [LLM.int8()](https://arxiv.org/abs/2208.07339v2), [SmoothQuant](https://arxiv.org/abs/2211.10438v7), [GPTQ](https://arxiv.org/abs/2210.17323v2), [AWQ](https://arxiv.org/abs/2306.00978v6): выделение измерений, согласованное масштабирование и локальная реконструкция. AWQ v6 датирована 25 апреля 2026. Рейтинг форматов из названий не выводится.
- [OmniQuant](https://arxiv.org/abs/2308.13137v3), [BitNet b1.58](https://arxiv.org/abs/2402.17764), [Atom](https://arxiv.org/abs/2310.19102v3): обучение параметров калибровки, специальная параметризация обучения и малобитное обслуживание не смешиваются.
- [Определения ggml](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h), [imatrix](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix), [ExLlamaV2](https://github.com/turboderp-org/exllamav2): кодирование, статистика входов и исполняющая реализация. Открытые страницы не фиксируют будущую версию ветки master; точный эксперимент потребует commit и runtime.
- [Wanda](https://arxiv.org/abs/2306.11695v3), [SparseGPT](https://arxiv.org/abs/2301.00774v3), [LLM-Pruner](https://arxiv.org/abs/2305.11627v3), [SliceGPT](https://arxiv.org/abs/2401.15024v2), [ShortGPT](https://arxiv.org/abs/2403.03853v3), [Lottery Ticket](https://arxiv.org/abs/1803.03635v5): критерии важности, согласованные размеры, восстановление и условия повторного обучения. Процент нулей не приравнен к ускорению.
- [Docker multi-stage](https://docs.docker.com/build/building/multi-stage/), [Kubernetes probes](https://kubernetes.io/docs/concepts/workloads/pods/probes/), [HPA](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/#algorithm-details): выборочная передача runtime-артефактов, готовность и асинхронное масштабирование. Страницы Kubernetes при проверке показывали v1.37; работающий кластер не поднимался.
- [NVIDIA MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/introduction.html), [MPS](https://docs.nvidia.com/deploy/mps/latest/architecture.html): профили, совместное выполнение и зависящие от версии возможности. MPS не объявляется синонимом time-slicing или неизменно отсутствующей изоляцией.

## Границы проверки

Прочитаны первичные описания и актуальные интерфейсы, самостоятельно пересчитаны учебные числа. Собственная тренировка LoRA, выбор лучшего ранга, нагрузочное измерение GPU, точность квантизованных LLM и развёртывание Kubernetes не выполнялись. Эти действия сформулированы в главах как практикум, а не отчёт о выполненном benchmark.
