"""Final reading corrections: precise figure descriptions and named consumers."""
import json,sys,re,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from edit_chapter import load,save,transform
course=json.loads((ROOT/'course.json').read_text(encoding='utf8'));by={c['id']:c for c in course['chapters']}
batch=json.loads((ROOT/'editorial/current-batch.json').read_text(encoding='utf8'));cov=load()
uses={
'adapt-variants':'выбор геометрии обновления и бюджета хранения; завершение сравнительной ветки методов PEFT',
'adapt-alternatives':'выбор места параметров при фиксированной основе; завершение ветки альтернатив LoRA',
'adapt-merging':'выпуск одной смеси либо обслуживание нескольких адаптеров; завершение ветки объединения',
'adapt-distillation':'подготовка ученика по выбранному интерфейсу учителя; завершение ветки переноса сигнала',
'adapt-continual':'оценка новых и прежних задач при последующих обновлениях; завершение ветки последовательного обучения',
'adapt-evaluation':'приёмка адаптированного комплекта перед обычным инференсом и выпуском',
'infer-context':'проектирование входа, префиксного кеша и проверки длинных запросов перед обслуживанием',
'infer-speculative':'проверка точности и стоимости ускоренного декодирования перед выпуском',
'infer-quantization':'выбор числового представления и проверка сжатого артефакта перед развёртыванием',
'infer-pruning':'выбор структуры сжатого графа и проверка исполняемого экспорта',
'infer-deployment':'мониторинг, нагрузочная проверка и организация приложений после выпуска'
}
for ident in batch['chapters']:
 rel=by[ident]['path']
 def change(t):
  # Keep destinations and prose wording; refresh only a link whose entire label
  # is an obsolete registry title. Descriptive authored labels remain unchanged.
  old_titles={'SFT: данные, шаблоны, masks и objective','Полное обучение или PEFT','LoRA: low-rank обновление и градиенты',
   'AI feedback и continual learning','Prefill, decode, KV-cache, MQA и GQA','Контекст, sliding window и prefix caching',
   'Greedy, temperature, top-p, beam и ограничения','Serving, batching и PagedAttention','Speculative decoding и rejection sampling',
   'Deployment, cold start, очереди и масштабирование'}
  def label(m):
   target=(ROOT/rel).parent.joinpath(m[2]).resolve()
   ch=next((x for x in by.values() if (ROOT/x['path']).resolve()==target),None)
   return f'[{ch["title"]}](<{m[2]}>)' if ch and m[1] in old_titles else m[0]
  t=re.sub(r'\[([^\]]+)\]\(<([^>]+)>\)',label,t)
  if ident in uses:t=t.replace('прикладная проверка выбранной ветки; переходы указаны в конце главы',uses[ident])
  if ident=='adapt-choice':t=t.replace('Слева схема сохраняет вход, эталон и ошибку. Ветви изменения контекста, обучающих данных и параметров сходятся только при общей оценке. Последний блок выбирает: сохранить кандидата, исправить гипотезу или откатить изменение.',
   'Верхний блок фиксирует вход, эталон и ошибку базовой системы. Три ветви проверяют доступность факта, формат или поведение и бюджет исполнения. Сравнение на общем тесте позволяет сохранить кандидата, исправить гипотезу или откатить изменение.')
  if ident=='infer-kv':
   t=t.replace('MHA имеет отдельные K/V для голов запросов; MQA делит одну KV-голову;',
    'Обычное многоголовое внимание (multi-head attention, MHA) имеет отдельные K/V для голов запросов; внимание с общими ключами и значениями (multi-query attention, MQA) делит одну KV-голову;')
   t=t.replace('![32 запроса и разные варианты совместного использования KV-голов]',
    '![Четыре головы запросов в учебных схемах MHA, GQA и MQA]')
   t=t.replace('[GQA](https://arxiv.org/abs/2305.13245) задаёт несколько групп','сгруппированное внимание ([grouped-query attention, GQA](https://arxiv.org/abs/2305.13245)) задаёт несколько групп')
   t=t.replace('**Prefill** обрабатывает','**Обработка входа (prefill)** обрабатывает').replace('проходом **decode**','проходом **декодирования (decode)**')
  if ident=='infer-quantization':
   t=t.replace('**PTQ** готовит','**Квантизация после обучения (post-training quantization, PTQ)** готовит')
   t=t.replace('**QAT** включает','**Обучение с имитацией квантизации (quantization-aware training, QAT)** включает')
   t=t.replace('**STE** задаёт','**Суррогатная передача производной (straight-through estimator, STE)** задаёт')
   t=t.replace('Сырой payload','Сырой объём кодов весов (payload)')
  if ident=='infer-pruning':t=t.replace('**Pruning** удаляет','**Прореживание (pruning)** удаляет')
  if ident=='adapt-preference':t=t.replace('через KL;','через расхождение Кульбака—Лейблера (KL);')
  if ident=='infer-deployment':t=t.replace('пользовательскому SLO','пользовательскому целевому уровню обслуживания (service level objective, SLO)')
  return t
 transform(rel,change,'Итоговое чтение: уточнены подписи, применение главы и русские названия; адреса источников и числовые условия сохранены.',cov)
save(cov)
