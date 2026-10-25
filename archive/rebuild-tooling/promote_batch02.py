"""Publish matching evidence for the reviewed batch; preserve earlier reports."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def dump(p,o):(ROOT/p).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
batch=read('editorial/current-batch.json');assert batch['id']=='lm-batch-02'
h=hashlib.sha256((ROOT/'Читать конспекты.html').read_bytes()).hexdigest()
browser=read('.qa-results/batch02/browser/browser-audit.json');portable=read('.qa-results/portable-checkout.json')
audit=read('.qa-results/batch02/current/audit.json');layout=read('.qa-results/batch02/layout/layout.json')
bounds=read('.qa-results/batch02/figure-bounds.json')
assert all(r['status']=='passed' for r in [browser,portable,audit,layout,bounds])
assert browser['htmlSha256']==portable['workingHtmlSha256']==layout['htmlSha256']==h
assert len(browser['batchChapters'])==20 and len(browser['batchFigures'])==37 and len(layout['chapters'])==20
cards=read('editorial/chapter-reviews.json')['chapters'];assert len(cards)==168
assert sum(c['content_acceptance']=='closed' for c in cards.values())==112
assert all(cards[k]['content_acceptance']=='closed' and cards[k]['source_units_pending_semantic_review']==0 for k in batch['chapters'])
dest=ROOT/'editorial/qa/batch02';dest.mkdir(parents=True,exist_ok=True)
for name in browser['batchFigures']:
 assert Path(name).name==name;shutil.copyfile(ROOT/'.qa-results/batch02/browser/qa'/name,dest/name)
for name in layout['screenshots']:
 assert Path(name).name==name;shutil.copyfile(ROOT/'.qa-results/batch02/layout'/name,dest/name)
published_browser={**browser,'protocolScreenshotCount':len(browser['screenshots']),
 'screenshots':browser['batchFigures'],'screenshotRoot':'qa/batch02',
 'screenshotSelection':'37 illustrations of the reviewed batch; general protocol screenshots remain in temporary QA results.'}
for name,report in [('audit',audit),('browser',published_browser),('portable',portable),('layout',layout),('figure-bounds',bounds)]:
 dump(f'editorial/batch02-{name}.json',{**report,'published_on':'2026-10-25'})
text=f'''# Приёмка пакета 02 — 25 октября 2026

Закрыты двадцать карточек: все 11 глав адаптации и девять глав инференса до развёртывания включительно. Глава мониторинга остаётся на следующий пакет. [Содержательное чтение](<batch02-review.md>) и [первичные источники](<batch02-research.md>) фиксируют конкретные исправления и границы выводов.

## Что переработано

Вместо склеенных слоёв написаны связные главы, отдельные сценарии и практикумы. Сверены 1312 исходных единиц; сохранены их ID и старые якоря. Новая авторская глава режимов обновления связывает SFT и LoRA. Раскрыты прямые/обратные пути адаптеров, разные цели предпочтений, перенос сигнала, контекст и кеш, планирование, точная спекулятивная коррекция, числовое/структурное сжатие и жизненный цикл запроса.

Подготовлены и просмотрены 37 авторских SVG. Исправлен обнаруженный независимым расчётом tie в квантизации: 1.3 при шаге .2 и округлении половин от нуля восстанавливается как 1.4; MSE примера остаётся .005. Иллюстрации, формулы и текст согласованы.

## Автоматические проверки

- [Аудит](<batch02-audit.json>): ошибок нет; 168 глав, пять модулей, 17 маршрутов, 153 архивных файла, привязки сайдкара и локальные ресурсы. В остальных материалах сохраняются {audit['content_review']['pending_spans']} pending-диапазонов; технический аудит не закрывает их.
- `node tools/build_course.cjs --check`: соответствуют 56 генерируемых файлов. Команда не пишет результаты.
- `tools/test_worked_examples.py`: 73 проверки, из них 34 новые группы адаптации/инференса. `test_portability.py`: четыре; `test_cv_expansion.py`: 14 групп. Учебные вычисления не являются обучением или benchmark модели.
- [Общий браузерный протокол Edge](<batch02-browser.json>): 168 глав, шесть проектов, 7078 исходных якорей, старые закладки, 37 рисунков пакета, увеличение и offline reload; HTTP-запросов и ошибок JavaScript нет.
- [Мобильная вёрстка и длинные главы](<batch02-layout.json>): проверены все 20 глав при ширине 390 px; сделаны шесть дополнительных снимков текста на desktop/mobile. Длинная формула прокручивается внутри своего блока, ширина страницы не увеличивается.
- [Границы подписей](<batch02-figure-bounds.json>): у 37 SVG нет выхода текста за рисунок или подписанный прямоугольник. Эта геометрическая проверка отделена от предметного просмотра.
- [Временный чистый клон Windows](<batch02-portable.json>): LF и CRLF проходят аудит и `--check`, две последовательные сборки одинаковы, отслеживаемых изменений нет. Индекс основного проекта не менялся.

SHA-256 проверенного локального HTML: `{h}`.

SHA-256 переносимого HTML: `{portable['htmlSha256']}`. Клон исключает {portable['excludedLocalExampleCount']} игнорируемых локальных примеров проектов; для его иной входной инвентаризации читатель пересобран. Оба хэша и причина различия записаны явно.

Снимки: [LoRA](<qa/batch02/batch02-adapt-lora-1.png>), [QLoRA](<qa/batch02/batch02-adapt-variants-3.png>), [спекулятивная коррекция](<qa/batch02/batch02-infer-speculative-2.png>), [DSpark](<qa/batch02/batch02-infer-speculative-4.png>), [квантизация на мобильном экране](<qa/batch02/infer-quantization-mobile-text.png>), [очередь и реплики](<qa/batch02/batch02-infer-deployment-3.png>). Полные перечни — в отчётах.

## Статус курса и ограничения

Закрыты **112 из 168 карточек**, открыты **56**. В это число входят 60 карточек CV, закрытых [прямым решением пользователя](<cv-user-acceptance.md>); независимая экспертная проверка CV не приписана этой партии. Остальные карточки не закрываются наличием метаданных или успешной сборкой.

Linux CI пока не подтверждён. Повторное чтение всех независимых маршрутов всего курса остаётся общей задачей. В пакете проверены свои предпосылки, переходы и выходы из углублений; обычный инференс не требует реализации LoRA. Обучение моделей, нагрузочные GPU-измерения, выполненная учащимся практика и рабочее развёртывание Kubernetes не проводились.
'''
(ROOT/'editorial/batch02-acceptance.md').write_text(text,encoding='utf8',newline='\n')
batch.update(status='complete',accepted_on='2026-10-25',remaining=[],acceptance='editorial/batch02-acceptance.md',
 limitations=['Linux CI не подтверждён; остаётся 56 карточек и общая приёмка независимых маршрутов.'])
dump('editorial/current-batch.json',batch);dump('editorial/batches/lm-batch-02.json',batch)
p=ROOT/'editorial/acceptance.md';old=p.read_text(encoding='utf8')
section='''## Пакет 02 — 25 октября 2026

Завершены двадцать карточек: 11 глав адаптации и девять глав инференса. Сверены 1312 исходных единиц, написаны предметные сценарии и практикумы, подготовлены 37 авторских SVG. Текущий статус — **112 из 168 карточек закрыты, 56 открыты**. [Содержательная проверка](<batch02-review.md>) и [техническая приёмка](<batch02-acceptance.md>) разделены; CV учтён по решению пользователя.

Аудит, 56 генерируемых файлов, 73 числовые проверки, четыре теста переносимости, 14 регрессионных групп CV, Edge/offline, мобильная вёрстка 20 глав и чистый LF/CRLF-клон Windows проходят. Linux CI и общая приёмка оставшихся материалов не объявляются выполненными. Предыдущие статусы ниже сохранены как история.

'''
if '## Пакет 02 — 25 октября 2026' not in old:old=old.replace('## Решение пользователя по CV — 25 октября 2026',section+'## Решение пользователя по CV — 25 октября 2026',1)
p.write_text(old,encoding='utf8',newline='\n')
print('Published batch 02; 112/168 cards closed; 43 reviewed screenshots; matching HTML '+h)
