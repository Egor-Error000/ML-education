"""Publish matching batch reports without replacing the separate CV acceptance."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2]
def read(rel):return json.loads((ROOT/rel).read_text(encoding='utf8'))
def dump(rel,value):(ROOT/rel).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
html_hash=hashlib.sha256((ROOT/'Читать конспекты.html').read_bytes()).hexdigest()
browser=read('.qa-results/batch01/browser/browser-audit.json')
portable=read('.qa-results/portable-checkout.json')
audit=read('.qa-results/batch01/current/audit.json')
assert browser['status']==portable['status']==audit['status']=='passed'
assert browser['htmlSha256']==portable.get('workingHtmlSha256',portable['htmlSha256'])==html_hash
assert browser['chaptersVisited']==audit['counts']['chapters']==168
assert len(browser['batchChapters'])==10 and len(browser['batchFigures'])==32
cards=read('editorial/chapter-reviews.json')['chapters']
batch=read('editorial/current-batch.json')
assert all(cards[k]['content_acceptance']=='closed' and cards[k]['source_units_pending_semantic_review']==0 for k in batch['chapters'])
closed=sum(c['content_acceptance']=='closed' for c in cards.values())
assert closed==32 and len(cards)==168
for name,value in [('batch01-audit.json',audit),('batch01-browser.json',browser),('batch01-portable.json',portable)]:
    value={**value,'published_on':'2026-10-25'}
    if name=='batch01-browser.json':
        # Preserve the full raw report in the temporary results; publish the batch
        # screenshots explicitly rather than duplicating all historical QA PNGs.
        value.update(protocolScreenshotCount=len(browser['screenshots']),
                     screenshots=browser['batchFigures'],screenshotRoot='qa/batch01',
                     screenshotSelection='32 illustrations of the ten reviewed chapters; other protocol screenshots remain in temporary results.')
    dump('editorial/'+name,value)
dest=ROOT/'editorial/qa/batch01';dest.mkdir(parents=True,exist_ok=True)
for name in browser['batchFigures']:
    assert Path(name).name==name
    shutil.copyfile(ROOT/'.qa-results/batch01/browser/qa'/name,dest/name)
review=f'''# Приёмка пакета 01 — 25 октября 2026

Завершены десять карточек: токенизация, состояния, позиции; адаптация BERT, прикладные головы, векторы предложений, поиск, языки/документы, оценка и эксплуатация. [Свидетельство чтения](<batch01-review.md>) перечисляет сохранённые исходные случаи, исправления, расчёты и ограничения. Источники для написания проверялись 21 октября; дата этого отчёта относится к итоговой приёмке.

## Содержание

Сверены 645 существенных исходных единиц; их идентификаторы и старые закладки сохранены в [карте покрытия](<coverage.md>). В сценариях заданы вопрос, результат, порядок, определения, формулы с предпосылками, пример, практика, потребители и ограничения. Подробная ветка BERT теперь целиком прошла авторскую содержательную приёмку; вводная и цели предобучения были приняты ранее.

Подготовлены 28 новых или переработанных авторских SVG. Вместе с четырьмя ранее существовавшими рисунками предметно просмотрены все 32 иллюстрации десяти глав. Объяснены, в частности, формы и головы BERT, нормировка CRF против Витерби, допустимые пары QA, общий контрастивный пул, SimCSE, Matryoshka, SPLADE, ColBERT, BGE-M3, Longformer, SciBERT/CodeBERT, пространственные входы LayoutLM, IG, очередь и миграция индекса. Утверждения о реальном ускорении не выводятся из размеров массивов.

## Автоматические проверки текущего курса

- [Аудит](<batch01-audit.json>): ошибок нет; 168 глав, пять модулей, 17 маршрутов, сохранность архива, целевые диапазоны, ссылки и ресурсы. Оставшиеся pending-диапазоны других глав не скрываются.
- `node tools/build_course.cjs --check`: соответствуют 56 генерируемых файлов; команда не пишет результаты.
- `python -X utf8 tools/test_worked_examples.py`: 39 числовых проверок, включая 18 проверок этого пакета. `tools/test_portability.py`: четыре проверки. Регрессионный `tools/test_cv_expansion.py`: 14 групп проверок.
- [Edge, общий браузерный протокол](<batch01-browser.json>): 168 глав и шесть проектов, 7078 исходных якорей, старые закладки, 32 рисунка пакета, увеличение, offline reload и мобильный вид. Внешних HTTP-запросов и ошибок JavaScript нет.
- `tools/check_batch_figures.cjs`: у 28 авторских SVG проверены границы текста; выходов за SVG или прямоугольники нет. Это геометрическая проверка подписей, отдельная от содержательного просмотра.
- [Временный чистый клон Windows](<batch01-portable.json>): аудит и `--check` проходят при LF и CRLF; две последовательные сборки совпадают, отслеживаемых изменений нет. Git-индекс основного проекта не менялся.

SHA-256 проверенного локального HTML: `{html_hash}`.

SHA-256 переносимого HTML: `{portable['htmlSha256']}`. В тестовом клоне отсутствуют {portable['excludedLocalExampleCount']} игнорируемых локальных примеров проектов; его ожидаемый читатель пересобран для этой входной инвентаризации. Это объясняет различие хэшей, а не объявляет разные файлы одинаковыми. Флаг и оба хэша сохранены в отчёте клона.

Снимки опубликованы, например, для [голов задач](<qa/batch01/batch01-bert-tasks-1.png>), [CRF](<qa/batch01/batch01-bert-tasks-2.png>), [ColBERT](<qa/batch01/batch01-bert-retrieval-4.png>), [Longformer](<qa/batch01/batch01-bert-domains-2.png>) и [очереди](<qa/batch01/batch01-bert-operations-3.png>). Полный перечень 32 снимков находится в браузерном отчёте.

## Общий статус и ограничения

Закрыты **32 из 168 карточек**, открыты **136**. Расширение CV сохраняет [собственное свидетельство](<cv-expansion-review.md>): новые материалы написаны и самостоятельно вычитаны, но его независимое экспертное чтение не объявляется закрытым этим пакетом. Исторические отчёты для 121 главы сохранены отдельно.

Linux CI ещё не подтверждён. Повторное содержательное чтение остальных модулей и всех независимых маршрутов не завершено. Обучение моделей, измерение пользовательских benchmarks и выполненная учащимся практика не проводились. Число глав, рисунков и тестов описывает инвентаризацию и объём проверок, а не заменяет предметную приёмку.
'''
(ROOT/'editorial/batch01-acceptance.md').write_text(review,encoding='utf8',newline='\n')
batch.update(status='complete',accepted_on='2026-10-25',remaining=[],
    acceptance='editorial/batch01-acceptance.md',
    limitations=['Linux CI не подтверждён; приёмка остальных 136 карточек продолжается.'])
dump('editorial/current-batch.json',batch)
p=ROOT/'editorial/acceptance.md';text=p.read_text(encoding='utf8')
heading='## Пакет 01 — 25 октября 2026'
section='''## Пакет 01 — 25 октября 2026

Завершён пакет из десяти карточек основ и BERT: связный текст, сценарии, 645 исходных единиц, предметные схемы и вычисления. Закрыты **32 из 168 карточек**; остальные **136** открыты. [Содержательное свидетельство](<batch01-review.md>) и [техническая приёмка пакета](<batch01-acceptance.md>) разделены. Аудит, 56 генерируемых файлов, 39 числовых проверок, четыре теста переносимости, 14 CV-проверок, Edge и чистый клон LF/CRLF проходят. Linux ещё не подтверждён. Отчёты расширения CV ниже сохранены.

'''
if heading not in text:text=text.replace('## Расширение CV — 25 октября 2026',section+'## Расширение CV — 25 октября 2026',1)
text=text.replace('## Редакторское чтение и визуальная проверка\n',
                  '## Редакторское чтение исторического среза 21 октября\n')
text=text.replace('## Оставшаяся работа\n',
                  '## Оставшаяся работа на историческом срезе 21 октября\n')
# A form-feed was accidentally introduced in the earlier CV hash line.
# Replace it only in that exact line, with the verified SHA-256; no text cleanup.
lines=text.split('\n')
for i,line in enumerate(lines):
    if line.startswith('SHA-256 окончательного HTML:') and '\x0c' in line:
        lines[i]=line.replace('\x0c','f')
p.write_text('\n'.join(lines),encoding='utf8',newline='\n')
print('Published batch 01:',closed,'/',len(cards),'cards; CV reports preserved; Linux unconfirmed.')
