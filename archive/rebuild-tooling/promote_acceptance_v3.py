"""Explicitly promote already inspected reports; never run from normal build."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[2]
def read(path):return json.loads(path.read_text(encoding='utf8'))
def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
html_hash=hashlib.sha256((ROOT/'Читать конспекты.html').read_bytes()).hexdigest()
browser=read(ROOT/'.qa-results/browser/browser-audit.json')
portable=read(ROOT/'.qa-results/portable-checkout.json')
audit=read(ROOT/'.qa-results/current/audit.json')
assert browser['status']==portable['status']==audit['status']=='passed'
assert browser['htmlSha256']==portable['htmlSha256']==html_hash,'Reports belong to another HTML revision'
history=ROOT/'archive/2026-10-20-acceptance-v2';history.mkdir(exist_ok=True)
for name in ['audit.json','browser-audit.json']:
    dest=history/name
    if not dest.exists():
        original=ROOT/'editorial'/name
        assert 'htmlSha256' not in read(original),'Historical report has already been replaced'
        shutil.copy2(original,dest)
old_browser=read(history/'browser-audit.json')
for name in old_browser.get('screenshots',[]):
    dest=history/'qa'/name;dest.parent.mkdir(exist_ok=True)
    if not dest.exists():shutil.copy2(ROOT/'editorial/qa'/name,dest)
(history/'README.md').write_text('# Исторические свидетельства v2\n\nОтчёты и снимки прежней рабочей проверки Windows сохранены до замены текущими свидетельствами. Они не подтверждали воспроизводимость чистого клона на Windows или Linux. Прежние содержательные заявления читаются с оговоркой в `editorial/acceptance-v2.md`.\n',encoding='utf8',newline='\n')
p=ROOT/'editorial/acceptance-v2.md';text=p.read_text(encoding='utf8')
if not text.startswith('# Исторический отчёт v2'):
    text='# Исторический отчёт v2\n\nЭтот текст сохраняет прежние заявления, часть которых оказалась неподтверждённой. Проверки выполнялись в рабочем окружении Windows; переносимость чистого клона и полная содержательная редактура тогда не были подтверждены. Текущий статус находится в [приёмке v3](<acceptance.md>).\n\n'+text
    text=text.replace('(<audit.json>)','(<../archive/2026-10-20-acceptance-v2/audit.json>)').replace('(<browser-audit.json>)','(<../archive/2026-10-20-acceptance-v2/browser-audit.json>)').replace('(<qa/','(<../archive/2026-10-20-acceptance-v2/qa/')
    p.write_text(text,encoding='utf8',newline='\n')
dump(ROOT/'editorial/audit.json',audit);dump(ROOT/'editorial/browser-audit.json',browser);dump(ROOT/'editorial/portable-checkout.json',portable)
for name in browser['screenshots']:shutil.copy2(ROOT/'.qa-results/browser/qa'/name,ROOT/'editorial/qa'/name)
cards=read(ROOT/'editorial/chapter-reviews.json')['chapters']
closed=[ident for ident,card in cards.items() if card['content_acceptance']=='closed']
rewritten=sum(card['prose']=='rewritten' for card in cards.values())
assert len(closed)==22 and rewritten==27,'Update the editorial description after further review'
text=f'''# Приёмка редакции v3: работа продолжается

Срез проверки: **21 октября 2026**. Прежние результаты относились к рабочему окружению Windows; чистый клон тогда не проверялся. [Исторический текст v2](<acceptance-v2.md>) и его отчёты сохранены отдельно.

## Автоматические проверки

- [Аудит](<audit.json>): ошибок нет. Сохранность архива, привязки сайдкара, ссылки, маршруты, старые закладки и ресурсы проверены отдельно от редакторского чтения.
- `node tools/build_course.cjs --check`: соответствуют все 38 генерируемых файлов. Команда ничего не записывает.
- [Чистый локальный клон Windows](<portable-checkout.json>): тестовый репозиторий создан из текущих файлов редакции; LF и CRLF проходят аудит и `--check`. Две последовательные сборки совпадают, отслеживаемых изменений в тестовом клоне нет.
- `tools/test_portability.py`: четыре проверки, включая смешанные окончания строк, сохранение BOM/пробелов, обнаружение содержательного изменения текста/SVG и точную проверку бинарных данных.
- `tools/test_worked_examples.py`: 21 проверка числовых примеров. Десять дополнительных арифметических проверок выполняет сам аудит.
- [Браузерный отчёт Windows/Edge](<browser-audit.json>): открыты 121 глава и шесть проектов; проверены 7078 исходных якорей, старые закладки, маршруты, увеличиваемые схемы, автономная перезагрузка и мобильный вид. Внешних HTTP-запросов и ошибок JavaScript нет.

Проверенный HTML имеет SHA-256 `{html_hash}`. Сборщик, браузер и проверка клона считают одни байты, включая завершающий LF. Проверки записывают временные результаты в указанный каталог; текущие свидетельства заменяются отдельным редакторским действием.

Workflow для Windows/Linux добавлен в `.github/workflows/course.yml`. **Успешный запуск на Linux ещё не подтверждён.** Локальная проверка LF/CRLF не эмулирует другую ОС.

## Что исправлено в инструментах и источниках

Текст и SVG хранятся с LF по `.gitattributes`; бинарные ресурсы не нормализуются. Архивные манифесты версии 2 проверяют SHA-256 после единственной замены CRLF → LF. Первоначальные байтовые хэши сохранены отдельно. Перед переходом исходный срез сопоставлен с Git. Старый HTML имел смешанные окончания строк; его нормализованное содержимое совпало с Git, поэтому восстановление CRLF по всему файлу не использовалось.

Технические обёртки удалены из активного учебного Markdown. Диапазоны и хэши находятся в `coverage.json`, старые якоря добавляются при сборке. Сохранность оригинала и содержательное покрытие новой редакцией имеют разные статусы. Устаревший диапазон останавливает сборку; неоднозначное назначение не исправляется автоматически.

Активные главы используют локальные рабочие изображения, без зависимости исполнения от архивных ресурсов. Сценарии — авторские данные; сборщик сохраняет списки, формулы и код и не выводит цель главы из первых абзацев. Новый рисунок BERT из первоисточника имеет атрибуцию и лицензионную запись в [реестре иллюстраций](<illustrations.json>).

## Редакторское чтение и визуальная проверка

Полностью заново изложены **27 глав**; остальные 94 прошли общий проход терминологии и удаления невидимого контекста. Наличие переписанного текста не закрывает его сверку с источниками.

Содержательная сверка закрыта для **22 из 121 главы**: attention; Transformer-блок; семейства архитектур; обучение; вводная BERT, подробный интерфейс и цели предобучения BERT; числовые преобразования; отбор признаков; все 13 глав кластеризации. В кластеризации сверены 253 существенные исходные единицы: геометрия, проекции, четыре семейства алгоритмов, метрики, устойчивость, замороженные представления, DEC/IDEC, изображения, тексты и системные применения. Для них [свидетельства чтения](<review-evidence.md>) перечисляют отдельные исходные случаи, исправления, расчёты и ограничения. Индивидуальные назначения сохранены в [карте покрытия](<coverage.md>).

Визуально просмотрены новые схемы [слоя BERT](<qa/bert-layer.png>), [входных представлений](<qa/bert-input-layers.png>), [независимого/совместного кодирования](<qa/bert-retrieval.png>), [K-Means](<qa/cl-kmeans-2d.png>), [дендрограмм](<qa/cl-linkage.png>), [плотности](<qa/cl-density.png>), [EM](<qa/cl-em.png>), [масштабирования](<qa/feat-numeric.png>), [отбора](<qa/feat-selection.png>), [управляющей MLP](<qa/gated-mlp.png>), [семейств Transformer](<qa/transformer-families.png>) и [обучения](<qa/training-paths.png>). Дополнительно просмотрены [геометрия](<qa/cl-geometry.png>), [проекции](<qa/cl-projection.png>), [метрики](<qa/cl-metrics.png>), [устойчивость](<qa/cl-stability.png>), [фазы DEC](<qa/cl-deep.png>), [ветви IDEC](<qa/cl-idec-branches.png>), [изображения и SLIC](<qa/cl-images.png>), [текстовые модели](<qa/cl-text.png>) и [IVF](<qa/cl-systems.png>). Исправлены пересечения подписей и несоответствия рисунка тексту. В главе целей BERT просмотрены 14 рисунков: [разветвление входа и целей](<qa/objective-mlm-targets.png>), [общие головы](<qa/objective-shared-heads.png>), [SpanBERT](<qa/objective-spanbert.png>), [ELECTRA](<qa/objective-electra.png>), [ALBERT](<qa/objective-albert.png>), [DistilBERT](<qa/objective-distilbert.png>), [DeBERTa-v3](<qa/objective-deberta.png>) и отдельные схемы сравниваемых кодировщиков. Проверен выход из сравнительной ветки к прикладному разделу. Автоматический мобильный прогон не заменяет предметного просмотра каждой длинной главы.

## Оставшаяся работа

**99 карточек остаются открытыми.** Требуются сверка оставшихся исходных случаев, чтение итоговых глав и предметная проверка иллюстраций. Общий проход терминов не означает связного переписывания всех глав. Повторное сквозное чтение независимых маршрутов и приёмка всего курса ещё не закрыты.

Также остаётся подтвердить чистый клон и браузерную проверку на Linux через CI. Запуск обучения моделей, пользовательские benchmarks и выполненная учащимся практика не проводились.

Инвентаризация текущего аудита: пять модулей, 121 глава, 7412 исходных единиц, {audit['counts']['figures']} используемых SVG. Эти числа описывают состав материалов; основание содержательной приёмки — чтение и свидетельства конкретных объяснений.
'''
(ROOT/'editorial/acceptance.md').write_text(text,encoding='utf8',newline='\n')
# Convert active text resources without changing archive originals or any other bytes.
for p in [ROOT/'.gitignore',ROOT/'editorial/migration-complete.json',*(ROOT/'оформление/transformer').glob('*.svg')]:
    data=p.read_bytes();canonical=data.replace(b'\r\n',b'\n')
    if data!=canonical:p.write_bytes(canonical)
print('Promoted matching reports; 22 closed cards, 99 open; Linux remains unconfirmed.')
