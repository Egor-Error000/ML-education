"""One-time migration from the immutable snapshot. Does not rebuild edited chapters.

After migration, course.json and the published Markdown files are the source of truth.
Use build_course.cjs for subsequent builds. --refresh is only for this migration's drafts.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import html
import json
import os
import re
import sys
from urllib.parse import unquote
from course_blueprint import MODULES, CHAPTERS, ROUTES
from migration_content import CONTENT, BRIDGES, EXERCISES
from migration_topics import topic

ROOT=Path(__file__).resolve().parents[2]
SNAP=ROOT/'archive/2026-10-20-before-rebuild'
FAMILY={'01-data':'data','02-regularization':'reg','03-categorical-features':'cat',
        '04-clustering':'cluster','05-computer-vision':'cv','07-bert':'bert'}
LM_FAMILY={'01-foundations':'found','02-adaptation':'adapt','03-inference':'infer','04-applications':'app','05-lifecycle':'agent'}
BY_ID={c['id']:c for c in CHAPTERS}
SIMPLE={c['source']:c['id'] for c in CHAPTERS if c['source']}
UNITS=[]
TARGET_BLOCKS=defaultdict(list)
SOURCE_DEFAULT={}
SOURCES={}
EDITORIAL=[]

def sha(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def write(path,text):
    path=ROOT/path
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text.rstrip()+'\n',encoding='utf-8')
def rel_link(origin,target):return Path(os.path.relpath(ROOT/target,(ROOT/origin).parent)).as_posix()
def source_key(p):
    parts=p.relative_to(SNAP).parts
    family=LM_FAMILY[parts[1]] if parts[0]=='06-llm' else FAMILY[parts[0]]
    letter='00' if p.name.startswith('00 ') else p.name[0]
    return family+'-'+letter

def split_blocks(lines,start,end):
    """Keep fenced code and display math intact; retain exact source line ranges."""
    index=start
    while index<end:
        while index<end and not lines[index].strip():index+=1
        if index==end:break
        a=index;fence=None;math=False
        while index<end:
            line=lines[index]
            if line.lstrip().startswith(('```','~~~')):
                mark=line.lstrip()[:3]
                fence=None if fence==mark else mark if fence is None else fence
            if fence is None and line.strip()=='$$':math=not math
            index+=1
            if not fence and not math and (index==end or not lines[index].strip()):break
        yield a,index,'\n'.join(lines[a:index]).strip()

def sections(p):
    lines=p.read_text(encoding='utf-8-sig').splitlines()
    headings=[i for i,l in enumerate(lines) if re.match(r'^#{2,6} ',l)]
    starts=[0]+headings+[len(lines)]
    top=0
    for a,b in zip(starts,starts[1:]):
        heading=lines[a] if a<len(lines) and re.match(r'^#{2,6} ',lines[a]) else ''
        if heading.startswith('## '):
            m=re.match(r'## (\d+)',heading);top=int(m[1]) if m else -1
        yield dict(a=a,b=b,heading=heading,top=top,lines=lines)

def destination(key,seg,text,line):
    h=seg['heading'];n=seg['top']
    if key=='reg-А' and n in [4,9]:return 'reg-theory'
    if key=='cv-А':
        if n in [4,5,6,8]:return 'cv-vit'
        if n==9:return 'cv-hierarchy'
        if n in [7,10,11]:return 'cv-pretraining'
        if n==12:return 'cv-pretraining' if re.search(r'SSL|DeiT|MAE|teacher|DINO',text) else 'cv-vit'
        return 'cv-classification'
    if key=='data-Б':
        if '2.4.' in h:return 'feat-numeric'
        if '4.3.' in h:
            return 'feat-autofe' if re.search(r'AutoFE|feature synthesis|candidate generator',text,re.I) else 'feat-selection'
    if key=='cluster-Б':
        return {1:'cl-kmeans',2:'cl-hierarchy',3:'cl-density',4:'cl-density',5:'cl-gmm',6:'cl-metrics'}.get(n,'cl-kmeans')
    if key=='cluster-Д':return 'cl-frozen' if n in [1,5] else 'cl-deep'
    if key=='infer-З':
        return {1:'app-state',2:'app-tools',3:'app-rag',4:'app-state',5:'app-evaluation'}.get(n,'app-state')
    if key=='infer-Е' and n==-1:return 'infer-generation'
    if key=='found-А':
        if n==1:
            if 'Почему lookup' in h:return 'lm-states'
            if 'Где заканчивается' in h:return 'lm-tokenizer'
            pivot=next((i for i in range(seg['a'],seg['b']) if seg['lines'][i].startswith('Пусть словарь')),seg['b'])
            return 'lm-states' if line>=pivot else 'lm-tokenizer'
        if n==2:return 'lm-position'
        if n==3:return 'lm-backward' if 'градиента CE' in h else 'lm-training'
        if n==4:return 'lm-generation'
        return 'lm-training'
    if key=='found-В':
        if n in [1,2]:return 'lm-transformer'
        if n==3:return 'lm-position'
        if n==4:return 'lm-families'
        if n==5:
            if '5.2.' in h:return 'lm-transformer'
            if '5.4.' in h:return 'lm-backward'
            if '5.6.' in h:
                return 'lm-bert' if text.startswith(('**BERT кратко','**Схема BERT')) else 'lm-families'
            return 'lm-training'
        if n==6:return 'lm-backward' if '6.4.' in h else 'lm-position'
        return 'lm-training'
    if key=='found-Г':
        if n in [1,2]:return 'lm-attention'
        if n==3:return 'lm-backward'
        if n==4:return 'lm-backward' if 'обучаются совместно' in h else 'lm-attention'
        if n==5:return 'lm-backward' if 'Backward' in h else 'lm-families'
        return 'lm-backward'
    if key=='found-Д':
        if n==1:return 'lm-states'
        if n==2:return 'lm-training' if 'Weight tying' in h else 'lm-output'
        if n==3:
            if 'loss mask' in h:return 'lm-training'
            if 'не заменяют K/V' in h:return 'infer-kv'
            return 'lm-generation'
        return 'lm-interpretation'
    return SIMPLE.get(key)

def old_index_target(old):
    if old.startswith('01-data/') or old.startswith('03-categorical-features/'):return '01-data-features/README.md'
    if old.startswith('02-regularization/'):return '02-regularization/README.md'
    if old.startswith('04-clustering/'):return '03-clustering/README.md'
    if old.startswith('05-computer-vision/'):return '04-computer-vision/README.md'
    if old.startswith('07-bert/'):return '05-language-models/02-bert-encoders/README.md'
    if old.startswith('06-llm/'):
        parts=old.split('/')
        fm=LM_FAMILY.get(parts[1])
        if fm:
            m=next(x for x in MODULES if x[0]=='lm');s=next(x for x in m[3] if x[0]==fm)
            return m[1]+'/'+s[1]+'/README.md'
        return '05-language-models/README.md'
    return None

def rewrite_links(text,old,new):
    # Support spaces in angle-bracket targets and ordinary percent-encoded targets.
    # A product such as [1-alpha+alpha(1-eta)^k](phi-a) inside TeX is not a link.
    protected=re.compile(r'(```[\s\S]*?```|~~~[\s\S]*?~~~|\$\$[\s\S]*?\$\$|\$[^$\n]+\$|`[^`\n]*`)')
    parts=protected.split(text)
    return ''.join(part if i%2 else re.sub(r'(\]\(<)([^>]+)(>\))|(\]\()([^\s)]+)(\))',lambda m:_link_sub(m,old,new),part) for i,part in enumerate(parts))

def _link_sub(m,old,new):
    groups=(m.group(1),m.group(2),m.group(3)) if m.group(1) else (m.group(4),m.group(5),m.group(6))
    pre,href,post=groups
    if re.match(r'^(?:[a-z][a-z0-9+.-]*:|#)',href,re.I):return m.group(0)
    path,_,fragment=unquote(href).partition('#')
    resolved=((ROOT/old).parent/path).resolve()
    if not resolved.is_relative_to(ROOT):target=resolved;fragment=fragment
    else:
        relative=resolved.relative_to(ROOT).as_posix()
        if relative in SOURCE_DEFAULT:target=BY_ID[SOURCE_DEFAULT[relative]]['path'];fragment=''
        elif relative.endswith('README.md') and old_index_target(relative):target=old_index_target(relative);fragment=''
        elif (SNAP/relative).exists():target=(SNAP/relative).relative_to(ROOT).as_posix()
        elif (ROOT/relative).exists() or relative in ['Читать конспекты.html','Учебные конспекты.md','README.md']:target=relative
        elif (SNAP/'root'/relative).exists():target=(SNAP/'root'/relative).relative_to(ROOT).as_posix()
        elif old_index_target(relative):target=old_index_target(relative)
        else:target=relative
    return pre+rel_link(new,target)+('#'+fragment if fragment else '')+post

def normalized_heading(text):
    if re.match(r'^#{2,6} ',text):
        lines=text.splitlines();lines[0]=re.sub(r'^(#{2,6}) (?:\d+(?:\.\d+)*\.?\s*|Вводная часть\.\s*)',r'\1 ',lines[0])
        return '\n'.join(lines)
    return text

def pedagogical_edit(text,key,new):
    """Remove editorial history from explanations without dropping the question."""
    replacements={
        'Возвращаем вопрос из чата о книгах и диалогах':'Книги и диалоги как обучающие данные',
        'В S7 дан одномерный пример':'Рассмотрим одномерный пример',
        'Вставленный в чате JSON':'Альтернативный JSON',
        'Так восстанавливается приведённое в чате приближение':'Получается приближение',
        'Так получается указанное в чате':'Получается',
        'Здесь стоит остановиться на вопросе из чата:':'Проверьте различие на вопросе:',
        'Вопрос чата о самостоятельном выборе subagents':'Вопрос о самостоятельном выборе участников',
        'В support-примере из чата':'В support-примере',
        'Ошибки из исходного обсуждения, которые не нужно повторять':'Ошибочные обобщения',
        'Числовой пример из чата с проверенной арифметикой':'Числовой пример с полным расчётом',
        'Memory strategies из чата':'Стратегии памяти',
        'из чата':'из разбираемого примера',
        'в чате':'в разбираемом примере',
        'из исходного обсуждения':'из разбираемой постановки',
        'В исходном обсуждении':'В разбираемой постановке',
        'Современный срез: 2024–2026':'Выбор encoder по механизму и ограничениям',
    }
    for old,replacement in replacements.items():text=text.replace(old,replacement)
    text=re.sub(r'\bВ чате\b','В разбираемом примере',text)
    text=text.replace('В S7','В разбираемом примере').replace('формулировка S7','формулировка примера').replace('строка старого чата с датой','неподтверждённая запись с датой')
    for a,b in {
        'В исходном S7':'В учебном примере',
        'Вопрос из S7':'Вопрос',
        'Запрос пользователя в S7 был точным: «сильно подробнее расписать шаг 2», желательно с полной математикой. Чтобы эта просьба не свелась к одной формуле, отдельно проследим адрес конкретного элемента.':'Проследим адрес конкретного элемента patch и все промежуточные индексы.',
        'Тезисы S7':'Утверждения',
        'Сохраним архитектурный пример S7':'Рассмотрим архитектурный пример',
        'Список S7 также связывал':'Варианты семейства связывают',
        'S7 описывал cascade как':'Ошибочная формулировка cascade:',
        'Модель из S7':'Монокулярная модель',
        'Другие направления из S7 и исправление Detic':'Другие способы связывать регионы и категории',
        'Архитектурные решения из S7 по конкретному ограничению':'Архитектурные решения по конкретному ограничению',
        'из S7':'из рассматриваемого примера',
        'Формула S7':'Формула',
        'фразы S7':'фразы',
        'В CV-чате S7 эти различия возникают':'Эти различия возникают',
        'дополнительное объяснение механизмов к S7;':'разбор механизмов;',
        'Фраза S7':'Фраза',
        'утверждение S7':'утверждение',
        'Формулировка S7':'Формулировка',
        'в S7 —':'здесь —',
        'заявления S7':'заявления',
        'В исходном SORT-примере S7 state':'В учебном SORT-примере state',
        'В примере S7':'В примере',
    }.items():text=text.replace(a,b)
    text=re.sub(r'(Контроль и практика) модулей?\s*\d+[–—-]\d+',r'\1',text)
    if key=='cv-З' and 'YOLO26 depth' in text:
        text='В [официальном описании YOLO26](https://docs.ultralytics.com/models/yolo26/) перечислены отдельные `*-depth` variants. Для depth нужны соответствующая task head, обученные веса и документированный контракт масштаба/единиц. Наличие общего backbone не превращает обычную detection head в depth estimator. Конкретный checkpoint оценивают depth-метриками, а не mAP; поддерживаемые входы и units проверяют по его model card.'
    family=key.split('-')[0]
    def chapter_ref(m):
        other=family+'-'+m.group(2)
        old=SOURCES.get(other)
        if not old:return m.group(0)
        target=BY_ID[SOURCE_DEFAULT[old]]
        return m.group(1)+' [«'+target['title']+'»](<'+rel_link(new,target['path'])+'>)'
    text=re.sub(r'(глав[аеуы]|главой)\s+([АБВГДЕЖЗИК])(?![А-Яа-я]|[–—-])',chapter_ref,text)
    return text

def ingest():
    files=[p for p in sorted(SNAP.rglob('*.md')) if p.parent.name=='Учебные конспекты' and p.name!='README.md']
    for p in files:
        key=source_key(p);old=p.relative_to(SNAP).as_posix();SOURCES[key]=old
        first=next(seg for seg in sections(p) if seg['heading'])
        SOURCE_DEFAULT[old]=destination(key,first,first['heading'],first['a'])
    duplicate={}
    for p in files:
        key=source_key(p);old=p.relative_to(SNAP).as_posix()
        for seg in sections(p):
            for a,b,text in split_blocks(seg['lines'],seg['a'],seg['b']):
                unit_id=sha(old+':'+str(a+1)+':'+text)[:18]
                record=dict(id=unit_id,source=old,source_lines=[a+1,b],source_sha256=sha(text),kind='heading' if text.startswith('#') else 'formula' if text.startswith('$$') else 'code' if text.startswith('```') else 'figure' if text.startswith('![') else 'explanation')
                substantive_preface=not seg['heading'] and (text.startswith(('**Схема','![')) or text.startswith('Общий encoder выдаёт'))
                if not seg['heading'] and not substantive_preface:
                    record.update(target='editorial/source-notes.md',anchor='u-'+unit_id,action='сохранить происхождение',status='archived-editorial')
                    EDITORIAL.append((record,text))
                else:
                    id=SOURCE_DEFAULT[old] if substantive_preface else destination(key,seg,text,a)
                    if not id:raise RuntimeError(f'Unassigned source {old}:{a+1}')
                    target=BY_ID[id]['path']
                    order,title=topic(key,seg['top'])
                    transformed=normalized_heading(text)
                    if title and text.startswith('## '):transformed='## '+title
                    # Rewrite existing links with the old file as base, then create
                    # new cross-chapter links with the new file as base.
                    transformed=pedagogical_edit(rewrite_links(transformed,old,target),key,target)
                    review=bool(re.search(r'Контроль|Самопровер|Проверь себя|Вопросы с ответами|небольшой расчёт',seg['heading'],re.I)) or order==9
                    record['topic_order']=90 if review else order
                    fingerprint=(id,sha(transformed))
                    if fingerprint in duplicate:
                        canonical=duplicate[fingerprint]
                        record.update(target=target,chapter=id,anchor=canonical['anchor'],action='объединить заголовки одной темы' if record['kind']=='heading' else 'объединить точный повтор',status='topic-heading-merged' if record['kind']=='heading' else 'exact-duplicate-verified',target_sha256=sha(transformed))
                    else:
                        record.update(target=target,chapter=id,anchor='u-'+unit_id,action='перенести с обновлением ссылок и изложения',status='transfer-verified',target_sha256=sha(transformed))
                        TARGET_BLOCKS[id].append((record,transformed))
                        duplicate[fingerprint]=record
                UNITS.append(record)
    return files

def overview_figure(c):
    """A pipeline figure is an actual vector asset, with explicit fitted/fixed labels."""
    from course_diagrams import chapter_diagram
    return chapter_diagram(ROOT,c)

def main():
    if (ROOT/'editorial/migration-complete.json').exists():
        raise SystemExit('Migration is locked after acceptance. Published Markdown and course.json are canonical; no files overwritten.')
    if (ROOT/'course.json').exists() and '--refresh' not in sys.argv:
        raise SystemExit('Migration already ran. Edit Markdown; use build_course.cjs. No files overwritten.')
    files=ingest()
    from course_diagrams import architecture_figures
    figures=architecture_figures(ROOT)
    for c in CHAPTERS:
        c['used_by']=[other['id'] for other in CHAPTERS if c['id'] in other['prerequisites']]
        c['source_files']=sorted(set(r['source'] for r,t in TARGET_BLOCKS[c['id']]))
        c['figures']=[]
        heading=f"# {c['title']}\n\n"
        req=', '.join(f"[{BY_ID[x]['title']}](<{rel_link(c['path'],BY_ID[x]['path'])}>)" for x in c['prerequisites']) or 'Векторы и матрицы, условная вероятность; пробелы закрываются по месту.'
        used=', '.join(f"[{BY_ID[x]['title']}](<{rel_link(c['path'],BY_ID[x]['path'])}>)" for x in c['used_by']) or 'Выбранная прикладная задача и итоговая проверка результата.'
        header=heading+f"**Уровень:** {c['level']}.\n\n**Предпосылки:** {req}\n\n**Используют:** {used}\n\n**Учебный вопрос:** {c['question']}\n\n"
        header+=BRIDGES.get(c['id'],'')+'\n\n'
        authored=CONTENT.get(c['id'],'')
        fig,caption=overview_figure(c);c['figures'].append(fig)
        body=header+authored+f"\n\n## Карта вычисления и границы этапов\n\n![{c['title']}: входы, преобразования и выходы](<{rel_link(c['path'],fig)}>)\n\n{caption}\n\n"
        records=sorted(TARGET_BLOCKS[c['id']],key=lambda pair:pair[0].get('topic_order',0))
        for record,text in (pair for pair in records if pair[0].get('topic_order',0)<90):
            body+=f"<!-- unit:{record['id']} -->\n<a id=\"{record['anchor']}\"></a>\n\n{text}\n\n<!-- /unit:{record['id']} -->\n\n"
        selected=figures.get(c['id'],[])
        if selected:
            body+='## Архитектуры: подробные схемы\n\n'
            for asset,title,explanation in selected:
                c['figures'].append(asset)
                body+=f'### {title}\n\n![{title}](<{rel_link(c["path"],asset)}>)\n\n{explanation}\n\n'
        for record,text in (pair for pair in records if pair[0].get('topic_order',0)>=90):
            body+=f"<!-- unit:{record['id']} -->\n<a id=\"{record['anchor']}\"></a>\n\n{text}\n\n<!-- /unit:{record['id']} -->\n\n"
        if c['id'] in EXERCISES:
            body+='## Самостоятельная практика\n\n**Практика:** '+EXERCISES[c['id']]+'\n\n'
        # New split chapters get their own observable checkpoint, even when a source
        # exercise is shared by several algorithms.
        body+=f"## Результат главы и переход дальше\n\nОтветьте на вопрос: **{c['question']}** Обоснуйте ответ конкретным вычислением из главы, назовите предположение и случай, когда оно нарушается.\n\n"
        if c['used_by']:
            body+='Следующие темы: '+', '.join(f"[{BY_ID[x]['title']}](<{rel_link(c['path'],BY_ID[x]['path'])}>)" for x in c['used_by'])+'.\n\n'
        body+=f'[Маршрут модуля](<{rel_link(c["path"],next(m[1] for m in MODULES if m[0]==c["module"])+"/README.md")}>) · [Происхождение и точные места переноса](<{rel_link(c["path"],"editorial/coverage.md")}>)\n'
        write(c['path'],body)
    notes='# Происхождение учебных материалов\n\nРедакционные вступления сохранены отдельно от учебного объяснения. Полный неизменённый срез — в архиве; отдельные содержательные блоки связаны с новой редакцией картой покрытия.\n\n'
    for r,text in EDITORIAL:
        notes+=f'<!-- unit:{r["id"]} -->\n<a id="{r["anchor"]}"></a>\n\n**Источник:** `{r["source"]}:{r["source_lines"][0]}`\n\n'+rewrite_links(text,r['source'],'editorial/source-notes.md')+f'\n\n<!-- /unit:{r["id"]} -->\n\n'
        r['target_sha256']=sha(rewrite_links(text,r['source'],'editorial/source-notes.md'))
    write('editorial/source-notes.md',notes)
    aliases={}
    families=['data','reg','cat','cluster','cv','found','adapt','infer','app','agent','bert']
    for ci,family in enumerate(families,1):
        group=[(k,p) for k,p in SOURCES.items() if k.startswith(family+'-')]
        group.sort(key=lambda kv:kv[1])
        module=next((c['module'] for c in CHAPTERS if c['id']==SOURCE_DEFAULT[group[0][1]]),None)
        if family in ['found','adapt','infer','app','agent','bert']:
            s=next(x for x in MODULES[-1][3] if x[0]==family)
            aliases[f'course-{ci}-0']='section-'+s[0]
        else:aliases[f'course-{ci}-0']='module-'+module
        for ni,(k,p) in enumerate(group,1):aliases[f'course-{ci}-{ni}']=SOURCE_DEFAULT[p]
    # Historical V0 was inserted between the module TOC and A; its legacy hash
    # is inspected from the snapshotted reader below rather than guessed.
    oldhtml=(SNAP/'root/Читать конспекты.html').read_text(encoding='utf-8-sig')
    for m in re.finditer(r'<article id="([^"]+)"[^>]*>\s*<div class="chapter-top">[\s\S]*?<a href="([^"]+)"',oldhtml):
        oldhref=unquote(m.group(2)).replace('\\','/')
        if oldhref in SOURCE_DEFAULT:aliases[m.group(1)]=SOURCE_DEFAULT[oldhref]
    manifest=dict(version=2,date='2026-10-20',modules=[dict(id=m[0],path=m[1],title=m[2],sections=[dict(id=s[0],path=s[1],title=s[2]) for s in m[3]]) for m in MODULES],chapters=CHAPTERS,routes=ROUTES,legacy_aliases=aliases,legacy_files=SOURCE_DEFAULT)
    write('course.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    write('editorial/coverage.json',json.dumps(dict(snapshot=SNAP.relative_to(ROOT).as_posix(),sources=len(files),units=UNITS),ensure_ascii=False,indent=2))
    report=['# Карта переноса содержательных блоков','',
        'Карта учитывает отдельные абзацы, формулы, код, вопросы, списки и схемы. Номер строки относится к неизменённому архивному срезу. Точный перенос подтверждается контрольной суммой; педагогическая проверка и проверка маршрутов описаны в отчёте приёмки. Старые редакционные вступления хранятся отдельно от учебного текста.','',
        'Исходные обсуждения S1–S7 и исправления связаны с главами через сохранённые вводные и архивные реестры. Эти координаты не являются отметками о выполненной практике.','']
    for c in CHAPTERS:
        report += [f'## {c["title"]}', '',f'Целевая глава: [{c["id"]}](<{rel_link("editorial/coverage.md",c["path"])}>).', '']
        if c['id'] in CONTENT:report+=['**Дополнение:** самостоятельное объяснение и пример, написанные при перестройке.','']
        for r in UNITS:
            if r.get('chapter')==c['id']:
                oldsrc=(SNAP/r['source']).relative_to(ROOT).as_posix()
                report.append(f'- [{r["source"]}:{r["source_lines"][0]}–{r["source_lines"][1]}](<{rel_link("editorial/coverage.md",oldsrc)}>) → [{r["kind"]}](<{rel_link("editorial/coverage.md",r["target"])}#{r["anchor"]}>); {r["action"]}; `{r["status"]}`.')
        report+=['']
    write('editorial/coverage.md','\n'.join(report))
    print(f'Migrated {len(files)} source lessons into {len(CHAPTERS)} chapters; {len(UNITS)} individually mapped blocks.')

if __name__=='__main__':main()
