"""Preserve the six BERT project protocols in the active task chapters."""
from pathlib import Path
import hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[2]
def sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def main():
    manifest=json.loads((ROOT/'course.json').read_text(encoding='utf-8'))
    if manifest.get('projects'):raise SystemExit('Project protocols already restored; no files overwritten.')
    cov=json.loads((ROOT/'editorial/coverage.json').read_text(encoding='utf-8'))
    old='07-bert/План — BERT и encoder-модели для NLP.md'
    lines=(ROOT/cov['snapshot']/old).read_text(encoding='utf-8-sig').splitlines()
    start=next(i for i,x in enumerate(lines) if x=='## Сквозные проекты и артефакты')
    end=next(i for i,x in enumerate(lines) if x=='## Как вести прогресс')
    by={c['id']:c for c in manifest['chapters']}
    mapping={1:'bert-tasks',2:'bert-tasks',3:'bert-retrieval',4:'bert-tasks',5:'bert-domains',6:'bert-operations'}
    blocks={};projects=[];current='bert-operations'
    i=start
    while i<end:
        if not lines[i].strip():i+=1;continue
        a=i
        if lines[i].startswith('### P'):
            n=int(re.match(r'### P(\d)',lines[i])[1]);current=mapping[n]
        if lines[i].startswith('## Контрольные задачи'):current='bert-evaluation'
        i+=1
        if not lines[a].startswith('#'):
            while i<end and lines[i].strip() and not re.match(r'^\d+\. ',lines[i]):i+=1
        text='\n'.join(lines[a:i]).strip()
        uid=sha(old+':'+str(a+1)+':'+text)[:18]
        record=dict(id=uid,source=old,source_lines=[a+1,i],source_sha256=sha(text),kind='project protocol',source_scope='project-protocols',target=by[current]['path'],chapter=current,anchor='u-'+uid,action='перенести полный протокол проекта',status='transfer-verified',target_sha256=sha(text))
        cov['units'].append(record)
        blocks.setdefault(current,[]).append(f'<!-- unit:{uid} -->\n<a id="u-{uid}"></a>\n\n{text}\n\n<!-- /unit:{uid} -->\n\n')
        if text.startswith('### P'):
            projects.append(dict(id='P'+str(n),title=text.removeprefix('### '),chapter=current,anchor='u-'+uid))
    cov.setdefault('source_scopes',[]).append(dict(source=old,lines=[start+1,end],purpose='Полные шесть проектов, режим теоретической подготовки и десять контрольных задач'))
    for id,parts in blocks.items():
        if old not in by[id]['source_files']:by[id]['source_files'].append(old)
        p=ROOT/by[id]['path'];text=p.read_text(encoding='utf-8')
        marker='## Результат главы и переход дальше'
        assert marker in text
        text=text.replace(marker,'## Проектная практика ветки\n\n'+''.join(parts)+marker)
        p.write_text(text,encoding='utf-8')
    manifest['projects']=projects
    (ROOT/'course.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'editorial/coverage.json').write_text(json.dumps(cov,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=(ROOT/'editorial/coverage.md').read_text(encoding='utf-8')
    report+='\n## Сквозные проекты исходного BERT-плана\n\nПолные протоколы и контрольные задачи перенесены в действующие главы; исходные строки учитываются отдельными единицами в coverage.json.\n\n'
    for project in projects:
        c=by[project['chapter']]
        report+=f'- [{project["title"]}](<../{c["path"]}#{project["anchor"]}>).\n'
    (ROOT/'editorial/coverage.md').write_text(report,encoding='utf-8')
    print(f'Restored {len(projects)} complete projects; source coverage units={len(cov["units"])}')
if __name__=='__main__':main()
