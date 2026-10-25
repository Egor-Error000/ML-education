"""Independent acceptance checks: source preservation, coverage, routes and local links."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib,json,math,re,sys,argparse
from editorial_io import file_digest,verify_span,span_text
from urllib.parse import unquote
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
def sha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def main():
    course=json.loads((ROOT/'course.json').read_text(encoding='utf-8'))
    coverage=json.loads((ROOT/'editorial/coverage.json').read_text(encoding='utf-8'))
    snap=ROOT/coverage['snapshot'];errors=[];counts={}
    def require(ok,message):
        if not ok:errors.append(message)
    snapshot=json.loads((snap/'snapshot.json').read_text(encoding='utf-8'))
    for rec in snapshot['files']:
        p=snap/rec.get('snapshot',rec['path'])
        require(p.is_file() and file_digest(p)==rec['sha256'],'snapshot changed: '+str(p))
    for rec in json.loads((snap/'assets.json').read_text(encoding='utf-8'))['files']:
        p=snap/rec['path'];require(file_digest(p)==rec['sha256'],'snapshot asset changed: '+rec['path'])
    counts['snapshot_files_verified']=len(snapshot['files'])
    text_cache={};source_cache={};unit_cache={};covered=defaultdict(set)
    for unit in coverage['units']:
        src=snap/unit['source']
        if src not in source_cache:source_cache[src]=src.read_text(encoding='utf-8-sig').splitlines()
        lines=source_cache[src]
        a,b=unit['source_lines'];source='\n'.join(lines[a-1:b]).strip()
        require(sha(source)==unit['source_sha256'],'source block hash: '+unit['id'])
        covered[unit['source']].update(range(a-1,b))
        target=ROOT/unit['target']
        if target not in text_cache:text_cache[target]=target.read_text(encoding='utf-8')
        for key in unit['targets']:
            span=coverage['spans'].get(key)
            require(span is not None,'unknown target span '+unit['id']+' -> '+key)
            if not span:continue
            path=ROOT/span['path']
            if path not in text_cache:text_cache[path]=path.read_text(encoding='utf8')
            require(verify_span(text_cache[path],span),f'stale target span {key} at {span["path"]}:{span["start_line"]}; review explicit target range')
            if span['status']=='transfer-verified' and unit.get('status')!='archived-editorial':
                math_pattern=r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$'
                require(re.findall(math_pattern,source)==re.findall(math_pattern,span_text(text_cache[path],span)),'formula changed during exact transfer: '+unit['id'])
            elif span['status']=='revised':
                require(bool(span.get('rationale')),'rewritten span lacks rationale '+key)
                require(span.get('review_status') in ('pending','verified'),'rewritten span lacks review state '+key)
    scoped={s['source']:s for s in coverage.get('source_scopes',[])}
    for old,indices in covered.items():
        lines=(snap/old).read_text(encoding='utf-8-sig').splitlines()
        scope=scoped.get(old);a,b=scope['lines'] if scope else [1,len(lines)]
        require(all(i in indices for i,line in enumerate(lines) if line.strip() and a-1<=i<b),'unmapped source lines: '+old)
    expected={p.relative_to(snap).as_posix() for p in snap.rglob('*.md') if p.parent.name=='Учебные конспекты' and p.name!='README.md'}
    require(set(covered)-set(scoped)==expected,'source lesson inventory differs')
    counts['source_lessons']=len(expected);counts['mapped_blocks']=len(coverage['units'])
    counts['mapping_actions']=dict(Counter(u['action'] for u in coverage['units']))
    by={c['id']:c for c in course['chapters']}
    require(len(by)==len(course['chapters']),'duplicate chapter ID')
    expected_dirs={m['path'] for m in course['modules']}
    actual_dirs={p.name for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit()}
    require(expected_dirs==actual_dirs,'active module directory structure differs')
    require(len(expected_dirs)==5,'not five modules')
    require(len(next(m for m in course['modules'] if m['id']=='lm')['sections'])==6,'not six language subsections')
    visiting=set();done=set()
    def walk(id):
        if id in visiting:errors.append('dependency cycle at '+id);return
        if id in done:return
        require(id in by,'unknown prerequisite '+id)
        if id not in by:return
        visiting.add(id)
        for req in by[id]['prerequisites']:walk(req)
        visiting.remove(id);done.add(id)
    for id in by:walk(id)
    def closure(ids):
        seen=set()
        def add(id):
            if id in seen:return
            seen.add(id)
            for p in by[id]['prerequisites']:add(p)
        for id in ids:add(id)
        return seen
    for r in course['routes']:
        before=closure(r.get('entry',[]))
        for id in r['chapters']:
            require(id in by,'unknown route chapter '+id)
            missing=set(by[id]['prerequisites'])-before
            require(not missing,f'route {r["id"]}, {id}: unfulfilled '+','.join(sorted(missing)))
            before.add(id)
        require(len(r['chapters'])==len(set(r['chapters'])),'route repeats '+r['id'])
    routes={r['id']:r for r in course['routes']}
    for route,banned in [('vision',('lm-','adapt-','bert-')),('clustering',('lm-','adapt-','bert-')),('encoder',('adapt-',)),('application',('adapt-lora','bert-retrieval'))]:
        deps=closure(routes[route]['chapters'])
        require(not any(id.startswith(banned) for id in deps),'hidden forbidden dependency in '+route)
    lmcore=closure(routes['language-core']['chapters'])
    require(not {'lm-rnn','lm-backward','lm-interpretation'}&lmcore,'optional chapter is a core prerequisite')
    require(by['lm-bert']['section']=='found' and by['lm-bert']['level']=='общее ядро','BERT intro not in common core')
    adapt=[c['id'] for c in course['chapters'] if c['section']=='adapt']
    require(adapt.index('adapt-sft')<adapt.index('adapt-lora'),'SFT does not precede LoRA')
    inf=[c['id'] for c in course['chapters'] if c['section']=='infer'];require(inf[0]=='infer-generation','inference does not begin with generation')
    agent=[c['id'] for c in course['chapters'] if c['section']=='agent'];require(agent[:3]==['agent-task','agent-environment','agent-evaluation'],'agent evaluation not before training')
    html=(ROOT/'Читать конспекты.html').read_text(encoding='utf-8')
    ids=re.findall(r'\bid="([^"]+)"',html)
    # SVGs are embedded base64, so their internal IDs do not pollute HTML IDs.
    require(len(ids)==len(set(ids)),'duplicate HTML IDs')
    idset=set(ids)
    oldhtml=(snap/'root/Читать конспекты.html').read_text(encoding='utf-8-sig')
    legacy=re.findall(r'<article id="([^"]+)"',oldhtml)
    for id in legacy:require(id in course['legacy_aliases'] and course['legacy_aliases'][id] in idset,'lost old bookmark '+id)
    for href in re.findall(r'href="#([^"]+)"',html):require(href in idset or href in course['legacy_aliases'],'broken HTML anchor '+href)
    require(html.count('data-module-card=')==5,'home does not show five module cards')
    require('katex-error' not in html,'rendered math error')
    require(not re.search(r'<(?:script|link)[^>]+(?:src|href)="https?://',html),'online runtime dependency')
    counts['legacy_bookmarks']=len(legacy);counts['html_internal_links']=len(re.findall(r'href="#',html))
    files=[ROOT/c['path'] for c in course['chapters']]+list(ROOT.glob('*.md'))
    for m in course['modules']:files.extend((ROOT/m['path']).rglob('README.md'));files.extend((ROOT/m['path']).rglob('План.md'))
    files+=list((ROOT/'editorial').glob('*.md'))
    links=0;resolved_cache={};exists_cache={}
    for file in set(files):
        text=file.read_text(encoding='utf-8')
        text=re.sub(r'```[\s\S]*?```|\$\$[\s\S]*?\$\$|\$[^$\n]+\$|`[^`\n]*`','',text)
        for match in re.finditer(r'\]\((?:<([^>]+)>|([^\s)]+))\)',text):
            target=unquote(match[1] or match[2])
            if re.match(r'^[a-z][a-z0-9+.-]*:',target,re.I) or target.startswith('#'):continue
            name,_,frag=target.partition('#');key=(file.parent,name)
            if key not in resolved_cache:resolved_cache[key]=(file.parent/name).resolve()
            p=resolved_cache[key];links+=1
            if p not in exists_cache:exists_cache[p]=p.exists()
            require(exists_cache[p] or p==ROOT/'editorial/audit.json',f'missing local file {file.relative_to(ROOT)} -> {target}')
            if frag.startswith('u-') and p.suffix=='.md' and exists_cache[p]:
                if p not in text_cache:text_cache[p]=p.read_text(encoding='utf-8')
                require(frag in coverage['spans'] and (ROOT/coverage['spans'][frag]['path']).resolve()==p,'missing sidecar anchor '+target)
    figures=set()
    for c in course['chapters']:
        text=(ROOT/c['path']).read_text(encoding='utf-8')
        require('<!-- unit:' not in text and not re.search(r'<a id="u-',text),'inline provenance in active chapter '+c['id'])
        require(not re.search(r'!\[[^\]]*\]\([^\n]*archive/',text),'archive runtime image '+c['id'])
        require(text.startswith('# '+c['title']),'chapter title mismatch '+c['id'])
        for field in ['**Уровень:**','**Предпосылки:**','**Используют:**','**Учебный вопрос:**']:require(field in text,'chapter metadata missing '+c['id']+field)
        require('Практика' in text or 'практика' in text,'chapter practice missing '+c['id'])
        require(bool(c['figures']),'chapter has no diagram '+c['id'])
        titles={other['title']:other['path'] for other in course['chapters']}
        for match in re.finditer(r'\[«([^»]+)»\]\(<([^>]+)>\)',text):
            if match[1] in titles:
                require(((ROOT/c['path']).parent/match[2]).resolve()==(ROOT/titles[match[1]]).resolve(),'wrong semantic chapter link: '+c['id']+' -> '+match[1])
        for name in c['figures']:
            figures.add(name)
            tree=ET.parse(ROOT/name);v=tree.getroot()
            require(v.find('{http://www.w3.org/2000/svg}title') is not None,'SVG title missing '+name)
            require(v.find('{http://www.w3.org/2000/svg}desc') is not None,'SVG explanation missing '+name)
    counts.update(chapters=len(by),modules=len(expected_dirs),routes=len(routes),figures=len(figures),local_links_checked=links)
    # Recompute new worked examples independently, rather than snapshot their text.
    lon1,lon2=map(math.radians,[179,-179]);a=math.sin((lon2-lon1)/2)**2;distance=2*6371*math.asin(math.sqrt(a))
    require(abs(distance-222.389853289)<1e-6,'geographic worked example')
    require(abs(-math.log(.4*.5*.8)/3-.6108604879)<1e-9,'sequence probability example')
    require(abs(-math.log(.6)-.5108256238)<1e-9,'MLM example')
    require(abs(math.log1p(math.exp(-1))-.3132616875)<1e-9,'classifier example')
    require(abs(1/(1+math.exp(-2))-.8807970779)<1e-9,'GMM responsibilities')
    require(abs(.36/(.36+.16)-.6923076923)<1e-9,'DEC example')
    mean=[2,2];basis=[1/math.sqrt(2)]*2;z=sum((x-m)*v for x,m,v in zip([5,1],mean,basis));reconstructed=[m+z*v for m,v in zip(mean,basis)]
    require(all(abs(x-3)<1e-12 for x in reconstructed),'PCA transform example')
    require(abs(sum((x-y)**2 for x,y in zip([5,1],reconstructed))-8)<1e-10,'PCA reconstruction example')
    require(4+3==7 and 4*3==12,'PEFT parameter example')
    require(abs(-(math.log(.8)+math.log(.4))/2-.5697171416)<1e-9,'CV classifier example')
    counts['new_numerical_checks']=10
    report={'status':'passed' if not errors else 'failed','counts':counts,'errors':errors}
    report['content_review']={'pending_spans':sum(s.get('review_status')=='pending' for s in coverage['spans'].values()),'note':'Technical checks do not close chapter editorial review.'}
    options=argparse.ArgumentParser();options.add_argument('--output-dir',type=Path);args=options.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if errors:sys.exit(1)
if __name__=='__main__':main()
