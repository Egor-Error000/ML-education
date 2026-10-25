"""One-time portable manifest migration and extraction of inline provenance.

Refuses a repeated run. Current lesson files remain the only source of text.
"""
from pathlib import Path
import hashlib,json,re,subprocess,sys,shutil,os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from editorial_io import canonical_bytes,sha,is_text
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def main():
    course=json.loads((ROOT/'course.json').read_text(encoding='utf8'))
    coverage=json.loads((ROOT/'editorial/coverage.json').read_text(encoding='utf8'))
    if coverage.get('version',0)>=3:raise SystemExit('Revision already initialized; no files written.')
    report=ROOT/'editorial/acceptance.md'
    report.write_text('# Статус редакции и воспроизводимости\n\nРедакция исправляется. Прежние проверки выполнены в рабочем окружении Windows. Чистый клон Windows/Linux не был проверен; прежние заявления не подтверждают переносимость.\n\nИсторический отчёт: [приёмка v2](<acceptance-v2.md>). Содержательные карточки новой редакции закрываются отдельно от автоматических проверок.\n',encoding='utf8')
    snap=ROOT/coverage['snapshot']
    for name in ['snapshot.json','assets.json']:
        p=snap/name;old=json.loads(p.read_text(encoding='utf8'))
        records=old['files'] if isinstance(old,dict) else old
        for rec in records:
            path=snap/rec.get('snapshot',rec['path']);data=path.read_bytes()
            if sha(data)!=rec['sha256']:
                raise RuntimeError('Original snapshot bytes not verified: '+str(path))
            blob=subprocess.check_output(['git','show','HEAD:'+path.relative_to(ROOT).as_posix()],cwd=ROOT)
            if canonical_bytes(blob)!=canonical_bytes(data):raise RuntimeError('Git content differs: '+str(path))
            rec['original_sha256']=rec.pop('sha256')
            rec['sha256']=sha(canonical_bytes(data) if is_text(path) else data)
            rec['hash_mode']='crlf-to-lf' if is_text(path) else 'bytes'
        obj=old if isinstance(old,dict) else {'files':records}
        obj['version']=2;obj['normalization']='CRLF -> LF only; BOM and all other bytes preserved'
        dump(p,obj)
    pattern=re.compile(r'<!-- unit:([a-f0-9]+) -->\s*<a id="u-[a-f0-9]+"></a>\s*([\s\S]*?)\s*<!-- /unit:\1 -->')
    targets={u['target'] for u in coverage['units']};spans={}
    for rel in sorted(targets):
        p=ROOT/rel;text=p.read_text(encoding='utf8');output=[];pos=0;offset=0
        for m in pattern.finditer(text):
            prefix=text[pos:m.start()];output.append(prefix);offset+=len(prefix)
            body=m[2].strip();start=offset;output.append(body);offset+=len(body)
            clean_prefix=''.join(output)[:start]
            spans['u-'+m[1]]={'path':rel,'start_line':clean_prefix.count('\n')+1,'end_line':clean_prefix.count('\n')+1+body.count('\n'),'sha256':sha(body.encode('utf8')),'status':'transfer-verified'}
            pos=m.end()
        output.append(text[pos:]);p.write_text(''.join(output),encoding='utf8',newline='\n')
    for u in coverage['units']:
        if u['anchor'] not in spans:raise RuntimeError('Missing span '+u['anchor'])
        u['targets']=[u['anchor']]
        u['historical_target_sha256']=u.pop('target_sha256')
    coverage['version']=3;coverage['spans']=spans
    coverage['coordinate_system']='1-based inclusive Markdown lines after CRLF -> LF; spans include internal blank lines'
    dump(ROOT/'editorial/coverage.json',coverage)
    # Bring runtime images out of the historical snapshot.
    for c in course['chapters']:
        p=ROOT/c['path'];text=p.read_text(encoding='utf8')
        def image(m):
            raw=m[2].strip('<>');source=(p.parent/raw).resolve()
            if not source.is_relative_to(ROOT/'archive'):return m[0]
            dest=ROOT/'оформление/transformer'/source.name
            if dest.exists() and dest.read_bytes()!=source.read_bytes():dest=dest.with_name(source.stem+'-'+sha(source.read_bytes())[:8]+source.suffix)
            if not dest.exists():shutil.copy2(source,dest)
            link=os.path.relpath(dest,p.parent).replace('\\','/')
            return '!['+m[1]+'](<'+link+'>)'
        # Image replacement is performed by the follow-up editor, which also
        # updates explicit span coordinates. Preserve current span integrity here.
        revised=re.sub(r'!\[([^\]]*)\]\(([^\n)]+)\)',image,text)
        if revised!=text:
            changes={}
            for key,s in spans.items():
                if s['path']!=c['path']:continue
                old='\n'.join(text.splitlines()[s['start_line']-1:s['end_line']])
                changes[key]=re.sub(r'!\[([^\]]*)\]\(([^\n)]+)\)',image,old)
            p.write_text(revised,encoding='utf8',newline='\n')
            for key,value in changes.items():spans[key]['sha256']=sha(value.encode('utf8'))
    dump(ROOT/'editorial/coverage.json',coverage)
    print('Portable manifests and sidecar initialized:',len(spans),'target spans')
if __name__=='__main__':main()
