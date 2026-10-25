"""Explicit editorial operations; never silently rehash a changed lesson.

Use transform() for a stated wording operation or rewrite() with an explicit
old-heading -> new-heading map. Unmapped rewrites fail instead of losing coverage.
"""
from pathlib import Path
import json,re,hashlib
from editorial_io import verify_span,span_text
ROOT=Path(__file__).resolve().parents[1]
def load():return json.loads((ROOT/'editorial/coverage.json').read_text(encoding='utf8'))
def save(c):
    (ROOT/'editorial/coverage.json').write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def digest(s):return hashlib.sha256(s.encode('utf8')).hexdigest()
def transform(rel,operation,rationale,coverage):
    path=ROOT/rel;text=path.read_text(encoding='utf8');spans={k:s for k,s in coverage['spans'].items() if s['path']==rel}
    lines=text.splitlines();starts={};ends={}
    # Different source IDs can refer to the same target span; wrap each range once.
    ranges={}
    for k,s in spans.items():
        if not verify_span(text,s):raise ValueError('Stale before edit: '+k)
        ranges.setdefault((s['start_line'],s['end_line']),[]).append(k)
    for i,((a,b),keys) in enumerate(ranges.items()):
        starts.setdefault(a,[]).append(f'<!-- EDIT:{i} -->');ends.setdefault(b,[]).append(f'<!-- /EDIT:{i} -->')
    wrapped='\n'.join('\n'.join(starts.get(i,[])+[line]+ends.get(i,[])) for i,line in enumerate(lines,1))+'\n'
    wrapped=operation(wrapped)
    pattern=re.compile(r'<!-- EDIT:(\d+) -->\n([\s\S]*?)\n<!-- /EDIT:\1 -->')
    out=[];pos=0
    for m in pattern.finditer(wrapped):
        out.append(wrapped[pos:m.start()]);start=''.join(out).count('\n')+1;body=m[2];out.append(body)
        keys=list(ranges.values())[int(m[1])]
        for key in keys:
            old=spans[key]['sha256'];s=spans[key]
            s.update(start_line=start,end_line=start+body.count('\n'),sha256=digest(body))
            if old!=s['sha256']:s.update(status='revised',rationale=rationale,review_status='pending')
        pos=m.end()
    out.append(wrapped[pos:]);result=''.join(out)
    if '<!-- EDIT:' in result:raise ValueError('Edit destroyed a range boundary: '+rel)
    path.write_text(result,encoding='utf8',newline='\n')
    return result
def sections(text):
    matches=list(re.finditer(r'^## (.+)$',text,re.M));result={}
    for i,m in enumerate(matches):
        end=matches[i+1].start() if i+1<len(matches) else len(text)
        body=text[m.start():end].rstrip('\n');a=text[:m.start()].count('\n')+1
        result[m[1]]=(a,a+body.count('\n'),body)
    return result
def rewrite(rel,new,heading_map,rationale,coverage):
    old=(ROOT/rel).read_text(encoding='utf8');dest=sections(new)
    for key,s in coverage['spans'].items():
        if s['path']!=rel:continue
        if not verify_span(old,s):raise ValueError('Stale before rewrite '+key)
        before='\n'.join(old.splitlines()[:s['start_line']])
        headings=re.findall(r'^#{2,3} (.+)$',before,re.M)
        heading=headings[-1] if headings else '__intro__'
        if heading not in heading_map:raise ValueError('Explicit mapping needed: '+heading)
        title=heading_map[heading]
        a,b,body=dest[title]
        s.update(start_line=a,end_line=b,sha256=digest(body),status='revised',rationale=rationale,review_status='pending',destination_section=title)
    (ROOT/rel).write_text(new.rstrip()+'\n',encoding='utf8',newline='\n')
