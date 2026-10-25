"""Inspect stale span addresses; relocate only unique byte-equivalent blocks.

Content changes require --span, --lines and --rationale. New hashes are always
marked pending, never accepted as evidence of semantic preservation.
"""
from pathlib import Path
import argparse,json
from edit_chapter import load,save,digest,ROOT
from editorial_io import verify_span,span_text
def main():
    p=argparse.ArgumentParser();p.add_argument('--chapter',required=True)
    p.add_argument('--apply-exact',action='store_true');p.add_argument('--span');p.add_argument('--lines');p.add_argument('--rationale')
    args=p.parse_args();coverage=load();course=json.loads((ROOT/'course.json').read_text(encoding='utf8'))
    chapter=next((c for c in course['chapters'] if c['id']==args.chapter),None)
    if not chapter:p.error('Unknown chapter')
    path=chapter['path'];text=(ROOT/path).read_text(encoding='utf8');lines=text.splitlines()
    spans={k:s for k,s in coverage['spans'].items() if s['path']==path};report=[]
    if args.span:
        if args.span not in spans or not args.lines or not args.rationale:p.error('--span requires a chapter span, --lines A:B and --rationale')
        a,b=map(int,args.lines.split(':'))
        if not 1<=a<=b<=len(lines):p.error('Invalid inclusive line range')
        s=spans[args.span];s.update(start_line=a,end_line=b)
        s.update(sha256=digest(span_text(text,s)),status='revised',review_status='pending',rationale=args.rationale)
        save(coverage);print('Changed content mapping recorded as pending:',args.span);return
    for key,s in spans.items():
        if verify_span(text,s):continue
        size=s['end_line']-s['start_line']+1
        matches=[i+1 for i in range(len(lines)-size+1) if digest('\n'.join(lines[i:i+size]))==s['sha256']]
        origins=[{'source':u['source'],'lines':u['source_lines']} for u in coverage['units'] if key in u['targets']]
        report.append({'span':key,'old_lines':[s['start_line'],s['end_line']],'exact_candidates':[[a,a+size-1] for a in matches],'origins':origins,'action':'unique exact relocation available' if len(matches)==1 else 'manual review required'})
        if args.apply_exact and len(matches)==1:s.update(start_line=matches[0],end_line=matches[0]+size-1)
    if args.apply_exact:save(coverage)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
