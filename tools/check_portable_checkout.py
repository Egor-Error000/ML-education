"""Verify a clean Git clone of the current edition without committing this project.

The temporary fixture repository and clone live under ignored .qa-results.
No project index, branch, remote or tag is changed. Runtime paths are arguments.
"""
from pathlib import Path
import argparse,json,subprocess,sys,tempfile,shutil,os,hashlib
from editorial_io import is_text,canonical_bytes
ROOT=Path(__file__).resolve().parents[1]
def run(args,cwd):
    result=subprocess.run(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(' '.join(map(str,args))+'\n'+result.stdout.decode('utf8',errors='replace'))
    return result.stdout.decode('utf8',errors='replace')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--node',default='node');args=parser.parse_args()
    base=ROOT/'.qa-results';base.mkdir(exist_ok=True)
    # Validate the actual working reader before preparing a distributable fixture.
    # Private ignored project examples are embedded locally, but are intentionally
    # absent from a clean clone. Its expected reader must use that input inventory.
    run([args.node,'tools/build_course.cjs','--check'],ROOT)
    working_html_sha256=hashlib.sha256(canonical_bytes((ROOT/'Читать конспекты.html').read_bytes())).hexdigest()
    names=run(['git','ls-files','--cached','--others','--exclude-standard','-z'],ROOT).split('\0')
    with tempfile.TemporaryDirectory(prefix='portable-',dir=base) as folder:
        work=Path(folder).resolve();assert work.is_relative_to(base.resolve())
        fixture=work/'fixture';checkout=work/'checkout';fixture.mkdir()
        for name in dict.fromkeys(names):
            src=ROOT/name
            if not name or not src.is_file():continue
            target=(fixture/name).resolve();assert target.is_relative_to(fixture)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
        local_examples=sorted(p.name for p in (ROOT/'Примеры проектов').glob('*.md') if p.name.lower()!='readme.md')
        fixture_examples=sorted(p.name for p in (fixture/'Примеры проектов').glob('*.md') if p.name.lower()!='readme.md')
        normalized_private_examples=local_examples!=fixture_examples
        if normalized_private_examples:
            run([args.node,'tools/build_course.cjs','--html-only'],fixture)
        run(['git','init','-q'],fixture)
        run(['git','-c','core.safecrlf=false','-c','core.autocrlf=false','add','.'],fixture)
        run(['git','-c','user.name=Course fixture','-c','user.email=fixture@invalid','-c','commit.gpgsign=false','commit','-qm','Fixture'],fixture)
        run(['git','-c','core.autocrlf=false','clone','-q','--no-hardlinks',str(fixture),str(checkout)],work)
        results=[]
        for mode in ['LF clean clone','CRLF working text']:
            if mode.startswith('CRLF'):
                for name in names:
                    p=checkout/name
                    if name and p.is_file() and is_text(p):p.write_bytes(canonical_bytes(p.read_bytes()).replace(b'\n',b'\r\n'))
            run([args.node,'tools/build_course.cjs','--check'],checkout)
            report=json.loads(run([sys.executable,'-X','utf8','tools/audit_course.py'],checkout))
            assert report['status']=='passed'
            before=canonical_bytes((checkout/'Читать конспекты.html').read_bytes())
            run([args.node,'tools/build_course.cjs'],checkout)
            run([args.node,'tools/build_course.cjs'],checkout)
            assert before==(checkout/'Читать конспекты.html').read_bytes(),'build bytes changed: '+mode
            run(['git','-c','core.safecrlf=false','diff','--exit-code'],checkout)
            results.append({'mode':mode,'build_check':'passed','audit':'passed','two_builds_identical':True,'tracked_changes':False})
        report={'status':'passed','host':sys.platform,'htmlSha256':hashlib.sha256(before).hexdigest(),'workingHtmlSha256':working_html_sha256,'fixtureReaderRegeneratedForIgnoredExamples':normalized_private_examples,'excludedLocalExampleCount':len(set(local_examples)-set(fixture_examples)),'checks':results,'linux_execution':'Use CI ubuntu-latest; this command does not emulate another OS.'}
        (base/'portable-checkout.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
        print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
