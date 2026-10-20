"""Create the immutable source snapshot before the one-time course migration."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / 'archive' / '2026-10-20-before-rebuild'
OLD_MODULES = ['01-data', '02-regularization', '03-categorical-features',
               '04-clustering', '05-computer-vision', '06-llm', '07-bert']

def complete_assets():
    """Keep the archived reader self-contained without editing any original byte."""
    for file in (SNAPSHOT / 'root').glob('*'):
        target=SNAPSHOT/file.name
        if file.is_file() and not target.exists():shutil.copy2(file,target)
    for relative in ['оформление/katex','оформление/transformer','archive/study-sources','archive/2026-10-12','archive/2026-10-17']:
        source=ROOT/relative;target=SNAPSHOT/relative
        if source.is_dir() and not target.exists():shutil.copytree(source,target)
    records=[]
    for base in ['оформление','archive']:
        for file in (SNAPSHOT/base).rglob('*'):
            if file.is_file():records.append(dict(path=file.relative_to(SNAPSHOT).as_posix(),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    manifest=SNAPSHOT/'assets.json'
    if not manifest.exists():manifest.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    if (SNAPSHOT / 'snapshot.json').exists():
        complete_assets()
        print('Snapshot already exists; source files were not changed.')
        return
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    sources = []
    for name in OLD_MODULES:
        source = (ROOT / name).resolve()
        target = (SNAPSHOT / name).resolve()
        # Verify both final absolute paths before any recursive move on Windows.
        if source.parent != ROOT or not target.is_relative_to(ROOT / 'archive'):
            raise RuntimeError(f'Unsafe migration paths: {source} -> {target}')
        if target.exists():
            raise RuntimeError(f'Incomplete snapshot requires inspection: {target}')
        for file in sorted(source.rglob('*')):
            if file.is_file():
                sources.append({'path': file.relative_to(ROOT).as_posix(),
                                'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                                'bytes': file.stat().st_size})
        shutil.move(str(source), str(target))
    for file in ROOT.glob('*'):
        if file.is_file() and file.suffix in ('.md', '.html'):
            target = SNAPSHOT / 'root' / file.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(file, target)
            sources.append({'path': file.name,
                            'snapshot': 'root/' + file.name,
                            'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                            'bytes': file.stat().st_size})
    (SNAPSHOT / 'snapshot.json').write_text(json.dumps({
        'date': '2026-10-20', 'purpose': 'Исходный срез до физической перестройки курса',
        'files': sources}, ensure_ascii=False, indent=2), encoding='utf-8')
    complete_assets()
    print(f'Snapshot preserved {len(sources)} files in {SNAPSHOT.relative_to(ROOT)}')

if __name__ == '__main__':
    main()
