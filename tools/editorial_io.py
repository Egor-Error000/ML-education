"""Portable byte hashing and explicit, checked Markdown span addressing."""
from pathlib import Path
import hashlib

TEXT_SUFFIXES={'.md','.txt','.html','.svg','.json','.css','.js','.cjs','.mjs','.py','.yml','.yaml','.xml'}
def canonical_bytes(data):
    return data.replace(b'\r\n',b'\n')
def sha(data):
    return hashlib.sha256(data).hexdigest()
def is_text(path):
    return Path(path).suffix.lower() in TEXT_SUFFIXES or Path(path).name in {'LICENSE','README'}
def file_digest(path):
    data=Path(path).read_bytes()
    return sha(canonical_bytes(data) if is_text(path) else data)
def span_text(text,span):
    # Lines are 1-based inclusive; the hash excludes only the separating newline.
    lines=text.replace('\r\n','\n').splitlines()
    return '\n'.join(lines[span['start_line']-1:span['end_line']])
def verify_span(text,span):
    return sha(span_text(text,span).encode('utf8'))==span['sha256']
