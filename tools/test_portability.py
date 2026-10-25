"""Regression tests for byte integrity and non-mutating, line-ending-aware checks."""
from pathlib import Path
import tempfile,unittest,subprocess,shutil,json,os
from editorial_io import file_digest,verify_span,sha
ROOT=Path(__file__).resolve().parents[1]
TEMP_ROOT=ROOT/'.qa-results'
TEMP_ROOT.mkdir(exist_ok=True)
class Portability(unittest.TestCase):
    def test_line_endings_and_damage(self):
        for suffix in ['.md','.svg','.html','.json']:
            with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
                p=Path(temp)/('sample'+suffix)
                p.write_bytes(b'\xef\xbb\xbfline 1\r\nline 2\n')
                original=file_digest(p)
                p.write_bytes(b'\xef\xbb\xbfline 1\nline 2\n')
                self.assertEqual(original,file_digest(p))
                for damaged in [b'\xef\xbb\xbfline 9\nline 2\n',b'line 1\nline 2\n',b'\xef\xbb\xbfline 1 \nline 2\n']:
                    p.write_bytes(damaged);self.assertNotEqual(original,file_digest(p))
    def test_binary_bytes(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            p=Path(temp)/'image.png';p.write_bytes(b'\r\n');a=file_digest(p)
            p.write_bytes(b'\n');self.assertNotEqual(a,file_digest(p))
    def test_explicit_spans(self):
        span={'start_line':2,'end_line':3,'sha256':sha('one\ntwo'.encode())}
        self.assertTrue(verify_span('heading\r\none\r\ntwo\r\n',span))
        self.assertFalse(verify_span('heading\none\nchanged\n',span))
    def test_source_has_no_markers(self):
        course=json.loads((ROOT/'course.json').read_text(encoding='utf8'))
        for c in course['chapters']:
            text=(ROOT/c['path']).read_text(encoding='utf8')
            self.assertNotIn('<!-- unit:',text,c['id'])
            self.assertNotIn('<a id="u-',text,c['id'])
if __name__=='__main__':unittest.main()
