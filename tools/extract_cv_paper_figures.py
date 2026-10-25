"""Reproduce locally stored paper-figure crops from the recorded PDF sources.

The course reader does not need Python or this tool. Source PDFs are deliberately
kept outside the course checkout. Download them separately using source_pdf from
editorial/cv-expansion.json, then provide their directory via --pdf-dir.
"""
from pathlib import Path
import argparse,json,hashlib

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pdf-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    import pypdfium2 as pdfium
    data=json.loads((ROOT/'editorial/cv-expansion.json').read_text(encoding='utf8'))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    count=0
    for fig in data['external_figures']:
        if fig['kind']!='primary-paper-figure':continue
        name=Path(fig['path']).stem;source=args.pdf_dir/(name+'.pdf')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=fig['source_pdf_sha256']:
            raise ValueError('Source PDF differs from reviewed version: '+name)
        # Corrected crops used scale 2.6; the remaining reviewed crops used 2.
        scale=2.6 if fig['visual_review']=='manually-corrected' else 2
        with pdfium.PdfDocument(source) as document:
            page=document[fig['pdf_page']-1]
            image=page.render(scale=scale).to_pil().convert('RGB')
            bounds=tuple(round(x*scale) for x in fig['crop_pdf_points'])
            image.crop(bounds).save(args.output_dir/(name+'.png'))
        count+=1
    print('Extracted source figures:',count)

if __name__=='__main__':main()
