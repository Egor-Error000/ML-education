import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from edit_chapter import load,save,transform
c=load();rel='05-language-models/04-inference/infer-quantization.md'
transform(rel,lambda t:t.replace('как (−1.4,−.4,.2,1.2), ошибки (0,−.1,0,−.1)','как (−1.4,−.4,.2,1.4), ошибки (0,−.1,0,+.1)'),
 'Независимый расчёт исправил последний tie: 6.5 округляется от нуля до 7; MSE остаётся .005.',c);save(c)
