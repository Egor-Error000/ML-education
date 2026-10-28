"""Keep the title and section references consistent during the unfinished edit."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
def load(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def save(p,v):(ROOT/p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
course=load('course.json')
chapter=next(x for x in course['chapters'] if x['id']=='feat-target')
chapter['title']='Целевые статистики: сглаживание, перекрёстное вычисление и WoE'
save('course.json',course)
coverage=load('editorial/coverage.json')
renames={'Target statistics и утечка':'Целевая статистика и утечка',
         'Shrinkage, smoothing и LOO':'Сглаживание и собственная метка',
         'Вложенный protocol':'Вложенный протокол'}
for span in coverage['spans'].values():
 if span['path']==chapter['path'] and span.get('destination_section') in renames:
  span['destination_section']=renames[span['destination_section']]
save('editorial/coverage.json',coverage)
print('Target title and section references synchronized; review remains open.')
