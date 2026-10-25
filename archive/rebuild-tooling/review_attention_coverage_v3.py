"""Explicit semantic destinations after reading every attention source unit."""
from pathlib import Path
import sys,json,re
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from edit_chapter import load,save,sections,digest
DESTINATIONS={
'a6758fef10d613db40':'От состояний к запросам, ключам и значениям',
'7b39c1e927ac3206a8':'От состояний к запросам, ключам и значениям',
'b7810854bed1f29459':'От состояний к запросам, ключам и значениям',
'0d1e795febdac668f8':'От состояний к запросам, ключам и значениям',
'e15ac0acfd7b6fc8f8':'От состояний к запросам, ключам и значениям',
'1734310b2c4e84c19c':'От состояний к запросам, ключам и значениям',
'ec2e37a261521ed168':'Оценки, масштаб, маска и нормировка',
'88082412b71709def5':'Полный расчёт одной головы',
'1dc45ebc16c937a223':'Полный расчёт одной головы',
'4b99f6c485f3d0e1d1':'Полный расчёт одной головы',
'5b4c1dfe5ae31c0384':'Полный расчёт одной головы',
'77cfefe222a8131f00':'Полный расчёт одной головы',
'118bd974b901fdd01b':'Полный расчёт одной головы',
'd2d6f59c79dd32ec3e':'Оценки, масштаб, маска и нормировка',
'7bbac9e77f2894aa56':'Полный расчёт одной головы',
'9055569f48530876d9':'Полный расчёт одной головы',
'd59f83ee87932f7b5c':'Полный расчёт одной головы',
'e350dfd78c6950c98a':'Полный расчёт одной головы',
'74e861f5dda4755884':'Полный расчёт одной головы',
'c9de5b540d4df478d3':'Полный расчёт одной головы',
'39c5fbc129b9c9039f':'Оценки, масштаб, маска и нормировка',
'f28e6c9698e5ba461a':'Полный расчёт одной головы',
'7fb739ee8659fe3b7d':'Полный расчёт одной головы',
'ad0be91d05702a1dd4':'Полный расчёт одной головы',
'f2b9e5f53623b32156':'Полный расчёт одной головы',
'5af7a2c4fbf289e60f':'Несколько голов и выходная проекция',
'8a67a379b275b6abe2':'Несколько голов и выходная проекция',
'3d76fbe00cc3b1c9aa':'Несколько голов и выходная проекция',
'df09dcb90ce1dc898a':'Несколько голов и выходная проекция',
'eb76b284d2875b3023':'MQA, GQA и границы понятия «голова»',
'acb204cca3c919e7b4':'Несколько голов и выходная проекция',
'8f48e5629ef7abcb66':'MQA, GQA и границы понятия «голова»',
'e580936b0ebef4f282':'Несколько голов и выходная проекция',
'471ad0a9e8a2269d36':'Несколько голов и выходная проекция',
'2d7e4415873fd180b3':'MQA, GQA и границы понятия «голова»',
'c65f073a126ce6933c':'MQA, GQA и границы понятия «голова»',
'60479933708a7de318':'От состояний к запросам, ключам и значениям',
'e82165c6d35716f535':'Несколько голов и выходная проекция',
'613fbe02ea741c6b6d':'Несколько голов и выходная проекция',
'ad502062fb5f9c8070':'Несколько голов и выходная проекция',
'e8bc9e5f3ef40d52dd':'Несколько голов и выходная проекция',
'bf72baf578fc82646e':'Несколько голов и выходная проекция',
'de8ec298f7e4e0991a':'Несколько голов и выходная проекция',
'a2f73fc0111443e197':'Несколько голов и выходная проекция',
'ad7af51cb30be504b8':'Несколько голов и выходная проекция',
}
def main():
    cov=load();rel='05-language-models/01-foundations/lm-attention.md';text=(ROOT/rel).read_text(encoding='utf8');parts=sections(text)
    for u in cov['units']:
        if u.get('chapter')!='lm-attention':continue
        dest=DESTINATIONS[u['id']];a,b,body=parts[dest];key=u['anchor'];span=cov['spans'][key]
        span.update(path=rel,start_line=a,end_line=b,sha256=digest(body),destination_section=dest,status='revised',review_status='verified',rationale='Сверены исходный блок и назначенный разбор: роли Q/K/V, масштаб при явных предпосылках, пример с causal маской, изменения V/K, локальный нулевой градиент, асимметрия softmax, головы/GQA и числовой concat сохранены без отсылки к чату.',evidence='editorial/review-evidence.md#attention')
        u['status']='semantic-reviewed';u['action']='объединить с сохранением нюансов'
    # Two old composite blocks deliberately have several destinations.
    extras={
      'a6758fef10d613db40':['Оценки, масштаб, маска и нормировка','Несколько голов и выходная проекция'],
      '0d1e795febdac668f8':['Оценки, масштаб, маска и нормировка'],
      'df09dcb90ce1dc898a':['MQA, GQA и границы понятия «голова»']}
    for ident,titles in extras.items():
        unit=next(u for u in cov['units'] if u['id']==ident)
        for i,title in enumerate(titles,1):
            key=f't-lm-attention-{ident}-{i}';a,b,body=parts[title]
            cov['spans'][key]={'path':rel,'start_line':a,'end_line':b,'sha256':digest(body),'status':'revised','review_status':'verified','destination_section':title,'rationale':'Продолжение составного исходного блока; старый якорь ведёт к вводной части, дополнительные цели раскрывают остальные операции.','evidence':'editorial/review-evidence.md#attention'}
            if key not in unit['targets']:unit['targets'].append(key)
    # Comments are source provenance, not omitted educational explanations.
    for u in cov['units']:
        src=ROOT/cov['snapshot']/u['source'];lines=src.read_text(encoding='utf-8-sig').splitlines();a,b=u['source_lines'];source='\n'.join(lines[a-1:b]).strip()
        if not re.fullmatch(r'<!--[\s\S]*?-->',source):continue
        u['action']='сохранить происхождение';u['status']='provenance-reviewed';u['provenance_only']=True
        span=cov['spans'][u['targets'][0]];target=ROOT/span['path'];title=target.read_text(encoding='utf8').splitlines()[0]
        span.update(start_line=1,end_line=1,sha256=digest(title),status='revised',review_status='verified',rationale='Исходная единица — служебный HTML-комментарий происхождения. Комментарий сохранён в архиве и координатах реестра; старый якорь ведёт к заголовку главы.')
    save(cov)
if __name__=='__main__':main()
