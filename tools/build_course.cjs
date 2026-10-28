/* Markdown + course.json are canonical. No network or npm install is needed. */
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {pathToFileURL}=require('url');
const root=path.resolve(__dirname,'..');
const {normalize,chapterWithAnchors}=require('./coverage.cjs');
const readText=file=>normalize(fs.readFileSync(file,'utf8'));
const coverage=JSON.parse(readText(path.join(root,'editorial/coverage.json')));
const scenarioData=JSON.parse(readText(path.join(root,'editorial/chapter-scenarios.json')));
const course=JSON.parse(fs.readFileSync(path.join(root,'course.json'),'utf8'));
const katex=require('./vendor/katex.cjs');
const check=process.argv.includes('--check');
const htmlOnly=process.argv.includes('--html-only');
const outputs=new Map();
const projectExamplesDir='Примеры проектов';
const projectExamplesPath=path.join(root,projectExamplesDir);
const projectOrder=["ALGORITHMS_AND_FUNCTIONS(SFT).md","ALGORITHMS_AND_FUNCTIONS(alimant).md","ALGORITHMS_AND_FUNCTIONS(distlation).md","ALGORITHMS_AND_FUNCTIONS(rag).md","ALGORITHMS_AND_FUNCTIONS(videocap_integrated).md"];
const projectRank=name=>{const i=projectOrder.indexOf(name);return i<0?projectOrder.length:i;};
const projectExamples=fs.existsSync(projectExamplesPath)
 ?fs.readdirSync(projectExamplesPath,{withFileTypes:true}).filter(entry=>entry.isFile()&&entry.name.toLowerCase().endsWith('.md')&&entry.name.toLowerCase()!=='readme.md').map(entry=>entry.name).sort((a,b)=>projectRank(a)-projectRank(b)||a.localeCompare(b,'ru'))
 :[];
const projectExampleIds=new Map(projectExamples.map(name=>[path.join(projectExamplesPath,name).toLowerCase(),'project-example-'+name.replace(/\.md$/i,'').normalize('NFKD').replace(/[^\p{L}\p{N}]+/gu,'-').replace(/^-|-$/g,'').toLowerCase()]));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const digest=s=>crypto.createHash('sha256').update(s).digest('hex');
const chapters=new Map(course.chapters.map(c=>[c.id,c]));
const relative=(from,to)=>path.relative(path.dirname(from),to).split(path.sep).join('/');
const link=(from,c)=>`[${c.title}](<${relative(from,c.path)}>)`;
const outputsText=(file,text)=>outputs.set(file,text.trimEnd()+'\n');
const sectionId=s=>'section-'+s.id;

function coverageIndex(){
 let text='# Карта происхождения и содержательной редакции\n\nИсточник каждой единицы указан до точного диапазона архивных строк. Цель задана диапазоном Markdown в `coverage.json`; якорь добавляется в HTML при сборке. Техническая сохранность и смысловая приёмка имеют разные статусы.\n\n';
 const groups=[...course.chapters.map(c=>({id:c.id,title:c.title})),{id:null,title:'Редакторские сведения происхождения'}];
 for(const group of groups){
  const units=coverage.units.filter(u=>(u.chapter||null)===group.id);
  if(!units.length)continue;
  text+=`## ${group.title}\n\n`;
  for(const u of units){
   const source=`../${coverage.snapshot}/${u.source}`;
   const destinations=u.targets.map(key=>{const s=coverage.spans[key];const target=s.path.startsWith('editorial/')?'../'+s.path:'../Читать конспекты.html#'+key;return `[${s.destination_section||s.path}:${s.start_line}–${s.end_line}](<${target}>) (${s.review_status||s.status})`;}).join('; ');
   text+=`- [${u.source}:${u.source_lines.join('–')}](<${source}>) → ${destinations}; ${u.action}; ID \`${u.id}\`.\n`;
  }
  text+='\n';
 }
 outputsText('editorial/coverage.md',text);
}

function indexes(){
 let home='# Курс ML: пять самостоятельных модулей\n\n**[Открыть HTML-читатель](<Читать конспекты.html>)** · [Порядок подготовки](<План подготовки курса.md>) · [Карта переноса](<editorial/coverage.md>) · [Содержательная приёмка](<editorial/acceptance.md>)\n\nВыберите задачу и обязательный маршрут. Углубления можно пропускать; полные ветки BERT находятся внутри языковых моделей. Markdown — редактируемый источник, HTML собирается автоматически.\n\n';
 for(const m of course.modules){
   const cs=course.chapters.filter(c=>c.module===m.id);
   home+=`- [${m.title}](<${m.path}/README.md>) — ${cs.length} глав.\n`;
   const file=m.path+'/README.md';
   let text=`# ${m.title}\n\n[Все модули](<${relative(file,'README.md')}>) · [Открыть в читателе](<${relative(file,'Читать конспекты.html')}#module-${m.id}>) · [План и сценарии](<План.md>)\n\n`;
   text+='Общее ядро образует маршрут подраздела. Основа ветки нужна для выбранной задачи; углубление имеет отдельный вход и возврат к продолжению. Предпосылки каждой главы ведут к конкретным материалам.\n\n';
   for(const s of m.sections){
     const group=cs.filter(c=>c.section===s.id);
     text+=`## ${s.title}\n\n`;
     if(s.path)text+=`[Оглавление подраздела](<${s.path}/README.md>)\n\n`;
     for(const c of group)text+=`- ${link(file,c)} — ${c.level}. ${c.question}\n`;
     text+='\n';
     if(s.path){
       const sf=`${m.path}/${s.path}/README.md`;
       const projects=s.id==='bert'?'\n\n## Сквозные проекты\n\n'+(course.projects||[]).map(p=>`- [${p.title}](<${relative(sf,chapters.get(p.chapter).path)}#${p.anchor}>)`).join('\n'):'';
       outputsText(sf,`# ${s.title}\n\n[Модуль «${m.title}»](<../README.md>) · [Сценарии глав](<План.md>)\n\n`+group.map(c=>`- ${link(sf,c)} — ${c.level}. ${c.question}`).join('\n')+'\n\n'+routesFor(sf,group)+projects);
     }
   }
   text+='## Маршруты\n\n'+routesFor(file,cs);
   outputsText(file,text);
 }
 home+='\n## Маршруты по задачам\n\n'+course.routes.map(r=>`- **${r.title}:** `+r.chapters.map(id=>link('README.md',chapters.get(id))).join(' → ')+(r.entry?.length?' Вход: '+r.entry.map(id=>link('README.md',chapters.get(id))).join(', ')+'.':'')).join('\n\n');
 home+='\n\n## Сборка и происхождение\n\n`node tools/build_course.cjs` пересобирает читатель и оглавления; `node tools/build_course.cjs --check` проверяет, что они соответствуют Markdown. Сборка использует локальные зависимости в `tools/vendor`.\n\nИсходный срез до перестройки хранится в `archive/2026-10-20-before-rebuild`. Перенос учтён по отдельным абзацам, формулам, примерам и схемам. История редакции хранится отдельно от объяснений. Наличие задания не означает выполненной практики.\n';
 outputsText('README.md',home);
 outputsText('Учебные конспекты.md',home.replace('# Курс ML: пять самостоятельных модулей','# Учебные материалы: оглавление'));
}
function routesFor(file,group){
 const ids=new Set(group.map(c=>c.id));
 return course.routes.filter(r=>r.chapters.some(id=>ids.has(id))).map(r=>`**${r.title}:** `+r.chapters.map(id=>link(file,chapters.get(id))).join(' → ')+(r.entry?.length?'\n\nВход в маршрут: '+r.entry.map(id=>link(file,chapters.get(id))).join(', ')+'.':'')).join('\n\n')+'\n';
}

function scenarios(){
 let all='# Сценарии новой редакции\n\nСценарии фиксируют целевую структуру и конкретные учебные действия. Сопоставление исходных единиц — в [карте переноса](<coverage.md>).\n\n';
 for(const m of course.modules){
  const cs=course.chapters.filter(c=>c.module===m.id);
  const modulePlan=`${m.path}/План.md`;
  let moduleText=`# План модуля «${m.title}»\n\n[Оглавление](<README.md>) · [Карта покрытия](<${relative(modulePlan,'editorial/coverage.md')}>) · [Проверка источников](<${relative(modulePlan,'editorial/research.md')}>)\n\n`;
  for(const s of m.sections){
   let sectionText=`# Сценарии: ${s.title}\n\n`;
   for(const c of cs.filter(c=>c.section===s.id)){
    const authored=scenarioData.chapters[c.id];
    if(!authored)throw Error('Missing authored scenario: '+c.id);
    const safe=`<a id="scenario-${c.id}"></a>\n\n## ${c.title}\n\n**Статус сценария:** ${authored.status}\n\n${authored.markdown}\n\n`;
    all+=safe;sectionText+=safe;moduleText+=safe;
   }
   if(s.path)outputsText(`${m.path}/${s.path}/План.md`,sectionText);
  }
  outputsText(modulePlan,moduleText);
  const name={'data':'План — данные и признаки.md','reg':'План — регуляризация.md','cluster':'План — кластеризация.md','cv':'План — компьютерное зрение.md','lm':'План — LLM и языковые модели.md'}[m.id];
  outputsText(name,`# План обновлённого модуля «${m.title}»\n\nДействующая структура, порядок и сценарии находятся в [плане модуля](<${modulePlan}>), [оглавлении](<${m.path}/README.md>) и [сценариях курса](<editorial/scenarios.md>). Это одна редакция, а не параллельный набор конкурирующих планов.\n\n`+routesFor(name,cs)+'\n[Первичные источники и статус проверки](<editorial/research.md>) · [Поблочная карта переноса](<editorial/coverage.md>).');
 }
 outputsText('editorial/scenarios.md',all);
}

(async()=>{
 const {Marked}=await import(pathToFileURL(path.join(__dirname,'vendor/marked.mjs')).href);
 indexes();scenarios();coverageIndex();
 const fileIds=new Map(course.chapters.map(c=>[path.resolve(root,c.path).toLowerCase(),c.id]));
 for(const m of course.modules){fileIds.set(path.resolve(root,m.path,'README.md').toLowerCase(),'module-'+m.id);for(const s of m.sections)if(s.path)fileIds.set(path.resolve(root,m.path,s.path,'README.md').toLowerCase(),sectionId(s));}
 fileIds.set(path.join(root,'README.md').toLowerCase(),'welcome');fileIds.set(path.join(root,'Учебные конспекты.md').toLowerCase(),'welcome');
 let currentFile,currentId,formulaCount=0,headingSeen;
 const renderMath=(latex,display)=>{formulaCount++;try{return katex.renderToString(latex,{displayMode:display,throwOnError:true,strict:false,trust:false,output:'htmlAndMathml'});}catch(error){throw Error(currentFile+': '+error.message);}};
 const md=new Marked({gfm:true,breaks:false});
 const slug=s=>s.replace(/<[^>]+>/g,'').toLowerCase().replace(/[^\p{L}\p{N}]+/gu,'-').replace(/^-|-$/g,'');
 md.use({extensions:[
  {name:'displayMath',level:'block',start:s=>s.indexOf('$$'),tokenizer(src){const m=/^\$\$([\s\S]+?)\$\$(?:\r?\n|$)/.exec(src);if(m)return{type:'displayMath',raw:m[0],text:m[1]};},renderer(t){return `<div class="math-display">${renderMath(t.text,true)}</div>\n`;}},
  {name:'inlineMath',level:'inline',start:s=>s.indexOf('$'),tokenizer(src){const m=/^\$([^$\n]+)\$(?!\$)/.exec(src);if(m)return{type:'inlineMath',raw:m[0],text:m[1]};},renderer(t){return renderMath(t.text,false);}}
 ],renderer:{
  heading(t){const base=currentId+'--'+slug(t.text);const n=(headingSeen.get(base)||0)+1;headingSeen.set(base,n);return `<h${t.depth} id="${base}${n>1?'-'+n:''}">${this.parser.parseInline(t.tokens)}</h${t.depth}>\n`;},
  image(t){const target=path.resolve(path.dirname(currentFile),decodeURIComponent(t.href));const types={'.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg'};const type=types[path.extname(target).toLowerCase()];if(!target.startsWith(root+path.sep)||!type)throw Error('Unsupported image '+target);return `<button type="button" class="diagram-open" data-enlarge aria-label="Развернуть: ${esc(t.text)}"><img class="diagram" src="data:${type};base64,${(type==='image/svg+xml'?Buffer.from(readText(target),'utf8'):fs.readFileSync(target)).toString('base64')}" alt="${esc(t.text)}"><span>Развернуть схему</span></button>`;},
   link(t){let href=t.href||'';if(!/^[a-z][a-z0-9+.-]*:/i.test(href)&&!href.startsWith('#')){const [p,frag]=href.split('#');const target=path.resolve(path.dirname(currentFile),decodeURIComponent(p));if(currentFile.startsWith(projectExamplesPath+path.sep)&&!fs.existsSync(target))return this.parser.parseInline(t.tokens);const exampleId=projectExampleIds.get(target.toLowerCase());if(exampleId){href='#'+exampleId+(frag?'--'+slug(frag):'');}else if(currentFile.startsWith(projectExamplesPath+path.sep)&&target.startsWith(projectExamplesPath+path.sep)){return this.parser.parseInline(t.tokens);}else{const id=fileIds.get(target.toLowerCase());href=id?'#'+(frag?.startsWith('u-')?frag:id):target===path.join(root,'Читать конспекты.html')?'#'+(frag||'welcome'):path.relative(root,target).split(path.sep).map(encodeURIComponent).join('/')+(frag?'#'+frag:'');}}else if(href.startsWith('#')&&!chapters.has(href.slice(1))&&!href.startsWith('#u-')&&!href.startsWith('#module-')&&!href.startsWith('#section-')&&!['#welcome','#project-examples'].includes(href)){href='#'+currentId+'--'+slug(href.slice(1));}return `<a href="${esc(href)}"${/^https?:/i.test(href)?' target="_blank" rel="noopener noreferrer"':''}>${this.parser.parseInline(t.tokens)}</a>`;}
 }});
 const render=(file,id,text)=>{currentFile=path.join(root,file);currentId=id;headingSeen=new Map();let source=text??outputs.get(file)??readText(currentFile);if(chapters.has(id))source=chapterWithAnchors(root,file,source,coverage);return md.parse(source);};
 const css=`:root{--paper:#fffdf7;--ink:#23353f;--accent:#166558;--line:#d6dfd9;--wash:#eff4ee}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:18px/1.7 Georgia,serif}a{color:var(--accent);text-underline-offset:3px}header{height:70px;position:sticky;top:0;z-index:4;background:var(--paper);border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:12px 26px;font:14px/1.4 'Segoe UI',sans-serif}header strong{font-size:21px}aside{position:fixed;top:70px;bottom:0;left:0;width:320px;background:var(--wash);overflow:auto;padding:20px;font:14px/1.45 'Segoe UI',sans-serif;border-right:1px solid var(--line)}aside input,aside select{width:100%;font:inherit;padding:10px;border:1px solid #aabdac;border-radius:6px;background:white;margin-bottom:12px}aside details{padding-left:8px;margin:9px 0}aside>details{padding-left:0}summary{cursor:pointer;font-weight:650;padding:6px 0}aside a{display:block;padding:7px 6px;text-decoration:none;border-radius:5px}aside a.active{background:#d6e8db;font-weight:650}aside small{display:block;font-size:11px;color:#52685d}main{margin-left:320px;padding:30px 4vw 70px}article,#welcome{max-width:960px;margin:auto}h1,h2,h3{line-height:1.28;color:#173642}h1{font-size:2em;font-weight:500}h2{margin-top:2em;font-size:1.45em}h3{margin-top:1.6em;font-size:1.15em}p{margin:1em 0}li{margin:.5em 0}.diagram{display:block;width:100%;height:auto;margin:24px 0;border:1px solid var(--line);border-radius:12px}.math-display{overflow:auto;padding:14px;background:#f1f4ec;border-radius:6px;margin:22px 0}.katex{font-size:1.04em}.chapter-top,.chapter-nav{font:14px/1.5 'Segoe UI',sans-serif;display:flex;justify-content:space-between;gap:16px;padding:15px 0;border-bottom:1px solid var(--line)}.chapter-nav{border-top:1px solid var(--line);margin:35px 0 15px;flex-wrap:wrap}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{display:block;border:1px solid var(--line);border-radius:12px;padding:23px;text-decoration:none;font:17px/1.5 'Segoe UI',sans-serif}.card strong{display:block;font-size:22px;margin-bottom:10px}.card span{color:#52685d}.pill{font:12px 'Segoe UI',sans-serif;background:#e1ecde;padding:5px 9px;border-radius:20px}pre{padding:18px;overflow:auto;background:#edf1ec;border-radius:6px}code{font:.86em Consolas,monospace;background:#edf1ec;padding:2px 3px}pre code{padding:0}table{border-collapse:collapse;display:block;overflow:auto;font-size:.92em}td,th{border:1px solid var(--line);padding:8px}blockquote{border-left:3px solid #8da99b;padding-left:18px;margin-left:0;color:#4f665b}[hidden]{display:none!important}article{overflow-wrap:anywhere}a[id^="u-"]{display:block;scroll-margin-top:90px}h1,h2,h3{scroll-margin-top:90px}.route-intro{font:15px/1.6 'Segoe UI',sans-serif;background:#edf2e8;padding:18px;border-radius:8px}.muted{color:#586d61}@media(max-width:950px){header{height:auto;position:relative}header span{display:none}aside{position:relative;top:auto;width:100%;max-height:38vh}main{margin-left:0;padding:22px 18px}.cards{grid-template-columns:1fr}body{font-size:17px}}@media print{header,aside,.chapter-nav{display:none}main{margin:0;padding:0}article[hidden]{display:block!important}#welcome{display:none}.diagram,.math-display{break-inside:avoid}}`;
 let htmlDoc=`<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="generator" content="tools/build_course.cjs"><title>Курс ML</title><link rel="stylesheet" href="оформление/katex/katex.min.css"><style>${css}\n.diagram-open{display:block;width:100%;border:0;background:transparent;padding:0;text-align:left;cursor:zoom-in}.diagram-open span{display:block;font-size:13px;color:#315a67;margin:4px 0 18px}.diagram-open:focus-visible{outline:3px solid #447cbb}#figure-view{width:min(96vw,1300px);max-width:96vw;border:1px solid #8093aa;border-radius:12px;padding:16px}#figure-view::backdrop{background:#10283cc9}.figure-scroll{overflow:auto;max-height:80vh}#figure-full{width:1120px;max-width:none;height:auto}#figure-close{margin-bottom:10px;padding:8px 16px}</style></head><body><header><a href="#welcome"><strong>Курс ML</strong></a><span>5 модулей · ${course.chapters.length} глав · работает без сети</span><a href="README.md">Markdown и планы</a></header><aside><label for="search">Найти главу</label><input id="search" type="search" placeholder="Название или учебный вопрос"><label for="route">Маршрут</label><select id="route"><option value="">Порядок подраздела</option>${course.routes.map(r=>`<option value="${r.id}">${esc(r.title)}</option>`).join('')}</select>`;
 for(const m of course.modules){htmlDoc+=`<details class="module" data-module="${m.id}" open><summary>${esc(m.title)}</summary><a href="#module-${m.id}">Оглавление модуля</a>`;for(const s of m.sections){if(s.path)htmlDoc+=`<details class="subsection"><summary>${esc(s.title)}</summary><a href="#${sectionId(s)}">Маршрут подраздела</a>`;for(const c of course.chapters.filter(c=>c.module===m.id&&c.section===s.id))htmlDoc+=`<a data-chapter="${c.id}" data-search="${esc(c.title+' '+c.question)}" href="#${c.id}">${esc(c.title)}<small>${esc(c.level)}</small></a>`;if(s.path)htmlDoc+='</details>';}htmlDoc+='</details>';}
  htmlDoc+='</aside><dialog id="figure-view" aria-label="Увеличенная схема"><button type="button" id="figure-close">Закрыть</button><div class="figure-scroll"><img id="figure-full" alt=""></div></dialog><main><section id="welcome"><p class="muted">Пять самостоятельных модулей</p><h1>От задачи к механизму,<br>вычислению и проверке</h1><p>Выберите модуль или маршрут. BERT входит в языковые модели: краткая глава — в общем ядре, полные задачи — во вложенной encoder-ветке.</p><div class="cards">'+course.modules.map(m=>`<a class="card" data-module-card="${m.id}" href="#module-${m.id}"><strong>${esc(m.title)}</strong><span>${m.sections.length>1?m.sections.map(s=>esc(s.title)).join(' · '):'Основы, методы и проверка результата'}</span></a>`).join('')+'</div><h2>Самостоятельные маршруты</h2>'+course.routes.map(r=>`<p><a data-start-route="${r.id}" href="#${r.chapters[0]}">${esc(r.title)}</a></p>`).join('')+(projectExamples.length?'<h2>Примеры проектов</h2><p><a href="#project-examples">Открыть локальные примеры проектов</a></p>':'')+'<p class="muted">Markdown — единственный источник учебного текста. Задания и вопросы предназначены для самостоятельной работы; наличие главы не означает освоения темы.</p></section>';
  if(projectExamples.length){htmlDoc+=`<article id="project-examples" hidden data-title="Примеры проектов"><h1>Примеры проектов</h1><p class="muted">Локальные материалы из папки «${esc(projectExamplesDir)}». Текст и изображения встроены в этот HTML при сборке.</p>`;if(fs.existsSync(path.join(projectExamplesPath,'README.md')))htmlDoc+=render(projectExamplesDir+'/README.md','project-examples-overview');for(const name of projectExamples){const file=projectExamplesDir+'/'+name,id=projectExampleIds.get(path.join(projectExamplesPath,name).toLowerCase());htmlDoc+=`<section class="project-example" id="${esc(id)}">${render(file,id)}</section>`;}htmlDoc+='</article>';}
 for(const m of course.modules){htmlDoc+=`<article id="module-${m.id}" hidden data-title="${esc(m.title)}">${render(m.path+'/README.md','module-'+m.id)}</article>`;for(const s of m.sections)if(s.path){let file=m.path+'/'+s.path+'/README.md';htmlDoc+=`<article id="${sectionId(s)}" hidden data-title="${esc(s.title)}">${render(file,sectionId(s))}</article>`;}}
 for(const c of course.chapters){const m=course.modules.find(m=>m.id===c.module),s=m.sections.find(s=>s.id===c.section);htmlDoc+=`<article id="${c.id}" hidden data-title="${esc(c.title)}"><div class="chapter-top"><a href="#module-${m.id}">${esc(m.title)}</a><a href="${c.path.split('/').map(encodeURIComponent).join('/')}">Исходный Markdown</a></div><p class="route-intro" data-route-context></p>${render(c.path,c.id)}<nav class="chapter-nav" aria-label="Продолжение"><a data-prev></a><a data-next></a><a data-return href="#${s.path?sectionId(s):'module-'+m.id}">К маршруту и выбору ветки</a></nav></article>`;}
 const client=JSON.stringify({chapters:course.chapters.map(({id,module,section,level,title})=>({id,module,section,level,title})),routes:course.routes,aliases:course.legacy_aliases}).replace(/</g,'\\u003c');
 htmlDoc+=`</main><script>const COURSE=${client};
const routeSelect=document.getElementById('route');
const figureView=document.getElementById('figure-view'),figureFull=document.getElementById('figure-full');
document.addEventListener('click',event=>{const button=event.target.closest('[data-enlarge]');if(button){const img=button.querySelector('img');figureFull.src=img.src;figureFull.alt=img.alt;figureFull.style.width=Math.max(1120,Math.min(3200,img.naturalWidth||1120))+'px';figureView.showModal();}});
document.getElementById('figure-close').addEventListener('click',()=>figureView.close());
function showChapter(){let id=decodeURIComponent(location.hash.slice(1)||'welcome');if(COURSE.aliases[id]){history.replaceState(null,'','#'+COURSE.aliases[id]);id=COURSE.aliases[id];}let anchor=document.getElementById(id);let target=anchor&&(anchor.matches('article,#welcome')?anchor:anchor.closest('article'));if(!target){target=document.getElementById('welcome');}document.querySelectorAll('article,#welcome').forEach(el=>el.hidden=el!==target);document.querySelectorAll('[data-chapter]').forEach(a=>a.classList.toggle('active',a.dataset.chapter===target.id));document.title=(target.dataset.title||'Курс ML')+' · Курс ML';document.querySelectorAll('aside a.active').forEach(a=>{let d=a.parentElement;while(d&&d.tagName!=='ASIDE'){if(d.tagName==='DETAILS')d.open=true;d=d.parentElement;}});const current=COURSE.chapters.find(c=>c.id===target.id);if(current){document.querySelectorAll('aside>details.module').forEach(d=>d.open=d.dataset.module===current.module);const active=document.querySelector('aside a.active');if(active){const side=active.closest('aside');if(active.offsetTop<side.scrollTop||active.offsetTop>side.scrollTop+side.clientHeight-70)side.scrollTop=Math.max(0,active.offsetTop-side.clientHeight+100);}const r=COURSE.routes.find(r=>r.id===routeSelect.value);let seq=r?r.chapters:COURSE.chapters.filter(c=>c.module===current.module&&c.section===current.section&&c.level!=='углубление').map(c=>c.id);let index=seq.indexOf(current.id);let next=index>=0?seq[index+1]:null,prev=index>=0?seq[index-1]:null;if(!r&&current.level==='углубление'){const siblings=COURSE.chapters.filter(c=>c.module===current.module&&c.section===current.section);const p=siblings.findIndex(c=>c.id===current.id);next=siblings.slice(p+1).find(c=>c.level!=='углубление')?.id;prev=siblings.slice(0,p).reverse().find(c=>c.level!=='углубление')?.id;}for(const [type,id2]of[['prev',prev],['next',next]]){const a=target.querySelector('[data-'+type+']');a.hidden=!id2;if(id2){a.href='#'+id2;a.textContent=(type==='prev'?'← ':'Продолжить: ')+COURSE.chapters.find(c=>c.id===id2).title;}}target.querySelector('[data-route-context]').textContent=r?(index>=0?'Маршрут «'+r.title+'»: '+(index+1)+' из '+seq.length:'Эта глава вне выбранного маршрута. Вернитесь к его оглавлению или выберите порядок подраздела.'):(current.level==='углубление'?'Углубление: его можно пропустить. Ссылка продолжения возвращает к следующей обязательной теме, если она есть.':'Порядок подраздела. Обязательную последовательность задачи задаёт выбранный маршрут; остальные ветки выбираются отдельно.');}const back=target.querySelector('[data-return]');if(back&&current&&current.level==='углубление'){const nextA=target.querySelector('[data-next]');back.href=nextA&&!nextA.hidden?nextA.getAttribute('href'):'#module-'+current.module;back.textContent=nextA&&!nextA.hidden?'Пропустить углубление и продолжить маршрут':'Выбрать следующую ветку модуля';}requestAnimationFrame(()=>{if(anchor&&anchor!==target)anchor.scrollIntoView();else window.scrollTo(0,0);});}
window.addEventListener('hashchange',showChapter);routeSelect.addEventListener('change',showChapter);document.querySelectorAll('[data-start-route]').forEach(a=>a.addEventListener('click',()=>{routeSelect.value=a.dataset.startRoute;showChapter();}));document.getElementById('search').addEventListener('input',function(){let q=this.value.toLocaleLowerCase('ru');document.querySelectorAll('[data-chapter]').forEach(a=>a.hidden=!a.dataset.search.toLocaleLowerCase('ru').includes(q));document.querySelectorAll('aside details').forEach(d=>{d.hidden=q&&!Array.from(d.querySelectorAll('[data-chapter]')).some(a=>!a.hidden);if(q)d.open=true;});});showChapter();</script></body></html>`;
 outputsText('Читать конспекты.html',htmlDoc);
 const allIds=new Set([...htmlDoc.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]));
 const missing=[...htmlDoc.matchAll(/href="#([^"]+)"/g)].map(x=>x[1]).filter(id=>!allIds.has(id)&&!course.legacy_aliases[id]);
 if(missing.length)throw Error('Unresolved HTML targets: '+[...new Set(missing)].join(', '));
 let stale=[];
 for(const [file,text]of outputs){
   if(htmlOnly&&file!=='Читать конспекты.html')continue;
  const target=path.join(root,file);
  if(check){if(!fs.existsSync(target)||readText(target)!==text)stale.push(file);}
  else{
   fs.mkdirSync(path.dirname(target),{recursive:true});
   const temporary=target+'.building';fs.writeFileSync(temporary,text);
   for(let attempt=0;;attempt++){
    try{fs.renameSync(temporary,target);break;}
    catch(error){if(attempt===4)throw error;Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,200);}
   }
  }
 }
 if(stale.length){console.error('Generated files differ from Markdown:',stale.join(', '));process.exitCode=1;return;}
 console.log(JSON.stringify({mode:check?'check':'build',modules:course.modules.length,chapters:course.chapters.length,formulas:formulaCount,generated:outputs.size,htmlSha256:digest(outputs.get('Читать конспекты.html'))},null,2));
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
