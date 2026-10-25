/* Browser acceptance for the generated offline reader. Requires Playwright. */
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert');
const {pathToFileURL}=require('url');
const root=path.resolve(__dirname,'..');
const packagePath=require.resolve('playwright',{paths:[root,process.env.COURSE_NODE_MODULES||path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules')]});
const {chromium}=require(packagePath);
const course=JSON.parse(fs.readFileSync(path.join(root,'course.json'),'utf8'));
const coverage=JSON.parse(fs.readFileSync(path.join(root,'editorial/coverage.json'),'utf8'));
let browser,reportDir,testedHtmlSha256;
const htmlFingerprint=()=>require('crypto').createHash('sha256').update(fs.readFileSync(path.join(root,'Читать конспекты.html'),'utf8').replace(/\r\n/g,'\n')).digest('hex');
(async()=>{
 const outputIndex=process.argv.indexOf('--output-dir');
 reportDir=outputIndex>=0?path.resolve(process.argv[outputIndex+1]):fs.mkdtempSync(path.join(os.tmpdir(),'course-qa-'));
 const out=path.join(reportDir,'qa');fs.mkdirSync(out,{recursive:true});
 testedHtmlSha256=htmlFingerprint();
 fs.writeFileSync(path.join(reportDir,'browser-audit.json'),JSON.stringify({status:'running',htmlSha256:testedHtmlSha256,host:process.platform},null,2)+'\n');
 const edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
 browser=await chromium.launch({headless:true,...(fs.existsSync(edge)?{executablePath:edge}:{})});
 const context=await browser.newContext({viewport:{width:1440,height:1000}});
 const network=[],errors=[];await context.route(/^https?:\/\//,route=>{network.push(route.request().url());return route.abort();});
 const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
 const url=pathToFileURL(path.join(root,'Читать конспекты.html')).href;
 await page.goto(url);await page.evaluate(()=>document.fonts.ready);
 assert.equal(await page.locator('[data-module-card]').count(),5);
 assert.equal(await page.locator('aside > details.module').count(),5);
 assert.equal(await page.locator('aside [data-module="lm"] > details.subsection').count(),6);
 await page.screenshot({path:path.join(out,'home-desktop.png')});
 const go=async id=>{await page.evaluate(id=>{location.hash=id;showChapter();},id);await page.waitForTimeout(30);};
 await go('course-11-1');assert.equal(await page.locator('article:not([hidden])').getAttribute('id'),'bert-interface');
 await go('course-5-1');assert.equal(await page.locator('article:not([hidden])').getAttribute('id'),'cv-classification');
 await go('course-5-10');assert.equal(await page.locator('article:not([hidden])').getAttribute('id'),'cv-input');
 await go('lm-bert');assert((await page.locator('#lm-bert .katex').count())>0);assert((await page.locator('#lm-bert').innerText()).includes('0.5108'));
 await page.screenshot({path:path.join(out,'bert-core.png')});
 await page.locator('#lm-bert .diagram').last().screenshot({path:path.join(out,'bert-architecture.png')});
 await go('bert-interface');
 const illustrations=[['bert-input-layers','Три источника входного вектора'],['bert-stack','Последовательность encoder-блоков'],['bert-layer','Развёрнутый post-LN слой исходного BERT'],['bert-retrieval','Bi-encoder и cross-encoder: где взаимодействуют тексты']];
 for(const [name,alt] of illustrations)await page.locator('#bert-interface img[alt="'+alt+'"]').screenshot({path:path.join(out,name+'.png')});
 await page.locator('#bert-interface img[alt="Развёрнутый post-LN слой исходного BERT"]').click();
 assert.equal(await page.locator('#figure-view').isVisible(),true);await page.keyboard.press('Escape');assert.equal(await page.locator('#figure-view').isVisible(),false);
 for(const [id,name] of [['cl-hierarchy','cl-linkage'],['cl-density','cl-density'],['cl-gmm','cl-em'],['cl-kmeans','cl-kmeans-2d'],['cl-geometry','cl-geometry'],['cl-projection','cl-projection'],['cl-metrics','cl-metrics'],['cl-stability','cl-stability'],['cl-images','cl-images'],['cl-text','cl-text'],['cl-systems','cl-systems'],['feat-numeric','feat-numeric'],['feat-selection','feat-selection']]){await go(id);await page.locator('#'+id+' img').last().screenshot({style:'header{visibility:hidden!important}',path:path.join(out,name+'.png')});}
 await go('cl-deep');
 for(const [i,name] of [[0,'cl-deep'],[1,'cl-idec-branches']])await page.locator('#cl-deep img').nth(i).screenshot({style:'header{visibility:hidden!important}',path:path.join(out,name+'.png')});
 await go('bert-objectives');
 const objectiveFigures=['mlm-targets','mlm-head','shared-heads','roberta','spanbert','electra','albert','distilbert','deberta','xlmr','modernbert','neobert','mmbert','ettin'];
 assert.equal(await page.locator('#bert-objectives img').count(),objectiveFigures.length);
 for(let i=0;i<objectiveFigures.length;i++)await page.locator('#bert-objectives img').nth(i).screenshot({style:'header{visibility:hidden!important}',path:path.join(out,'objective-'+objectiveFigures[i]+'.png')});
 await page.locator('#bert-objectives a[href="#bert-objectives--переход-к-прикладной-задаче"]').first().click();
 assert.equal(await page.locator('article:not([hidden])').getAttribute('id'),'bert-objectives');
 await go('lm-transformer');await page.locator('#lm-transformer img[alt="Две ветви управляющей MLP с числовым примером"]').screenshot({path:path.join(out,'gated-mlp.png')});
 await go('lm-families');await page.locator('#lm-families img').first().screenshot({path:path.join(out,'transformer-families.png')});
 await go('lm-training');await page.locator('#lm-training img').first().screenshot({path:path.join(out,'training-paths.png')});
 const anchors=coverage.units.filter(u=>u.chapter).map(u=>({id:u.anchor,chapter:u.chapter}));
 const broken=await page.evaluate(entries=>entries.filter(e=>document.getElementById(e.id)?.closest('article')?.id!==e.chapter),anchors);assert.deepEqual(broken,[]);
 await go('reg-structure');assert.equal(await page.locator('#reg-structure [data-next]').getAttribute('href'),'#reg-selection');
 await go('lm-rnn');assert.equal(await page.locator('#lm-rnn [data-return]').getAttribute('href'),'#module-lm');
 await page.selectOption('#route','vision');await go('cv-classification');assert.equal(await page.locator('#cv-classification [data-next]').getAttribute('href'),'#cv-detection');
 await page.selectOption('#route','application');await go('app-rag');assert.equal(await page.locator('#app-rag [data-next]').getAttribute('href'),'#app-evaluation');
 await page.selectOption('#route','');
 const unit=coverage.units.find(u=>u.chapter==='lm-attention');await go(unit.anchor);assert.equal(await page.locator('article:not([hidden])').getAttribute('id'),'lm-attention');
 for(const project of course.projects||[]){
   await go(project.anchor);
   const state=await page.evaluate(()=>({hash:location.hash,visible:[...document.querySelectorAll('article:not([hidden]),#welcome:not([hidden])')].map(x=>x.id)}));
   assert.deepEqual(state.visible,[project.chapter],project.id+' navigation: '+JSON.stringify(state));
 }
 const all=await page.evaluate(ids=>{const failures=[];for(const id of ids){location.hash=id;showChapter();const a=document.getElementById(id);if(a.hidden)failures.push(id+': hidden');for(const img of a.querySelectorAll('img'))if(!img.complete||!img.naturalWidth)failures.push(id+': broken image');}return failures;},course.chapters.map(c=>c.id));
 assert.deepEqual(all,[]);
 const batch=JSON.parse(fs.readFileSync(path.join(root,'editorial/current-batch.json'),'utf8'));
 const batchFigures=[];
 for(const id of batch.chapters){
   await go(id);
   const imgs=page.locator('#'+id+' img');
   for(let i=0;i<await imgs.count();i++){
     const name=batch.id.replace('lm-','').replace('batch-','batch')+'-'+id+'-'+(i+1)+'.png';
     await imgs.nth(i).screenshot({style:'header{visibility:hidden!important}',path:path.join(out,name)});
     batchFigures.push(name);
   }
   const source=fs.readFileSync(path.join(root,course.chapters.find(c=>c.id===id).path),'utf8');
   if(/\$[^$\n]+\$|\$\$/m.test(source))assert((await page.locator('#'+id+' .katex').count())>0,id+' must render its formulas');
 }
 await go('bert-adaptation');
 const advancedExit=page.locator('#bert-adaptation a').filter({hasText:'Пропустить'});
 if(await advancedExit.count()){
   await advancedExit.first().click();
   assert.equal(await page.locator('article:not([hidden])').getAttribute('id'),'bert-adaptation');
 }
 await go('df-geography'); // Unknown hash must recover to home.
 assert.equal(await page.locator('#welcome').isVisible(),true);
 await go('feat-geography');await page.screenshot({path:path.join(out,'geography.png')});
 await go('cv-semantic');await page.locator('#cv-semantic img[alt="U-Net: skip concatenation"]').screenshot({path:path.join(out,'unet-skip.png')});
 await context.setOffline(true);await page.reload();assert.equal(await page.locator('#cv-semantic').isVisible(),true);assert.equal(await page.locator('.katex-error').count(),0);
 await page.setViewportSize({width:390,height:844});await go('welcome');await page.screenshot({path:path.join(out,'home-mobile.png')});
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'mobile document overflows');
 await go('lm-bert');assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),'mobile chapter overflows');
 assert.deepEqual(network,[]);assert.deepEqual(errors,[]);
 const htmlSha256=htmlFingerprint();assert.equal(htmlSha256,testedHtmlSha256,'HTML changed during browser verification; rerun on a fixed revision');
 const report={status:'passed',htmlSha256,host:process.platform,browser:'Chromium / Edge headless',chaptersVisited:course.chapters.length,projectsVisited:(course.projects||[]).length,topModules:5,languageSubsections:6,legacyBookmarks:['course-11-1','course-5-1','course-5-10'],sourceAnchors:true,allMappedChapterAnchors:anchors.length,objectiveFigures:objectiveFigures.length,objectiveBranchReturn:true,batchChapters:batch.chapters,batchFigures,figureZoom:true,skipAdvanced:true,selectedRoute:true,offline:true,mobile:true,networkRequests:network,pageErrors:errors,screenshots:fs.readdirSync(out).filter(x=>x.endsWith('.png'))};
 fs.writeFileSync(path.join(reportDir,'browser-audit.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report,null,2));await browser.close();
})().catch(e=>{
 if(reportDir)fs.writeFileSync(path.join(reportDir,'browser-audit.json'),JSON.stringify({status:'failed',htmlSha256:testedHtmlSha256,host:process.platform,error:e.stack},null,2)+'\n');
 console.error(e.stack);process.exitCode=1;
}).finally(async()=>{if(browser)await browser.close();});
