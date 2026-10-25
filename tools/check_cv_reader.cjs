/* Read-only acceptance of the expanded CV routes, offline figures and mobile UI. */
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert'),crypto=require('crypto');
const {pathToFileURL}=require('url');
const root=path.resolve(__dirname,'..');
const pw=require(require.resolve('playwright',{paths:[root,process.env.COURSE_NODE_MODULES||path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules')]}));
const course=JSON.parse(fs.readFileSync(path.join(root,'course.json'),'utf8'));
const cv=course.chapters.filter(c=>c.module==='cv');
const external=JSON.parse(fs.readFileSync(path.join(root,'editorial/cv-expansion.json'),'utf8')).external_figures;
const out=path.resolve(root,'.qa-results/cv-expansion/cv-browser');fs.mkdirSync(out,{recursive:true});
const htmlSha256=crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'Читать конспекты.html'))).digest('hex');
let browser;
(async()=>{
 const edge='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
 browser=await pw.chromium.launch({headless:true,...(fs.existsSync(edge)?{executablePath:edge}:{})});
 console.log('CV browser launched');
 const context=await browser.newContext({viewport:{width:1440,height:1000}});
 const requests=[],errors=[];
 await context.route(/^https?:\/\//,r=>{requests.push(r.request().url());return r.abort();});
 const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(path.join(root,'Читать конспекты.html')).href);await page.evaluate(()=>document.fonts.ready);
 console.log('Offline reader loaded');
 assert.equal(await page.locator('aside [data-module="cv"] > details.subsection').count(),9);
 const go=async id=>{await page.evaluate(id=>{location.hash=id;showChapter();},id);};
 const routeResults=[];
 for(const r of course.routes.filter(r=>r.id.startsWith('vision'))){
  await page.selectOption('#route',r.id);
  for(const i of [...new Set([0,Math.floor(r.chapters.length/2),r.chapters.length-1])]){
   const id=r.chapters[i];await go(id);
   assert.equal(await page.locator('article:not([hidden])').getAttribute('id'),id);
   const next=page.locator('#'+id+' [data-next]');
   if(i+1<r.chapters.length)assert.equal(await next.getAttribute('href'),'#'+r.chapters[i+1]);
   else assert(await next.isHidden());
  }
  routeResults.push({id:r.id,chapters:r.chapters.length});
  console.log('Route checked:',r.id);
 }
 await page.selectOption('#route','');
 const chapterChecks=await page.evaluate(async ids=>{
  const results=[];
  for(const id of ids){
   const article=document.getElementById(id),imgs=[...article.querySelectorAll('img')];
   await Promise.all(imgs.map(i=>i.decode()));
   results.push({id,answers:article.textContent.includes('Самопроверка с ответами'),images:imgs.length,imagesValid:imgs.every(i=>i.naturalWidth>0&&i.src.startsWith('data:')),mathErrors:article.querySelectorAll('.katex-error').length});
  }
  return results;
 },cv.map(c=>c.id));
 assert(chapterChecks.every(c=>c.answers&&c.imagesValid&&!c.mathErrors));
 let imagesChecked=chapterChecks.reduce((n,c)=>n+c.images,0);
 const screenshots=['cv-resnet','cv-yolo','cv-mask2former','cv-video','cv-nerf','cv-gaussians','cv-text-recognition','cv-latent-diffusion','cv-cases'];
 for(const c of cv.filter(c=>screenshots.includes(c.id))){
  await go(c.id);
  assert((await page.locator('#'+c.id).innerText()).includes('Самопроверка с ответами'));
  assert.equal(await page.locator('#'+c.id+' .katex-error').count(),0);
  if(screenshots.includes(c.id)){
   await page.locator('#'+c.id+' h1').scrollIntoViewIfNeeded();
   await page.screenshot({path:path.join(out,c.id+'-desktop.png')});
  }
 }
 await go('cv-yolo');
 const authorFigure=page.locator('#cv-yolo img[alt*="RangeKing"]');await authorFigure.scrollIntoViewIfNeeded();await authorFigure.click();
 assert(await page.locator('#figure-view').isVisible());
 assert.equal(await page.locator('#figure-full').evaluate(i=>i.clientWidth),3200,'large source figure must retain readable detail');
 await page.screenshot({path:path.join(out,'yolov8-zoom.png')});await page.keyboard.press('Escape');
 await context.setOffline(true);await page.reload();await page.evaluate(()=>document.fonts.ready);
 assert.equal(await page.locator('article:not([hidden])').getAttribute('id'),'cv-yolo');
 await page.setViewportSize({width:390,height:844});
 for(const id of ['module-cv','cv-yolo','cv-ddpm','cv-pointnet','cv-text-recognition']){
  await go(id);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),id+' mobile overflow');
  await page.screenshot({path:path.join(out,id+'-mobile.png')});
 }
 await page.fill('#search','Gaussian');
 assert(await page.locator('aside a[data-chapter="cv-gaussians"]').isVisible());
 assert.deepEqual(requests,[]);assert.deepEqual(errors,[]);
 assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'Читать конспекты.html'))).digest('hex'),htmlSha256,'HTML changed during verification');
 const report={status:'passed',htmlSha256,cvChapters:cv.length,cvSections:9,routeResults,imagesChecked,externalFigures:external.length,offline:true,mobile:true,figureZoom:true,search:true,pageErrors:errors,networkRequests:requests};
 fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
})().catch(e=>{console.error(e.stack);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();});
