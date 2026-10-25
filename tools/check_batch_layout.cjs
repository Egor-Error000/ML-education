/* Read-only checks of long text and mobile layout for a stated chapter batch. */
const fs=require('fs'),path=require('path'),os=require('os'),assert=require('assert/strict');
const {pathToFileURL}=require('url'),crypto=require('crypto');
const root=path.resolve(__dirname,'..');
const {chromium}=require(require.resolve('playwright',{paths:[root,path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules')]}));
(async()=>{
 const oi=process.argv.indexOf('--output-dir');
 const out=oi<0?fs.mkdtempSync(path.join(os.tmpdir(),'course-layout-')):path.resolve(process.argv[oi+1]);fs.mkdirSync(out,{recursive:true});
 const batch=JSON.parse(fs.readFileSync(path.join(root,'editorial/current-batch.json'),'utf8'));
 const fingerprint=()=>crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'Читать конспекты.html'),'utf8').replace(/\r\n/g,'\n')).digest('hex');
 const initial=fingerprint(),browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try{
  const context=await browser.newContext(),page=await context.newPage(),network=[],errors=[];
  await context.route(/^https?:/,r=>{network.push(r.request().url());return r.abort();});page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(pathToFileURL(path.join(root,'Читать конспекты.html')).href);
  const go=async id=>{await page.evaluate(x=>{location.hash=x;showChapter();},id);await page.locator('#'+id).waitFor({state:'visible'});};
  const screenshots=[];
  await page.setViewportSize({width:1280,height:900});
  for(const id of ['adapt-lora','infer-quantization','infer-deployment'].filter(x=>batch.chapters.includes(x))){
   await go(id);await page.locator('#'+id+' h2').nth(2).scrollIntoViewIfNeeded();
   const name=id+'-desktop-text.png';await page.screenshot({path:path.join(out,name)});screenshots.push(name);
  }
  await page.setViewportSize({width:390,height:844});
  const checked=[];
  for(const id of batch.chapters){
   await go(id);
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2),id+' mobile document overflow');
   assert.equal(await page.locator('#'+id+' .katex-error').count(),0,id+' formula parse error');
   const state=await page.evaluate(id=>{const a=document.getElementById(id);return {height:a.scrollHeight,images:[...a.querySelectorAll('img')].length,missingAlt:[...a.querySelectorAll('img')].filter(i=>!i.alt.trim()).length};},id);
   assert.equal(state.missingAlt,0);checked.push({id,...state});
   if(['adapt-lora','infer-quantization','infer-deployment'].includes(id)){
    await page.locator('#'+id+' h2').nth(2).scrollIntoViewIfNeeded();const name=id+'-mobile-text.png';await page.screenshot({path:path.join(out,name)});screenshots.push(name);
   }
  }
  assert.deepEqual(network,[]);assert.deepEqual(errors,[]);assert.equal(fingerprint(),initial);
  const report={status:'passed',htmlSha256:initial,batch:batch.id,viewport:{width:390,height:844},chapters:checked,networkRequests:network,pageErrors:errors,screenshots};
  fs.writeFileSync(path.join(out,'layout.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({status:'passed',chapters:checked.length,screenshots:screenshots.length,htmlSha256:initial}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
