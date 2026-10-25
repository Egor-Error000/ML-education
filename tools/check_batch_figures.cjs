/* Inspect the authored SVG text in Chromium, without contacting the network. */
const fs=require('fs'),path=require('path'),os=require('os');
const root=path.resolve(__dirname,'..');
const {chromium}=require(require.resolve('playwright',{paths:[root,path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules')]}));
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try{
  const page=await browser.newPage();
  const registry=JSON.parse(fs.readFileSync(path.join(root,'editorial/illustrations.json'),'utf8'));
  const result=[];
  const arg=process.argv.indexOf('--production');
  const production=arg>=0?process.argv[arg+1]:'tools/draw_batch10_figures.py';
  const figures=registry.figures.filter(x=>x.production===production);
  if(!figures.length)throw Error('No registered illustrations for '+production);
  for(const figure of figures){
   await page.setContent(fs.readFileSync(path.join(root,figure.path),'utf8'));
   const issues=await page.evaluate(()=>{
    const svg=document.querySelector('svg'),v=svg.viewBox.baseVal,issues=[];
    for(const text of svg.querySelectorAll('text')){
     const b=text.getBBox(),r=text.previousElementSibling;
     if(b.x<0||b.y<0||b.x+b.width>v.width||b.y+b.height>v.height)issues.push('outside SVG: '+text.textContent);
     if(r?.tagName==='rect'&&r.getAttribute('rx')){
      const x=+r.getAttribute('x'),y=+r.getAttribute('y'),w=+r.getAttribute('width'),h=+r.getAttribute('height');
      if(b.x<x+3||b.y<y||b.x+b.width>x+w-3||b.y+b.height>y+h-3)issues.push('outside box: '+text.textContent);
     }
    }
    return issues;
   });
   if(issues.length)result.push({path:figure.path,issues});
  }
  const report={status:result.length?'failed':'passed',production,checked:figures.length,issues:result};
  const oi=process.argv.indexOf('--output');
  if(oi>=0){const out=path.resolve(process.argv[oi+1]);fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify(report,null,2)+'\n');}
  console.log(JSON.stringify(report,null,2));if(result.length)process.exitCode=1;
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
