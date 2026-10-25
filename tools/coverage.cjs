/* Sidecar anchors are injected into HTML only. No lesson source is rewritten. */
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const normalize=text=>text.replace(/\r\n/g,'\n');
const digest=text=>crypto.createHash('sha256').update(text).digest('hex');
function chapterWithAnchors(root,file,text,coverage){
 const lines=normalize(text).split('\n'),at=new Map();
 const relevant=Object.entries(coverage.spans).filter(([,s])=>s.path===file);
 for(const [id,s] of relevant){
  const block=lines.slice(s.start_line-1,s.end_line).join('\n');
  if(digest(block)!==s.sha256){
   const origin=coverage.units.find(u=>u.targets.includes(id));
   throw Error(`Stale coverage span ${id} in ${file}:${s.start_line}-${s.end_line}; source ${origin?.source}:${origin?.source_lines.join('-')}. Run python tools/review_coverage.py --chapter ${coverage.units.find(u=>u.targets.includes(id)&&u.chapter)?.chapter||'CHAPTER'} for exact candidates; changed text requires an explicit reviewed range.`);
  }
  if(!at.has(s.start_line))at.set(s.start_line,[]);
  at.get(s.start_line).push(id);
 }
 return lines.map((line,i)=>(at.has(i+1)?at.get(i+1).map(id=>`<a id="${id}"></a>`).join('\n')+'\n\n':'')+line).join('\n');
}
module.exports={normalize,digest,chapterWithAnchors};
