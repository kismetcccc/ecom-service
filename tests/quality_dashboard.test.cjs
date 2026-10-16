const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('app/server/static/quality.html','utf8');
const context = {module:{exports:{}}};
vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
const {qualityModel,qualityDelta} = context.module.exports;
const report = () => ({report_type:'engineering_contract',schema_version:1,test_sha256:'same',suites:[{name:'react',tests:1,passed:1,wall_ms:10,status:'completed',cases:[{status:'passed'}]}]});
test('dashboard recomputes counts instead of trusting supplied summary',()=>{
  const data=report();data.summary={tests:999,passed:999};
  assert.equal(qualityModel(data).total,1);
  data.suites[0].cases[0].status='failed';
  assert.throws(()=>qualityModel(data));
});
test('dashboard refuses incomparable or empty baselines',()=>{
  const a=qualityModel(report()),other=report();other.test_sha256='different';
  assert.equal(qualityDelta(a,qualityModel(other)),null);
  assert.equal(qualityDelta(a,qualityModel(report())),0);
});
test('timeout cannot show a passing gate',()=>{
  const data=report();data.suites[0].status='timeout';assert.equal(qualityModel(data).gate,false);
});
test('quantitative dashboard preserves unknown and rejects fabricated numeric shapes',()=>{
  const d={report_type:'quantitative_metrics',schema_version:1,cases:[],metrics:[{category:'performance',key:'ttft',label:'TTFT',value:null,unit:'ms',samples:0,scope:'non-streaming'}]};
  assert.equal(qualityModel(d).data.metrics[0].value,null);
  d.metrics[0].value=-1;assert.throws(()=>qualityModel(d));
  d.metrics[0].value=Infinity;assert.throws(()=>qualityModel(d));
  d.metrics[0].value=1.2;d.metrics[0].unit='ratio';assert.throws(()=>qualityModel(d));
  d.metrics[0].value=0;d.metrics.push({...d.metrics[0]});assert.throws(()=>qualityModel(d));
});
test('actual generated report can be imported',()=>{
  const path='reports/quant-overview.json';
  if(fs.existsSync(path))assert.equal(qualityModel(JSON.parse(fs.readFileSync(path,'utf8'))).kind,'quantitative');
});
