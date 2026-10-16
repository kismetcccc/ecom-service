const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {webcrypto} = require('node:crypto');
const script = fs.readFileSync('app/server/static/index.html', 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];

function page() {
  class Element {
    constructor() { this.children = []; this.events = {}; this.style = {}; this.value = ''; this.textContent = ''; this.open = false; this.lastElementChild = {}; this.classList = {add(){}}; }
    addEventListener(name, fn) { this.events[name] = fn; }
    append(...nodes) { this.children.push(...nodes); }
    appendChild(node) { this.append(node); }
    replaceChildren(...nodes) { this.children = nodes; }
    setAttribute() {} focus() {} remove() {} close() { this.open = false; }
    showModal() { this.open = true; } reportValidity() { return true; }
  }
  const elements = new Map();
  const node = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const storage = () => { const map = new Map(); return {getItem:k=>map.get(k)||null,setItem:(k,v)=>map.set(k,v),removeItem:k=>map.delete(k)}; };
  const calls = [];
  const responses = [];
  const context = vm.createContext({Headers, AbortController, crypto:webcrypto, console, confirm:()=>true,
    setTimeout:()=>1, clearTimeout(){}, setInterval(){}, localStorage:storage(), sessionStorage:storage(),
    document:{getElementById:node, createElement:()=>new Element(), querySelectorAll:()=>[], querySelector:()=>null, body:new Element()},
    fetch:async(url, options)=> {
      if (url === '/health/ready') return new Response(JSON.stringify({capabilities:{}}));
      calls.push({url, options});
      const result = responses.shift();
      if (result instanceof Error) throw result;
      return new Response(JSON.stringify(result?.data || {}), {status:result?.status || 200});
    }});
  vm.runInContext(script, context);
  vm.runInContext("state.token='test-token'; state.userId='new-user'; addMessage=()=>({}); showToast=()=>{};", context);
  const descendants = node => [node, ...node.children.flatMap(descendants)];
  return {context,node,calls,responses,button:text=>descendants(node('dialog-body')).find(n=>n.textContent===text)};
}

test('catalog renders safe text and checkout retries the same payload after response loss', async()=> {
  const p=page();
  p.responses.push({data:{products:[{product_id:'P1',name:'<script>unsafe</script>',category:'鞋',description:'说明',price:19.9,stock:3,specs:{}}]}});
  await vm.runInContext('showShop()',p.context);
  assert.equal(p.node('dialog-title').textContent,'挑选商品');
  assert.ok(p.button('<script>unsafe</script>')); // textContent, never HTML interpolation
  p.button('选择并下单').events.click();
  p.button('下一步：确认订单').events.click();
  p.responses.push(new Error('network disconnected'));
  await p.button('确认创建订单').events.click();
  assert.ok(vm.runInContext("sessionStorage.getItem('xiaofei-purchase-new-user')",p.context));
  p.responses.push({data:{order:{order_id:'ORD-TEST',status:'purchased',total:19.9}}});
  await p.button('使用原凭据重试并核实').events.click();
  const posts=p.calls.filter(c=>c.url==='/v1/orders');
  assert.equal(posts.length,2);
  assert.equal(posts[0].options.body,posts[1].options.body);
  assert.equal(JSON.parse(posts[0].options.body).user_id,undefined);
  assert.equal(vm.runInContext("sessionStorage.getItem('xiaofei-purchase-new-user')",p.context),null);
  assert.equal(p.node('dialog-title').textContent,'购买成功');
});

test('stale confirmation cannot place an order for a switched account',async()=> {
  const p=page();
  vm.runInContext("confirmPurchase({product_id:'P1',quantity:1,expected_price:19.9,idempotency_key:'checkout-key-001'})",p.context);
  const button=p.button('确认创建订单');
  vm.runInContext("state.viewVersion++; state.userId='other-user';",p.context);
  await button.events.click();
  assert.equal(p.calls.length,0);
});

test('price conflicts require fresh confirmation and release the pending request',async()=> {
  const p=page();
  vm.runInContext("confirmPurchase({product_id:'P1',quantity:1,expected_price:19.9,idempotency_key:'checkout-key-001'})",p.context);
  p.responses.push({status:409,data:{detail:'价格已变化'}});
  await p.button('确认创建订单').events.click();
  assert.equal(p.button('请返回商品重新核对').disabled,true);
  assert.equal(vm.runInContext("sessionStorage.getItem('xiaofei-purchase-new-user')",p.context),null);
});

test('purchase text opens the catalog even before a model reply',async()=> {
  const p=page();
  vm.runInContext("var opened=0; showShop=async()=>{opened++}; addTyping=()=>({remove(){}});",p.context);
  p.responses.push({data:{result:{reply:'请选择商品',intent:'product_consult'}}});
  await vm.runInContext("sendMessage('我想买一双鞋')",p.context);
  assert.equal(vm.runInContext('opened',p.context),1);
});

test('purchased orders expose details cancellation and simulated refund', async()=> {
  const p=page();
  p.responses.push({data:{orders:[{order_id:'ORD-NEW',status:'purchased',purchase_mode:'simulation',
    items_summary:'测试商品',total:19.9,created_at:'2026-10-12'}]}});
  await vm.runInContext('showOrders()',p.context);
  assert.ok(p.button('已购买 · ¥19.90'));
  assert.ok(p.button('申请模拟退款'));
  assert.ok(p.button('取消订单并释放库存'));
  p.responses.push({data:{order:{order_id:'ORD-NEW',status:'purchased',purchase_mode:'simulation',total:19.9,
    items:[{name:'测试商品',quantity:1,price:19.9}],created_at:'2026-10-12'}}});
  await p.button('订单详情').events.click();
  assert.equal(p.node('dialog-title').textContent,'订单详情');
  assert.ok(p.button('尚未发货，暂无物流单号。'));
});

test('refunded orders cannot be cancelled or refunded again', async()=> {
  const p=page();
  p.responses.push({data:{orders:[{order_id:'ORD-NEW',status:'refunded',purchase_mode:'simulation',
    items_summary:'测试商品',total:19.9,created_at:'2026-10-12'}]}});
  await vm.runInContext('showOrders()',p.context);
  assert.ok(p.button('已退款 · ¥19.90'));
  assert.equal(p.button('申请模拟退款'),undefined);
  assert.equal(p.button('取消订单并释放库存'),undefined);
});
