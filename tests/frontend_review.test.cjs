// CR-06/07/08：执行实际页面脚本，用隔离 DOM/网络替身验证异步行为。
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('app/server/static/index.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page() {
  const nodes = new Map();
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, {
      value: '', style: {}, lastElementChild: {}, classList: {add() {}},
      children: [], listeners: {},
      addEventListener(type, action) {this.listeners[type] = action;}, focus() {}, close() {},
      replaceChildren(...children) {this.children = children;},
      append(...children) {this.children.push(...children);},
      prepend(...children) {this.children.unshift(...children);},
      setAttribute() {}, showModal() {},
      remove() {}, scrollHeight: 10,
    });
    return nodes.get(id);
  };
  const storage = () => {
    const values = new Map();
    return {getItem: key => values.get(key) || null, setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key)};
  };
  const pending = [];
  const messages = [];
  const context = vm.createContext({
    Headers, AbortController, console, localStorage: storage(), sessionStorage: storage(),
    setTimeout: () => 1, clearTimeout() {}, setInterval() {},
    document: {getElementById: node, createElement: () => node(Symbol()), querySelectorAll: () => [], querySelector: () => null},
    fetch: (url) => url === '/health/ready'
      ? Promise.resolve(new Response(JSON.stringify({active_sessions: 0, capabilities: {}})))
      : new Promise(resolve => pending.push(resolve)),
    record: (role, text) => {messages.push({role, text}); return {remove() {}, classList: {add() {}}, querySelector: () => ({classList: {add() {}}})};},
  });
  vm.runInContext(script, context);
  vm.runInContext("addMessage = record; state.token = 'token-a';", context);
  return {context, pending, messages, node};
}

test('late chat response is ignored after a session switch', async () => {
  const p = page();
  const sent = vm.runInContext("sendMessage('question')", p.context);
  vm.runInContext("state.sessionId = 'another'; clearScreen();", p.context);
  p.pending.shift()(new Response(JSON.stringify({result: {reply: 'old private answer'}})));
  await sent;
  assert.equal(p.messages.some(m => m.text.includes('old private answer')), false);
  assert.equal(vm.runInContext('state.busy', p.context), false);
});

test('upstream failure preserves login and restores the draft', async () => {
  const p = page();
  const sent = vm.runInContext("sendMessage('retry this question')", p.context);
  p.pending.shift()(new Response(JSON.stringify({detail: 'upstream unavailable'}), {status: 502}));
  await sent;
  assert.equal(vm.runInContext('state.token', p.context), 'token-a');
  assert.equal(p.node('message').value, 'retry this question');
  assert.equal(p.node('send-btn').disabled, false);
});

test('refund requires explicit acknowledgement and suppresses duplicate submit', async () => {
  const p = page();
  vm.runInContext(`state.userId = 'owner'; renderRefundPreview({order_id:'ORD-1', amount:10,
    reason:'test', preview_token:'opaque', expires_at: Date.now()/1000+600}, state.viewVersion, 'owner');`, p.context);
  const content = p.node('dialog-body').children;
  const label = content.find(n => n.textContent?.startsWith('我已核对'));
  const checkbox = label.children[0];
  const submit = content.find(n => n.textContent === '明确确认并提交');
  assert.equal(submit.disabled, true);
  await submit.listeners.click();
  assert.equal(p.pending.length, 0);
  checkbox.checked = true;
  checkbox.listeners.change();
  assert.equal(submit.disabled, false);
  const sent = submit.listeners.click();
  await submit.listeners.click();
  assert.equal(p.pending.length, 1);
  assert.equal(vm.runInContext("sessionStorage.getItem('xiaofei-refund-owner')", p.context), 'opaque');
  p.pending.shift()(new Response(JSON.stringify({application_id:'RF-1', order_id:'ORD-1', status:'processing'})));
  await sent;
  assert.equal(vm.runInContext("sessionStorage.getItem('xiaofei-refund-owner')", p.context), null);
  assert.ok(p.messages.some(m => m.text.includes('RF-1')));
});

test('uncertain refund keeps original token and offers read-only resolution', async () => {
  const p = page();
  vm.runInContext(`state.userId = 'owner'; renderRefundPreview({order_id:'ORD-1', amount:10,
    reason:'test', preview_token:'opaque', expires_at: Date.now()/1000+600}, state.viewVersion, 'owner');`, p.context);
  const content = p.node('dialog-body').children;
  const checkbox = content.find(n => n.textContent?.startsWith('我已核对')).children[0];
  const submit = content.find(n => n.textContent === '明确确认并提交');
  checkbox.checked = true;
  const sent = submit.listeners.click();
  p.pending.shift()(new Response(JSON.stringify({detail:'unavailable'}), {status:503}));
  await sent;
  assert.equal(vm.runInContext("sessionStorage.getItem('xiaofei-refund-owner')", p.context), 'opaque');
  assert.equal(submit.disabled, true);
  assert.ok(p.node('dialog-body').children.some(n => n.textContent === '核实本次申请'));
});

test('expired refund preview cannot submit', async () => {
  const p = page();
  vm.runInContext(`renderRefundPreview({order_id:'ORD-1', amount:10, reason:'test',
    preview_token:'opaque', expires_at:1}, state.viewVersion, 'owner');`, p.context);
  const content = p.node('dialog-body').children;
  content.find(n => n.textContent?.startsWith('我已核对')).children[0].checked = true;
  await content.find(n => n.textContent === '明确确认并提交').listeners.click();
  assert.equal(p.pending.length, 0);
});

test('late refund receipt is hidden after account switch', async () => {
  const p = page();
  vm.runInContext(`state.userId='owner'; renderRefundPreview({order_id:'ORD-1', amount:10,
    reason:'test', preview_token:'opaque', expires_at:Date.now()/1000+600}, state.viewVersion, 'owner');`, p.context);
  const content = p.node('dialog-body').children;
  content.find(n => n.textContent?.startsWith('我已核对')).children[0].checked = true;
  const sent = content.find(n => n.textContent === '明确确认并提交').listeners.click();
  vm.runInContext("state.userId='other'; clearScreen();", p.context);
  p.pending.shift()(new Response(JSON.stringify({application_id:'PRIVATE-RF', order_id:'ORD-1', status:'processing'})));
  await sent;
  assert.equal(p.messages.some(m => m.text.includes('PRIVATE-RF')), false);
});
