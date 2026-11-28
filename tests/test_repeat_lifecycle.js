// Run: node tests/test_repeat_lifecycle.js
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(require("node:path").join(__dirname, "../javascript/contextMenus.js"), "utf8");
const start = source.indexOf("    function startRepeat(");
const end = source.indexOf("    function cancelTxt2img(", start);
assert.ok(start >= 0 && end > start);

function setup({initial = null, submitId = null, reply = {completed: false, active: false, queued: false}} = {}) {
    let now = 1000, stored = initial, clicks = 0, cancelled = false, warnings = 0, tick;
    const context = vm.createContext({
        localGet: () => stored,
        nativeInfinite: () => false,
        idleGenerateButton: () => ({click: () => { clicks++; stored = submitId; }}),
        request: (url, data, ok) => ok(reply),
        setInterval: callback => { tick = callback; return 1; },
        Date: {now: () => now},
        console: {warn: () => {}},
        alert: () => { warnings++; },
        cancel: () => { cancelled = true; },
    });
    vm.runInContext(source.slice(start, end) + "\nstartRepeat('txt2img', cancel);", context);
    return {
        advance(ms) { now += ms; if (!cancelled) tick(); },
        clear() { stored = null; },
        get clicks() { return clicks; },
        get cancelled() { return cancelled; },
        get warnings() { return warnings; },
    };
}

{
    const state = setup();
    state.advance(500);
    assert.equal(state.clicks, 1);
    state.advance(40001);
    assert.equal(state.cancelled, true);
    assert.equal(state.clicks, 1);
    assert.equal(state.warnings, 1);
}
{
    const state = setup({initial: "forgotten"});
    state.clear();
    state.advance(500);
    state.advance(40001);
    assert.equal(state.cancelled, true);
    assert.equal(state.clicks, 0);
}
{
    const state = setup({initial: "stale"});
    state.advance(500);
    state.advance(40001);
    assert.equal(state.cancelled, true);
    assert.equal(state.clicks, 0);
}
{
    const state = setup({initial: "finished", reply: {completed: true, active: false, queued: false}, submitId: "next"});
    state.clear();
    state.advance(500);
    state.advance(500);
    assert.equal(state.clicks, 1);
    assert.equal(state.cancelled, false);
}
{
    const state = setup({initial: "running", reply: {completed: false, active: true, queued: false}});
    for (let i = 0; i < 100; i++) state.advance(1000);
    assert.equal(state.cancelled, false);
    assert.equal(state.clicks, 0);
}
console.log("5 repeat lifecycle scenarios passed");
