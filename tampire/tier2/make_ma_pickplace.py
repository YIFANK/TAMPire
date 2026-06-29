"""Build the multi-agent HTML dashboard for the fixed-base long-horizon SORT
(trace_pickplace.json), matching the recovery dashboard's look. Per-item goals
(in(item, bin)) tracked by Gemma's in_bin checks. Writes runs/ma_pickplace.html.

    python -m tampire.tier2.make_ma_pickplace
"""
import json
import os

R = "/Users/yifankang/TAMPire/runs"
TRACE = os.path.join(R, "trace_pickplace.json")
OUT = os.path.join(R, "ma_pickplace.html")

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>TAMPire — Multi-Agent Sort</title>
<style>
:root{
  --font-sans:'Lato','Inter',-apple-system,Segoe UI,Roboto,sans-serif;
  --font-mono:'SFMono-Regular',Menlo,Consolas,monospace;
  --bg:#0a0e14; --surface-0:#0c1119; --surface-1:#121823; --border:#1f2a38;
  --border-strong:#2c3a4d; --border-danger:#5a2330;
  --text-primary:#e6edf6; --text-secondary:#aebacd; --text-muted:#6f7d93;
  --text-success:#5dcaa5; --bg-success:#0e2a20; --text-danger:#ff6b78;
  --radius:7px;
}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1100px 650px at 75% -12%,#13243800,#070a10),var(--bg);
  color:var(--text-primary);font-family:var(--font-sans);padding:18px 22px}
.tm{max-width:1180px;margin:0 auto}
.top-bar{display:flex;align-items:baseline;gap:12px;margin-bottom:12px}
.top-title{font-size:22px;font-weight:900;letter-spacing:.3px}
.top-title .o{color:#85b7eb}
.top-sub{font-size:12.5px;color:var(--text-muted)}
.top-badge{margin-left:auto;font-size:11px;padding:4px 11px;border-radius:999px;background:var(--bg-success);color:var(--text-success);font-weight:700}
.top-badge .d{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--text-success);margin-right:5px;vertical-align:1px;animation:pulse 1.3s infinite}
.speed{margin-left:10px;font-size:11px;padding:4px 11px;border-radius:999px;background:#0d2438;border:1px solid #1d4d6b;color:#7fd0ff;font-family:var(--font-mono)}
@keyframes pulse{50%{opacity:.35}}

.dash{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:10px}
.agent-card{background:var(--surface-1);border-radius:12px;border:1px solid var(--border);padding:13px 15px;transition:.3s}
.agent-card.active{border-color:var(--ac);box-shadow:0 0 0 1px var(--ac),0 8px 24px rgba(0,0,0,.3)}
.agent-header{display:flex;align-items:center;gap:8px;margin-bottom:11px}
.agent-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;background:var(--ac)}
.agent-card.active .agent-dot{box-shadow:0 0 9px var(--ac);animation:pulse 1s infinite}
.agent-name{font-size:13.5px;font-weight:700}
.agent-role{font-size:11px;color:var(--text-muted);margin-left:auto;font-family:var(--font-mono)}
.agent-state{font-size:12px;padding:5px 11px;border-radius:var(--radius);margin-bottom:10px;font-weight:600;min-height:28px}
.c-green{--ac:#5dcaa5}.c-amber{--ac:#ef9f27}.c-blue{--ac:#85b7eb}
.c-green .agent-state{background:#0c2c23;color:#9fe1cb}
.c-amber .agent-state{background:#3a2810;color:#fac775}
.c-blue .agent-state{background:#0c2c4a;color:#b5d4f4}
.agent-log{font-size:11px;line-height:1.5;color:var(--text-secondary);font-family:var(--font-mono)}
.agent-log-entry{padding:3px 0;border-bottom:1px solid var(--border);display:flex;gap:7px;opacity:0;transform:translateY(4px);transition:.25s}
.agent-log-entry.on{opacity:1;transform:none}
.agent-log-entry:last-child{border-bottom:none}
.log-ts{color:var(--text-muted);flex-shrink:0;min-width:30px}
.log-msg{flex:1}.log-msg.bad{color:#ff9aa6}.log-msg.good{color:#9fe1cb}

.goal-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.goal-pill{font-size:11.5px;padding:4px 10px;border-radius:var(--radius);border:1px solid var(--border);color:var(--text-secondary);display:flex;align-items:center;gap:5px;font-family:var(--font-mono);transition:.3s}
.goal-pill.done{border-color:#1d9e75;color:#9fe1cb}
.goal-pill .ck{width:13px;height:13px;border-radius:4px;border:1.5px solid var(--text-muted);display:inline-flex;align-items:center;justify-content:center;font-size:10px;color:#0a0e14}
.goal-pill.done .ck{background:#1d9e75;border-color:#1d9e75}

.obs-row{display:grid;grid-template-columns:1.25fr 1fr;gap:10px;margin-bottom:10px}
.obs-panel,.timeline{background:var(--surface-1);border-radius:12px;border:1px solid var(--border);padding:13px 15px}
.section-label{font-size:10.5px;color:var(--text-muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;font-weight:700}
.obs-wrap{position:relative;border-radius:var(--radius);overflow:hidden;background:#05080d}
.obs-wrap img{width:100%;display:block}
.obs-cap{position:absolute;left:8px;bottom:8px;right:8px;background:#000a;border:1px solid var(--border);border-radius:6px;padding:5px 9px;font-size:11px;backdrop-filter:blur(4px)}
.obs-cap .k{font-size:9px;letter-spacing:1px;text-transform:uppercase;color:var(--text-muted)}

.tl-entries{position:relative;padding-left:22px}
.tl-line{position:absolute;left:5px;top:6px;bottom:6px;width:1.5px;background:var(--border-strong)}
.tl-entry{display:flex;align-items:flex-start;gap:7px;padding:4px 0;position:relative;font-size:11.5px;opacity:0;transform:translateX(-6px);transition:.25s;font-family:var(--font-mono)}
.tl-entry.on{opacity:1;transform:none}
.tl-dot{width:10px;height:10px;border-radius:50%;border:2px solid;flex-shrink:0;position:absolute;left:-21px;top:4px;background:var(--surface-1)}
.tl-from{font-weight:700;color:var(--text-primary);min-width:52px;flex-shrink:0}
.tl-arrow{color:var(--text-muted)}.tl-to{color:var(--text-secondary);min-width:48px;flex-shrink:0}
.tl-msg{color:var(--text-secondary);flex:1}.tl-msg.bad{color:#ff9aa6}.tl-msg.good{color:#9fe1cb}
.tl-time{color:var(--text-muted);font-size:10.5px;flex-shrink:0}

.input-bar{background:var(--surface-1);border-radius:12px;border:1px solid var(--border);padding:9px 12px;display:flex;gap:9px;align-items:center}
.input-bar .term{color:var(--text-muted);font-family:var(--font-mono)}
.input-bar input{flex:1;border:none;background:none;font-size:13px;color:var(--text-primary);outline:none;font-family:var(--font-mono)}
.input-bar input::placeholder{color:var(--text-muted)}
.input-btn{font-size:12px;padding:5px 13px;border-radius:var(--radius);border:1px solid var(--border-strong);background:none;color:var(--text-primary);cursor:pointer}
.input-btn:hover{background:var(--surface-0)}
.input-btn.stop{border-color:var(--border-danger);color:var(--text-danger)}
.ctrl{display:flex;gap:8px;align-items:center;margin-top:10px}
.play{background:#85b7eb;color:#04121f;border:none;border-radius:9px;padding:8px 18px;font-weight:800;cursor:pointer}
.win{margin-left:auto;font-weight:800;color:var(--text-success);opacity:0;transition:.4s}.win.on{opacity:1}
</style></head>
<body><div class="tm">
  <div class="top-bar">
    <span class="top-title">TAMP<span class="o">ire</span></span>
    <span class="top-sub" id="sub">fixed-base long-horizon sort</span>
    <span class="top-badge"><span class="d"></span>3 agents active</span>
    <span class="speed" id="speed">— ms/check</span>
  </div>

  <div class="dash">
    <div class="agent-card c-green" id="ag-perc">
      <div class="agent-header"><span class="agent-dot"></span><span class="agent-name">Gemma perception</span><span class="agent-role">Gemma-4</span></div>
      <div class="agent-state" id="st-perc">idle</div>
      <div class="agent-log" id="log-perc"></div>
    </div>
    <div class="agent-card c-amber" id="ag-plan">
      <div class="agent-header"><span class="agent-dot"></span><span class="agent-name">TAMP planner</span><span class="agent-role">symbolic</span></div>
      <div class="agent-state" id="st-plan">idle</div>
      <div class="goal-row" id="goals"></div>
      <div class="agent-log" id="log-plan"></div>
    </div>
    <div class="agent-card c-blue" id="ag-motor">
      <div class="agent-header"><span class="agent-dot"></span><span class="agent-name">Motor controller</span><span class="agent-role">OSC</span></div>
      <div class="agent-state" id="st-motor">idle</div>
      <div class="agent-log" id="log-motor"></div>
    </div>
  </div>

  <div class="obs-row">
    <div class="obs-panel">
      <div class="section-label">observation · <span id="obs-task">robosuite PickPlace</span></div>
      <div class="obs-wrap"><img id="vid"/><div class="obs-cap"><div class="k" id="capk">—</div><div id="capt">…</div></div></div>
    </div>
    <div class="timeline">
      <div class="section-label">message bus</div>
      <div class="tl-entries"><div class="tl-line"></div><div id="bus"></div></div>
    </div>
  </div>

  <div class="input-bar">
    <span class="term">›_</span>
    <input type="text" placeholder="override: skip item / re-order bins / pause / set goal..." readonly />
    <button class="input-btn">send</button>
    <button class="input-btn stop">e-stop</button>
  </div>
  <div class="ctrl"><button class="play" id="play">▶ Play</button><button class="play" id="restart" style="background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border)">⟲ Restart</button>
    <span class="win" id="win">✓ NATIVE SUCCESS · 3/3 SORTED</span></div>
</div>
<script>
const T = __TRACE__;
const $=s=>document.querySelector(s);
const FR=T.frames, EV=T.events, N=FR.length, FPS=12;
$("#sub").textContent=T.subtitle||"fixed-base long-horizon sort";
$("#obs-task").textContent=(T.task||"robosuite PickPlace");
$("#speed").textContent=(T.metrics&&T.metrics.ms_each?T.metrics.ms_each:"~900")+" ms / Gemma check";

// items in sort order, parsed from holding()/in_bin() checks
const ITEMS=[]; EV.forEach(e=>{const m=/(?:holding|in_bin)\(([A-Za-z]+)\)/.exec(e.text||""); if(m&&!ITEMS.includes(m[1]))ITEMS.push(m[1]);});
const BIN={Cereal:"bin 3",Can:"bin 4",Bread:"bin 2"};

const AG={perception:"perc",plan:"plan",replan:"plan",action:"motor",check:"perc",success:"plan"};
function ts(k){return (k/FPS).toFixed(1)+"s";}
function shorten(s,n){s=s.replace(/[“”]/g,'"');return s.length>n?s.slice(0,n-1)+"…":s;}
function curItem(e){const m=/(?:holding|in_bin)\(([A-Za-z]+)\)/.exec(e.text||"");return m?m[1]:null;}

function busRow(e){
  if(e.kind==="perception") return {f:"Gemma",t:"Planner",m:"items localized = 3",c:"#5dcaa5"};
  if(e.kind==="plan") return {f:"Planner",t:"Motor",m:shorten(e.text.replace(/^plan:\s*/,""),30),c:"#ef9f27"};
  if(e.kind==="action") return {f:"Planner",t:"Motor",m:shorten(e.text,30),c:"#85b7eb"};
  if(e.kind==="check"){const it=curItem(e);const hold=/holding/.test(e.text);
    return {f:"Gemma",t:"Planner",m:(hold?"holding":"in_bin")+"("+(it?it.toLowerCase():"obj")+")="+(e.ok?"true":"false"),c:e.ok?"#5dcaa5":"#e24b4a",bad:!e.ok,good:e.ok};}
  if(e.kind==="success") return {f:"Planner",t:"—",m:"all items sorted ✓",c:"#5dcaa5",good:true};
  return null;
}
const logEls={perc:$("#log-perc"),plan:$("#log-plan"),motor:$("#log-motor")};
EV.forEach((e,i)=>{
  const ag=AG[e.kind]; if(ag){
    const r=document.createElement("div");r.className="agent-log-entry";r.id="le"+i;
    const cls=e.kind==="check"?(e.ok?"good":"bad"):"";
    r.innerHTML=`<span class="log-ts">${ts(e.k)}</span><span class="log-msg ${cls}">${shorten(e.text.replace(/—.*/,'').replace(/"[^"]*"/,''),46)}</span>`;
    logEls[ag].appendChild(r);
  }
  const b=busRow(e); if(b){
    const r=document.createElement("div");r.className="tl-entry";r.id="be"+i;
    r.innerHTML=`<span class="tl-dot" style="border-color:${b.c}"></span><span class="tl-from">${b.f}</span><span class="tl-arrow">→</span><span class="tl-to">${b.t}</span><span class="tl-msg ${b.bad?'bad':b.good?'good':''}">${b.m}</span><span class="tl-time">${ts(e.k)}</span>`;
    $("#bus").appendChild(r);
  }
});
// per-item goals
ITEMS.forEach(it=>{const p=document.createElement("span");p.className="goal-pill";p.id="g-"+it;
  const b=BIN[it]||"bin";p.innerHTML=`<span class="ck">✓</span>in(${it.toLowerCase()}, ${b})`;$("#goals").appendChild(p);});

let fi=0,raf=null,shown=new Set();
function setStates(rev){
  const last=rev[rev.length-1]||{};
  const lastBy=k=>{for(let i=rev.length-1;i>=0;i--)if(AG[rev[i].kind]===k)return rev[i];return null;};
  const pe=lastBy("perc"),pl=lastBy("plan"),mo=lastBy("motor");
  $("#st-perc").textContent=pe?(pe.kind==="check"?shorten(pe.text.replace(/—.*/,'').replace(/"[^"]*"/,''),38):"verifying from pixels"):"watching";
  $("#st-plan").textContent=pl?shorten(pl.text.replace(/^plan:\s*/,""),34):"planning sort order";
  $("#st-motor").textContent=mo?shorten(mo.text.replace(/\(.*/,''),34):"idle";
  ["perc","plan","motor"].forEach(k=>$("#ag-"+k).classList.toggle("active",AG[(last.kind)]===k));
  // goals satisfied when in_bin(X) check ok
  rev.forEach(e=>{if(e.kind==="check"&&/in_bin/.test(e.text)&&e.ok){const it=curItem(e);if(it)$("#g-"+it)&&$("#g-"+it).classList.add("done");}});
  if(rev.some(e=>e.kind==="success"))$("#win").classList.add("on");
}
function frame(){
  $("#vid").src="data:image/jpeg;base64,"+FR[fi];
  EV.forEach((e,i)=>{if(!shown.has(i)&&e.k<=fi){shown.add(i);
    const le=$("#le"+i);if(le)le.classList.add("on");
    const be=$("#be"+i);if(be)be.classList.add("on");
    $("#capk").textContent=e.kind;$("#capt").textContent=shorten(e.text,64);}});
  setStates(EV.filter(e=>e.k<=fi));
  fi++;
  if(fi>=N){stop();$("#play").textContent="▶ Replay";return;}
  raf=setTimeout(frame,1000/FPS);
}
function stop(){if(raf)clearTimeout(raf);raf=null;$("#play").textContent="▶ Play";}
function start(){if(fi>=N)restart();$("#play").textContent="❚❚ Pause";frame();}
function restart(){stop();fi=0;shown=new Set();
  document.querySelectorAll(".agent-log-entry,.tl-entry").forEach(e=>e.classList.remove("on"));
  document.querySelectorAll(".goal-pill").forEach(e=>e.classList.remove("done"));
  $("#win").classList.remove("on");$("#vid").src="data:image/jpeg;base64,"+FR[0];}
$("#play").onclick=()=>{if(raf)stop();else start();};
$("#restart").onclick=restart;
restart();
</script>
</body></html>
"""


def main():
    T = json.load(open(TRACE))
    html = HTML.replace("__TRACE__", json.dumps(T))
    with open(OUT, "w") as f:
        f.write(html)
    print("wrote", OUT, f"({len(html)//1024} KB, {len(T['frames'])} frames, {len(T['events'])} events)")


if __name__ == "__main__":
    main()
