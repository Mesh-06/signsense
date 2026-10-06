const IMG = 48, BG = -1;   // 43-class model: no "no sign" class yet (set BG = 43 after the 44-class retrain)
const NAMES = ["Speed limit 20","Speed limit 30","Speed limit 50","Speed limit 60","Speed limit 70","Speed limit 80",
"End of speed limit 80","Speed limit 100","Speed limit 120","No passing","No passing for heavy vehicles",
"Right-of-way at next intersection","Priority road","Yield","Stop","No vehicles","Heavy vehicles prohibited",
"No entry","General caution","Dangerous curve left","Dangerous curve right","Double curve","Bumpy road",
"Slippery road","Road narrows on right","Road work","Traffic signals","Pedestrians","Children crossing",
"Bicycles crossing","Beware of ice or snow","Wild animals crossing","End of all speed and passing limits",
"Turn right ahead","Turn left ahead","Ahead only","Go straight or right","Go straight or left","Keep right",
"Keep left","Roundabout mandatory","End of no passing","End of no passing for heavy vehicles"];
const SPEED = {0:20,1:30,2:50,3:60,4:70,5:80,7:100,8:120};
const category = id => [13,14,17].includes(id) ? 'critical' : id in SPEED ? 'speed'
  : (id>=18 && id<=31) ? 'warning' : [9,10,15,16].includes(id) ? 'prohibitory'
  : (id>=33 && id<=40) ? 'mandatory' : 'info';

const $ = s => document.querySelector(s);
let model, INFO = null, voiceOn = false, camOn = false, busy = false, lastMs = 0, buf = [], spoken = {};

// ---------- model ----------
(async () => {
  try {
    await tf.ready();
    model = await tf.loadGraphModel('model/model.json');
    const w = predictRaw(tf.zeros([1, IMG, IMG, 3])); await w.data(); w.dispose();   // warm-up
    $('#status').textContent = `Model ready • ${tf.getBackend()}`;
    try { INFO = await (await fetch('model/info.json')).json(); } catch {}
    loadGallery(); loop();
  } catch (e) {
    $('#status').textContent = 'Model failed to load: ' + e.message;
  }
})();

function predictRaw(x) {
  const o = model.predict(x);
  return Array.isArray(o) ? o[0] : (o.shape ? o : Object.values(o)[0]);
}

async function classify(source) {
  const t0 = performance.now();
  const out = tf.tidy(() => {
    const x = tf.image.resizeBilinear(tf.browser.fromPixels(source).toFloat(), [IMG, IMG], false, true);
    return predictRaw(x.expandDims(0));
  });
  const p = Array.from(await out.data()); out.dispose();
  lastMs = performance.now() - t0; return p;
}

// native-resolution square crop (no browser scaling)
const cv = document.createElement('canvas'), cx = cv.getContext('2d');
function crop(src, zoom = 1, ox = .5, oy = .5) {
  const w = src.videoWidth || src.naturalWidth || src.width, h = src.videoHeight || src.naturalHeight || src.height;
  const s = Math.min(w, h) / zoom;
  const sx = Math.min(Math.max(ox * w - s / 2, 0), w - s), sy = Math.min(Math.max(oy * h - s / 2, 0), h - s);
  cv.width = cv.height = Math.round(s); cx.drawImage(src, sx, sy, s, s, 0, 0, cv.width, cv.height); return cv;
}

// ---------- UI ----------
function show(p) {
  const order = [...p.keys()].sort((a, b) => p[b] - p[a]), id = order[0], conf = p[id];
  const ok = conf >= +$('#thr').value && id !== BG;
  $('#name').textContent = ok ? NAMES[id] : 'No clear sign';
  $('#badge').className = ok ? category(id) : 'info';
  $('#badge').textContent = ok ? category(id).toUpperCase() : '—';
  $('#conf').style.width = (conf * 100) + '%';
  $('#top3').innerHTML = order.slice(0, 3).map(i => `<li>${NAMES[i]} — ${(p[i]*100).toFixed(1)}%</li>`).join('');
  $('#info').textContent = `${INFO?.name ?? ''} • params ${INFO?.params ?? '?'} • test acc ${INFO?.acc ? (INFO.acc*100).toFixed(2)+'%' : '?'} • last inference ${lastMs.toFixed(1)} ms • ${tf.getBackend()}`;
  return ok ? id : null;
}

// ---------- smoothing + voice ----------
function confirmed(id) {
  buf.push(id); if (buf.length > 8) buf.shift();
  if (id === null) return null;
  return buf.filter(b => b === id).length >= 6 ? id : null;   // 6 of the last 8 frames agree
}
function speak(text, urgent) {
  if (!voiceOn) return;
  const u = new SpeechSynthesisUtterance(text); u.rate = urgent ? 1.1 : 1; u.pitch = urgent ? 1.2 : 1;
  if (urgent) speechSynthesis.cancel();
  speechSynthesis.speak(u);
}
function alertFor(id) {
  const now = Date.now(); if (now - (spoken[id] || 0) < 5000) return; spoken[id] = now;
  const cat = category(id); let msg = (cat === 'warning' ? 'Caution: ' : '') + NAMES[id];
  if (cat === 'speed') {
    const mine = +$('#speed').value;
    msg = `Speed limit ${SPEED[id]} kilometres per hour` + (mine > SPEED[id] ? `. You are over the limit. Slow down.` : '');
  }
  if (cat === 'critical') msg += '! Be ready.';
  $('#log').insertAdjacentHTML('afterbegin', `<li>${new Date().toLocaleTimeString()} — ${msg}</li>`);
  speak(msg, cat === 'critical');
}
function handle(p) { const c = confirmed(show(p)); if (c !== null) alertFor(c); }

// ---------- inputs ----------
async function loop() {
  if (camOn && model && !busy) {
    busy = true;
    try { handle(await classify(crop($('#video'), +$('#zoom').value))); } catch (e) { console.error(e); }
    busy = false;
  }
  setTimeout(loop, 150);
}
$('#startCam').onclick = async () => {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } } });
    $('#video').srcObject = s; camOn = true; $('#startCam').style.display = 'none';
  } catch (e) { alert('Camera unavailable: ' + e.message + '\nCamera needs HTTPS or localhost, and permission.'); }
};
$('#zoom').oninput = e => $('#reticle').style.width = $('#reticle').style.height = (100 / e.target.value) + '%';
$('#zoom').oninput({ target: $('#zoom') });

const runUpload = async () => {
  if (!model) return;
  show(await classify(crop($('#preview'), +$('#uz').value, +$('#ox').value, +$('#oy').value)));
};
$('#file').onchange = e => { $('#preview').onload = runUpload; $('#preview').src = URL.createObjectURL(e.target.files[0]); };
['uz', 'ox', 'oy'].forEach(i => $('#' + i).oninput = runUpload);

$('#voice').onclick = () => {
  voiceOn = !voiceOn;
  $('#voice').textContent = voiceOn ? 'Voice alerts on' : 'Enable voice alerts';
  if (voiceOn) speak('Voice alerts enabled', false);   // the click satisfies the iOS user-gesture rule
};
$('#thr').oninput = e => $('#thrv').textContent = e.target.value;
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button,.tab').forEach(x => x.classList.remove('on'));
  b.classList.add('on'); $('#' + b.dataset.tab).classList.add('on');
});

// ---------- samples + parity ----------
const loadImg = src => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
async function loadGallery() {
  const exp = await (await fetch('samples/expected.json')).json();
  for (const e of exp) {
    const i = await loadImg('samples/' + e.file); i.title = 'True: ' + NAMES[e.y];
    i.onclick = async () => show(await classify(i)); $('#gallery').appendChild(i);
  }
}
$('#parity').onclick = async () => {
  const exp = await (await fetch('samples/expected.json')).json(); let ok = 0, lines = [];
  for (const e of exp) {
    const p = await classify(await loadImg('samples/' + e.file)); const id = p.indexOf(Math.max(...p));
    const pass = id === e.pred && Math.abs(p[id] - e.prob) < 0.02; ok += pass;
    lines.push(`${e.file}  python=${e.pred} (${e.prob.toFixed(3)})  browser=${id} (${p[id].toFixed(3)})  ${pass ? '✓' : '✗'}`);
  }
  $('#parityOut').textContent = lines.join('\n') + `\n\n${ok}/${exp.length} match`;
};
