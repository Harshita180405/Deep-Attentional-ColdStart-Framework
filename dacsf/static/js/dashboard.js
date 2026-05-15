/* ─────────────────────────────────────────────────────────────────────────────
   dashboard.js  —  Frontend logic for the DACSF training dashboard
───────────────────────────────────────────────────────────────────────────── */

let polling       = null;
let chartObj      = null;
let currentMetric = 'ndcg';
let history       = [];

// ── Chart initialisation ──────────────────────────────────────────────────────

function initChart() {
  const ctx = document.getElementById('chart').getContext('2d');
  chartObj  = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'NDCG@10',
        data: [],
        borderColor: '#1b998b',
        backgroundColor: 'rgba(27,153,139,.08)',
        borderWidth: 2,
        pointRadius: 3,
        pointBackgroundColor: '#1b998b',
        tension: 0.4,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,.04)' },
          ticks: { color: '#4a6375', font: { family: 'Space Mono', size: 10 } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,.04)' },
          ticks: { color: '#4a6375', font: { family: 'Space Mono', size: 10 } }
        }
      }
    }
  });
}

function switchChart(metric, el) {
  currentMetric = metric;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  updateChart();
}

function updateChart() {
  if (!chartObj || !history.length) return;
  const colors = { ndcg: '#1b998b', loss: '#e84855', mae: '#e84855', rmse: '#ffba08' };
  const c      = colors[currentMetric] || '#1b998b';
  chartObj.data.labels                       = history.map(h => h.epoch);
  chartObj.data.datasets[0].data            = history.map(h => h[currentMetric]);
  chartObj.data.datasets[0].label           = currentMetric.toUpperCase();
  chartObj.data.datasets[0].borderColor     = c;
  chartObj.data.datasets[0].pointBackgroundColor = c;
  chartObj.data.datasets[0].backgroundColor = c + '18';
  chartObj.update('none');
}

// ── Status badge ──────────────────────────────────────────────────────────────

function setStatus(s) {
  const badge = document.getElementById('status-badge');
  const label = document.getElementById('status-text');
  badge.className = 'status-badge ' + s;
  const labels = {
    idle: 'Idle', loading: 'Loading data…',
    training: 'Training', ready: 'Model ready', error: 'Error'
  };
  label.textContent = labels[s] || s;
}

// ── Toast notification ────────────────────────────────────────────────────────

function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2800);
}

// ── Start training ────────────────────────────────────────────────────────────

function startTraining() {
  const cfg = {
    epochs:  parseInt(document.getElementById('cfg-epochs').value),
    dim:     parseInt(document.getElementById('cfg-dim').value),
    lr:      parseFloat(document.getElementById('cfg-lr').value),
    dropout: parseFloat(document.getElementById('cfg-dropout').value),
    tau:     parseInt(document.getElementById('cfg-tau').value),
    k:       parseInt(document.getElementById('cfg-k').value),
  };

  history = [];
  document.getElementById('results-card').style.display = 'none';
  document.getElementById('reco-card').style.display    = 'none';
  document.getElementById('btn-train').disabled = true;
  document.getElementById('btn-reco').disabled  = true;

  fetch('/api/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  })
  .then(r => r.json())
  .then(data => {
    if (!data.ok) { toast('⚠ ' + (data.msg || 'Could not start training.')); return; }
    startPolling();
    toast('Training started!');
  })
  .catch(err => toast('❌ Network error: ' + err));
}

// ── Polling ───────────────────────────────────────────────────────────────────

function startPolling() {
  if (polling) clearInterval(polling);
  polling = setInterval(pollState, 1200);
}

function pollState() {
  fetch('/api/state')
    .then(r => r.json())
    .then(s => {
      setStatus(s.status);
      updateProgress(s.progress, s.epoch, s.total_epochs);
      updateLiveMetrics(s.metrics);
      updateLog(s.log);
      history = s.history || [];
      updateChart();

      if (s.status === 'ready') {
        clearInterval(polling);
        polling = null;
        showResults(s.results);
        document.getElementById('btn-train').disabled = false;
        document.getElementById('btn-reco').disabled  = false;
        document.getElementById('max-user').textContent = s.n_users - 1;
        toast('✅ Training complete!');
      }

      if (s.status === 'error') {
        clearInterval(polling);
        document.getElementById('btn-train').disabled = false;
        toast('❌ Error: ' + (s.error || 'Unknown error.'));
      }
    })
    .catch(() => { /* network hiccup — keep polling */ });
}

// ── UI update helpers ─────────────────────────────────────────────────────────

function updateProgress(pct, ep, total) {
  document.getElementById('prog-bar').style.width   = pct + '%';
  document.getElementById('ep-pct').textContent     = pct + '%';
  document.getElementById('ep-num').textContent     = ep;
  document.getElementById('ep-total').textContent   = total;
}

function updateLiveMetrics(m) {
  if (!m || m.loss === undefined) return;
  document.getElementById('m-loss').textContent = m.loss;
  document.getElementById('m-ndcg').textContent = m.ndcg;
  document.getElementById('m-mae').textContent  = m.mae;
  document.getElementById('m-rmse').textContent = m.rmse;
}

let lastLogLen = 0;
function updateLog(logs) {
  if (!logs || logs.length === lastLogLen) return;
  lastLogLen = logs.length;
  const el   = document.getElementById('terminal');
  el.innerHTML = logs.map(l => `<div class="log-line">${l}</div>`).join('');
  el.scrollTop = el.scrollHeight;
}

function showResults(r) {
  if (!r || r.mae === undefined) return;
  document.getElementById('results-card').style.display = 'block';
  document.getElementById('r-mae').textContent     = r.mae;
  document.getElementById('r-rmse').textContent    = r.rmse;
  document.getElementById('r-ndcg').textContent    = r.ndcg;
  document.getElementById('r-hr').textContent      = r.hr;
  document.getElementById('r-cu-ndcg').textContent = r.cold_user_ndcg;
  document.getElementById('r-cu-hr').textContent   = r.cold_user_hr;
  document.getElementById('r-ci-ndcg').textContent = r.cold_item_ndcg;
  document.getElementById('r-ci-hr').textContent   = r.cold_item_hr;
}

// ── Recommendations ───────────────────────────────────────────────────────────

function getRecos() {
  const uid   = parseInt(document.getElementById('rec-user').value);
  const top_k = parseInt(document.getElementById('rec-k').value);

  fetch('/api/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_idx: uid, top_k }),
  })
  .then(r => r.json())
  .then(data => {
    if (data.error) { toast('❌ ' + data.error); return; }
    renderRecos(data.recommendations, uid);
  })
  .catch(err => toast('❌ Network error: ' + err));
}

function renderRecos(recos, uid) {
  document.getElementById('reco-card').style.display  = 'block';
  document.getElementById('reco-user-title').textContent = uid;

  const list = document.getElementById('reco-list');
  list.innerHTML = recos.map(r => `
    <div class="reco-card">
      <div class="reco-rank">${r.rank}</div>
      <div class="reco-info">
        <div class="reco-title">${r.title}</div>
        <div class="reco-score">Score: ${r.score} &nbsp;·&nbsp; Item ID: ${r.item_id}</div>
        <div class="attn-bars">
          <div class="attn-bar-wrap">
            <div class="attn-label">Text α=${r.attention.text}</div>
            <div class="attn-bar"     style="width:${r.attention.text*100}%"></div>
          </div>
          <div class="attn-bar-wrap">
            <div class="attn-label">Image α=${r.attention.image}</div>
            <div class="attn-bar img" style="width:${r.attention.image*100}%"></div>
          </div>
          <div class="attn-bar-wrap">
            <div class="attn-label">Category α=${r.attention.category}</div>
            <div class="attn-bar cat" style="width:${r.attention.category*100}%"></div>
          </div>
        </div>
      </div>
    </div>
  `).join('');

  document.getElementById('reco-card').scrollIntoView({ behavior: 'smooth' });
  toast(`✦ ${recos.length} recommendations generated`);
}

// ── Init ──────────────────────────────────────────────────────────────────────
initChart();
