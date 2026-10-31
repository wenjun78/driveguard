/* DriveGuard fleet console — vanilla JS over the /api endpoints. No build step. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "html") n.innerHTML = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    for (const k of kids) n.append(k instanceof Node ? k : document.createTextNode(k ?? ""));
    return n;
  };
  const fmt = {
    pct: (v, d = 1) => (v == null || Number.isNaN(v) ? "—" : (100 * v).toFixed(d) + "%"),
    num: (v, d = 3) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d)),
    int: (v) => (v == null ? "—" : Number(v).toLocaleString()),
    sci: (v) => (v == null ? "—" : Math.abs(v) >= 1e5 ? Number(v).toExponential(2) : Number(v).toLocaleString()),
    date: (s) => s || "—",
    ts: (s) => (s ? new Date(s).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—"),
  };
  const palette = { ink: "#16221c", accent: "#c8ee73", green: "#4f9a5c", red: "#d9534f", amber: "#e3a008", blue: "#4a7fb5", grey: "#9aa89d", line: "#e3e9df" };
  const charts = {};
  const chart = (id, cfg) => {
    const ctx = $("#" + id);
    if (!ctx) return;
    Chart.defaults.font.family = "'DM Sans', Arial, sans-serif";
    Chart.defaults.color = "#5b6a60";
    Chart.defaults.animation = false;
    const existing = charts[id];
    if (existing && existing.config.type === cfg.type) {
      existing.data = cfg.data;
      existing.options = cfg.options || {};
      existing.update();
      return;
    }
    if (existing) existing.destroy();
    charts[id] = new Chart(ctx, cfg);
  };
  const cache = {};
  async function api(path, opts) {
    if (!opts && cache[path]) return cache[path];
    const r = await fetch(path, opts);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || r.statusText);
    if (!opts) cache[path] = j;
    return j;
  }
  function setStatus(j) {
    const s = $("#status"), t = $("#status-text"), b = $("#banner");
    s.classList.remove("live", "snapshot");
    if (j.source === "databricks" && j.live_ok) {
      s.classList.add("live");
      t.textContent = `live · ${j.data.workspace || "databricks"} · ${j.data.catalog}.${j.data.schema}`;
      b.hidden = true;
    } else {
      s.classList.add("snapshot");
      t.textContent = "demo snapshot";
      b.hidden = false;
      b.textContent = j.data.note || "Serving the bundled snapshot. Point the console at a Databricks workspace to go live.";
      if (j.data.primary === "databricks" && !j.live_ok) b.textContent = `Databricks workspace unreachable (${j.data.last_error || "unknown"}) — showing the bundled snapshot meanwhile.`;
    }
  }

  // ---------------------------------------------------------------- overview
  async function renderOverview() {
    const { data: d } = await api("/api/overview");
    $("#ov-asof").textContent = d.as_of;
    const k = $("#ov-kpis"); k.innerHTML = "";
    const kpi = (v, l, cls = "") => k.append(el("div", { class: "kpi " + cls }, el("b", {}, v), el("span", {}, l)));
    kpi(fmt.int(d.alerts), `drives flagged today · ${fmt.pct(d.alert_budget, 1)} budget`, "accent");
    kpi(fmt.int(d.drives_scored), "drives scored today");
    kpi(fmt.int(d.failures_last_30d), "failures in the last 30 days");
    const m = (d.matured || []).slice(-7);
    const rec = m.length ? m.reduce((a, r) => a + (r.recall_at_budget ?? 0), 0) / m.length : null;
    kpi(fmt.pct(rec, 0), `matured recall @ budget · last ${m.length || 0} snapshots`);
    const lead = m.length ? m.map((r) => r.lead_time_median_days).filter((x) => x != null) : [];
    kpi(lead.length ? Math.round(lead.reduce((a, b) => a + b, 0) / lead.length) + " d" : "—", "median lead time of first alert");
    $("#ov-budget").textContent = `${d.horizon_days}-day horizon · ${fmt.pct(d.alert_budget, 1)} of the fleet per day`;
    const apd = d.alerts_per_day || [];
    chart("chart-alerts", {
      type: "bar",
      data: { labels: apd.map((r) => r.date.slice(5)), datasets: [
        { label: "failed within 30 d (hit)", data: apd.map((r) => r.hits), backgroundColor: palette.red, borderRadius: 2 },
        { label: "stayed healthy (false alarm)", data: apd.map((r) => r.false_alarms), backgroundColor: palette.green, borderRadius: 2 },
        { label: "outcome pending", data: apd.map((r) => r.pending), backgroundColor: "#cfd9cc", borderRadius: 2 } ] },
      options: { plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } }, tooltip: { callbacks: { afterLabel: (c) => `${fmt.int(apd[c.dataIndex].drives)} drives scored` } } }, scales: { x: { stacked: true, grid: { display: false }, ticks: { maxTicksLimit: 12 } }, y: { stacked: true, beginAtZero: true, grid: { color: palette.line } } } },
    });
    const mat = d.matured || [];
    chart("chart-matured", {
      type: "line",
      data: {
        labels: mat.map((r) => r.snapshot_date.slice(5)),
        datasets: [
          { label: "recall @ budget", data: mat.map((r) => r.recall_at_budget), borderColor: palette.green, backgroundColor: palette.green, tension: 0.3, pointRadius: 2 },
          { label: "precision @ budget", data: mat.map((r) => r.precision_at_budget), borderColor: palette.blue, backgroundColor: palette.blue, tension: 0.3, pointRadius: 2 },
          { label: "drive recall", data: mat.map((r) => r.drive_recall_at_budget), borderColor: palette.amber, backgroundColor: palette.amber, borderDash: [4, 3], tension: 0.3, pointRadius: 0 },
        ],
      },
      options: { plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } } }, scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } }, y: { min: 0, max: 1, ticks: { callback: (v) => fmt.pct(v, 0) }, grid: { color: palette.line } } } },
    });
    const box = $("#ov-models"); box.innerHTML = "";
    const bars = el("div", { class: "bars" });
    const max = Math.max(1, ...(d.fleet_by_model || []).map((r) => r.drives));
    for (const r of d.fleet_by_model || []) {
      bars.append(el("div", { class: "bar" }, el("span", { class: "mono", style: "text-align:left;font-size:12px" }, r.model), el("i", {}, el("b", { style: `width:${(100 * r.drives) / max}%` })), el("span", { class: "mono" }, `${fmt.int(r.drives)} · ${r.alerts}`)));
    }
    box.append(bars);
  }

  // ---------------------------------------------------------------- alerts
  let selectedSerial = null;
  let driveReq = 0;
  async function renderAlerts(date) {
    const { data: d } = await api("/api/alerts" + (date ? `?date=${date}` : ""));
    $("#al-date").textContent = d.date || "—";
    const sel = $("#al-select");
    if (!sel.options.length || sel.dataset.filled !== "1") {
      sel.innerHTML = "";
      for (const x of (d.dates || []).slice().reverse()) sel.append(el("option", { value: x }, x));
      sel.dataset.filled = "1";
      sel.onchange = () => renderAlerts(sel.value);
    }
    sel.value = d.date;
    const note = $("#al-note"); note.innerHTML = "";
    if (d.matured) note.textContent = "Horizon elapsed — the hindsight column shows what actually happened.";
    else {
      note.append("Horizon not yet elapsed — outcomes pending. ");
      const dates = d.dates || [];
      const maturedDates = dates.filter((x) => { const t = new Date(x); t.setDate(t.getDate() + 30); return t <= new Date(d.dates[d.dates.length - 1]); });
      if (maturedDates.length) note.append(el("a", { href: "#alerts", onclick: (e) => { e.preventDefault(); renderAlerts(maturedDates[maturedDates.length - 1]); } }, "Jump to the latest snapshot with known outcomes →"));
    }
    const tb = $("#al-table tbody"); tb.innerHTML = "";
    for (const r of d.rows) {
      const tr = el("tr", { class: "clickable", onclick: () => { selectedSerial = r.serial; [...tb.children].forEach((x) => x.classList.remove("selected")); tr.classList.add("selected"); renderDrive(r.serial, d.date); } },
        el("td", { class: "mono" }, String(r.rank)),
        el("td", { class: "mono" }, r.serial),
        el("td", {}, r.model || "—"),
        el("td", { class: "mono" }, fmt.num(r.score, 3)),
        el("td", {}, el("div", { class: "chips" }, ...(r.reason_codes || []).map((c) => el("span", { class: "chip" }, c)))),
        el("td", {}, el("span", { class: "pill " + r.outcome }, r.outcome === "failed" && r.days_to_failure != null ? `failed in ${Math.round(r.days_to_failure)} d` : r.outcome)));
      tb.append(tr);
    }
    if (!d.rows.length) tb.append(el("tr", {}, el("td", { colspan: "6", class: "muted" }, "No alerts for this snapshot.")));
    if (d.rows.length && (!selectedSerial || !d.rows.some((r) => r.serial === selectedSerial))) { selectedSerial = d.rows[0].serial; tb.children[0].classList.add("selected"); renderDrive(selectedSerial, d.date); }
  }
  async function renderDrive(serial, asOf) {
    const my = ++driveReq;
    $("#dr-title").textContent = serial; $("#dr-sub").textContent = "loading…"; $("#dr-meta").innerHTML = "";
    let d;
    try { d = (await api(`/api/drive/${encodeURIComponent(serial)}`)).data; } catch (e) { $("#dr-sub").textContent = e.message; return; }
    if (my !== driveReq) return; // a newer selection superseded this one
    $("#dr-sub").textContent = `${d.model} · ${fmt.num(d.capacity_tb, 0)} TB`;
    const meta = $("#dr-meta");
    const item = (v, l) => meta.append(el("div", {}, el("b", {}, v), el("span", {}, l)));
    item(d.first_seen, "first observed"); item(d.last_seen, "last observed");
    item(d.failed ? d.failure_date : "no", d.failed ? "failed on" : "failure recorded");
    const h = d.history || [];
    const flagged = h.filter((r) => r.flagged).length;
    item(String(flagged), "flagged days in window");
    const labels = h.map((r) => r.date.slice(5));
    const line = (key, label, color, axis = "y") => ({ label, data: h.map((r) => r[key]), borderColor: color, backgroundColor: color, tension: 0.25, pointRadius: 0, borderWidth: 1.6, yAxisID: axis });
    chart("chart-drive", {
      type: "line",
      data: { labels, datasets: [line("smart_5_raw", "5 reallocated", palette.red), line("smart_197_raw", "197 pending", palette.amber), line("smart_198_raw", "198 offline uncorr.", palette.blue), line("smart_187_raw", "187 reported uncorr.", "#7a4fb5"), line("smart_194_raw", "194 temp °C", palette.grey, "y2")] },
      options: { plugins: { legend: { position: "bottom", labels: { boxWidth: 10, font: { size: 10 } } } }, scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } }, y: { beginAtZero: true, grid: { color: palette.line } }, y2: { position: "right", grid: { display: false }, suggestedMin: 20, suggestedMax: 45 } } },
    });
    chart("chart-drive-score", {
      type: "bar",
      data: { labels, datasets: [{ label: "score", data: h.map((r) => r.score), backgroundColor: h.map((r) => (r.flagged ? palette.red : palette.green)), borderRadius: 2 }] },
      options: { plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: (c) => (h[c.dataIndex].flagged ? "flagged (within budget)" : "") } } }, scales: { x: { display: false }, y: { beginAtZero: true, suggestedMax: 0.05, grid: { color: palette.line } } } },
    });
  }

  // ---------------------------------------------------------------- monitoring
  async function renderMonitoring() {
    const { data: d } = await api("/api/monitoring");
    const mat = d.matured || [];
    chart("chart-mon-matured", {
      type: "line",
      data: { labels: mat.map((r) => r.snapshot_date.slice(5)), datasets: [
        { label: "recall @ budget", data: mat.map((r) => r.recall_at_budget), borderColor: palette.green, backgroundColor: palette.green, tension: 0.3, pointRadius: 2 },
        { label: "precision @ budget", data: mat.map((r) => r.precision_at_budget), borderColor: palette.blue, backgroundColor: palette.blue, tension: 0.3, pointRadius: 2 },
        { label: "false alerts / 1k healthy", data: mat.map((r) => r.false_alerts_per_1000_drive_days), borderColor: palette.grey, borderDash: [3, 3], tension: 0.3, pointRadius: 0, yAxisID: "y2" } ] },
      options: { plugins: { legend: { position: "bottom", labels: { boxWidth: 10 } } }, scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } }, y: { min: 0, max: 1, ticks: { callback: (v) => fmt.pct(v, 0) }, grid: { color: palette.line } }, y2: { position: "right", beginAtZero: true, grid: { display: false } } } },
    });
    chart("chart-mon-lead", {
      type: "bar",
      data: { labels: mat.map((r) => r.snapshot_date.slice(5)), datasets: [{ label: "median lead time (days)", data: mat.map((r) => r.lead_time_median_days), backgroundColor: palette.amber, borderRadius: 4 }] },
      options: { plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: (c) => `${fmt.pct(mat[c.dataIndex].share_caught_with_7d_lead, 0)} caught with ≥7 d lead · ${mat[c.dataIndex].failing_drives} failing drives` } } }, scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } }, y: { beginAtZero: true, grid: { color: palette.line } } } },
    });
    const tb = $("#mon-table tbody"); tb.innerHTML = "";
    const runs = (d.runs || []).slice().reverse();
    for (const r of runs) {
      tb.append(el("tr", {}, el("td", { class: "mono" }, fmt.ts(r.monitored_at)), el("td", { class: "mono" }, r.snapshot_date), el("td", { class: "mono" }, fmt.int(r.drives_scored)), el("td", { class: "mono" }, fmt.num(r.score_psi, 3)),
        el("td", {}, el("div", { class: "chips" }, ...(r.features_drifted || []).map((c) => el("span", { class: "chip" }, c)))), el("td", {}, el("div", { class: "chips" }, ...(r.features_warned || []).map((c) => el("span", { class: "chip" }, c)))),
        el("td", { class: "mono" }, fmt.pct(r.matured_recent_recall_at_budget, 0)), el("td", {}, el("span", { class: "pill " + (r.retrain_recommended ? "yes" : "no"), title: (r.retrain_reasons || []).join("; ") }, r.retrain_recommended ? "retrain" : "keep"))));
    }
    if (!runs.length) tb.append(el("tr", {}, el("td", { colspan: "8", class: "muted" }, "No monitoring runs yet.")));
    const psi = $("#mon-psi"); psi.innerHTML = "";
    const latest = runs[0];
    const entries = Object.entries((latest && latest.psi_by_feature) || {}).sort((a, b) => (b[1] ?? -1) - (a[1] ?? -1));
    for (const [f, v] of entries) psi.append(el("div", { class: v == null ? "" : v >= 0.2 ? "alert" : v >= 0.1 ? "warn" : "" }, el("span", { class: "mono" }, f), el("span", { class: "mono" }, fmt.num(v, 3))));
    if (!entries.length) psi.append(el("span", { class: "muted" }, "No PSI values recorded."));
  }

  // ---------------------------------------------------------------- model
  async function renderModel() {
    const { data: d } = await api("/api/model");
    $("#md-title").textContent = `${d.name} · champion v${d.champion.version}`;
    $("#md-lede").textContent = `${d.horizon_days}-day horizon · alert budget ${fmt.pct(d.alert_budget, 1)} · registered ${fmt.ts(d.champion.created)} · ${d.champion.promotion || ""}`;
    const b = d.split || {};
    $("#md-split").textContent = b.train_end ? `train ≤ day ${b.train_end} · val ${b.val_start}–${b.val_end} · test ${b.test_start}–${b.end_day} · ${b.gap}-day gaps` : "";
    const rows = [["PR-AUC", "pr_auc", (v) => fmt.num(v, 3)], ["ROC-AUC", "roc_auc", (v) => fmt.num(v, 3)], ["Recall @ budget", "recall_at_budget", fmt.pct], ["Precision @ budget", "precision_at_budget", fmt.pct], ["Drive recall @ budget", "drive_recall_at_budget", fmt.pct], ["Median lead time (d)", "lead_time_median_days", (v) => fmt.num(v, 0)], ["Caught with ≥ 7 d lead", "share_caught_with_7d_lead", (v) => fmt.pct(v, 0)], ["False alerts / 1k healthy drive-days", "false_alerts_per_1000_drive_days", (v) => fmt.num(v, 2)], ["Failing drives", "failing_drives", (v) => fmt.num(v, 0)], ["Labeled drive-days", "rows_labeled", fmt.int]];
    const tb = $("#md-metrics tbody"); tb.innerHTML = "";
    const rep = d.report || {};
    for (const [label, key, f] of rows) tb.append(el("tr", {}, el("td", {}, label), el("td", { class: "mono" }, f((rep.val || {})[key])), el("td", { class: "mono" }, f((rep.test || {})[key])), el("td", { class: "mono" }, f((rep.baseline_test || {})[key]))));
    const imp = (d.importance || []).slice(0, 18);
    chart("chart-imp", {
      type: "bar",
      data: { labels: imp.map((x) => x[0]), datasets: [{ data: imp.map((x) => x[1]), backgroundColor: palette.green, borderRadius: 3 }] },
      options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { grid: { color: palette.line } }, y: { grid: { display: false }, ticks: { font: { family: "'IBM Plex Mono'", size: 10 } } } } },
    });
    const vt = $("#md-versions tbody"); vt.innerHTML = "";
    for (const v of d.versions || []) {
      const isChamp = v.version === d.champion.version;
      vt.append(el("tr", {}, el("td", { class: "mono" }, "v" + v.version), el("td", { class: "mono" }, fmt.ts(v.created)), el("td", { class: "mono" }, fmt.num(v.val_pr_auc, 4)), el("td", { class: "mono" }, fmt.num(v.test_pr_auc, 4)), el("td", { class: "mono" }, (v.run_id || "").slice(0, 8)), el("td", {}, isChamp ? el("span", { class: "pill healthy" }, "@champion") : "")));
    }
  }

  // ---------------------------------------------------------------- pipeline
  async function renderPipeline() {
    const { data: d } = await api("/api/pipeline");
    const st = $("#pl-stages"); st.innerHTML = "";
    const stages = [["LAND", "bronze", "new daily files → typed Delta table"], ["CLEAN", "silver", "1 row per drive-day, lifecycle"], ["FEATURE", "gold", "time-safe windows, censored label"], ["SCORE", "score", "rank the fleet at budget, reason codes"], ["WATCH", "monitor", "PSI, matured outcomes, retrain flag"], ["GATE", "condition", "retrain_recommended == true?"], ["TRAIN", "retrain", "XGBoost → MLflow → UC → champion gate"]];
    for (const [k, b, s] of stages) st.append(el("div", { class: "stage" }, el("small", {}, k), el("b", {}, b), el("span", {}, s)));
    const rp = $("#pl-replay"); rp.innerHTML = "";
    const item = (box, v, l) => box.append(el("div", {}, el("b", {}, v), el("span", {}, l)));
    item(rp, d.replay && d.replay.anchor ? d.replay.anchor : "not set", "anchor (data day @ real day)");
    item(rp, d.replay && d.replay.cursor ? d.replay.cursor : "latest day in gold", "data day that is 'today'");
    const sv = $("#pl-serving"); sv.innerHTML = "";
    if (d.serving) { item(sv, d.serving.name, "endpoint"); item(sv, d.serving.state || d.serving.error || "—", "state"); } else sv.append(el("p", { class: "muted" }, d.note || "No endpoint information."));
    const jobs = $("#pl-jobs"); jobs.innerHTML = "";
    if (!(d.jobs || []).length) { jobs.append(el("div", { class: "card" }, el("p", { class: "muted" }, d.note || "No jobs found."))); return; }
    const wrap = el("div", { class: "jobs" });
    for (const j of d.jobs) {
      const t = el("table", { class: "table" }, el("thead", {}, el("tr", {}, el("th", {}, "Run"), el("th", {}, "Started"), el("th", {}, "Duration"), el("th", {}, "State"), el("th", {}, "Result"))));
      const tb = el("tbody");
      for (const r of j.runs || []) tb.append(el("tr", {}, el("td", { class: "mono" }, r.url ? el("a", { href: r.url, target: "_blank", rel: "noopener" }, String(r.run_id)) : String(r.run_id)), el("td", { class: "mono" }, fmt.ts(r.start)), el("td", { class: "mono" }, r.duration_s ? Math.round(r.duration_s) + " s" : "—"), el("td", { class: "mono" }, r.state || "—"), el("td", {}, el("span", { class: "pill " + (r.result === "SUCCESS" ? "healthy" : r.result ? "failed" : "pending") }, r.result || "running"))));
      if (!(j.runs || []).length) tb.append(el("tr", {}, el("td", { colspan: "5", class: "muted" }, "No runs yet.")));
      t.append(tb);
      wrap.append(el("div", { class: "card job" }, el("h4", {}, j.name, el("small", {}, `job ${j.job_id}${j.schedule ? " · " + j.schedule + (j.paused === "PAUSED" ? " (paused)" : "") : ""}`)), t));
    }
    jobs.append(wrap);
  }

  // ---------------------------------------------------------------- predict
  function showResult(d, source) {
    const box = $("#pf-result"); box.innerHTML = "";
    const p = d.p_fail_within_horizon;
    box.append(el("div", { class: "card-head" }, el("h3", {}, d.serial ? `Drive ${d.serial}` : "Hand-entered values"), el("span", { class: "muted" }, source)));
    box.append(el("div", { class: "result-p " + (p >= 0.5 ? "hi" : p >= 0.1 ? "mid" : "lo") }, fmt.pct(p, 1)));
    box.append(el("p", { class: "muted" }, "probability of failure within the next 30 days"));
    box.append(el("p", {}, el("b", {}, "Reason codes "), el("span", { class: "muted" }, "(top TreeSHAP contributions toward failure)")));
    box.append(el("div", { class: "chips" }, ...(d.reason_codes || []).map((c) => el("span", { class: "chip" }, c))));
    box.append(el("p", { class: "muted small", style: "margin-top:12px" }, `Served by: ${d.served_by}` + (d.local_p != null && Math.abs(d.local_p - p) > 1e-6 ? ` · local copy says ${fmt.pct(d.local_p, 1)}` : "")));
    if (d.inputs) box.append(el("p", { class: "muted small" }, "Non-zero inputs: " + Object.entries(d.inputs).map(([k, v]) => `${k}=${v}`).join(", ")));
  }
  async function predict(body) {
    const box = $("#pf-result"); box.innerHTML = ""; box.append(el("p", { class: "skeleton" }, "scoring…"));
    try {
      const j = await api("/api/predict", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
      showResult(j.data, j.source === "databricks" ? "live workspace" : "demo snapshot");
    } catch (e) { box.innerHTML = ""; box.append(el("p", { class: "error" }, e.message)); }
  }
  $("#pf-serial").addEventListener("submit", (e) => { e.preventDefault(); const s = new FormData(e.target).get("serial").trim(); if (s) predict({ serial: s }); });
  $("#pf-manual").addEventListener("submit", (e) => {
    e.preventDefault();
    const fd = new FormData(e.target); const features = {}; let model = "UNKNOWN";
    for (const [k, v] of fd.entries()) { if (k === "model") model = v || "UNKNOWN"; else if (v !== "") features[k] = Number(v); }
    predict({ model, features });
  });
  $("#pf-preset").addEventListener("click", () => {
    const vals = { smart_9_raw: 41000, smart_194_raw: 34, smart_5_raw: 180, smart_5_delta_30d: 140, smart_197_raw: 64, smart_197_delta_7d: 22, smart_197_nzdays_30d: 18, smart_198_raw: 12, smart_187_raw: 40, smart_188_raw: 3, smart_199_raw: 0 };
    for (const [k, v] of Object.entries(vals)) { const i = $(`#pf-manual [name="${k}"]`); if (i) i.value = v; }
  });

  // ---------------------------------------------------------------- routing
  const renderers = { overview: renderOverview, alerts: () => renderAlerts(), monitoring: renderMonitoring, model: renderModel, pipeline: renderPipeline, predict: async () => {} };
  const rendered = {};
  async function show(view) {
    if (!renderers[view]) view = "overview";
    document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== "view-" + view));
    document.querySelectorAll("#tabs a").forEach((a) => a.classList.toggle("active", a.dataset.view === view));
    if (!rendered[view]) {
      rendered[view] = true;
      try { await renderers[view](); } catch (e) { rendered[view] = false; const sec = $("#view-" + view); sec.prepend(el("p", { class: "error" }, "Could not load: " + e.message)); }
    }
  }
  window.addEventListener("hashchange", () => show(location.hash.slice(1)));
  (async () => {
    try { setStatus(await api("/api/status")); } catch (e) { $("#status-text").textContent = "api unavailable"; }
    show(location.hash.slice(1) || "overview");
  })();
})();
