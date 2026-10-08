/* EEG page: submit via fetch to /api/eeg/analyze and render the result in place.
 *
 * Progressive enhancement: the form still posts to /analyze_eeg without JS,
 * and if the API cannot be reached this falls back to that plain submit.
 * Every value from the server is escaped before it touches innerHTML.
 */
(function () {
  "use strict";

  var form = document.getElementById("eeg-form");
  var live = document.getElementById("eeg-live");
  if (!form || !live || !window.fetch || !window.FormData) return;

  var BANDS = ["delta", "theta", "alpha", "beta", "gamma"];
  var TONES = { good: "var(--success)", bad: "var(--danger)" };

  function esc(v) {
    return String(v === null || v === undefined ? "—" : v)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function fixed(v, d) {
    return v === null || v === undefined ? "—" : Number(v).toFixed(d);
  }
  function h2(text) {
    return '<h2 style="font-family: var(--font-heading); font-size: 22px; margin: 8px 0 14px;">' + text + "</h2>";
  }
  function metric(label, value, color) {
    return '<div class="metric-box"><div class="metric-label">' + esc(label) +
      '</div><div class="metric-value"' + (color ? ' style="color:' + color + '"' : "") +
      ">" + esc(value) + "</div></div>";
  }
  function list(items) {
    return "<ul style=\"padding-left: 18px; font-size: 13.5px; line-height: 1.7;\">" +
      (items || []).map(function (i) { return "<li>" + esc(i) + "</li>"; }).join("") + "</ul>";
  }

  function savedPanel(saved) {
    if (!saved) return "";
    if (saved.error) {
      return '<div class="records-error"><strong>⚠ Result shown but NOT saved to the record:</strong> ' +
        esc(saved.error) + "</div>";
    }
    if (saved.unsaved) {
      return '<div class="records-notes"><strong>Not stored:</strong> ' + esc(saved.unsaved) + "</div>";
    }
    var html = '<div class="records-bound"><span>✅ <strong>Saved to record</strong> as <a href="' +
      esc(saved.url) + '">' + esc(saved.study_id) + '</a></span><a href="' + esc(saved.pdf_url) +
      '">📥 Stored PDF</a><a href="' + esc(saved.patient_url) + '">Patient history →</a></div>';
    if (saved.note) {
      html += '<div class="records-error"><strong>⚠ Check the patient:</strong> ' + esc(saved.note) + "</div>";
    }
    var pr = saved.previous_report;
    if (pr) {
      html += h2("🗂️ Previous EEG report") +
        '<div style="color: var(--text-muted); font-size: 13px; margin-bottom: 12px;"><a href="' + esc(pr.url) +
        '" style="color:#a5b4fc">' + esc(pr.study_id) + "</a> · " + esc(pr.study_date) +
        " — included in this report's PDF.</div>" +
        '<div class="compare-images" style="margin-bottom:16px;">' +
        pr.images.map(function (i) {
          return '<figure><img src="' + esc(i.url) + '" alt="Previous ' + esc(i.caption) + '" loading="lazy">' +
            "<figcaption>Previous " + esc(i.caption) + " · " + esc(pr.study_date) + "</figcaption></figure>";
        }).join("") + "</div>" +
        '<div class="table-scroll" style="max-height:none;margin-bottom:24px;"><table class="data-table">' +
        "<thead><tr><th>Previous report</th><th>Details</th></tr></thead><tbody>" +
        pr.details.map(function (d) {
          return "<tr><td>" + esc(d[0]) + "</td><td>" + esc(d[1]) + "</td></tr>";
        }).join("") + "</tbody></table></div>";
    }
    if (saved.previous && saved.since_last_visit && saved.since_last_visit.length) {
      html += h2("🔁 Since last visit") +
        '<div class="table-scroll" style="max-height:none;margin-bottom:24px;"><table class="data-table">' +
        "<thead><tr><th>Measure</th><th>Before</th><th>Now</th><th>Note</th></tr></thead><tbody>" +
        saved.since_last_visit.map(function (r) {
          return '<tr class="' + (r.changed ? "row-changed" : "") + '"><td>' + esc(r.label) +
            "</td><td>" + esc(r.before) + "</td><td>" + esc(r.after) + "</td><td>" + esc(r.note) + "</td></tr>";
        }).join("") + "</tbody></table></div>";
    }
    return html;
  }

  function windowTable(j) {
    var peakIndex = j.peak ? j.peak.index : -1;
    var rows = j.windows.map(function (w) {
      var style = w.index === peakIndex ? ' style="outline: 1px solid #f59e0b;"' :
        (w.flagged ? ' class="row-changed"' : "");
      var rsp = BANDS.map(function (b) { return "<td>" + (w.rsp ? fixed(w.rsp[b], 3) : "—") + "</td>"; }).join("");
      return "<tr" + style + "><td>" + fixed(w.start_s, 1) + "–" + fixed(w.stop_s, 1) + "</td>" + rsp +
        "<td>" + fixed(w.spectral_entropy, 3) + "</td><td>" + fixed(w.spikes, 2) + "</td><td>" +
        fixed(w.score, 3) + "</td><td>" + esc(w.artifact || "") + "</td></tr>";
    }).join("");
    return h2("📋 Every Window (" + j.windows.length + ")") +
      '<div class="table-scroll"><table class="data-table"><thead><tr><th>Time (s)</th>' +
      BANDS.map(function (b) { return "<th>" + b + "</th>"; }).join("") +
      "<th>Entropy</th><th>Spikes/s</th><th>Score</th><th>Artifact</th></tr></thead><tbody>" +
      rows + "</tbody></table></div>";
  }

  function pdfForm(fields) {
    var inputs = Object.keys(fields).map(function (k) {
      return '<input type="hidden" name="' + esc(k) + '" value="' + esc(fields[k] === null ? "" : fields[k]) + '">';
    }).join("");
    return '<form action="/download_eeg_report" method="POST" style="margin: 24px 0;">' + inputs +
      '<button type="submit" class="btn-primary" style="width:100%;padding:14px 24px;background:var(--eeg-gradient);">' +
      "📥 Download Complete EEG Report (PDF)</button></form>";
  }

  function render(j) {
    var v = j.verdict, tone = TONES[v.tone] || "#f59e0b";
    var patientRows = j.patient.map(function (p) {
      return '<div class="meta-summary-item"><dt>' + esc(p[0]) + "</dt><dd>" + esc(p[1]) + "</dd></div>";
    }).join("");
    var html = '<div class="result-card">' +
      h2("🗂️ Study Details") + '<dl class="meta-summary">' + patientRows + "</dl>" +
      savedPanel(j.saved) +
      h2("🧬 Analysis Result") +
      '<div style="color: var(--text-muted); font-size: 13px; margin-bottom: 20px;">' +
      esc(j.task) + " · " + esc(j.representation_label) + " · Source: " + esc(j.provenance) +
      " · " + esc(j.duration_s) + " s</div>";
    if (!j.model_used) {
      html += '<div style="background: rgba(251,191,36,0.12); border: 1px solid #f59e0b; border-radius: 8px; padding: 14px 16px; margin-bottom: 20px; font-size: 14px;">⚠️ <strong>' +
        esc(j.not_assessed) + "</strong> Every measurement below is computed from the signal itself and remains valid. This is <strong>not</strong> a statement that the recording is normal.</div>";
    }
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:20px;">' +
      metric("Windows", j.n_windows) +
      metric("Episodes", j.model_used ? j.summary.episodes : "—",
        j.model_used && j.summary.episodes ? "var(--danger)" : "var(--success)") +
      metric("Anomaly Burden", j.model_used ? fixed(j.summary.burden_pct, 1) + "%" : "—") +
      metric("Highest Window", j.peak ? fixed(j.peak.score, 2) : "—",
        j.peak && j.peak.score >= 0.5 ? "#f59e0b" : null) +
      metric("Artifact Windows", j.artifact_windows) + "</div>";
    html += '<div style="background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.35); border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; font-size: 14px;">' +
      esc(j.summary.headline) +
      (j.peak ? '<br><span style="color: var(--text-muted);">Highest window score: <strong>' + fixed(j.peak.score, 2) +
        "</strong> at " + fixed(j.peak.start_s, 1) + "–" + fixed(j.peak.stop_s, 1) + " s</span>" : "") + "</div>";
    if (j.summary.peak_note) {
      html += '<div class="records-notes" style="margin-bottom: 20px;">🔎 ' + esc(j.summary.peak_note) + "</div>";
    }
    html += '<div class="metric-box" style="margin: 16px 0 20px; border-color:' + tone + ';"><div class="metric-label">EEG Classification</div>' +
      '<div class="metric-value" style="color:' + tone + '; font-size: 30px;">' + esc(v.label) + "</div>" +
      '<div style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">' + esc(v.detail) + "</div></div>" +
      h2("🩺 Clinical Impression") + '<p style="font-size:14px;line-height:1.75;margin-bottom:24px;">' + esc(j.notes.impression) + "</p>" +
      '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px;margin-bottom:28px;">' +
      '<div class="metric-box" style="text-align:left;"><div class="metric-label">📋 Recommendations</div>' + list(j.notes.recommendations) + "</div>" +
      '<div class="metric-box" style="text-align:left;border-color:#f59e0b;"><div class="metric-label">⚠️ Precautions</div>' + list(j.notes.precautions) + "</div></div>" +
      h2("📊 Timeline") + '<img src="' + esc(j.timeline_url) + '" alt="EEG timeline" style="width:100%;border-radius:12px;margin-bottom:28px;background:#fff;">';
    if (j.episodes.length) {
      html += h2("🚨 Detected Episodes") + '<div class="table-scroll" style="max-height:none;margin-bottom:24px;"><table class="data-table">' +
        "<thead><tr><th>Onset (s)</th><th>Offset (s)</th><th>Duration (s)</th><th>Peak score</th><th>Spikes/s</th></tr></thead><tbody>" +
        j.episodes.map(function (e) {
          return "<tr><td>" + fixed(e.onset_s, 1) + "</td><td>" + fixed(e.offset_s, 1) + "</td><td>" + fixed(e.duration_s, 1) +
            "</td><td>" + fixed(e.peak_score, 3) + "</td><td>" + fixed(e.spike_rate_per_s, 2) + "</td></tr>";
        }).join("") + "</tbody></table></div>";
    }
    html += pdfForm(j.report_form) + windowTable(j) + "</div>";
    live.innerHTML = html;
    live.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function showError(message) {
    live.innerHTML = '<div class="records-error"><strong>⚠ Analysis Error:</strong> ' + esc(message) + "</div>";
    live.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var button = form.querySelector('button[type="submit"]');
    var label = button ? button.innerHTML : "";
    if (button) { button.disabled = true; button.innerHTML = "⏳ Analysing every window…"; }
    document.querySelectorAll(".result-card, .eeg-server-error").forEach(function (el) { el.remove(); });

    fetch("/api/eeg/analyze", { method: "POST", body: new FormData(form) })
      .then(function (resp) {
        return resp.json().then(function (j) { return { status: resp.status, body: j }; });
      })
      .then(function (r) {
        if (r.body && r.body.ok) render(r.body);
        else showError((r.body && r.body.error) || "The analysis failed (HTTP " + r.status + ").");
      })
      .catch(function () {
        // API unreachable: fall back to the classic server-rendered submit.
        form.submit();
      })
      .finally(function () {
        if (button) { button.disabled = false; button.innerHTML = label; }
      });
  });
})();
