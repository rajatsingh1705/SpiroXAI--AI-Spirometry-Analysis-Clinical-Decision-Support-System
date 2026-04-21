/**
 * SpiroXAI — Frontend Application Logic
 * Demo mode: No authentication. Direct API calls to FastAPI backend.
 */

const API = "http://localhost:8000";

// ── Startup ───────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  showView("dashboard");
  loadRecords();

  // ADDED: History search filter
  const searchInput = document.getElementById("historySearch");
  if (searchInput) {
    searchInput.addEventListener("input", filterHistory);
  }
});

// ── View Routing ──────────────────────────────────────────────────────────────
function showView(view) {
  document.getElementById("view-dashboard").style.display = view === "dashboard" ? "block" : "none";
  document.getElementById("view-history").style.display   = view === "history"   ? "block" : "none";

  // Update active nav
  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle("active", el.id === `nav-${view}`);
  });

  if (view === "history") loadHistory();
}

// ── Auto-calculations ─────────────────────────────────────────────────────────
function autoCalcBMI() {
  const w = parseFloat(document.getElementById("inp-Weight").value);
  const h = parseFloat(document.getElementById("inp-Height").value);
  if (w > 0 && h > 0) {
    const bmi = (w / ((h / 100) ** 2)).toFixed(1);
    document.getElementById("inp-BMI").value = bmi;
  } else {
    document.getElementById("inp-BMI").value = "";
  }
}

function autoCalcRatio() {
  const fev1 = parseFloat(document.getElementById("ref-FEV1").value);
  const fvc  = parseFloat(document.getElementById("ref-FVC").value);
  if (fev1 > 0 && fvc > 0) {
    document.getElementById("ref-Ratio").value = (fev1 / fvc).toFixed(2);
  } else {
    document.getElementById("ref-Ratio").value = "";
  }
}

// ── Sample Data ───────────────────────────────────────────────────────────────
function loadSample(type) {
  const samples = {
    normal: {
      Age: 35, Sex: "1", Weight: 75, Height: 175, Race: "White",
      Baseline_PEF_Ls: 9.50, Baseline_FEF2575_Ls: 4.80,
      Baseline_Extrapolated_Volume: 0.08, Baseline_Forced_Expiratory_Time: 4.00,
      Baseline_Number_Acceptable_Curves: 5,
    },
    obstruction: {
      Age: 62, Sex: "1", Weight: 85, Height: 172, Race: "White",
      Baseline_PEF_Ls: 2.10, Baseline_FEF2575_Ls: 0.35,
      Baseline_Extrapolated_Volume: 0.30, Baseline_Forced_Expiratory_Time: 12.00,
      Baseline_Number_Acceptable_Curves: 3,
    },
    restriction: {
      Age: 48, Sex: "0", Weight: 58, Height: 155, Race: "White",
      Baseline_PEF_Ls: 3.20, Baseline_FEF2575_Ls: 3.10,
      Baseline_Extrapolated_Volume: 0.04, Baseline_Forced_Expiratory_Time: 1.50,
      Baseline_Number_Acceptable_Curves: 5,
    },
  };

  const data = samples[type];
  if (!data) return;

  // Fill demographics
  document.getElementById("inp-Age").value = data.Age;
  document.getElementById("inp-Sex").value = data.Sex;
  document.getElementById("inp-Weight").value = data.Weight;
  document.getElementById("inp-Height").value = data.Height;
  document.getElementById("inp-Race").value = data.Race;
  autoCalcBMI();

  // Fill spirometry
  const spiroFields = [
    "Baseline_PEF_Ls", "Baseline_FEF2575_Ls", "Baseline_Extrapolated_Volume",
    "Baseline_Forced_Expiratory_Time", "Baseline_Number_Acceptable_Curves"
  ];
  spiroFields.forEach(f => {
    document.getElementById(`inp-${f}`).value = data[f];
  });

  // Clear reference fields
  document.getElementById("ref-FEV1").value = "";
  document.getElementById("ref-FVC").value = "";
  document.getElementById("ref-Ratio").value = "";

  clearAllFieldErrors();
  showToast(`Loaded ${type} sample data`, "info");
}

// ── Prediction ────────────────────────────────────────────────────────────────
async function runPrediction() {
  clearAllFieldErrors();
  hideError("form-error");

  // Gather form values
  const age    = parseFloat(document.getElementById("inp-Age").value);
  const sex    = parseFloat(document.getElementById("inp-Sex").value);
  const weight = parseFloat(document.getElementById("inp-Weight").value);
  const height = parseFloat(document.getElementById("inp-Height").value);
  const bmi    = parseFloat(document.getElementById("inp-BMI").value);

  // Basic client-side checks
  if (isNaN(age) || isNaN(weight) || isNaN(height)) {
    showError("form-error", "Please fill in all demographic fields.");
    return;
  }

  const spiroFields = [
    "Baseline_PEF_Ls", "Baseline_FEF2575_Ls", "Baseline_Extrapolated_Volume",
    "Baseline_Forced_Expiratory_Time", "Baseline_Number_Acceptable_Curves"
  ];

  const patientName = document.getElementById("patientName").value.trim() || "Unknown"; // ADDED

  const payload = {
    patient_name: patientName, // ADDED
    Age: age,
    Sex: sex,
    Weight: weight,
    Height: height,
    BMI: bmi,
  };

  for (const f of spiroFields) {
    const el = document.getElementById(`inp-${f}`);
    if (!el || el.value === "") {
      showError("form-error", `Please enter a value for ${f.replace(/Baseline_/g, "").replace(/_/g, " ")}`);
      return;
    }
    payload[f] = parseFloat(el.value);
  }

  // Race one-hot encoding
  const race = document.getElementById("inp-Race").value;
  payload.Race_Black = race === "Black" ? 1 : 0;
  payload.Race_Mexican_American = race === "Mexican American" ? 1 : 0;
  payload.Race_Other_hispanic = race === "Other Hispanic" ? 1 : 0;
  payload.Race_Other_race_including_multi_racial = race === "Other" ? 1 : 0;
  payload.Race_White = race === "White" ? 1 : 0;

  showSpinner(true);

  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const json = await res.json();

    if (!res.ok || json.status === "error") {
      // Handle validation errors
      if (json.errors && Array.isArray(json.errors)) {
        json.errors.forEach(err => {
          showFieldError(err.field, err.message);
        });
        showToast("Validation failed. Please check the highlighted fields.", "error");
      } else {
        showError("form-error", json.detail || json.message || "Prediction failed.");
        showToast("Prediction failed", "error");
      }
      return;
    }

    // Success
    renderResults(json);
    showToast("Diagnosis complete!", "success");
    loadRecords(); // Refresh records

  } catch (e) {
    showError("form-error", "Could not reach the backend. Is the server running on port 8000?");
    showToast("Connection error", "error");
  } finally {
    showSpinner(false);
  }
}

// ── Render Results ────────────────────────────────────────────────────────────
function renderResults(json) {
  const pred = json.prediction;
  const expl = json.explanation || {};

  // Show results, hide empty state
  document.getElementById("results-panel").style.display = "flex";
  document.getElementById("empty-results").style.display = "none";

  const cls  = pred.predicted_class;
  const slug = cls.toLowerCase();
  const conf = pred.confidence_pct;

  // Badge
  const icons = { normal: "✅", obstruction: "⚠️", restriction: "🔴" };
  document.getElementById("badge-icon").textContent  = icons[slug] || "❓";
  document.getElementById("badge-class").textContent = cls;
  document.getElementById("badge-conf").textContent  = `${conf}% confidence`;
  document.getElementById("prediction-badge").className = `prediction-badge ${slug}`;

  // ADDED: Display patient name in result panel
  const patientNameDisplay = json.patient_name || pred.patient_name || "Unknown";
  let nameEl = document.getElementById("result-patient-name");
  if (!nameEl) {
    nameEl = document.createElement("div");
    nameEl.id = "result-patient-name";
    nameEl.style.cssText = "font-size:1.05rem;font-weight:600;margin-bottom:.5rem;color:var(--text)";
    document.getElementById("prediction-badge").before(nameEl);
  }
  nameEl.textContent = `Patient: ${patientNameDisplay}`;
  // /ADDED

  // Timestamp
  if (pred.timestamp) {
    const dt = new Date(pred.timestamp + "Z");
    document.getElementById("result-timestamp").textContent = dt.toLocaleString();
  }

  // Heuristic warning
  document.getElementById("heuristic-warning").style.display =
    pred.is_heuristic ? "block" : "none";

  // Clinical subtext
  const subtexts = {
    Normal: "<strong>Normal Pattern</strong><br/>Spirometry values are within healthy baseline ranges for the patient's demographics.",
    Obstruction: "<strong>Obstructive Pattern</strong><br/>Reduced mid-expiratory flow (FEF 25-75%) and/or prolonged expiratory time suggest airway obstruction, consistent with conditions such as asthma or COPD.",
    Restriction: "<strong>Restrictive Pattern</strong><br/>Reduced lung volumes with preserved or elevated flow rates suggest a restrictive process, consistent with pulmonary fibrosis or chest wall disorders.",
  };
  document.getElementById("prediction-subtext").innerHTML = subtexts[cls] || "";

  // Probability bars
  const probs = pred.probabilities;
  const barsEl = document.getElementById("prob-bars");
  barsEl.innerHTML = "";
  for (const [clsName, prob] of Object.entries(probs)) {
    const pct = Math.round(prob * 100);
    const s = clsName.toLowerCase();
    const row = document.createElement("div");
    row.className = "prob-bar-row";
    row.innerHTML = `
      <span class="prob-bar-label">${clsName}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill ${s}" style="width:0%" data-target="${pct}"></div>
      </div>
      <span class="prob-bar-pct">${pct}%</span>
    `;
    barsEl.appendChild(row);
  }
  // Animate bars
  requestAnimationFrame(() => {
    document.querySelectorAll(".prob-bar-fill").forEach(el => {
      el.style.width = el.dataset.target + "%";
    });
  });

  // Explanation text
  document.getElementById("explanation-text").textContent =
    expl.text_summary || "No explanation available.";

  // SHAP table (top 5 features)
  const topFeatures = (expl.top_features || []).slice(0, 5);
  const shapBody = document.getElementById("shap-body");
  shapBody.innerHTML = "";

  if (topFeatures.length > 0) {
    document.getElementById("shap-table").style.display = "table";
    topFeatures.forEach((f, i) => {
      const arrow = f.direction === "positive"
        ? '<span class="arrow-up">▲ Increases risk</span>'
        : '<span class="arrow-down">▼ Decreases risk</span>';
      const label = f.feature.replace(/_/g, " ").replace("Baseline ", "");
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td class="feat-name-cell" title="${f.feature}">${label}</td>
        <td>${f.value}</td>
        <td>${f.contribution > 0 ? "+" : ""}${f.contribution.toFixed(4)}</td>
        <td>${arrow}</td>
      `;
      shapBody.appendChild(tr);
    });
  } else {
    document.getElementById("shap-table").style.display = "none";
    document.getElementById("shap-table-container").innerHTML =
      '<p style="color:var(--muted);font-size:.85rem">Feature importance not available (SHAP requires XGBoost model)</p>';
  }

  // Scroll to results
  document.getElementById("results-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Records (Dashboard bottom + History view) ─────────────────────────────────
async function loadRecords() {
  const loading = document.getElementById("records-loading");
  const empty   = document.getElementById("records-empty");
  const table   = document.getElementById("records-table");
  const tbody   = document.getElementById("records-body");

  loading.style.display = "block";
  empty.style.display   = "none";
  table.style.display   = "none";

  try {
    const res = await fetch(`${API}/records`);
    const json = await res.json();
    const records = json.records || [];

    loading.style.display = "none";

    if (records.length === 0) {
      empty.style.display = "block";
      return;
    }

    table.style.display = "table";
    tbody.innerHTML = "";

    // Show max 10 in dashboard
    records.slice(0, 10).forEach(r => {
      const dt = new Date(r.created_at).toLocaleString();
      const cls = r.predicted_class || "—";
      const slug = cls.toLowerCase();
      const conf = r.confidence_normal || r.confidence_obstruction || r.confidence_restriction;
      const maxConf = Math.max(r.confidence_normal || 0, r.confidence_obstruction || 0, r.confidence_restriction || 0);
      const confPct = Math.round(maxConf * 100);
      const sexLabel = r.patient_sex === 1 ? "M" : "F";

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${dt}</td>
        <td>${r.patient_name || "Unknown"}</td><!-- ADDED -->
        <td>${r.patient_age || "—"}</td>
        <td>${sexLabel}</td>
        <td>${r.pef || "—"}</td>
        <td>${r.fef2575 || "—"}</td>
        <td><span class="diag-pill ${slug}">${cls}</span></td>
        <td>${confPct}%</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    loading.textContent = "Could not load records. Is the backend running?";
  }
}

async function loadHistory() {
  const loading = document.getElementById("history-loading");
  const empty   = document.getElementById("history-empty");
  const table   = document.getElementById("history-table");
  const tbody   = document.getElementById("history-body");

  loading.style.display = "block";
  empty.style.display   = "none";
  table.style.display   = "none";

  try {
    const res = await fetch(`${API}/records`);
    const json = await res.json();
    const records = json.records || [];

    loading.style.display = "none";

    if (records.length === 0) {
      empty.style.display = "block";
      return;
    }

    table.style.display = "table";
    tbody.innerHTML = "";

    records.forEach(r => {
      const dt = new Date(r.created_at).toLocaleString();
      const cls = r.predicted_class || "—";
      const slug = cls.toLowerCase();
      const sexLabel = r.patient_sex === 1 ? "M" : "F";

      const tr = document.createElement("tr");
      // ADDED: data attributes for search filtering
      tr.dataset.patientName = (r.patient_name || "Unknown").toLowerCase();
      tr.dataset.diagnosis = (cls || "").toLowerCase();
      // /ADDED
      tr.innerHTML = `
        <td>${dt}</td>
        <td>${r.patient_name || "Unknown"}</td><!-- ADDED -->
        <td>${r.patient_age || "—"}</td>
        <td>${sexLabel}</td>
        <td>${r.race || "—"}</td>
        <td>${r.bmi || "—"}</td>
        <td>${r.pef || "—"}</td>
        <td>${r.fef2575 || "—"}</td>
        <td><span class="diag-pill ${slug}">${cls}</span></td>
        <td>${r.confidence_normal ? Math.round(r.confidence_normal * 100) + "%" : "—"}</td>
        <td>${r.confidence_obstruction ? Math.round(r.confidence_obstruction * 100) + "%" : "—"}</td>
        <td>${r.confidence_restriction ? Math.round(r.confidence_restriction * 100) + "%" : "—"}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    loading.textContent = "Could not load history. Is the backend running?";
  }
}

// ── Toast Notifications ───────────────────────────────────────────────────────
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;

  const icons = { success: "✅", error: "❌", info: "ℹ️" };
  toast.innerHTML = `<span class="toast-icon">${icons[type] || "ℹ️"}</span><span>${message}</span>`;

  container.appendChild(toast);

  // Trigger animation
  requestAnimationFrame(() => toast.classList.add("show"));

  // Auto-remove after 4s
  setTimeout(() => {
    toast.classList.remove("show");
    toast.classList.add("hide");
    setTimeout(() => toast.remove(), 400);
  }, 4000);
}

// ── Field Error Handling ──────────────────────────────────────────────────────
function showFieldError(field, message) {
  const errEl = document.getElementById(`err-${field}`);
  if (errEl) {
    errEl.textContent = message;
    errEl.style.display = "block";
  }
  // Highlight input
  const inp = document.getElementById(`inp-${field}`);
  if (inp) inp.classList.add("input-error");
}

function clearAllFieldErrors() {
  document.querySelectorAll(".field-error").forEach(el => {
    el.textContent = "";
    el.style.display = "none";
  });
  document.querySelectorAll(".input-error").forEach(el => {
    el.classList.remove("input-error");
  });
}

// ── UI Helpers ────────────────────────────────────────────────────────────────
function showError(id, msg) {
  const el = document.getElementById(id);
  if (el) { el.textContent = msg; el.style.display = "block"; }
}

function hideError(id) {
  const el = document.getElementById(id);
  if (el) { el.textContent = ""; el.style.display = "none"; }
}

function showSpinner(show) {
  document.getElementById("global-spinner").style.display = show ? "flex" : "none";
}

// ADDED: Filter history rows by patient name or diagnosis
function filterHistory() {
  const query = document.getElementById("historySearch").value.trim().toLowerCase();
  const tbody = document.getElementById("history-body");
  const noResults = document.getElementById("history-no-results");
  if (!tbody) return;

  const rows = tbody.querySelectorAll("tr");
  let visibleCount = 0;

  rows.forEach(tr => {
    const name = tr.dataset.patientName || "";
    const diag = tr.dataset.diagnosis || "";
    const match = !query || name.includes(query) || diag.includes(query);
    tr.style.display = match ? "" : "none";
    if (match) visibleCount++;
  });

  if (noResults) {
    noResults.style.display = (rows.length > 0 && visibleCount === 0) ? "block" : "none";
  }
}
// /ADDED
