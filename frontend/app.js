const form = document.getElementById("customer-form");
const ADDONS = [...form.querySelectorAll("[data-addon]")];
const MULTIPLE_LINES = form.elements.MultipleLines;

const PRESETS = {
  risky: {
    gender: "Female", SeniorCitizen: "0", Partner: "No", Dependents: "No",
    tenure: 3, MonthlyCharges: 89.5, Contract: "Month-to-month",
    PaymentMethod: "Electronic check", PaperlessBilling: "Yes",
    PhoneService: "Yes", MultipleLines: "No", InternetService: "Fiber optic",
    OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No",
    TechSupport: "No", StreamingTV: "Yes", StreamingMovies: "No",
  },
  loyal: {
    gender: "Male", SeniorCitizen: "0", Partner: "Yes", Dependents: "Yes",
    tenure: 60, MonthlyCharges: 65, Contract: "Two year",
    PaymentMethod: "Bank transfer (automatic)", PaperlessBilling: "No",
    PhoneService: "Yes", MultipleLines: "Yes", InternetService: "DSL",
    OnlineSecurity: "Yes", OnlineBackup: "Yes", DeviceProtection: "Yes",
    TechSupport: "Yes", StreamingTV: "No", StreamingMovies: "No",
  },
  basic: {
    gender: "Female", SeniorCitizen: "0", Partner: "Yes", Dependents: "No",
    tenure: 24, MonthlyCharges: 20, Contract: "One year",
    PaymentMethod: "Mailed check", PaperlessBilling: "No",
    PhoneService: "Yes", MultipleLines: "No", InternetService: "No",
    OnlineSecurity: "No internet service", OnlineBackup: "No internet service",
    DeviceProtection: "No internet service", TechSupport: "No internet service",
    StreamingTV: "No internet service", StreamingMovies: "No internet service",
  },
};

// Keep dependent fields valid: without internet every add-on is
// "No internet service", and without phone service MultipleLines is
// "No phone service". The API rejects any other combination.
function setLocked(select, locked, lockedValue) {
  const existing = [...select.options].find((o) => o.value === lockedValue);
  if (locked) {
    if (!existing) select.add(new Option(lockedValue, lockedValue));
    select.value = lockedValue;
    select.disabled = true;
  } else {
    if (existing) {
      existing.remove();
      select.value = "No";
    }
    select.disabled = false;
  }
}

function syncDependents() {
  const noInternet = form.elements.InternetService.value === "No";
  ADDONS.forEach((s) => setLocked(s, noInternet, "No internet service"));
  setLocked(MULTIPLE_LINES, form.elements.PhoneService.value === "No", "No phone service");
}

function updateOutputs() {
  const tenure = Number(form.elements.tenure.value);
  document.getElementById("tenure-out").textContent =
    `${tenure} ${tenure === 1 ? "month" : "months"}`;
  document.getElementById("charges-out").textContent =
    `$${Number(form.elements.MonthlyCharges.value).toFixed(2)}`;
}

function readForm() {
  const data = {};
  for (const el of form.elements) {
    if (!el.name) continue;
    data[el.name] = el.value;
  }
  data.SeniorCitizen = Number(data.SeniorCitizen);
  data.tenure = Number(data.tenure);
  data.MonthlyCharges = Number(data.MonthlyCharges);
  return data;
}

function applyPreset(name) {
  const preset = PRESETS[name];
  // Set the parent fields first so the dependent selects get the right options.
  form.elements.InternetService.value = preset.InternetService;
  form.elements.PhoneService.value = preset.PhoneService;
  syncDependents();
  for (const [key, value] of Object.entries(preset)) {
    form.elements[key].value = String(value);
  }
  updateOutputs();
  predict();
}

const pct = (p) => `${(p * 100).toFixed(1)}%`;

function renderPrediction(result) {
  const level = result.risk_level;
  const badge = document.getElementById("risk-badge");
  badge.textContent = `${level} risk`;
  badge.className = `badge ${level}`;

  document.getElementById("probability").textContent = pct(result.churn_probability);
  const fill = document.getElementById("meter-fill");
  fill.style.width = `${result.churn_probability * 100}%`;
  fill.style.background = `var(--${level})`;
  document.getElementById("meter").setAttribute(
    "aria-label", `Churn probability ${pct(result.churn_probability)}`);

  document.getElementById("verdict").textContent = result.will_churn
    ? "The model predicts this customer is likely to cancel."
    : "The model predicts this customer is likely to stay.";

  const maxImpact = Math.max(0.5, ...result.factors.map((f) => Math.abs(f.impact)));
  const factors = document.getElementById("factors");
  factors.replaceChildren(...result.factors.map((f) => {
    const row = document.createElement("div");
    row.className = "factor";
    const direction = f.impact > 0 ? "up" : "down";
    const width = (Math.abs(f.impact) / maxImpact) * 50;
    row.innerHTML = `
      <div class="name" title="${f.label}: ${f.value}"><b>${f.label}</b> <em>${f.value}</em></div>
      <div class="bar-track"><div class="bar ${direction}" style="width:${width}%"></div></div>`;
    row.title = `${f.label} = ${f.value} ${direction === "up" ? "raises" : "lowers"} the risk`;
    return row;
  }));
  if (!result.factors.length) {
    factors.textContent = "This customer is close to the average profile.";
  }

  const recs = document.getElementById("recommendations");
  recs.replaceChildren(...result.recommendations.map((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    return li;
  }));
}

let pending;
let requestId = 0;

async function predict() {
  const id = ++requestId;
  const error = document.getElementById("error");
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail?.[0]?.msg || "Prediction failed");
    if (id !== requestId) return; // a newer request has already been sent
    error.hidden = true;
    renderPrediction(body);
  } catch (err) {
    if (id !== requestId) return;
    error.textContent = `${err.message}. The server may be waking up; it will retry when you change a field.`;
    error.hidden = false;
  }
}

function schedulePredict() {
  clearTimeout(pending);
  pending = setTimeout(predict, 150);
}

async function loadModelInfo() {
  try {
    const info = await (await fetch("/model-info")).json();
    const t = info.test;
    document.getElementById("model-summary").textContent =
      `${info.model} trained on ${info.train_rows.toLocaleString()} customers and tested on ` +
      `${info.test_rows.toLocaleString()} it never saw. ${pct(info.churn_rate)} of customers in the data cancelled.`;
    const stats = [
      ["Recall", t.recall], ["Precision", t.precision], ["F1", t.f1],
      ["Accuracy", t.accuracy], ["ROC-AUC", t.roc_auc],
    ];
    document.getElementById("stats").replaceChildren(...stats.map(([label, value]) => {
      const div = document.createElement("div");
      div.className = "stat";
      div.innerHTML = `<b>${label === "ROC-AUC" ? value.toFixed(2) : pct(value)}</b><span>${label}</span>`;
      return div;
    }));
  } catch {
    /* model info is optional */
  }
}

form.addEventListener("input", (event) => {
  if (["InternetService", "PhoneService"].includes(event.target.name)) syncDependents();
  updateOutputs();
  schedulePredict();
});
form.addEventListener("submit", (event) => event.preventDefault());
document.querySelectorAll("[data-preset]").forEach((btn) =>
  btn.addEventListener("click", () => applyPreset(btn.dataset.preset)));

syncDependents();
updateOutputs();
predict();
loadModelInfo();
