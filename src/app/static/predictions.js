const HORIZON_LABELS = { "30min": "30 minutes", "1h": "1 hour", "1d": "1 day", "7d": "7 days" };
const HORIZON_ORDER = ["30min", "1h", "1d", "7d"];

const mwFormatter = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });
const timeFormatter = new Intl.DateTimeFormat("en-GB", {
  weekday: "short", hour: "2-digit", minute: "2-digit", timeZoneName: "short",
});

function formatMw(value) {
  return `${mwFormatter.format(value)} MW`;
}

function formatTime(iso) {
  return timeFormatter.format(new Date(iso));
}

function renderHorizonCard(name, horizon) {
  const template = document.getElementById("horizon-card-template");
  const node = template.content.cloneNode(true);

  node.querySelector(".horizon-name").textContent = HORIZON_LABELS[name] ?? name;
  node.querySelector(".target-time").textContent = `Target: ${formatTime(horizon.target_time)}`;
  node.querySelector(".point-number").textContent = mwFormatter.format(horizon.point_mw);

  const { p10_mw: p10, p50_mw: p50, p90_mw: p90 } = horizon;
  const padding = (p90 - p10) * 0.15 || 1;
  const scaleMin = p10 - padding;
  const scaleMax = p90 + padding;
  const toPct = (value) => ((value - scaleMin) / (scaleMax - scaleMin)) * 100;

  const fill = node.querySelector(".interval-fill");
  fill.style.left = `${toPct(p10)}%`;
  fill.style.width = `${toPct(p90) - toPct(p10)}%`;
  node.querySelector(".p50-marker").style.left = `${toPct(p50)}%`;

  node.querySelector(".p10-label").textContent = formatMw(p10);
  node.querySelector(".p50-label").textContent = formatMw(p50);
  node.querySelector(".p90-label").textContent = formatMw(p90);

  return node;
}

function renderNdfPanel(comparison) {
  const panel = document.getElementById("ndf-panel");
  if (!comparison) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  document.getElementById("ndf-time").textContent = formatTime(comparison.cardinal_point_time);
  document.getElementById("ndf-forecast").textContent = formatMw(comparison.ndf_forecast_mw);
  document.getElementById("ndf-ours").textContent = formatMw(comparison.our_forecast_mw);
  document.getElementById("ndf-combined").textContent = formatMw(comparison.combined_forecast_mw);
  document.getElementById("ndf-note").textContent = comparison.note;
}

async function loadPredictions() {
  const status = document.getElementById("status-line");
  const grid = document.getElementById("horizon-grid");
  const button = document.getElementById("refresh-button");

  status.textContent = "Loading live data…";
  button.disabled = true;
  try {
    const response = await fetch("/api/predict");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    grid.innerHTML = "";
    for (const name of HORIZON_ORDER) {
      const horizon = data.horizons[name];
      if (horizon) grid.appendChild(renderHorizonCard(name, horizon));
    }
    renderNdfPanel(data.ndf_comparison);

    status.textContent =
      `Issue time (latest available demand): ${formatTime(data.issue_time)} — ` +
      `weather fetched: ${formatTime(data.weather_as_of)}`;
  } catch (err) {
    status.textContent = `Couldn't load live predictions: ${err.message}`;
  } finally {
    button.disabled = false;
  }
}

document.getElementById("refresh-button").addEventListener("click", loadPredictions);
loadPredictions();
