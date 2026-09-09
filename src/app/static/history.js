const HISTORY_CHART_WIDTH = 800;
const HISTORY_CHART_HEIGHT = 320;
const HISTORY_CHART_PADDING = { top: 16, right: 16, bottom: 28, left: 56 };
const HISTORY_SERIES = [
  { key: "actual_demand_mw", cls: "history-line-actual" },
  { key: "ndf_forecast_mw", cls: "history-line-ndf" },
  { key: "blended_forecast_mw", cls: "history-line-blended" },
];
const HISTORY_MIN_ZOOM_DAYS = 3; // shortest range a drag-zoom can select

let historyAllDays = [];
let historyViewStart = 0;
let historyViewEnd = 0; // inclusive indices into historyAllDays

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  return el;
}

function pathFromSeries(values, xScale, yScale) {
  let d = "";
  let drawing = false;
  values.forEach((value, i) => {
    if (value === null || value === undefined) {
      drawing = false;
      return;
    }
    const x = xScale(i).toFixed(1);
    const y = yScale(value).toFixed(1);
    d += `${drawing ? "L" : "M"}${x} ${y} `;
    drawing = true;
  });
  return d.trim();
}

function attachDragZoom(svg, days, xScale, pad, innerWidth) {
  let startIndex = null;
  let selectionRect = null;

  const indexFromClientX = (clientX) => {
    const rect = svg.getBoundingClientRect();
    const scale = HISTORY_CHART_WIDTH / rect.width;
    const svgX = (clientX - rect.left) * scale;
    const fraction = (svgX - pad.left) / innerWidth;
    const index = Math.round(fraction * (days.length - 1));
    return Math.min(days.length - 1, Math.max(0, index));
  };

  svg.addEventListener("pointerdown", (event) => {
    startIndex = indexFromClientX(event.clientX);
    selectionRect = svgEl("rect", {
      x: xScale(startIndex).toFixed(1), y: pad.top,
      width: 0, height: HISTORY_CHART_HEIGHT - pad.top - pad.bottom,
      class: "history-selection",
    });
    svg.appendChild(selectionRect);
    svg.setPointerCapture(event.pointerId);
  });

  svg.addEventListener("pointermove", (event) => {
    if (startIndex === null) return;
    const currentIndex = indexFromClientX(event.clientX);
    const x1 = xScale(Math.min(startIndex, currentIndex));
    const x2 = xScale(Math.max(startIndex, currentIndex));
    selectionRect.setAttribute("x", x1.toFixed(1));
    selectionRect.setAttribute("width", Math.max(0, x2 - x1).toFixed(1));
  });

  const finishDrag = (event) => {
    if (startIndex === null) return;
    const endIndex = indexFromClientX(event.clientX);
    const lo = Math.min(startIndex, endIndex);
    const hi = Math.max(startIndex, endIndex);
    startIndex = null;
    selectionRect?.remove();
    if (hi - lo >= HISTORY_MIN_ZOOM_DAYS) {
      setHistoryView(historyViewStart + lo, historyViewStart + hi);
    }
  };
  svg.addEventListener("pointerup", finishDrag);
  svg.addEventListener("pointercancel", finishDrag);
}

function renderHistoryChart(days) {
  const container = document.getElementById("history-chart");
  container.innerHTML = "";
  if (!days.length) {
    container.textContent = "No historical comparison data available in this deployment.";
    return;
  }

  const width = HISTORY_CHART_WIDTH;
  const height = HISTORY_CHART_HEIGHT;
  const pad = HISTORY_CHART_PADDING;
  const innerWidth = width - pad.left - pad.right;
  const innerHeight = height - pad.top - pad.bottom;

  const allValues = days.flatMap((d) =>
    HISTORY_SERIES.map((s) => d[s.key]).filter((v) => v !== null && v !== undefined)
  );
  const yMin = Math.min(...allValues);
  const yMax = Math.max(...allValues);
  const yPadding = (yMax - yMin) * 0.08 || 1;
  const scaleMin = yMin - yPadding;
  const scaleMax = yMax + yPadding;

  const xScale = (i) => pad.left + (days.length === 1 ? 0 : (i / (days.length - 1)) * innerWidth);
  const yScale = (v) => pad.top + innerHeight - ((v - scaleMin) / (scaleMax - scaleMin)) * innerHeight;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "history-chart-svg",
    role: "img",
    "aria-label": "Actual demand vs NDF vs our blended forecast",
  });

  const ticks = 4;
  for (let t = 0; t <= ticks; t++) {
    const value = scaleMin + (t / ticks) * (scaleMax - scaleMin);
    const y = yScale(value);
    svg.appendChild(
      svgEl("line", {
        x1: pad.left, x2: width - pad.right, y1: y.toFixed(1), y2: y.toFixed(1),
        class: "history-gridline",
      })
    );
    const label = svgEl("text", {
      x: pad.left - 8, y: (y + 4).toFixed(1), class: "history-axis-label", "text-anchor": "end",
    });
    label.textContent = `${Math.round(value / 1000)} GW`;
    svg.appendChild(label);
  }

  const labelIndices = [0, Math.floor((days.length - 1) / 2), days.length - 1];
  labelIndices.forEach((i, position) => {
    const anchor = position === 0 ? "start" : position === labelIndices.length - 1 ? "end" : "middle";
    const label = svgEl("text", {
      x: xScale(i).toFixed(1), y: height - 6, class: "history-axis-label", "text-anchor": anchor,
    });
    label.textContent = days[i].date;
    svg.appendChild(label);
  });

  for (const series of HISTORY_SERIES) {
    const values = days.map((d) => d[series.key]);
    const d = pathFromSeries(values, xScale, yScale);
    if (d) svg.appendChild(svgEl("path", { d, class: `history-line ${series.cls}`, fill: "none" }));
  }

  attachDragZoom(svg, days, xScale, pad, innerWidth);
  container.appendChild(svg);
}

function updateRangeButtons() {
  const fullRange = historyViewStart === 0 && historyViewEnd === historyAllDays.length - 1;
  const viewLength = historyViewEnd - historyViewStart + 1;
  document.querySelectorAll(".history-range-btn").forEach((button) => {
    const isAll = button.dataset.days === "all";
    const matches = isAll ? fullRange : !fullRange && Number(button.dataset.days) === viewLength;
    button.classList.toggle("active", matches);
  });
  document.getElementById("history-reset-zoom").hidden = fullRange;
}

function setHistoryView(start, end) {
  historyViewStart = Math.max(0, start);
  historyViewEnd = Math.min(historyAllDays.length - 1, end);
  renderHistoryChart(historyAllDays.slice(historyViewStart, historyViewEnd + 1));
  updateRangeButtons();
}

function applyPresetRange(daysArg) {
  if (daysArg === "all") {
    setHistoryView(0, historyAllDays.length - 1);
  } else {
    const count = Number(daysArg);
    setHistoryView(historyAllDays.length - count, historyAllDays.length - 1);
  }
}

document.querySelectorAll(".history-range-btn").forEach((button) => {
  button.addEventListener("click", () => applyPresetRange(button.dataset.days));
});
document.getElementById("history-reset-zoom").addEventListener("click", () => applyPresetRange("all"));

async function loadHistory() {
  const chart = document.getElementById("history-chart");
  const note = document.getElementById("history-note");
  try {
    const response = await fetch("/api/history");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    historyAllDays = data.days;
    setHistoryView(0, historyAllDays.length - 1);
    note.textContent = data.days.length
      ? `${data.weather_upper_bound_note} ${data.scottish_transfer_advisory}`
      : "Historical comparison not available in this deployment.";
  } catch (err) {
    chart.textContent = `Couldn't load history: ${err.message}`;
  }
}

loadHistory();
