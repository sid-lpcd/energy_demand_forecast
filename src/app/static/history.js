const HISTORY_CHART_WIDTH = 800;
const HISTORY_CHART_HEIGHT = 260;
const HISTORY_CHART_PADDING = { top: 16, right: 16, bottom: 8, left: 56 };
const HISTORY_DIFF_HEIGHT = 130;
const HISTORY_DIFF_PADDING = { top: 10, right: 16, bottom: 28, left: 56 };
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

function makeXScale(days, pad, innerWidth) {
  return (i) => pad.left + (days.length === 1 ? 0 : (i / (days.length - 1)) * innerWidth);
}

function buildMainChart(days) {
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

  const xScale = makeXScale(days, pad, innerWidth);
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

  for (const series of HISTORY_SERIES) {
    const values = days.map((d) => d[series.key]);
    const d = pathFromSeries(values, xScale, yScale);
    if (d) svg.appendChild(svgEl("path", { d, class: `history-line ${series.cls}`, fill: "none" }));
  }

  return { svg, xScale, yScale, pad, innerWidth, innerHeight };
}

function buildDiffChart(days) {
  const width = HISTORY_CHART_WIDTH;
  const height = HISTORY_DIFF_HEIGHT;
  const pad = HISTORY_DIFF_PADDING;
  const innerWidth = width - pad.left - pad.right;
  const innerHeight = height - pad.top - pad.bottom;

  const blendDiff = days.map((d) => (d.actual_demand_mw == null ? null : d.blended_forecast_mw - d.actual_demand_mw));
  const ndfDiff = days.map((d) => (d.actual_demand_mw == null ? null : d.ndf_forecast_mw - d.actual_demand_mw));
  const allAbs = [...blendDiff, ...ndfDiff].filter((v) => v !== null).map(Math.abs);
  const bound = (Math.max(...allAbs, 100)) * 1.15;

  const xScale = makeXScale(days, pad, innerWidth);
  const yScale = (v) => pad.top + innerHeight / 2 - (v / bound) * (innerHeight / 2);

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "history-chart-svg history-diff-svg",
    role: "img",
    "aria-label": "Forecast error vs actual demand, NDF and our blend",
  });

  [bound, 0, -bound].forEach((value) => {
    const y = yScale(value);
    svg.appendChild(
      svgEl("line", {
        x1: pad.left, x2: width - pad.right, y1: y.toFixed(1), y2: y.toFixed(1),
        class: value === 0 ? "history-gridline history-zero-line" : "history-gridline",
      })
    );
    const label = svgEl("text", {
      x: pad.left - 8, y: (y + 3).toFixed(1), class: "history-axis-label", "text-anchor": "end",
    });
    label.textContent = value === 0 ? "0" : `${value > 0 ? "+" : ""}${Math.round(value)} MW`;
    svg.appendChild(label);
  });

  const labelIndices = [0, Math.floor((days.length - 1) / 2), days.length - 1];
  labelIndices.forEach((i, position) => {
    const anchor = position === 0 ? "start" : position === labelIndices.length - 1 ? "end" : "middle";
    const label = svgEl("text", {
      x: xScale(i).toFixed(1), y: height - 6, class: "history-axis-label", "text-anchor": anchor,
    });
    label.textContent = days[i].date;
    svg.appendChild(label);
  });

  const ndfPath = pathFromSeries(ndfDiff, xScale, yScale);
  if (ndfPath) svg.appendChild(svgEl("path", { d: ndfPath, class: "history-line history-line-ndf", fill: "none" }));
  const blendPath = pathFromSeries(blendDiff, xScale, yScale);
  if (blendPath) svg.appendChild(svgEl("path", { d: blendPath, class: "history-line history-line-blended", fill: "none" }));

  return { svg, xScale, yScale, pad, innerWidth, innerHeight };
}

function formatMw(value) {
  return value == null ? "–" : `${Math.round(value).toLocaleString()} MW`;
}

function formatDiff(value) {
  if (value == null) return "";
  const rounded = Math.round(value);
  return ` (${rounded >= 0 ? "+" : ""}${rounded})`;
}

function attachChartInteractions(main, diff, days) {
  const { svg: mainSvg, xScale, pad, innerWidth } = main;
  const tooltip = document.getElementById("history-tooltip");
  const wrapper = document.getElementById("history-charts");

  let dragStartIndex = null;
  let selectionRect = null;

  const mainCrosshair = svgEl("line", {
    x1: 0, x2: 0, y1: HISTORY_CHART_PADDING.top, y2: HISTORY_CHART_HEIGHT - HISTORY_CHART_PADDING.bottom,
    class: "history-crosshair", visibility: "hidden",
  });
  mainSvg.appendChild(mainCrosshair);
  const diffCrosshair = svgEl("line", {
    x1: 0, x2: 0, y1: HISTORY_DIFF_PADDING.top, y2: HISTORY_DIFF_HEIGHT - HISTORY_DIFF_PADDING.bottom,
    class: "history-crosshair", visibility: "hidden",
  });
  diff.svg.appendChild(diffCrosshair);

  const indexFromClientX = (clientX) => {
    const rect = mainSvg.getBoundingClientRect();
    const scale = HISTORY_CHART_WIDTH / rect.width;
    const svgX = (clientX - rect.left) * scale;
    const fraction = (svgX - pad.left) / innerWidth;
    const index = Math.round(fraction * (days.length - 1));
    return Math.min(days.length - 1, Math.max(0, index));
  };

  function positionTooltip(clientX, clientY) {
    const rect = wrapper.getBoundingClientRect();
    tooltip.style.left = `${clientX - rect.left + 14}px`;
    tooltip.style.top = `${clientY - rect.top + 14}px`;
  }

  function showTooltip(index, clientX, clientY) {
    const day = days[index];
    const x = xScale(index).toFixed(1);
    mainCrosshair.setAttribute("x1", x);
    mainCrosshair.setAttribute("x2", x);
    mainCrosshair.setAttribute("visibility", "visible");
    diffCrosshair.setAttribute("x1", x);
    diffCrosshair.setAttribute("x2", x);
    diffCrosshair.setAttribute("visibility", "visible");

    const ndfDiff = day.actual_demand_mw == null ? null : day.ndf_forecast_mw - day.actual_demand_mw;
    const blendDiff = day.actual_demand_mw == null ? null : day.blended_forecast_mw - day.actual_demand_mw;
    tooltip.innerHTML = `
      <div class="history-tooltip-date">${day.date}</div>
      <div><span class="history-swatch history-swatch-actual"></span>Actual: ${formatMw(day.actual_demand_mw)}</div>
      <div><span class="history-swatch history-swatch-ndf"></span>NDF: ${formatMw(day.ndf_forecast_mw)}${formatDiff(ndfDiff)}</div>
      <div><span class="history-swatch history-swatch-blended"></span>Blend: ${formatMw(day.blended_forecast_mw)}${formatDiff(blendDiff)}</div>
    `;
    tooltip.hidden = false;
    positionTooltip(clientX, clientY);
  }

  function hideTooltip() {
    tooltip.hidden = true;
    mainCrosshair.setAttribute("visibility", "hidden");
    diffCrosshair.setAttribute("visibility", "hidden");
  }

  function handleMove(event) {
    const index = indexFromClientX(event.clientX);
    if (dragStartIndex !== null) {
      const x1 = xScale(Math.min(dragStartIndex, index));
      const x2 = xScale(Math.max(dragStartIndex, index));
      selectionRect.setAttribute("x", x1.toFixed(1));
      selectionRect.setAttribute("width", Math.max(0, x2 - x1).toFixed(1));
      return;
    }
    showTooltip(index, event.clientX, event.clientY);
  }

  mainSvg.addEventListener("pointerdown", (event) => {
    dragStartIndex = indexFromClientX(event.clientX);
    selectionRect = svgEl("rect", {
      x: xScale(dragStartIndex).toFixed(1), y: HISTORY_CHART_PADDING.top,
      width: 0, height: HISTORY_CHART_HEIGHT - HISTORY_CHART_PADDING.top - HISTORY_CHART_PADDING.bottom,
      class: "history-selection",
    });
    mainSvg.appendChild(selectionRect);
    mainSvg.setPointerCapture(event.pointerId);
    hideTooltip();
  });

  const finishDrag = (event) => {
    if (dragStartIndex === null) return;
    const endIndex = indexFromClientX(event.clientX);
    const lo = Math.min(dragStartIndex, endIndex);
    const hi = Math.max(dragStartIndex, endIndex);
    dragStartIndex = null;
    selectionRect?.remove();
    if (hi - lo >= HISTORY_MIN_ZOOM_DAYS) setHistoryView(historyViewStart + lo, historyViewStart + hi);
  };

  mainSvg.addEventListener("pointermove", handleMove);
  diff.svg.addEventListener("pointermove", handleMove);
  mainSvg.addEventListener("pointerup", finishDrag);
  mainSvg.addEventListener("pointercancel", finishDrag);
  mainSvg.addEventListener("pointerleave", () => dragStartIndex === null && hideTooltip());
  diff.svg.addEventListener("pointerleave", () => dragStartIndex === null && hideTooltip());
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

  const chartContainer = document.getElementById("history-chart");
  const diffContainer = document.getElementById("history-diff-chart");
  chartContainer.innerHTML = "";
  diffContainer.innerHTML = "";
  document.getElementById("history-tooltip").hidden = true;

  const days = historyAllDays.slice(historyViewStart, historyViewEnd + 1);
  if (!days.length) {
    chartContainer.textContent = "No historical comparison data available in this deployment.";
    updateRangeButtons();
    return;
  }

  const main = buildMainChart(days);
  const diff = buildDiffChart(days);
  chartContainer.appendChild(main.svg);
  diffContainer.appendChild(diff.svg);
  attachChartInteractions(main, diff, days);

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
