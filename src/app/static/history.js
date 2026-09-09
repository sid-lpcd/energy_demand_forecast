const HISTORY_CHART_WIDTH = 800;
const HISTORY_CHART_HEIGHT = 320;
const HISTORY_CHART_PADDING = { top: 16, right: 16, bottom: 28, left: 56 };
const HISTORY_SERIES = [
  { key: "actual_demand_mw", cls: "history-line-actual" },
  { key: "ndf_forecast_mw", cls: "history-line-ndf" },
  { key: "blended_forecast_mw", cls: "history-line-blended" },
];

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
    "aria-label": "Actual demand vs NDF vs our blended forecast, trailing 12 months",
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

  container.appendChild(svg);
}

async function loadHistory() {
  const chart = document.getElementById("history-chart");
  const note = document.getElementById("history-note");
  try {
    const response = await fetch("/api/history");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    renderHistoryChart(data.days);
    note.textContent = data.days.length
      ? `${data.weather_upper_bound_note} ${data.scottish_transfer_advisory}`
      : "Historical comparison not available in this deployment.";
  } catch (err) {
    chart.textContent = `Couldn't load history: ${err.message}`;
  }
}

loadHistory();
