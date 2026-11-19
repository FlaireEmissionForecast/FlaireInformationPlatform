/*
 * Statistics dialog module.
 * The main page owns the Show statistics action and supplies the current
 * forecast/spot-price payloads. This module owns dialog markup, styling,
 * statistics calculations, controls and Plotly rendering.
 */

// Allow statistics and regression calculations to use available matched
// observations when some day-ahead spot prices have not yet been published.
const ALLOW_PARTIAL_SPOT_PRICE_COVERAGE = true;

const STATISTICS_STYLES = `
.statistics-modal {
  --stats-surface: #fff;
  --stats-text: #263238;
  --stats-muted: #66727c;
  --stats-border: #dce2e8;
  --stats-header: #f2f5f8;
  --stats-row-alt: #f8fafc;
  --stats-shadow: 0 20px 60px rgba(0, 0, 0, .24);
  position: fixed; inset: 0; z-index: 1000; display: flex;
  align-items: center; justify-content: center; padding: 8px;
  box-sizing: border-box; background: rgba(12, 20, 28, .58);
}
.statistics-modal[hidden] { display: none; }
body.dark-theme .statistics-modal {
  --stats-surface: #1c2229; --stats-text: #edf2f7; --stats-muted: #aab5c0;
  --stats-border: #3a4652; --stats-header: #252e38; --stats-row-alt: #202831;
  --stats-shadow: 0 20px 60px rgba(0, 0, 0, .55);
}
.statistics-dialog {
  width: min(1360px, 100%); max-height: min(94vh, 1160px); min-height: 0;
  display: flex; flex-direction: column; overflow: hidden;
  border: 2px solid var(--stats-border); border-radius: 14px;
  background: var(--stats-surface); color: var(--stats-text); box-shadow: var(--stats-shadow);
}
.statistics-scroll-region { flex: 1 1 auto; min-height: 0; overflow-y: auto; overscroll-behavior: contain; }
.statistics-header {
  display: flex; justify-content: space-between; align-items: flex-start; gap: 18px;
  padding: 18px 22px 12px; border-bottom: 1px solid var(--stats-border);
}
.statistics-header h2 { margin: 0; color: var(--stats-text); font-size: 21px; font-weight: 600; }
.statistics-subtitle { margin: 8px 0 0; color: var(--stats-muted); font-size: 13px; line-height: 1.45; }
.statistics-close {
  flex: 0 0 auto; width: 36px; height: 36px; border: 1px solid var(--stats-border);
  border-radius: 8px; background: transparent; color: var(--stats-text);
  font-size: 24px; line-height: 1; cursor: pointer;
}
.statistics-close:focus-visible, .statistics-metric-select:focus-visible,
.statistics-window-select:focus-visible, .statistics-view-button:focus-visible {
  outline: 3px solid #4a90e2; outline-offset: 2px;
}
.statistics-status { margin: 0; padding: 10px 22px; color: var(--stats-muted); font-size: 12px; line-height: 1.5; }
.statistics-status.is-warning { color: #a65a00; background: rgba(245, 158, 11, .09); }
body.dark-theme .statistics-status.is-warning { color: #ffc36b; }
.statistics-table-wrap {
  width: 100%; overflow: auto; border-top: 1px solid var(--stats-border);
  border-bottom: 1px solid var(--stats-border);
}
.statistics-table {
  width: 100%; min-width: 900px; border-collapse: separate; border-spacing: 0;
  color: var(--stats-text); font-size: 12px; font-variant-numeric: tabular-nums;
}
.statistics-table th, .statistics-table td {
  padding: 7px 7px; border-bottom: 1px solid var(--stats-border);
  text-align: center; white-space: nowrap;
}
.statistics-table thead th {
  position: sticky; top: 0; z-index: 1; background: var(--stats-header);
  color: var(--stats-muted); font-size: 10px; font-weight: 700;
  letter-spacing: .035em; text-transform: uppercase;
}
.statistics-table th:first-child, .statistics-table td:first-child {
  position: sticky; left: 0; text-align: left;
}
.statistics-table thead th:first-child { z-index: 2; background: var(--stats-header); }
.statistics-table tbody th { background: var(--stats-surface); font-weight: 600; }
.statistics-table tbody tr:nth-child(even) td,
.statistics-table tbody tr:nth-child(even) th { background: var(--stats-row-alt); }
.statistics-table tbody tr:last-child th, .statistics-table tbody tr:last-child td { border-bottom: 0; }
.statistics-error-header { min-width: 100px; }
.statistics-error-header-content { display: inline-flex; align-items: center; justify-content: center; gap: 5px; }
.statistics-metric-select, .statistics-window-select {
  max-width: 130px; padding: 4px 7px; border: 1px solid var(--stats-border);
  border-radius: 6px; background: var(--stats-surface); color: var(--stats-text);
  font: inherit; font-size: 11px; text-transform: none; letter-spacing: normal;
}
.statistics-charts {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px; padding: 14px 16px 4px;
}
.statistics-chart-card {
  min-width: 0; overflow: hidden; border: 1px solid var(--stats-border);
  border-radius: 10px; background: var(--stats-surface);
}
.statistics-chart-card h3 {
  margin: 0; padding: 12px 14px 0; color: var(--stats-text); font-size: 14px; font-weight: 600;
}
.statistics-chart-description {
  min-height: 30px; margin: 4px 14px 0; color: var(--stats-muted);
  font-size: 11px; line-height: 1.4;
}
.statistics-chart-toolbar {
  display: flex; flex-flow: row wrap; align-items: center; justify-content: space-between;
  gap: 8px; padding: 8px 10px 0; min-width: 0;
}
.statistics-view-selector {
  display: flex; flex: 1 1 max-content; align-items: center; gap: 5px;
  min-width: 0; max-width: 100%; overflow-x: auto; overscroll-behavior-x: contain;
  scrollbar-width: thin; scroll-snap-type: x proximity; padding: 2px 2px 5px;
}
.statistics-view-button {
  flex: 0 0 auto; padding: 5px 9px; border: 1px solid var(--stats-border);
  border-radius: 999px; background: transparent; color: var(--stats-text);
  font: inherit; font-size: 11px; cursor: pointer; white-space: nowrap; scroll-snap-align: start;
}
.statistics-view-button[aria-pressed="true"] { background: #4a90e2; border-color: #4a90e2; color: #fff; }
.statistics-window-control {
  display: flex; flex: 0 0 auto; align-items: center; gap: 6px; padding-left: 4px;
  color: var(--stats-muted); font-size: 11px; white-space: nowrap;
}
.statistics-window-control[hidden] { display: none; }
.statistics-window-select { margin: 0; padding: 4px 5px; }
.statistics-chart { width: 100%; height: 315px; min-width: 0; }
.statistics-chart-empty {
  display: flex; align-items: center; justify-content: center; min-height: 230px;
  margin: 0; padding: 18px; box-sizing: border-box; color: var(--stats-muted);
  font-size: 12px; line-height: 1.5; text-align: center;
}
.statistics-footnote { margin: 0; padding: 10px 22px 16px; color: var(--stats-muted); font-size: 11px; line-height: 1.55; }
@media (max-width: 700px) {
  .statistics-modal { padding: 4px; }
  .statistics-dialog { max-height: 96vh; border-radius: 10px; }
  .statistics-charts { grid-template-columns: minmax(0, 1fr); gap: 12px; padding: 10px 9px 2px; }
  .statistics-chart { height: 285px; }
  .statistics-header { padding: 14px 14px 10px; }
  .statistics-header h2 { font-size: 19px; }
  .statistics-status { padding: 9px 14px; }
  .statistics-footnote { padding: 10px 14px 14px; }
  .statistics-table th, .statistics-table td { padding: 5px 5px; }
}
@media (prefers-reduced-motion: reduce) {
  .statistics-view-button { transition: none; }
}
`;

const STATISTICS_MARKUP = `
<div id="statistics-modal" class="statistics-modal" hidden>
  <section class="statistics-dialog" role="dialog" aria-modal="true"
           aria-labelledby="statistics-title" aria-describedby="statistics-subtitle" tabindex="-1">
    <div class="statistics-header">
      <div>
        <h2 id="statistics-title">statistics</h2>
        <p id="statistics-subtitle" class="statistics-subtitle"></p>
      </div>
      <button type="button" id="statistics-close" class="statistics-close" aria-label="Close statistics">&times;</button>
    </div>
    <p id="statistics-status" class="statistics-status" role="status" aria-live="polite">Preparing statistics…</p>
    <div class="statistics-scroll-region">
      <div class="statistics-table-wrap">
        <table class="statistics-table" aria-label="Descriptive statistics, regression and selected error metric">
          <thead><tr>
            <th scope="col">Data set</th><th scope="col">Count</th>
            <th scope="col">Minimum</th><th scope="col">Maximum</th>
            <th scope="col">Average</th><th scope="col">Median</th>
            <th scope="col">Std. dev.</th><th scope="col">Pearson r</th>
            <th scope="col">OLS R²</th>
            <th scope="col" class="statistics-error-header">
              <span class="statistics-error-header-content">
                Error metric
                <select id="statistics-metric-select" class="statistics-metric-select" aria-label="Select error metric">
                  <option value="MAE" selected>MAE</option>
                  <option value="MAPE">MAPE</option>
                  <option value="RMSE">RMSE</option>
                </select>
              </span>
            </th>
          </tr></thead>
          <tbody id="statistics-table-body"></tbody>
        </table>
      </div>
      <div class="statistics-charts">
        <section class="statistics-chart-card" aria-labelledby="statistics-relationship-title">
          <h3 id="statistics-relationship-title">Relationship</h3>
          <div class="statistics-chart-toolbar">
            <nav id="statistics-relationship-selector" class="statistics-view-selector" aria-label="Relationship graph views">
              <button type="button" class="statistics-view-button" data-view="scatter" aria-pressed="true">Series vs spot</button>
              <button type="button" class="statistics-view-button" data-view="regression-stats">Error vs spot</button>
              <button type="button" class="statistics-view-button" data-view="residual-spot">Residuals vs spot</button>
              <button type="button" class="statistics-view-button" data-view="rolling">Rolling correlation</button>
            </nav>
              <label id="statistics-window-control"
                    class="statistics-window-control"
                    hidden>
                <span id="statistics-window-label">Window</span>
                <select id="statistics-window-select"
                        class="statistics-window-select"
                        aria-label="Rolling window or error metric"
                        data-spot-error-metric="MAE">
                  <option value="6">6</option>
                  <option value="12" selected>12</option>
                  <option value="24">24</option>
                </select>
              </label>
          </div>
          <p id="statistics-relationship-description" class="statistics-chart-description"></p>
          <div id="statistics-relationship-chart" class="statistics-chart" role="img" aria-label="Selected relationship graph"></div>
        </section>
        <section class="statistics-chart-card" aria-labelledby="statistics-errors-title">
          <h3 id="statistics-errors-title">Errors</h3>
          <div class="statistics-chart-toolbar">
            <nav id="statistics-errors-selector" class="statistics-view-selector" aria-label="Error graph views">
              <button type="button" class="statistics-view-button" data-view="residual-time" aria-pressed="true">Residuals over time</button>
              <button type="button" class="statistics-view-button" data-view="histogram" aria-pressed="false">Histogram</button>
              <button type="button" class="statistics-view-button" data-view="residual-fitted" aria-pressed="false">Residuals vs pred.</button>
              <button type="button" class="statistics-view-button" data-view="hourly" aria-pressed="false">Hourly profile</button>
            </nav>
          </div>
          <p id="statistics-errors-description" class="statistics-chart-description"></p>
          <div id="statistics-errors-chart" class="statistics-chart" role="img" aria-label="Selected residual graph"></div>
        </section>
      </div>
      <p id="statistics-footnote" class="statistics-footnote"></p>
    </div>
  </section>
</div>`;

function normalizeTimestampKey(timestamp) {
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? null : date.getTime();
}

function getNumericEntries(values) {
  return Object.entries(values || {})
    .filter(([timestamp, value]) => timestamp && value !== null && value !== undefined &&
      typeof value !== "boolean" && !(typeof value === "string" && value.trim() === ""))
    .map(([timestamp, value]) => ({
      timestamp,
      key: normalizeTimestampKey(timestamp),
      value: Number(value)
    }))
    .filter(entry => entry.key !== null && Number.isFinite(entry.value))
    .sort((a, b) => a.key - b.key);
}

function mergeEntriesByTimestamp(entries) {
  const byTimestamp = new Map();
  entries.forEach(entry => byTimestamp.set(entry.key, entry));
  return Array.from(byTimestamp.values()).sort((a, b) => a.key - b.key);
}

function hasContinuousTimestamps(entries, intervalMinutes = 60) {
  if (entries.length < 2) return entries.length === 1;
  const intervalMilliseconds = intervalMinutes * 60 * 1000;
  const sorted = [...entries].sort((a, b) => a.key - b.key);
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].key - sorted[i - 1].key !== intervalMilliseconds) return false;
  }
  return true;
}

function calculateDescriptiveStatistics(entries) {
  const values = entries.map(entry => entry.value).filter(Number.isFinite);
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const count = sorted.length;
  const average = sorted.reduce((sum, value) => sum + value, 0) / count;
  const middle = Math.floor(count / 2);
  const median = count % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  const standardDeviation = count > 1
    ? Math.sqrt(sorted.reduce((sum, value) => sum + (value - average) ** 2, 0) / (count - 1))
    : null;
  return { count, minimum: sorted[0], maximum: sorted[count - 1], average, median, standardDeviation };
}

function calculateRegressionStatistics(
  seriesEntries,
  spotByTimestamp,
  hasFullCoverage,
  allowPartialCoverage = false
) {
  if (
    (!hasFullCoverage && !allowPartialCoverage) ||
    seriesEntries.length < 2
  ) {
    return null;
  }

  const pairs = seriesEntries
    .filter(entry => spotByTimestamp.has(entry.key))
    .map(entry => ({
      key: entry.key,
      timestamp: entry.timestamp,
      x: spotByTimestamp.get(entry.key),
      y: entry.value
    }));

  if (
    pairs.length < 2 ||
    (!allowPartialCoverage && pairs.length !== seriesEntries.length)
  ) {
    return null;
  }

  const meanX = pairs.reduce((sum, pair) => sum + pair.x, 0) / pairs.length;
  const meanY = pairs.reduce((sum, pair) => sum + pair.y, 0) / pairs.length;
  const sumXX = pairs.reduce((sum, pair) => sum + (pair.x - meanX) ** 2, 0);
  const sumYY = pairs.reduce((sum, pair) => sum + (pair.y - meanY) ** 2, 0);
  const sumXY = pairs.reduce((sum, pair) => sum + (pair.x - meanX) * (pair.y - meanY), 0);

  if (sumXX <= 0 || sumYY <= 0) return null;

  const correlation = Math.max(-1, Math.min(1, sumXY / Math.sqrt(sumXX * sumYY)));
  const slope = sumXY / sumXX;
  const intercept = meanY - slope * meanX;

  const pairsWithResiduals = pairs.map(pair => {
    const fitted = intercept + slope * pair.x;
    return { ...pair, fitted, residual: pair.y - fitted };
  });

  const residualSumSquares = pairsWithResiduals.reduce(
    (sum, pair) => sum + pair.residual ** 2,
    0
  );

  const rSquared = Math.max(0, Math.min(1, 1 - residualSumSquares / sumYY));

  return {
    correlation,
    rSquared,
    count: pairsWithResiduals.length,
    slope,
    intercept,
    pairs: pairsWithResiduals
  };
}

function formatStatistic(value, digits = 2) {
  if (!Number.isFinite(value)) return "N/A";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
}

function formatCoefficient(value) {
  if (!Number.isFinite(value)) return "N/A";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 4, maximumSignificantDigits: 4 }).format(value);
}

function pearsonCorrelation(points) {
  if (points.length < 2) return null;
  const meanX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
  const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
  let sumXX = 0, sumYY = 0, sumXY = 0;
  points.forEach(point => {
    const dx = point.x - meanX;
    const dy = point.y - meanY;
    sumXX += dx * dx;
    sumYY += dy * dy;
    sumXY += dx * dy;
  });
  if (sumXX <= 0 || sumYY <= 0) return null;
  return Math.max(-1, Math.min(1, sumXY / Math.sqrt(sumXX * sumYY)));
}

function metricResult(regression, metric, unit) {
  const pairs = regression?.pairs || [];
  if (!pairs.length) return "N/A";
  const errors = pairs.map(pair => pair.residual).filter(Number.isFinite);
  if (!errors.length) return "N/A";
  if (metric === "MAE") {
    const value = errors.reduce((sum, error) => sum + Math.abs(error), 0) / errors.length;
    return `${formatStatistic(value)}${unit ? ` ${unit}` : ""}`;
  }
  if (metric === "RMSE") {
    const value = Math.sqrt(errors.reduce((sum, error) => sum + error * error, 0) / errors.length);
    return `${formatStatistic(value)}${unit ? ` ${unit}` : ""}`;
  }

  // Exclude zero and near-zero actual values to avoid unstable percentage errors.
  const maxAbsActual = Math.max(...pairs.map(pair => Math.abs(pair.y)));
  const nearZeroThreshold = Math.max(1e-9, maxAbsActual * 1e-6);
  const eligible = pairs.filter(pair => Math.abs(pair.y) > nearZeroThreshold);
  const excludedCount = pairs.length - eligible.length;
  if (!eligible.length) return `N/A (${excludedCount} excluded)`;
  const mape = eligible.reduce((sum, pair) => sum + Math.abs(pair.residual / pair.y), 0) / eligible.length * 100;
  return `${formatStatistic(mape)}% (${excludedCount} excluded)`;
}

function getHour(timestamp) {
  // API timestamps are local wall-clock strings; prefer their literal hour to
  // avoid shifting hourly buckets through the browser's timezone conversion.
  const match = String(timestamp).match(/T(\d{2}):/);
  if (match) return Number(match[1]);
  const parsed = new Date(timestamp);
  return Number.isNaN(parsed.getTime()) ? null : parsed.getHours();
}

export class StatisticsDialog {
  constructor({ getTheme = () => "light", getLangVer } = {}) {
    this.getTheme = getTheme;
    this.getLangVer = getLangVer;
    this.metric = "MAE";
    this.relationshipView = "scatter";
    this.errorView = "residual-time";
    this.rollingWindow = 12;
    this.model = null;
    this.returnFocusElement = null;
    this.relationshipRenderToken = 0;
    this.errorRenderToken = 0;

    if (!document.getElementById("statistics-dialog-styles")) {
      const style = document.createElement("style");
      style.id = "statistics-dialog-styles";
      style.textContent = STATISTICS_STYLES;
      document.head.appendChild(style);
    }
    document.body.insertAdjacentHTML("beforeend", STATISTICS_MARKUP);
    this.cacheElements();
    this.bindEvents();
    window.addEventListener("resize", () => this.resizeVisibleCharts(), { passive: true });
  }

  cacheElements() {
    const byId = id => document.getElementById(id);
    this.modal = byId("statistics-modal");
    this.dialog = this.modal.querySelector(".statistics-dialog");
    this.closeButton = byId("statistics-close");
    this.title = byId("statistics-title");
    this.subtitle = byId("statistics-subtitle");
    this.status = byId("statistics-status");
    this.tableBody = byId("statistics-table-body");
    this.metricSelect = byId("statistics-metric-select");
    this.footnote = byId("statistics-footnote");
    this.relationshipSelector = byId("statistics-relationship-selector");
    this.errorsSelector = byId("statistics-errors-selector");
    this.relationshipDescription = byId("statistics-relationship-description");
    this.errorsDescription = byId("statistics-errors-description");
    this.relationshipChart = byId("statistics-relationship-chart");
    this.errorsChart = byId("statistics-errors-chart");
    this.windowControl = byId("statistics-window-control");
    this.windowSelect = byId("statistics-window-select");
  }

  bindEvents() {
    this.closeButton.addEventListener("click", () => this.close());

    this.modal.addEventListener("click", event => {
      if (event.target === this.modal) this.close();
    });

    document.addEventListener("keydown", event => {
      if (event.key === "Escape" && !this.modal.hidden) this.close();
    });

    this.metricSelect.addEventListener("change", () => {
      this.metric = this.metricSelect.value;
      this.renderTable();
    });

    this.windowSelect.addEventListener("change", () => {
      if (this.relationshipView === "rolling") {
        const parsed = Number(this.windowSelect.value);

        if ([6, 12, 24, 48].includes(parsed)) {
          this.rollingWindow = parsed;
        }
      } else if (this.relationshipView === "regression-stats") {
        this.windowSelect.dataset.spotErrorMetric =
          this.windowSelect.value;
      }

      if (
        this.relationshipView === "rolling" ||
        this.relationshipView === "regression-stats"
      ) {
        this.renderRelationshipChart();
      }
    });

    this.relationshipSelector.addEventListener("click", event => {
      const button = event.target.closest("button[data-view]");

      if (!button || !this.relationshipSelector.contains(button)) return;

      this.relationshipView = button.dataset.view;
      this.updateSelectorState();
      this.renderRelationshipChart();
    });

    this.errorsSelector.addEventListener("click", event => {
      const button = event.target.closest("button[data-view]");

      if (!button || !this.errorsSelector.contains(button)) return;

      this.errorView = button.dataset.view;
      this.updateSelectorState();
      this.renderErrorsChart();
    });
  }

  open() {
    this.returnFocusElement = document.activeElement;
    this.modal.hidden = false;
    this.closeButton.focus();
    requestAnimationFrame(() => this.resizeVisibleCharts());
  }

  close() {
    this.modal.hidden = true;
    if (this.returnFocusElement?.focus) this.returnFocusElement.focus();
  }

  setLoading(message) {
    this.model = null;
    this.subtitle.textContent = "Preparing the current data set…";
    this.tableBody.replaceChildren();
    this.footnote.textContent = "";
    this.setStatus(message);
    this.clearCharts("Charts will appear when the regression data is ready.");
  }

  setStatus(message, isWarning = false) {
    this.status.textContent = message;
    this.status.classList.toggle("is-warning", isWarning);
  }

  clearCharts(message) {
    this.setChartEmpty(this.relationshipChart, message);
    this.setChartEmpty(this.errorsChart, message);
  }

  setChartEmpty(container, message) {
    if (window.Plotly) {
      try { window.Plotly.purge(container); } catch (error) {
        console.warn("Could not clear statistics chart:", error);
      }
    }
    const placeholder = document.createElement("p");
    placeholder.className = "statistics-chart-empty";
    placeholder.textContent = message;
    container.replaceChildren(placeholder);
  }

  render(forecastPayload, spotPayload, { seriesKey = "", theme } = {}) {
    const data = forecastPayload?.data || {};

    const historyEntries = getNumericEntries(data.history);
    const forecastEntries = getNumericEntries(data.forecast);

    const combinedEntries = mergeEntriesByTimestamp([
      ...historyEntries,
      ...forecastEntries
    ]);

    const hasContinuousSeries = hasContinuousTimestamps(
      combinedEntries,
      60
    );

    const spotData = spotPayload?.data || {};

    const spotHistoryEntries = getNumericEntries(spotData.history);
    const spotForecastEntries = getNumericEntries(spotData.forecast);

    const spotEntries = mergeEntriesByTimestamp([
      ...spotHistoryEntries,
      ...spotForecastEntries
    ]);

    const spotByTimestamp = new Map(
      spotEntries.map(entry => [entry.key, entry.value])
    );

    // Count series timestamps for which a spot price is available.
    const matchedCount = combinedEntries.filter(
      entry => spotByTimestamp.has(entry.key)
    ).length;

    // Count series timestamps for which a spot price is missing.
    const missingSpotCount = Math.max(
      0,
      combinedEntries.length - matchedCount
    );

    // Full coverage still requires complete spot-price coverage
    // and continuous hourly series timestamps.
    const hasFullCoverage =
      combinedEntries.length > 0 &&
      matchedCount === combinedEntries.length &&
      hasContinuousSeries;

    // When enabled, regression calculations may use the available
    // matched observations instead of requiring complete spot coverage.
    // The existing continuous-hourly-series requirement is retained.
    const allowPartialRegression =
      ALLOW_PARTIAL_SPOT_PRICE_COVERAGE &&
      hasContinuousSeries;

    // Calculate regression statistics using the available matched
    // observations when partial coverage is enabled.
    const historyRegression = calculateRegressionStatistics(
      historyEntries,
      spotByTimestamp,
      hasFullCoverage,
      allowPartialRegression
    );

    const forecastRegression = calculateRegressionStatistics(
      forecastEntries,
      spotByTimestamp,
      hasFullCoverage,
      allowPartialRegression
    );

    const combinedRegression = calculateRegressionStatistics(
      combinedEntries,
      spotByTimestamp,
      hasFullCoverage,
      allowPartialRegression
    );

    const seriesName =
      this.getLangVer(forecastPayload?.metadata?.name) ||
      seriesKey ||
      "Current series";

    const unit =
      this.getLangVer(forecastPayload?.metadata?.unit) || "";

    const timestamps = combinedEntries.map(entry => entry.timestamp);

    const dateRange = timestamps.length
      ? `${timestamps[0].replace("T", " ")} – ${timestamps[timestamps.length - 1].replace("T", " ")}`
      : "Date range unavailable";

    // Store the complete data model for table and graph rendering.
    this.model = {
      historyEntries,
      forecastEntries,
      combinedEntries,
      spotEntries,
      historyRegression,
      forecastRegression,
      combinedRegression,
      hasFullCoverage,
      hasContinuousSeries,
      matchedCount,
      missingSpotCount,
      seriesCount: combinedEntries.length,
      unit,
      forecastKeys: new Set(
        forecastEntries.map(entry => entry.key)
      )
    };

    this.title.textContent = `${seriesName} statistics`;
    this.subtitle.textContent = `Date range: ${dateRange}`;

    // Render the statistics table and all available graphs.
    this.renderTable();
    this.renderCurrentCharts();

    // ---------------------------------------------------------
    // Status and explanatory text
    // ---------------------------------------------------------

    if (hasFullCoverage) {
      this.setStatus(
        `Spot-price coverage complete: ${matchedCount}/${combinedEntries.length} timestamps matched with ${seriesName.toLowerCase()} data. Regression and residual metrics use these aligned observations.`
      );

      this.footnote.textContent =
        "OLS uses series value as the dependent variable and spot price (€/MWh) as the independent variable. MAE and RMSE use the regression residuals in the series unit. MAPE excludes actual values at or below the scale-relative near-zero threshold (1 × 10⁻⁶ of the maximum absolute actual value, with a minimum threshold of 1 × 10⁻⁹); excluded observations are shown in the metric column. Standard deviation is the sample standard deviation (n − 1). Results are descriptive and do not imply causation.";

    } else if (
      missingSpotCount > 0 &&
      allowPartialRegression
    ) {
      const missingPriceMessage = missingSpotCount === 1
        ? "One day-ahead spot price is missing because it has not yet been published by ENTSO-E."
        : `Some (${missingSpotCount}/${forecastEntries.length}) day-ahead spot prices are missing because they have not yet been published by ENTSO-E.`;

      this.setStatus(
        `Calculations were completed using the currently available data. ${missingPriceMessage} Day-ahead spot prices can be queried after 14:30 EET/EEST from the FLIP API. Regression statistics and regression-based graphs use the available matched observations, provided there are enough valid observations to calculate them.`,
        true
      );

      this.footnote.textContent =
        "OLS and regression-based graphs use only timestamps for which both a series value and a spot price are available. MAE and RMSE use the resulting regression residuals in the series unit. MAPE excludes actual values at or below the scale-relative near-zero threshold (1 × 10⁻⁶ of the maximum absolute actual value, with a minimum threshold of 1 × 10⁻⁹); excluded observations are shown in the metric column. Standard deviation is the sample standard deviation (n − 1). Results are descriptive and do not imply causation.";

    } else {
      const coverageText = combinedEntries.length
        ? `${matchedCount} of ${combinedEntries.length} series timestamps matched`
        : "no valid series timestamps were found";

      const missingPriceText = missingSpotCount > 0
        ? ` Some (${missingSpotCount}/${forecastEntries.length}) day-ahead spot price${missingSpotCount === 1 ? " is" : "s are"} missing because ${missingSpotCount === 1 ? "it has" : "they have"} not yet been published by ENTSO-E. They are usually available after 14:30 (local time).`
        : "";

      const continuityText = hasContinuousSeries
        ? ""
        : " The series contains missing or irregular time intervals.";

      this.setStatus(
        `Regression was skipped. Spot-price coverage: ${coverageText}.${missingPriceText}${continuityText} Descriptive statistics remain available.`,
        true
      );

      this.footnote.textContent =
        "Complete spot-price coverage and continuous hourly series timestamps are required for regression. Error metrics are calculated from valid fitted regression observations only. Spot-price descriptive statistics use all valid spot observations returned for the displayed date range. Standard deviation is the sample standard deviation (n − 1).";
    }
  }

  renderTable() {
    if (!this.model) return;
    const model = this.model;
    const rows = [
      ["History", model.historyEntries, model.historyRegression],
      ["Forecast", model.forecastEntries, model.forecastRegression],
      ["Combined series", model.combinedEntries, model.combinedRegression],
      ["Spot prices (available)", model.spotEntries, null]
    ];
    const fragment = document.createDocumentFragment();
    rows.forEach(([label, entries, regression]) => {
      const stats = calculateDescriptiveStatistics(entries);
      const values = [
        label,
        stats ? String(stats.count) : "0",
        stats ? formatStatistic(stats.minimum) : "N/A",
        stats ? formatStatistic(stats.maximum) : "N/A",
        stats ? formatStatistic(stats.average) : "N/A",
        stats ? formatStatistic(stats.median) : "N/A",
        stats ? formatStatistic(stats.standardDeviation) : "N/A",
        label === "Spot prices (available)" ? "—" : (regression ? formatStatistic(regression.correlation) : "N/A"),
        label === "Spot prices (available)" ? "—" : (regression ? formatStatistic(regression.rSquared) : "N/A"),
        label === "Spot prices (available)" ? "—" : metricResult(regression, this.metric, model.unit)
      ];
      const row = document.createElement("tr");
      values.forEach((value, index) => {
        const cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.scope = "row";
        cell.textContent = value;
        row.appendChild(cell);
      });
      fragment.appendChild(row);
    });
    this.tableBody.replaceChildren(fragment);
  }

  updateSelectorState() {
    const update = (selector, selected) => {
      selector.querySelectorAll("button[data-view]").forEach(button => {
        button.setAttribute(
          "aria-pressed",
          String(button.dataset.view === selected)
        );
      });
    };

    update(this.relationshipSelector, this.relationshipView);
    update(this.errorsSelector, this.errorView);

    const isRolling = this.relationshipView === "rolling";
    const isSpotError = this.relationshipView === "regression-stats";

    this.windowControl.hidden = !isRolling && !isSpotError;

    const label = this.windowControl.querySelector(
      "#statistics-window-label"
    );

    const setOptions = (values, selectedValue) => {
      const options = values.map(value => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        return option;
      });

      this.windowSelect.replaceChildren(...options);
      this.windowSelect.value = String(selectedValue);
    };

    if (isRolling) {
      label.textContent = "Window";

      setOptions(
        ["6", "12", "24", "48"],
        this.rollingWindow
      );
    } else if (isSpotError) {
      label.textContent = "Error metric";

      const availableMetrics = ["MAE", "MAPE", "RMSE"];

      let selectedMetric =
        this.windowSelect.dataset.spotErrorMetric || "MAE";

      if (!availableMetrics.includes(selectedMetric)) {
        selectedMetric = "MAE";
      }

      this.windowSelect.dataset.spotErrorMetric = selectedMetric;

      setOptions(
        availableMetrics,
        selectedMetric
      );
    }
  }

  renderCurrentCharts() {
    this.updateSelectorState();
    this.renderRelationshipChart();
    this.renderErrorsChart();
  }

  getChartTheme() {
    const isDark = (this.getTheme?.() || "light") === "dark";

    return {
      isDark,
      template: isDark ? "plotly_dark" : "plotly_white",
      grid: isDark ? "#444b54" : "#e5e7eb",
      text: isDark ? "#edf2f7" : "#263238",
      muted: isDark ? "#aab5c0" : "#66727c",
      fit: isDark ? "#f2f2f2" : "#263238",
      history: "rgb(74, 144, 226)",
      forecast: "rgb(255, 140, 0)"
    };
  }

  commonLayout(theme, extra = {}) {
    return {
      template: theme.template,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      autosize: true,
      margin: { l: 58, r: 18, t: 42, b: 58 },
      font: { size: 11, color: theme.text },
      legend: { orientation: "h", x: 0.5, xanchor: "center", y: 1.02, yanchor: "bottom" },
      hovermode: "closest",
      ...extra
    };
  }

  renderRelationshipChart() {
    const token = ++this.relationshipRenderToken;
    const model = this.model;
    const theme = this.getChartTheme();

    const zeroShape = (axis = "y") => axis === "x"
      ? { type: "line", xref: "x", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1, line: { color: theme.fit, width: 1.5, dash: "dash" } }
      : { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: theme.fit, width: 1.5, dash: "dash" } };

    if (!window.Plotly) {
      this.relationshipDescription.textContent = "";
      this.setChartEmpty(
        this.relationshipChart,
        "Plotly is not available. Reload the page and try again."
      );
      return;
    }

    if (!model?.combinedRegression?.pairs?.length) {
      this.relationshipDescription.textContent = "";
      this.setChartEmpty(
        this.relationshipChart,
        "A complete set of matched, non-constant series and spot-price observations is required for regression graphs."
      );
      return;
    }

    const regression = model.combinedRegression;
    const pairs = regression.pairs;

    const historyPairs = pairs.filter(
      pair => !model.forecastKeys.has(pair.key)
    );

    const forecastPairs = pairs.filter(
      pair => model.forecastKeys.has(pair.key)
    );

    const unit = model.unit || "Series value";

    let traces = [];
    let layout = this.commonLayout(theme);

    // ---------------------------------------------------------
    // 1. Series vs spot
    // ---------------------------------------------------------
    if (this.relationshipView === "scatter") {
      this.relationshipDescription.textContent =
        "Series values against matched spot prices, with the combined-series least-squares fit.";

      if (historyPairs.length) {
        traces.push(
          this.relationshipScatterTrace(
            historyPairs,
            "History",
            theme.history,
            unit
          )
        );
      }

      if (forecastPairs.length) {
        traces.push(
          this.relationshipScatterTrace(
            forecastPairs,
            "Forecast",
            theme.forecast,
            unit
          )
        );
      }

      const minSpot = Math.min(...pairs.map(pair => pair.x));
      const maxSpot = Math.max(...pairs.map(pair => pair.x));

      traces.push({
        type: "scatter",
        mode: "lines",
        name: "OLS best fit",
        x: [minSpot, maxSpot],
        y: [
          regression.intercept + regression.slope * minSpot,
          regression.intercept + regression.slope * maxSpot
        ],
        line: {
          color: theme.fit,
          width: 2.5,
          dash: "dash"
        },
        hovertemplate:
          `Fitted series: %{y:.3f} ${unit}<extra>OLS fit</extra>`
      });

      const sign = regression.intercept >= 0 ? "+" : "−";

      const annotation =
        `Series = ${formatCoefficient(regression.slope)} × spot ${sign} ${formatCoefficient(Math.abs(regression.intercept))}<br>` +
        `R² = ${formatStatistic(regression.rSquared)} · Pearson r = ${formatStatistic(regression.correlation)}`;

      layout = this.commonLayout(theme, {
        xaxis: {
          title: { text: "Spot price (€/MWh)" },
          gridcolor: theme.grid,
          tickfont: { color: theme.muted }
        },
        yaxis: {
          title: { text: unit },
          gridcolor: theme.grid,
          tickfont: { color: theme.muted }
        },
        annotations: [{
          xref: "paper",
          yref: "paper",
          x: 0.98,
          y: 0.02,
          xanchor: "right",
          yanchor: "bottom",
          showarrow: false,
          text: annotation,
          align: "left",
          bgcolor: theme.isDark
            ? "rgba(28,34,41,.86)"
            : "rgba(255,255,255,.86)",
          bordercolor: theme.isDark ? "#56616d" : "#dce2e8",
          borderwidth: 1,
          borderpad: 4,
          font: {
            size: 10,
            color: theme.text
          }
        }]
      });

    // ---------------------------------------------------------
    // 2. Rolling correlation
    // ---------------------------------------------------------
    } else if (this.relationshipView === "rolling") {
      this.relationshipDescription.textContent =
        `Rolling Pearson correlation using ${this.rollingWindow} consecutive matched observations.`;

      if (pairs.length < this.rollingWindow) {
        this.setChartEmpty(
          this.relationshipChart,
          `At least ${this.rollingWindow} matched observations are required for this rolling window; ${pairs.length} are available.`
        );
        return;
      }

      const rolling = [];

      for (
        let i = this.rollingWindow - 1;
        i < pairs.length;
        i++
      ) {
        const slice = pairs.slice(
          i - this.rollingWindow + 1,
          i + 1
        );

        rolling.push({
          timestamp: pairs[i].timestamp,
          value: pearsonCorrelation(slice)
        });
      }

      traces = [{
        type: "scatter",
        mode: "lines+markers",
        name: `Rolling Pearson r (${this.rollingWindow})`,
        x: rolling.map(item => item.timestamp),
        y: rolling.map(item => item.value),
        connectgaps: false,
        line: {
          color: theme.history,
          width: 2
        },
        marker: {
          color: theme.history,
          size: 4
        },
        hovertemplate:
          "Time: %{x}<br>Pearson r: %{y:.3f}<extra></extra>"
      }];

      layout = this.commonLayout(theme, {
        xaxis: {
          type: "date",
          title: { text: "Time" },
          tickformat: "%d.%m.%y",
          tickfont: { color: theme.muted },
          gridcolor: theme.grid
        },
        yaxis: {
          title: { text: "Pearson correlation" },
          range: [-1, 1],
          tickmode: "linear",
          dtick: 0.5,
          tickfont: { color: theme.muted },
          gridcolor: theme.grid
        },
        shapes: [{
          type: "line",
          xref: "paper",
          x0: 0,
          x1: 1,
          yref: "y",
          y0: 0,
          y1: 0,
          line: {
            color: theme.fit,
            width: 1.4,
            dash: "dash"
          }
        }]
      });

    // ---------------------------------------------------------
    // Residuals vs. spot price
    // ---------------------------------------------------------
    } else if (this.relationshipView === "residual-spot") {

      const validPairs = pairs.filter(
        pair =>
          Number.isFinite(pair.x) &&
          Number.isFinite(pair.residual)
      );

      if (!validPairs.length) {
        this.setChartEmpty(
          this.relationshipChart,
          "No valid regression residuals are available."
        );
        return;
      }

      this.relationshipDescription.textContent =
        `Regression residuals against spot price; ${validPairs.length} matched observations. ` +
        "Positive residuals indicate underestimation; negative residuals indicate overestimation.";

      const validHistoryPairs = historyPairs.filter(
        pair =>
          Number.isFinite(pair.x) &&
          Number.isFinite(pair.residual)
      );

      const validForecastPairs = forecastPairs.filter(
        pair =>
          Number.isFinite(pair.x) &&
          Number.isFinite(pair.residual)
      );

      // ---------------------------------------------------------
      // 1. Individual observation traces.
      // ---------------------------------------------------------
      const createResidualTrace = (group, name, color) => ({
        type: "scatter",
        mode: "markers",
        name,
        x: group.map(pair => pair.x),
        y: group.map(pair => pair.residual),

        customdata: group.map(pair => [
          pair.timestamp,
          pair.y,
          pair.fitted
        ]),

        marker: {
          color,
          size: 7,
          opacity: 0.33
        },

        hovertemplate:
          `Spot price: %{x:.2f} €/MWh<br>` +
          `Residual: %{y:.3f} ${unit}<br>` +
          `Time: %{customdata[0]}<extra>${name}</extra>`
      });

      if (validHistoryPairs.length) {
        traces.push(
          createResidualTrace(
            validHistoryPairs,
            "History",
            theme.history
          )
        );
      }

      if (validForecastPairs.length) {
        traces.push(
          createResidualTrace(
            validForecastPairs,
            "Forecast",
            theme.forecast
          )
        );
      }

      // ---------------------------------------------------------
      // 2. Moving-median smoothing.
      //
      // Sort observations by spot price and calculate a
      // local median over neighbouring observations.
      // ---------------------------------------------------------
      const calculateMovingMedian = observations => {
        const sorted = [...observations].sort(
          (a, b) => a.x - b.x
        );

        if (sorted.length < 5) {
          return [];
        }

        // Use approximately 20% of observations,
        // with a minimum of 5 and maximum of 31.
        let windowSize = Math.min(
          31,
          Math.max(5, Math.ceil(sorted.length * 0.20))
        );

        // Prefer an odd-sized window.
        if (windowSize % 2 === 0) {
          windowSize += 1;
        }

        windowSize = Math.min(
          windowSize,
          sorted.length
        );

        const halfWindow = Math.floor(windowSize / 2);

        return sorted.map((pair, index) => {
          const start = Math.max(
            0,
            index - halfWindow
          );

          const end = Math.min(
            sorted.length,
            index + halfWindow + 1
          );

          const residuals = sorted
            .slice(start, end)
            .map(item => item.residual)
            .sort((a, b) => a - b);

          const middle = Math.floor(
            residuals.length / 2
          );

          const median = residuals.length % 2 === 0
            ? (
                residuals[middle - 1] +
                residuals[middle]
              ) / 2
            : residuals[middle];

          return {
            x: pair.x,
            residual: median,
            count: residuals.length
          };
        });
      };

      // ---------------------------------------------------------
      // 3. Add separate moving-median lines.
      // ---------------------------------------------------------
      const createTrendTrace = (group, name, color) => {
        const trend = calculateMovingMedian(group);

        if (!trend.length) {
          return;
        }

        traces.push({
          type: "scatter",
          mode: "lines",
          name: `${name} moving median`,
          x: trend.map(point => point.x),
          y: trend.map(point => point.residual),

          customdata: trend.map(point => point.count),

          line: {
            color,
            width: 2.8,
            shape: "linear"
          },

          showlegend: false,

          hovertemplate:
            `Spot price: %{x:.2f} €/MWh<br>` +
            `Moving median residual: %{y:.3f} ${unit}<br>` +
            `Observations in window: %{customdata}` +
            `<extra>${name} trend</extra>`
        });
      };

      createTrendTrace(
        validHistoryPairs,
        "History",
        theme.history
      );

      createTrendTrace(
        validForecastPairs,
        "Forecast",
        theme.forecast
      );

      // ---------------------------------------------------------
      // 4. Axis ranges.
      // ---------------------------------------------------------
      const minSpot = Math.min(
        ...validPairs.map(pair => pair.x)
      );

      const maxSpot = Math.max(
        ...validPairs.map(pair => pair.x)
      );

      const spotPadding =
        (maxSpot - minSpot) * 0.03 ||
        Math.max(Math.abs(minSpot) * 0.03, 0.5);

      const maxAbsResidual = Math.max(
        ...validPairs.map(pair => Math.abs(pair.residual))
      );

      const residualPadding =
        maxAbsResidual * 0.08 || 1;

      const xRange = [
        minSpot - spotPadding,
        maxSpot + spotPadding
      ];

      // Keep zero centred to make positive and negative
      // residuals visually comparable.
      const yRange = [
        -maxAbsResidual - residualPadding,
        maxAbsResidual + residualPadding
      ];

      // ---------------------------------------------------------
      // 5. Chart layout.
      // ---------------------------------------------------------
      layout = this.commonLayout(theme, {
        hovermode: "closest",

        xaxis: {
          title: {
            text: "Spot price (€/MWh)"
          },
          range: xRange,
          gridcolor: theme.grid,
          tickfont: { color: theme.muted },
          zeroline: false
        },

        yaxis: {
          title: {
            text: `Residual (${unit})`
          },
          range: yRange,
          gridcolor: theme.grid,
          tickfont: { color: theme.muted },
          zeroline: false
        },

        shapes: [zeroShape("y")],

        annotations: [{
          xref: "paper",
          yref: "paper",
          x: 0.02,
          y: 0.98,
          xanchor: "left",
          yanchor: "top",
          showarrow: false,
          text:
            "Above zero: underestimation<br>" +
            "Below zero: overestimation<br>" +
            "Lines: moving median",
          align: "left",
          font: {
            size: 10,
            color: theme.muted
          }
        }]
      });

    // ---------------------------------------------------------
    // 4. Error metric by spot-price range
    // ---------------------------------------------------------
    } else {
      const availableMetrics = ["MAE", "MAPE", "RMSE"];

      let metric =
        this.windowSelect.dataset.spotErrorMetric || "MAE";

      if (!availableMetrics.includes(metric)) {
        metric = "MAE";
      }

      this.relationshipDescription.textContent =
        `${metric} by spot-price range, shown as separate lines for history and forecast. ` +
        "Each point represents the error metric within a spot-price band.";

      if (!pairs.length) {
        this.setChartEmpty(
          this.relationshipChart,
          "No matched observations are available for the selected period."
        );
        return;
      }

      // Determine spot-price range and number of bands.
      const minSpot = Math.min(...pairs.map(pair => pair.x));
      const maxSpot = Math.max(...pairs.map(pair => pair.x));

      const binCount = minSpot === maxSpot
        ? 1
        : Math.min(
            8,
            Math.max(1, Math.ceil(Math.sqrt(pairs.length / 2)))
          );

      const binWidth = binCount > 1
        ? (maxSpot - minSpot) / binCount
        : 0;

      // Use the same near-zero threshold as the statistics table
      // when calculating MAPE.
      const maxAbsActual = Math.max(
        ...pairs.map(pair => Math.abs(pair.y))
      );

      const nearZeroThreshold = Math.max(
        1e-9,
        maxAbsActual * 1e-6
      );

      // Create the spot-price bands.
      const bins = Array.from(
        { length: binCount },
        (_, index) => {
          const start = binCount === 1
            ? minSpot
            : minSpot + index * binWidth;

          const end = index === binCount - 1
            ? maxSpot
            : start + binWidth;

          return {
            start,
            end,
            history: [],
            forecast: []
          };
        }
      );

      // Assign observations to their spot-price bands.
      pairs.forEach(pair => {
        let index = binCount === 1
          ? 0
          : Math.floor((pair.x - minSpot) / binWidth);

        index = Math.min(
          binCount - 1,
          Math.max(0, index)
        );

        const bin = bins[index];

        if (model.forecastKeys.has(pair.key)) {
          bin.forecast.push(pair);
        } else {
          bin.history.push(pair);
        }
      });

      // ---------------------------------------------------------
      // Merge bins containing fewer than N observations.
      // ---------------------------------------------------------
      const MIN_BIN_SAMPLES = 3;

      const getBinCount = bin =>
        bin.history.length + bin.forecast.length;

      while (bins.length > 1) {
        const sparseIndex = bins.findIndex(
          bin => getBinCount(bin) < MIN_BIN_SAMPLES
        );

        if (sparseIndex === -1) {
          break;
        }

        let neighborIndex;

        if (sparseIndex === 0) {
          neighborIndex = 1;
        } else if (sparseIndex === bins.length - 1) {
          neighborIndex = sparseIndex - 1;
        } else {
          const leftCount = getBinCount(bins[sparseIndex - 1]);
          const rightCount = getBinCount(bins[sparseIndex + 1]);

          neighborIndex = leftCount <= rightCount
            ? sparseIndex - 1
            : sparseIndex + 1;
        }

        const leftIndex = Math.min(sparseIndex, neighborIndex);

        const leftBin = bins[leftIndex];
        const rightBin = bins[leftIndex + 1];

        const mergedBin = {
          start: leftBin.start,
          end: rightBin.end,
          history: [
            ...leftBin.history,
            ...rightBin.history
          ],
          forecast: [
            ...leftBin.forecast,
            ...rightBin.forecast
          ]
        };

        bins.splice(leftIndex, 2, mergedBin);
      }

      // Calculate the selected error metric for a group.
      const calculateBinMetric = observations => {
        const valid = observations.filter(
          pair =>
            Number.isFinite(pair.residual) &&
            Number.isFinite(pair.y)
        );

        const count = valid.length;

        if (count === 0) {
          return {
            value: null,
            count: 0,
            includedCount: 0,
            excludedCount: 0
          };
        }

        const errors = valid.map(pair => pair.residual);

        if (metric === "MAE") {
          const value = errors.reduce(
            (sum, error) => sum + Math.abs(error),
            0
          ) / count;

          return {
            value,
            count,
            includedCount: count,
            excludedCount: 0
          };
        }

        if (metric === "RMSE") {
          const value = Math.sqrt(
            errors.reduce(
              (sum, error) => sum + error * error,
              0
            ) / count
          );

          return {
            value,
            count,
            includedCount: count,
            excludedCount: 0
          };
        }

        // MAPE: exclude zero and near-zero actual values.
        const eligible = valid.filter(
          pair => Math.abs(pair.y) > nearZeroThreshold
        );

        const excludedCount = count - eligible.length;

        if (!eligible.length) {
          return {
            value: null,
            count,
            includedCount: 0,
            excludedCount
          };
        }

        const value = eligible.reduce(
          (sum, pair) =>
            sum + Math.abs(pair.residual / pair.y),
          0
        ) / eligible.length * 100;

        return {
          value,
          count,
          includedCount: eligible.length,
          excludedCount
        };
      };

      // MAPE cannot be calculated if every actual value is
      // zero or below the near-zero exclusion threshold.
      if (
        metric === "MAPE" &&
        !pairs.some(
          pair => Math.abs(pair.y) > nearZeroThreshold
        )
      ) {
        this.setChartEmpty(
          this.relationshipChart,
          "MAPE is unavailable because all actual values are zero or near zero."
        );
        return;
      }

      // Position each metric at the midpoint of its spot-price band.
      const binCenters = bins.map(bin =>
        (bin.start + bin.end) / 2
      );

      const metricHover = metric === "MAPE"
        ? "MAPE: %{y:.2f}%"
        : `${metric}: %{y:.3f} ${unit}`;

      const countHover = metric === "MAPE"
        ? "<br>Observations: %{customdata[2]}" +
          "<br>Included in MAPE: %{customdata[3]}" +
          "<br>Excluded near-zero actuals: %{customdata[4]}"
        : "<br>Observations: %{customdata[2]}";

      // Build a line trace for History or Forecast.
      const createMetricTrace = (group, name, color) => {
        const results = bins.map(bin =>
          calculateBinMetric(bin[group])
        );

        return {
          type: "scatter",
          mode: "lines+markers",
          name,
          x: binCenters,
          y: results.map(result => result.value),

          customdata: results.map((result, index) => [
            bins[index].start,
            bins[index].end,
            result.count,
            result.includedCount,
            result.excludedCount
          ]),

          connectgaps: false,

          line: {
            color,
            width: 2.5,
            shape: "linear"
          },

          marker: {
            color,
            size: 8,
            symbol: "circle"
          },

          hovertemplate:
            `Spot-price range: %{customdata[0]:.1f} – %{customdata[1]:.1f} €/MWh<br>` +
            `${metricHover}` +
            `${countHover}` +
            `<extra>${name}</extra>`
        };
      };

      if (historyPairs.length) {
        traces.push(
          createMetricTrace(
            "history",
            "History",
            theme.history
          )
        );
      }

      if (forecastPairs.length) {
        traces.push(
          createMetricTrace(
            "forecast",
            "Forecast",
            theme.forecast
          )
        );
      }

      const metricAxisTitle = metric === "MAPE"
        ? "MAPE (%)"
        : `${metric} (${unit})`;

      layout = this.commonLayout(theme, {
        hovermode: "x",
        xaxis: {
          title: {
            text: "Spot price (€/MWh)"
          },
          gridcolor: theme.grid,
          tickfont: { color: theme.muted },
          zeroline: false,
          automargin: true
        },

        yaxis: {
          title: {
            text: metricAxisTitle
          },
          gridcolor: theme.grid,
          tickfont: { color: theme.muted },
          rangemode: "tozero",
          zeroline: true
        },

        showlegend: true
      });
    }

    this.drawPlot(
      this.relationshipChart,
      traces,
      layout,
      token,
      "relationship"
    );
  }

  relationshipScatterTrace(pairs, name, color, unit) {
    return {
      type: "scatter", mode: "markers", name,
      x: pairs.map(pair => pair.x), y: pairs.map(pair => pair.y),
      customdata: pairs.map(pair => pair.timestamp),
      marker: { color, size: 7, opacity: 0.84 },
      hovertemplate: `Spot price: %{x:.2f} €/MWh<br>Series: %{y:.3f} ${unit}<br>Time: %{customdata}<extra>${name}</extra>`
    };
  }

  actualPredictedTrace(pairs, name, color, unit) {
    return {
      type: "scatter", mode: "markers", name,
      x: pairs.map(pair => pair.fitted), y: pairs.map(pair => pair.y),
      customdata: pairs.map(pair => [pair.timestamp, pair.residual]),
      marker: { color, size: 7, opacity: 0.84 },
      hovertemplate: `Predicted: %{x:.3f} ${unit}<br>Actual: %{y:.3f} ${unit}<br>Residual: %{customdata[1]:.3f} ${unit}<br>Time: %{customdata[0]}<extra>${name}</extra>`
    };
  }

  renderErrorsChart() {
    const token = ++this.errorRenderToken;
    const model = this.model;
    const theme = this.getChartTheme();
    if (!window.Plotly) {
      this.errorsDescription.textContent = "";
      this.setChartEmpty(this.errorsChart, "Plotly is not available. Reload the page and try again.");
      return;
    }
    const regression = model?.combinedRegression;
    if (!regression?.pairs?.length) {
      this.errorsDescription.textContent = "";
      this.setChartEmpty(this.errorsChart, "Residual graphs require a valid matched regression data set.");
      return;
    }

    const pairs = regression.pairs;
    const historyPairs = pairs.filter(pair => !model.forecastKeys.has(pair.key));
    const forecastPairs = pairs.filter(pair => model.forecastKeys.has(pair.key));
    const unit = model.unit || "Series value";
    let traces = [];
    let layout = this.commonLayout(theme);
    const zeroShape = (axis = "y") => axis === "x"
      ? { type: "line", xref: "x", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1, line: { color: theme.fit, width: 1.5, dash: "dash" } }
      : { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: theme.fit, width: 1.5, dash: "dash" } };

    if (this.errorView === "residual-time") {
      this.errorsDescription.textContent = "Chronological residuals (actual minus predicted), with history and forecast shown separately.";
      if (historyPairs.length) traces.push(this.residualTimeTrace(historyPairs, "History", theme.history, unit));
      if (forecastPairs.length) traces.push(this.residualTimeTrace(forecastPairs, "Forecast", theme.forecast, unit));
      layout = this.commonLayout(theme, {
        hovermode: "x unified",
        xaxis: { 
          type: "date",
          title: { text: "Time" },
          tickformat: "%H:%M<br>%d.%m.%y",
          gridcolor: theme.grid,
          tickfont: { color: theme.muted },
        },
        yaxis: { 
          title: { text: `Residual (${unit})` },
          gridcolor: theme.grid,
          tickfont: { color: theme.muted },
        },
        shapes: [zeroShape("y")]
      });
    } else if (this.errorView === "histogram") {
      // Keep only finite residuals.
      const values = pairs
        .map(pair => pair.residual)
        .filter(Number.isFinite);

      if (!values.length) {
        this.setChartEmpty(
          this.errorsChart,
          "No valid residuals are available."
        );
        return;
      }

      this.errorsDescription.textContent =
        `Distribution of regression residuals; ${values.length} matched observations, ` +
        "with a fitted normal distribution for comparison.";

      // ---------------------------------------------------------
      // 1. Calculate histogram bins.
      // ---------------------------------------------------------
      let min = Math.min(...values);
      let max = Math.max(...values);

      if (min === max) {
        const padding = Math.max(Math.abs(min) * 0.05, 0.5);
        min -= padding;
        max += padding;
      }

      const binCount = Math.min(
        35,
        Math.max(8, Math.ceil(Math.sqrt(values.length)))
      );

      const binWidth = (max - min) / binCount;

      const bins = Array.from(
        { length: binCount },
        (_, index) => ({
          start: min + index * binWidth,
          end: index === binCount - 1
            ? max
            : min + (index + 1) * binWidth,
          count: 0
        })
      );

      values.forEach(value => {
        const index = Math.min(
          binCount - 1,
          Math.max(
            0,
            Math.floor((value - min) / binWidth)
          )
        );

        bins[index].count += 1;
      });

      // ---------------------------------------------------------
      // 2. Create histogram trace.
      // ---------------------------------------------------------
      traces = [{
        type: "bar",
        name: "Residual observations",
        x: bins.map(bin => (bin.start + bin.end) / 2),
        y: bins.map(bin => bin.count),
        width: bins.map(() => binWidth * 0.94),
        customdata: bins.map(bin => [
          bin.start,
          bin.end
        ]),
        marker: {
          color: theme.history,
          line: {
            color: theme.fit,
            width: 0.5
          }
        },
        opacity: 0.72,
        hovertemplate:
          "Residual range: %{customdata[0]:.3f} to %{customdata[1]:.3f}" +
          "<br>Observations: %{y}<extra></extra>"
      }];

      // ---------------------------------------------------------
      // 3. Calculate fitted normal distribution.
      // ---------------------------------------------------------
      const mean = values.reduce(
        (sum, value) => sum + value,
        0
      ) / values.length;

      const variance = values.length > 1
        ? values.reduce(
            (sum, value) => sum + (value - mean) ** 2,
            0
          ) / (values.length - 1)
        : 0;

      const standardDeviation = Math.sqrt(variance);

      // Only draw the normal curve when a meaningful
      // standard deviation can be calculated.
      if (
        values.length > 1 &&
        Number.isFinite(standardDeviation) &&
        standardDeviation > 0
      ) {
        const curvePointCount = 100;

        const curveX = Array.from(
          { length: curvePointCount },
          (_, index) =>
            min + (max - min) *
            index / (curvePointCount - 1)
        );

        // Scale normal density to expected histogram counts.
        const curveY = curveX.map(x => {
          const z = (x - mean) / standardDeviation;

          const density =
            Math.exp(-0.5 * z * z) /
            (standardDeviation * Math.sqrt(2 * Math.PI));

          return density * values.length * binWidth;
        });

        traces.push({
          type: "scatter",
          mode: "lines",
          name: "Fitted normal distribution",
          x: curveX,
          y: curveY,
          line: {
            color: theme.fit,
            width: 2,
            dash: "solid"
          },
          hovertemplate:
            `Residual: %{x:.3f} ${unit}` +
            "<br>Expected observations per bin: %{y:.2f}" +
            "<extra></extra>"
        });
      }

      // ---------------------------------------------------------
      // 4. Configure chart layout.
      // ---------------------------------------------------------

      layout = this.commonLayout(theme, {
        barmode: "overlay",
        xaxis: {
          title: {
            text: `Residual (${unit})`
          },
          tickfont: { color: theme.muted },
          range: [min, max],
          gridcolor: theme.grid
        },
        yaxis: {
          title: {
            text: "Observation count"
          },
          gridcolor: theme.grid,
          tickfont: { color: theme.muted },
          rangemode: "tozero"
        },
        shapes: [
          zeroShape("x")
        ],
        annotations: [{
          xref: "paper",
          yref: "paper",
          x: 0.98,
          y: 0.98,
          xanchor: "right",
          yanchor: "top",
          showarrow: false,
          text:
            `n = ${values.length}` +
            `<br>Mean = ${formatStatistic(mean, 3)}` +
            `<br>SD = ${formatStatistic(standardDeviation, 3)}`,
          align: "right",
          font: {
            size: 11,
            color: theme.muted
          }
        }]
      });
    } else if (this.errorView === "residual-fitted") {
      this.errorsDescription.textContent = "Residuals plotted against fitted values to reveal curvature, changing variance and outliers.";
      if (historyPairs.length) traces.push(this.residualFittedTrace(historyPairs, "History", theme.history, unit));
      if (forecastPairs.length) traces.push(this.residualFittedTrace(forecastPairs, "Forecast", theme.forecast, unit));
      layout = this.commonLayout(theme, {
        xaxis: {
          title: { text: `Fitted value (${unit})` },
          gridcolor: theme.grid,
          tickfont: { color: theme.muted }
        },
        yaxis: { 
          title: { text: `Residual (${unit})` }, 
          gridcolor: theme.grid,
          tickfont: { color: theme.muted }
        },
        shapes: [zeroShape("y")]
      });
    } else {
      this.errorsDescription.textContent = "Mean residual by local clock hour. Hover over each bar to see its observation count.";
      const buckets = Array.from({ length: 24 }, () => []);
      pairs.forEach(pair => {
        const hour = getHour(pair.timestamp);
        if (Number.isInteger(hour) && hour >= 0 && hour <= 23) buckets[hour].push(pair.residual);
      });
      const means = buckets.map(values => values.length
        ? values.reduce((sum, value) => sum + value, 0) / values.length
        : null);
      const labels = Array.from({ length: 24 }, (_, hour) => String(hour).padStart(2, "0"));
      const counts = buckets.map(values => values.length);
      traces = [{
        type: "bar", name: "Average residual", x: labels, y: means,
        customdata: counts,
        marker: { color: means.map(value => value === null ? (theme.isDark ? "#68737d" : "#aab5c0") : value >= 0 ? "rgb(255, 140, 0)" : "rgb(74, 144, 226)") },
        hovertemplate: "Hour: %{x}:00<br>Average residual: %{y:.3f} " + unit + "<br>Observations: %{customdata}<extra></extra>"
      }];
      layout = this.commonLayout(theme, {
        xaxis: { 
          title: { text: "Hour of day" },
          categoryorder: "array",
          categoryarray: labels,
          gridcolor: theme.grid,
          tickfont: { color: theme.muted }
      },
        yaxis: {
          title: { text: `Avg. residual (${unit})` },
          gridcolor: theme.grid,
          tickfont: { color: theme.muted }
        },
        shapes: [zeroShape("y")],
        showlegend: false
      });
    }

    this.drawPlot(this.errorsChart, traces, layout, token, "errors");
  }

  residualTimeTrace(pairs, name, color, unit) {
    return {
      type: "scatter", mode: "lines+markers", name,
      x: pairs.map(pair => pair.timestamp), y: pairs.map(pair => pair.residual),
      customdata: pairs.map(pair => [pair.y, pair.fitted, pair.x]),
      connectgaps: false,
      line: { color, width: 1.7 }, marker: { color, size: 5, opacity: 0.9 },
      hovertemplate: `%{customdata[0]:.3f} ${unit}<br>Predicted: %{customdata[1]:.3f} ${unit}<br>Residual: %{y:.3f} ${unit}<br>Spot price: %{customdata[2]:.2f} €/MWh<extra>${name}</extra>`
    };
  }

  residualFittedTrace(pairs, name, color, unit) {
    return {
      type: "scatter", mode: "markers", name,
      x: pairs.map(pair => pair.fitted), y: pairs.map(pair => pair.residual),
      customdata: pairs.map(pair => [pair.timestamp, pair.y]),
      marker: { color, size: 7, opacity: 0.84 },
      hovertemplate: `Fitted: %{x:.3f} ${unit}<br>Residual: %{y:.3f} ${unit}<br>Actual: %{customdata[1]:.3f} ${unit}<br>Time: %{customdata[0]}<extra>${name}</extra>`
    };
  }

  drawPlot(container, traces, layout, token, chartName) {
    const latestToken = chartName === "relationship" ? this.relationshipRenderToken : this.errorRenderToken;
    if (token !== latestToken) return;
    const config = { responsive: true, displayModeBar: false, displaylogo: false, scrollZoom: false };
    window.Plotly.react(container, traces, layout, config).then(() => {
      const currentToken = chartName === "relationship" ? this.relationshipRenderToken : this.errorRenderToken;
      if (token === currentToken && !this.modal.hidden) window.Plotly.Plots.resize(container);
    }).catch(error => {
      console.error(`Could not render statistics ${chartName} chart:`, error);
      this.setChartEmpty(container, "The selected graph could not be rendered.");
    });
  }

  refreshTheme() {
    if (this.model && !this.modal.hidden) this.renderCurrentCharts();
  }

  resizeVisibleCharts() {
    if (!window.Plotly || this.modal.hidden) return;
    [this.relationshipChart, this.errorsChart].forEach(container => {
      if (container.data) {
        try { window.Plotly.Plots.resize(container); } catch (error) {
          console.warn("Could not resize statistics chart:", error);
        }
      }
    });
  }
}
