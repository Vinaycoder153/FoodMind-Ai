/**
 * FoodMind AI – Lightweight Chart library (Canvas-based)
 * Replaces Chart.js with zero external dependencies.
 * Supports: "doughnut" and "bar" chart types.
 */

/* ------------------------------------------------------------------ */
/* Tiny global Chart class that mimics the Chart.js API used in app.js */
/* ------------------------------------------------------------------ */
class Chart {
  constructor(ctx, config) {
    this.ctx = ctx;
    this.config = config;
    this._draw();
  }

  destroy() {
    const canvas = this.ctx.canvas;
    this.ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  _draw() {
    const type = this.config.type;
    if (type === "doughnut") {
      this._drawDoughnut();
    } else if (type === "bar") {
      this._drawBar();
    }
  }

  /* ---------------------------------------------------------------- */
  /* Doughnut chart                                                     */
  /* ---------------------------------------------------------------- */
  _drawDoughnut() {
    const { ctx, config } = this;
    const canvas = ctx.canvas;
    const W = canvas.width  = canvas.offsetWidth  || 300;
    const H = canvas.height = canvas.offsetHeight || 240;

    ctx.clearRect(0, 0, W, H);

    const dataset = config.data.datasets[0];
    const data   = dataset.data;
    const colors = dataset.backgroundColor;
    const labels = config.data.labels || [];

    const total  = data.reduce((a, b) => a + b, 0);
    if (total === 0) return;

    const legendH = 40;
    const cx = W / 2;
    const cy = (H - legendH) / 2;
    const outerR = Math.min(cx, cy) - 10;
    const innerR = outerR * 0.68;

    let startAngle = -Math.PI / 2;
    const slices = [];

    data.forEach((val, i) => {
      const sweep = (val / total) * 2 * Math.PI;
      slices.push({ startAngle, sweep, color: colors[i], label: labels[i], val });
      startAngle += sweep;
    });

    // Draw slices
    slices.forEach(({ startAngle: sa, sweep, color }) => {
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, outerR, sa, sa + sweep);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
    });

    // Draw hole
    ctx.beginPath();
    ctx.arc(cx, cy, innerR, 0, 2 * Math.PI);
    ctx.fillStyle = "#161b27";
    ctx.fill();

    // Center text (largest slice %)
    const dominant = slices.reduce((a, b) => (b.val > a.val ? b : a));
    ctx.fillStyle = "#e2e8f0";
    ctx.font      = `bold ${Math.round(outerR * 0.28)}px 'Segoe UI', sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(`${dominant.val}%`, cx, cy);

    // Legend
    const legendY = H - legendH + 8;
    const itemW   = W / slices.length;
    slices.forEach(({ color, label, val }, i) => {
      const lx = i * itemW + itemW / 2;
      ctx.fillStyle = color;
      ctx.fillRect(lx - 28, legendY, 10, 10);
      ctx.fillStyle = "#8892aa";
      ctx.font      = "11px 'Segoe UI', sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(`${label} ${val}%`, lx - 14, legendY);
    });
  }

  /* ---------------------------------------------------------------- */
  /* Bar chart                                                          */
  /* ---------------------------------------------------------------- */
  _drawBar() {
    const { ctx, config } = this;
    const canvas = ctx.canvas;
    const W = canvas.width  = canvas.offsetWidth  || 300;
    const H = canvas.height = canvas.offsetHeight || 240;

    ctx.clearRect(0, 0, W, H);

    const datasets = config.data.datasets;
    const labels   = config.data.labels || [];
    const maxYOpt  = config.options?.scales?.y?.max;

    // Flatten all values to find max
    let allVals = [];
    datasets.forEach(ds => allVals = allVals.concat(ds.data));
    const maxVal = maxYOpt || Math.ceil(Math.max(...allVals) * 1.2) || 100;

    const padLeft   = 42;
    const padRight  = 16;
    const padTop    = 20;
    const legendH   = datasets.length > 1 ? 28 : 0;
    const padBottom = 34 + legendH;

    const chartW = W - padLeft - padRight;
    const chartH = H - padTop - padBottom;

    // Grid lines
    const gridCount = 5;
    ctx.strokeStyle = "#2d3652";
    ctx.lineWidth   = 1;
    ctx.fillStyle   = "#8892aa";
    ctx.font        = "10px 'Segoe UI', sans-serif";
    ctx.textAlign   = "right";
    ctx.textBaseline = "middle";

    for (let i = 0; i <= gridCount; i++) {
      const yVal = (maxVal / gridCount) * i;
      const y    = padTop + chartH - (yVal / maxVal) * chartH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + chartW, y);
      ctx.stroke();
      ctx.fillText(`${Math.round(yVal)}%`, padLeft - 4, y);
    }

    // Bars
    const numGroups  = labels.length;
    const numDatasets = datasets.length;
    const groupW     = chartW / numGroups;
    const gap        = 6;
    const barW       = (groupW - gap * (numDatasets + 1)) / numDatasets;

    datasets.forEach((ds, dsIdx) => {
      ds.data.forEach((val, i) => {
        const barH  = (val / maxVal) * chartH;
        const gx    = padLeft + i * groupW;
        const bx    = gx + gap + dsIdx * (barW + gap);
        const by    = padTop + chartH - barH;

        const color = Array.isArray(ds.backgroundColor)
          ? ds.backgroundColor[i]
          : ds.backgroundColor;

        ctx.fillStyle = color;
        const r = Math.min(4, barW / 2, barH / 2);
        _roundRect(ctx, bx, by, barW, barH, r);
        ctx.fill();
      });
    });

    // X-axis labels
    ctx.fillStyle   = "#8892aa";
    ctx.font        = "11px 'Segoe UI', sans-serif";
    ctx.textAlign   = "center";
    ctx.textBaseline = "top";
    labels.forEach((label, i) => {
      const lx = padLeft + i * groupW + groupW / 2;
      ctx.fillText(label, lx, padTop + chartH + 6);
    });

    // Legend (for grouped bars)
    if (datasets.length > 1) {
      const legendY = H - legendH + 6;
      let lx = padLeft;
      datasets.forEach((ds) => {
        ctx.fillStyle = Array.isArray(ds.backgroundColor) ? ds.backgroundColor[0] : ds.backgroundColor;
        ctx.fillRect(lx, legendY, 10, 10);
        ctx.fillStyle   = "#8892aa";
        ctx.font        = "11px 'Segoe UI', sans-serif";
        ctx.textAlign   = "left";
        ctx.textBaseline = "top";
        ctx.fillText(ds.label, lx + 14, legendY);
        lx += ctx.measureText(ds.label).width + 32;
      });
    }
  }
}

/* ------------------------------------------------------------------ */
/* Helper: rounded rectangle                                           */
/* ------------------------------------------------------------------ */
function _roundRect(ctx, x, y, w, h, r) {
  if (h <= 0) return;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
