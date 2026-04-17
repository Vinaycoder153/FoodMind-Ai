/* ===================================================================
   FoodMind AI – Frontend Application Logic
   =================================================================== */

const API_BASE = "http://localhost:8000";

const SAMPLE_REVIEWS = [
  "The food was too oily! Every bite dripped with grease. Very disappointed.",
  "बहुत ज़्यादा तेल था, खाना खराब लगा। Quantity bhi bahut kam thi.",
  "ಎಣ್ಣೆ ತುಂಬಾ ಜಾಸ್ತಿ ಇತ್ತು, ರುಚಿ ಚೆನ್ನಾಗಿದೆ ಆದರೆ ಪ್ರಮಾಣ ಕಡಿಮೆ ಇತ್ತು.",
  "Spice level is way too high, I couldn't even finish the dish.",
  "Amazing biryani! Perfect flavors and great portion size. Loved it!",
  "Too much masala in the curry. My stomach is upset now.",
  "Good quantity but the oil was excessive. Please reduce it.",
  "Bland taste, no spices at all. Not worth the price.",
  "Excellent food! Fresh ingredients, perfect cooking. Will definitely come back.",
  "Small portion for such a high price. Very unsatisfying.",
  "The chicken was perfectly cooked but way too oily. Needs improvement.",
  "Worst food ever. Tasteless, cold, and overcooked. Never ordering again.",
  "Great taste and reasonable quantity. Just reduce the oil a bit.",
  "teekha bahut zyada tha, thoda kam karo please.",
  "Ruchi bahut acchi thi! Swad excellent. Loved the flavors.",
  "Too greasy, dripping with oil. Very unhealthy preparation.",
  "Nice portion size and amazing taste. Happy with my order!",
  "Kharab taste, masala galatho. Very bad experience.",
  "The food quality has been declining over the past few weeks.",
  "Best restaurant in town! Everything was perfect today.",
];

/* ------------------------------------------------------------------ */
/* DOM helpers                                                          */
/* ------------------------------------------------------------------ */
const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ */
/* Review count updater                                                 */
/* ------------------------------------------------------------------ */
function updateReviewCount() {
  const text = $("reviewsInput").value.trim();
  const count = text ? text.split("\n").filter((l) => l.trim()).length : 0;
  $("reviewCount").textContent = `${count} review${count !== 1 ? "s" : ""}`;
}

$("reviewsInput").addEventListener("input", updateReviewCount);

/* ------------------------------------------------------------------ */
/* Load sample reviews                                                  */
/* ------------------------------------------------------------------ */
$("loadSample").addEventListener("click", () => {
  $("reviewsInput").value = SAMPLE_REVIEWS.join("\n");
  updateReviewCount();
});

/* ------------------------------------------------------------------ */
/* CSV Upload                                                           */
/* ------------------------------------------------------------------ */
const uploadZone = $("uploadZone");
const csvFile = $("csvFile");

uploadZone.addEventListener("click", () => csvFile.click());
uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("drag-over");
});
uploadZone.addEventListener("dragleave", () =>
  uploadZone.classList.remove("drag-over"),
);
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleCSVUpload(file);
});

csvFile.addEventListener("change", () => {
  if (csvFile.files[0]) handleCSVUpload(csvFile.files[0]);
});

async function handleCSVUpload(file) {
  const status = $("uploadStatus");
  status.textContent = `⏳ Uploading ${file.name}…`;
  status.className = "upload-status";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/upload-csv`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed");
    }
    const data = await res.json();
    const processed = data.insights?.total_reviews || 0;
    status.textContent = `✅ ${file.name} uploaded — ${processed} reviews processed`;
    renderResults(data);
  } catch (err) {
    status.className = "upload-status error";
    status.textContent = `❌ ${err.message}`;
  }
  status.classList.remove("hidden");
}

/* ------------------------------------------------------------------ */
/* Main Analyse action                                                  */
/* ------------------------------------------------------------------ */
$("analyseBtn").addEventListener("click", analyseReviews);

async function analyseReviews() {
  const raw = $("reviewsInput").value.trim();
  if (!raw) {
    alert("Please paste at least one review or load sample reviews.");
    return;
  }

  const reviews = raw.split("\n").filter((l) => l.trim());
  if (!reviews.length) return;

  // Show loading
  $("loadingBar").classList.remove("hidden");
  $("analyseBtn").disabled = true;
  $("results").classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/mvp/analyse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reviews }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Analysis failed");
    }

    const data = await res.json();
    renderResults(data);
  } catch (err) {
    alert(
      `Error: ${err.message}\n\nMake sure the backend is running at ${API_BASE}`,
    );
  } finally {
    $("loadingBar").classList.add("hidden");
    $("analyseBtn").disabled = false;
  }
}

/* ------------------------------------------------------------------ */
/* Render all results                                                   */
/* ------------------------------------------------------------------ */
function renderResults(data) {
  const insights = data.insights || {};
  const topIssues = data.top_issues || [];
  const recommendations = data.recommendations || [];
  const impact = data.impact_simulation || {};
  const comparison = data.before_vs_after || {};

  renderOverview(insights, impact, topIssues);
  renderReviewList(topIssues);
  renderRecommendations(recommendations);
  renderCustomisations(insights.strengths || []);
  renderImpact(impact, comparison);
  renderFeedbackLoop(comparison);

  $("results").classList.remove("hidden");
  $("results").scrollIntoView({ behavior: "smooth" });
}

/* ------------------------------------------------------------------ */
/* Overview                                                             */
/* ------------------------------------------------------------------ */
let sentimentChartInst = null;
let complaintChartInst = null;

function renderOverview(insights, impact, topIssues) {
  const sentiment = insights.sentiment_distribution || {};
  const languages = insights.languages || {};
  const featureSummary = insights.feature_summary || {};
  const dominantIssue = topIssues[0] || null;

  // Metric cards
  $("metTotalReviews").textContent = insights.total_reviews || 0;
  $("metPositive").textContent = `${sentiment.positive || 0}%`;
  $("metNegative").textContent = `${sentiment.negative || 0}%`;
  $("metNeutral").textContent = `${sentiment.neutral || 0}%`;
  $("metCurrentRating").textContent = `${impact.current_rating || "—"} ⭐`;
  $("metPredictedRating").textContent = `${impact.predicted_rating || "—"} ⭐`;

  // Alerts
  const alertContainer = $("alertBanners");
  alertContainer.innerHTML = "";
  (insights.alerts || []).forEach((alert) => {
    const div = document.createElement("div");
    div.className = `alert-banner ${alert.level}`;
    div.innerHTML = `<span>${alert.message}</span>`;
    alertContainer.appendChild(div);
  });

  // Trend
  const trendEl = $("trendBadge");
  trendEl.className = `trend-badge ${impact.improvement_pct >= 0 ? "improving" : "stable"}`;
  trendEl.textContent = dominantIssue
    ? `Top issue: ${dominantIssue.feature} · language mix: ${
        Object.entries(languages)
          .filter(([, v]) => v)
          .map(([k]) => k)
          .join(", ") || "unknown"
      }`
    : `No major complaint clusters detected`;

  // Sentiment doughnut chart
  const sentimentCtx = document
    .getElementById("sentimentChart")
    .getContext("2d");
  if (sentimentChartInst) sentimentChartInst.destroy();
  sentimentChartInst = new Chart(sentimentCtx, {
    type: "doughnut",
    data: {
      labels: ["Positive", "Negative", "Neutral", "Mixed"],
      datasets: [
        {
          data: [
            sentiment.positive || 0,
            sentiment.negative || 0,
            sentiment.neutral || 0,
            sentiment.mixed || 0,
          ],
          backgroundColor: ["#34d399", "#f87171", "#64748b", "#fbbf24"],
          borderWidth: 0,
          hoverOffset: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#8892aa", padding: 16, font: { size: 12 } },
        },
        tooltip: {
          callbacks: { label: (ctx) => ` ${ctx.label}: ${ctx.raw}%` },
        },
      },
      cutout: "68%",
    },
  });

  // Complaint bar chart
  const features = Object.keys(featureSummary);
  const values = features.map((f) => featureSummary[f].negative || 0);
  const colors = features.map((f) => {
    const negativeCount = featureSummary[f].negative || 0;
    if (negativeCount >= 2) return "#f87171";
    if (negativeCount >= 1) return "#fbbf24";
    return "#6c63ff";
  });

  const complaintCtx = document
    .getElementById("complaintChart")
    .getContext("2d");
  if (complaintChartInst) complaintChartInst.destroy();
  complaintChartInst = new Chart(complaintCtx, {
    type: "bar",
    data: {
      labels: features.map((f) => f.charAt(0).toUpperCase() + f.slice(1)),
      datasets: [
        {
          label: "Negative Mentions",
          data: values,
          backgroundColor: colors,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8892aa" }, grid: { color: "#2d3652" } },
        y: {
          ticks: { color: "#8892aa" },
          grid: { color: "#2d3652" },
          max: Math.max(5, ...values) + 1,
        },
      },
    },
  });
}

/* ------------------------------------------------------------------ */
/* Review list                                                          */
/* ------------------------------------------------------------------ */
function renderReviewList(reviews) {
  const container = $("reviewList");
  container.innerHTML = "";

  reviews.forEach((r, idx) => {
    const div = document.createElement("div");
    div.className = "review-item negative";

    const examples = (r.examples || [])
      .map((e) => `<li>${escapeHtml(e)}</li>`)
      .join("");

    div.innerHTML = `
      <div class="review-header">
        <span class="review-number">#${idx + 1}</span>
        <span class="sentiment-pill negative">${featureIcon(r.feature)} ${escapeHtml(r.feature)}</span>
      </div>
      <div class="review-text">Impact score: ${escapeHtml(String(r.impact_score || 0))}</div>
      <div class="review-tags">
        <span class="tag negative">Frequency: ${escapeHtml(String(Math.round((r.frequency || 0) * 100)))}%</span>
        <span class="tag negative">Severity: ${escapeHtml(String(r.severity || 0))}</span>
        <span class="tag negative">Confidence: ${escapeHtml(String(r.confidence || 0))}</span>
      </div>
      ${examples ? `<div class="review-text"><ul>${examples}</ul></div>` : ""}
    `;
    container.appendChild(div);
  });
}

/* ------------------------------------------------------------------ */
/* Recommendations                                                      */
/* ------------------------------------------------------------------ */
function renderRecommendations(recs) {
  const container = $("recommendationsList");
  container.innerHTML = "";

  if (!recs.length) {
    container.innerHTML =
      "<p style='color:var(--text-dim)'>No significant issues detected. Keep collecting feedback. 🎉</p>";
    return;
  }

  recs.forEach((rec, idx) => {
    const conf = rec.confidence || 0;
    const pct = `${conf}%`;
    const deg = `${(conf / 100) * 360}deg`;

    const div = document.createElement("div");
    div.className = "rec-card";
    div.innerHTML = `
      <div>
        <span class="rec-feature-tag">${featureIcon(rec.feature)} ${rec.feature}</span>
        <div class="rec-action">${idx + 1}. ${escapeHtml(rec.fix)}</div>
        <div class="rec-detail">Priority: ${escapeHtml(rec.priority)} · ${escapeHtml(rec.effort_vs_impact)}</div>
        <div class="rec-reasoning">📊 ${escapeHtml(rec.rationale)}</div>
      </div>
      <div class="rec-confidence">
        <div class="confidence-ring" style="--pct:${deg}">${pct}</div>
        <div class="confidence-label">Lift +${escapeHtml(String(rec.expected_rating_lift || 0))}</div>
      </div>
    `;
    container.appendChild(div);
  });
}

/* ------------------------------------------------------------------ */
/* Customisations                                                        */
/* ------------------------------------------------------------------ */
function renderCustomisations(items) {
  const container = $("customisationsList");
  container.innerHTML = "";

  items.forEach((item) => {
    const div = document.createElement("div");
    div.className = "custom-card";
    div.innerHTML = `
      <div class="custom-tag">${escapeHtml(item.feature)}</div>
      <div class="custom-name">Strength score ${escapeHtml(String(item.strength_score || 0))}</div>
      <div class="custom-desc">${escapeHtml(String(item.positive_mentions || 0))} positive mentions · ${escapeHtml(String(Math.round((item.frequency || 0) * 100)))}% of reviews</div>
      <span class="custom-price">Confidence ${escapeHtml(String(item.confidence || 0))}</span>
    `;
    container.appendChild(div);
  });
}

/* ------------------------------------------------------------------ */
/* Impact Simulator                                                     */
/* ------------------------------------------------------------------ */
let beforeAfterChartInst = null;

function renderImpact(impact, comparison) {
  if (!impact.current_rating) return;

  const summaryEl = $("impactSummary");
  summaryEl.innerHTML = `
    Current rating: <strong>${escapeHtml(String(impact.current_rating || "—"))} ⭐</strong><br>
    Predicted rating: <strong>${escapeHtml(String(impact.predicted_rating || "—"))} ⭐</strong><br>
    Improvement: <strong>${escapeHtml(String(impact.improvement_pct || 0))}%</strong>
  `;

  $("beforeRating").textContent = impact.current_rating;
  $("afterRating").textContent = impact.predicted_rating;
  $("beforeStars").textContent = starString(impact.current_rating);
  $("afterStars").textContent = starString(impact.predicted_rating);

  // Before/after bar chart
  const before = comparison.before?.sentiment_distribution || {};
  const after = comparison.after?.expected_sentiment_distribution || {};
  const features = Object.keys(before);

  const ctx = document.getElementById("beforeAfterChart").getContext("2d");
  if (beforeAfterChartInst) beforeAfterChartInst.destroy();
  beforeAfterChartInst = new Chart(ctx, {
    type: "bar",
    data: {
      labels: features.map((f) => f.charAt(0).toUpperCase() + f.slice(1)),
      datasets: [
        {
          label: "Before (%)",
          data: features.map((f) => before[f] || 0),
          backgroundColor: "rgba(248,113,113,0.7)",
          borderRadius: 5,
        },
        {
          label: "After (%)",
          data: features.map((f) => after[f] || 0),
          backgroundColor: "rgba(52,211,153,0.7)",
          borderRadius: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top", labels: { color: "#8892aa" } },
      },
      scales: {
        x: { ticks: { color: "#8892aa" }, grid: { color: "#2d3652" } },
        y: {
          ticks: { color: "#8892aa", callback: (v) => `${v}%` },
          grid: { color: "#2d3652" },
          max: 100,
        },
      },
    },
  });
}

/* ------------------------------------------------------------------ */
/* Feedback loop dashboard                                              */
/* ------------------------------------------------------------------ */
function renderFeedbackLoop(impact) {
  const container = $("feedbackLoop");
  container.innerHTML = "";

  const before = impact.before?.sentiment_distribution || {};
  const after = impact.after?.expected_sentiment_distribution || {};
  const features = Object.keys(before);
  if (!features.length) return;

  // Before card
  const beforeCard = document.createElement("div");
  beforeCard.className = "fl-card";
  beforeCard.innerHTML = `<div class="fl-header before">😞 BEFORE — Current State</div>`;
  features.forEach((f) => {
    const pct = before[f] || 0;
    beforeCard.innerHTML += `
      <div class="fl-row">
        <span class="fl-feature">${featureIcon(f)} ${f}</span>
        <div class="fl-bar-wrap"><div class="fl-bar before-bar" style="width:${pct}%"></div></div>
        <span class="fl-pct" style="color:var(--red)">${pct}%</span>
      </div>
    `;
  });

  // After card
  const afterCard = document.createElement("div");
  afterCard.className = "fl-card";
  afterCard.innerHTML = `<div class="fl-header after">✅ AFTER — Predicted State</div>`;
  features.forEach((f) => {
    const pct = after[f] || 0;
    afterCard.innerHTML += `
      <div class="fl-row">
        <span class="fl-feature">${featureIcon(f)} ${f}</span>
        <div class="fl-bar-wrap"><div class="fl-bar after-bar" style="width:${pct}%"></div></div>
        <span class="fl-pct" style="color:var(--green)">${pct}%</span>
      </div>
    `;
  });

  container.appendChild(beforeCard);
  container.appendChild(afterCard);
}

/* ------------------------------------------------------------------ */
/* Utility helpers                                                      */
/* ------------------------------------------------------------------ */
function featureIcon(feature) {
  const icons = {
    oil_level: "🛢️",
    spice_level: "🌶️",
    quantity: "⚖️",
    texture: "🥘",
    taste: "👅",
    price_value: "💸",
    freshness: "🧊",
    service: "🧑‍🍳",
    hygiene: "🧼",
  };
  return icons[feature] || "🔍";
}

function sentimentEmoji(sentiment) {
  return sentiment === "positive"
    ? "😊"
    : sentiment === "negative"
      ? "😞"
      : "😐";
}

function starString(rating) {
  const full = Math.floor(rating);
  const half = rating - full >= 0.5;
  return "⭐".repeat(full) + (half ? "✨" : "");
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* ------------------------------------------------------------------ */
/* Init                                                                 */
/* ------------------------------------------------------------------ */
updateReviewCount();
