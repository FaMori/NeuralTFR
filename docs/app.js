/* ===================================================================
   NeuralTFR — TFR Model Comparison Dashboard
   =================================================================== */

(() => {
  "use strict";

  // ── Color & Style Config (light-theme friendly) ──
  const MODEL_COLORS = {
    NeuralTFR: { line: "#2563eb", fill: "rgba(37,99,235,0.12)" },
    NeuralTFR_New: { line: "#f97316", fill: "rgba(249,115,22,0.12)" },
    WPP: { line: "#08b224ff", fill: "rgba(8,145,178,0.10)" },
    gbd: { line: "#059669", fill: "rgba(5,150,105,0.10)" },
  };

  const MODEL_LABELS = {
    NeuralTFR: "NeuralTFR",
    NeuralTFR_New: "NeuralTFR New",
    WPP: "WPP (UN)",
    gbd: "GBD",
  };

  const HISTORICAL_MEAN_COLOR = "rgba(51, 65, 85, 0.6)";

  // High contrast palette for different sources/methods
  const POINT_COLORS = ["#ef4444", "#eab308", "#10b981", "#06b6d4", "#6366f1", "#d946ef", "#f43f5e"];
  const POINT_SHAPES = ["circle", "triangle", "rect", "rectRounded", "rectRot", "cross", "star"];

  // ── State ──
  let DATA = null;
  let chart = null;
  let map = null;
  let geojsonLayer = null;
  let worldGeoJson = null;

  let selectedCountryId = null;
  let horizonYear = 2042;
  let enabledModels = new Set();

  let showHistPoints = true;
  let activeTab = "chart";

  const methodStyleMap = new Map();
  let styleCounter = 0;
  let countryListScrollTop = 0;

  // ── DOM refs ──
  const $loading = document.getElementById("loading");
  const $countryPanel = document.getElementById("country-panel");
  const $countryList = document.getElementById("country-list");
  const $countrySearch = document.getElementById("country-search");
  const $modelToggles = document.getElementById("model-toggles");
  const $chartTitle = document.getElementById("chart-title");
  const $chartCanvas = document.getElementById("main-chart");
  const $chartLegend = document.getElementById("chart-legend");
  const $resetZoom = document.getElementById("btn-reset-zoom");
  const $countryCount = document.getElementById("country-count");
  const $horizonSlider = document.getElementById("horizon-slider");
  const $horizonValue = document.getElementById("horizon-value");

  // View sections
  const $tabChart = document.getElementById("tab-chart");
  const $tabMap = document.getElementById("tab-map");
  const $viewChart = document.getElementById("view-chart");
  const $viewMap = document.getElementById("view-map");
  const $mapLegend = document.getElementById("map-legend");

  const $horizonLabelText = document.getElementById("horizon-label-text");
  const $dataOptionsPanel = document.getElementById("data-options-panel");

  const $togglePoints = document.getElementById("toggle-historical-points");

  // ── Helpers ──
  function interpolateColor(color1, color2, factor) {
    const hex = (color) => [parseInt(color.slice(1,3), 16), parseInt(color.slice(3,5), 16), parseInt(color.slice(5,7), 16)];
    const c1 = hex(color1);
    const c2 = hex(color2);
    const result = c1.map((c, i) => Math.round(c + factor * (c2[i] - c)));
    return `#${result.map(c => c.toString(16).padStart(2, '0')).join('')}`;
  }
  function getStyleForMethod(source, method) {
    const key = `${source || 'Unknown'}-${method || 'Unknown'}`;
    if (!methodStyleMap.has(key)) {
      methodStyleMap.set(key, {
        shape: POINT_SHAPES[styleCounter % POINT_SHAPES.length],
        color: POINT_COLORS[styleCounter % POINT_COLORS.length]
      });
      styleCounter++;
    }
    return { key, ...methodStyleMap.get(key) };
  }

  function getModelMaxYear(models) {
    if (!DATA?.modelYearRanges) return 2042;

    let maxYear = 2042;
    let foundRange = false;
    for (const model of models) {
      const range = DATA.modelYearRanges[model];
      if (!range?.max) continue;
      maxYear = Math.max(maxYear, range.max);
      foundRange = true;
    }
    return foundRange ? maxYear : 2042;
  }

  // ── Init ──
  async function init() {
    if (!window.FERTCAST_DATA) {
      $loading.innerHTML = `<p style="color:#e11d48">Error: data.js not loaded. Run <code>python prepare_data.py</code> first.</p>`;
      return;
    }
    DATA = window.FERTCAST_DATA;

    // Collect all model names from forecast only
    const availableModels = Object.keys(DATA.forecast || {});
    // Filter to allowed specific comparison models
    const allowed = new Set(["WPP", "gbd", "NeuralTFR", "NeuralTFR_New"]);
    const allModels = new Set(availableModels.filter(m => allowed.has(m)));

    // Default model
    enabledModels = new Set();
    if (allModels.has("NeuralTFR")) enabledModels.add("NeuralTFR");
    if (allModels.has("NeuralTFR_New")) enabledModels.add("NeuralTFR_New");
    //if (allModels.has("WPP")) enabledModels.add("WPP");
    if (enabledModels.size === 0 && allModels.size > 0) enabledModels.add(Array.from(allModels)[0]);

    $countryCount.textContent = DATA.countries.length;

    // Set horizon slider limits
    if (DATA.modelYearRanges) {
      const globalMin = 2024;
      const globalMax = getModelMaxYear(allModels);
      $horizonSlider.min = globalMin;
      $horizonSlider.max = globalMax;

      window.horizonYearChart = Math.min(2042, globalMax);
      window.horizonYearMap = Math.min(2030, globalMax);
      horizonYear = window.horizonYearChart;
      $horizonSlider.value = horizonYear;
      $horizonValue.textContent = horizonYear;

      const $ticks = document.querySelector(".horizon-ticks");
      if ($ticks) {
        $ticks.innerHTML = `<span>${globalMin}</span><span>${Math.round((globalMin + globalMax) / 2)}</span><span>${globalMax}</span>`;
      }
    }

    renderCountryList();
    renderModelToggles(allModels);
    initChart();

    // Load GeoJSON for the map
    try {
      const res = await fetch('https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json');
      worldGeoJson = await res.json();
      initMap();
    } catch (e) {
      console.error("Failed to load map GeoJSON", e);
    }

    bindEvents();

    if (DATA.countries.length > 0) {
      selectCountry(DATA.countries[0].id);
    }

    $loading.classList.add("fade-out");
    setTimeout(() => { $loading.style.display = "none"; }, 500);
  }

  // ── Country List ──
  function renderCountryList(filter = "") {
    const lowerFilter = filter.toLowerCase();
    const filtered = DATA.countries.filter(c =>
      c.name.toLowerCase().includes(lowerFilter)
    );
    $countryList.innerHTML = filtered.map(c => `
      <div class="country-item ${c.id === selectedCountryId ? 'active' : ''}"
           data-id="${c.id}" title="${c.name}">${c.name}</div>
    `).join("");
  }

  function updateSliderLimits() {
    let lim = getModelMaxYear(
      enabledModels.size > 0 ? enabledModels : Object.keys(DATA.modelYearRanges || {})
    );
    if (activeTab === "chart" && selectedCountryId) {
      lim = 1950;
      const cid = String(selectedCountryId);
      Object.keys(DATA.forecast || {}).forEach(m => {
        if (DATA.forecast[m][cid] && enabledModels.has(m)) {
           DATA.forecast[m][cid].forEach(d => { if (d.year > lim) lim = d.year; });
        }
      });
      if (lim === 1950) {
        lim = getModelMaxYear(
          enabledModels.size > 0 ? enabledModels : Object.keys(DATA.modelYearRanges || {})
        );
      }
    }
    $horizonSlider.max = lim;
    if (parseInt($horizonSlider.value) > lim || horizonYear > lim) {
      horizonYear = lim;
      $horizonSlider.value = lim;
      $horizonValue.textContent = lim;
    } else {
      $horizonSlider.value = horizonYear; // Force visual refresh of slider thumb
    }
    const $ticks = document.querySelector(".horizon-ticks");
    if ($ticks) {
      $ticks.innerHTML = `<span>2024</span><span>${Math.round((2024 + lim) / 2)}</span><span>${lim}</span>`;
    }
  }

  function selectCountry(id) {
    selectedCountryId = id;
    const country = DATA.countries.find(c => c.id === id);
    $chartTitle.textContent = country ? country.name : "Unknown";
    document.querySelectorAll(".country-item").forEach(el => {
      el.classList.toggle("active", parseInt(el.dataset.id) === id);
    });
    updateSliderLimits();
    if (activeTab === "chart") updateChart();
  }

  // ── Model Toggles ──
  function renderModelToggles(allModels) {
    const ordered = [
      "NeuralTFR",
      "NeuralTFR_New",
      ...Array.from(allModels).filter(m => m !== "NeuralTFR" && m !== "NeuralTFR_New").sort()
    ];
    $modelToggles.innerHTML = ordered.filter(m => allModels.has(m)).map(model => {
      const color = MODEL_COLORS[model]?.line || "#888";
      const label = MODEL_LABELS[model] || model;
      return `
        <label class="model-toggle" data-model="${model}">
          <input type="checkbox" value="${model}" ${enabledModels.has(model) ? "checked" : ""}>
          <span class="toggle-indicator" style="background: ${enabledModels.has(model) ? color : 'transparent'}; border-color: ${color}"></span>
          <span class="model-name">${label}</span>
        </label>
      `;
    }).join("");
  }

  // ── Events ──
  function bindEvents() {
    $countrySearch.addEventListener("input", (e) => {
      renderCountryList(e.target.value);
    });

    $countryList.addEventListener("click", (e) => {
      const item = e.target.closest(".country-item");
      if (item) selectCountry(parseInt(item.dataset.id));
    });

    $horizonSlider.addEventListener("input", (e) => {
      horizonYear = parseInt(e.target.value);
      $horizonValue.textContent = horizonYear;
      if (activeTab === "chart") { window.horizonYearChart = horizonYear; updateChart(); }
      else if (activeTab === "map") { window.horizonYearMap = horizonYear; updateMap(); }
    });

    $modelToggles.addEventListener("change", (e) => {
      if (e.target.type !== "checkbox") return;
      const model = e.target.value;
      if (e.target.checked) enabledModels.add(model);
      else enabledModels.delete(model);

      const label = e.target.closest(".model-toggle");
      const indicator = label.querySelector(".toggle-indicator");
      const color = MODEL_COLORS[model]?.line || "#888";
      indicator.style.background = e.target.checked ? color : "transparent";

      updateSliderLimits();
      if (activeTab === "chart") updateChart();
      else if (activeTab === "map") updateMap();
    });

    $togglePoints.addEventListener("change", (e) => {
      showHistPoints = e.target.checked;
      if (activeTab === "chart") updateChart();
    });

    $resetZoom.addEventListener("click", () => {
      if (chart) chart.resetZoom();
    });

    // Tab switching
    $tabChart.addEventListener("click", () => switchTab("chart"));
    $tabMap.addEventListener("click", () => switchTab("map"));
  }

  function switchTab(tabId) {
    if (activeTab === "chart" && tabId === "map") {
      countryListScrollTop = $countryList.scrollTop;
    }
    activeTab = tabId;
    if (tabId === "chart") {
      if (activeTab === "map") window.horizonYearMap = horizonYear;
      horizonYear = window.horizonYearChart || 2042;
      $horizonLabelText.textContent = "Forecast until";
      $dataOptionsPanel.style.display = "block";
      $horizonSlider.value = horizonYear;
      $horizonValue.textContent = horizonYear;

      $tabChart.classList.add("active");
      $tabMap.classList.remove("active");
      $viewChart.classList.add("active");
      $viewMap.classList.remove("active");
      $countryPanel.style.display = "block";
      $countryList.scrollTop = countryListScrollTop;
      updateSliderLimits();
      updateChart();
    } else {
      if (activeTab === "chart") window.horizonYearChart = horizonYear;
      horizonYear = window.horizonYearMap || 2030;
      $horizonLabelText.textContent = "Selected Map Year";
      $dataOptionsPanel.style.display = "none";
      $horizonSlider.value = horizonYear;
      $horizonValue.textContent = horizonYear;

      $tabMap.classList.add("active");
      $tabChart.classList.remove("active");
      $viewMap.classList.add("active");
      $viewChart.classList.remove("active");
      $countryPanel.style.display = "none";
      updateSliderLimits();
      if (map) {
        setTimeout(() => {
          map.invalidateSize();
          updateMap();
        }, 10);
      }
    }
  }

  // ── Chart ──
  function initChart() {
    const ctx = $chartCanvas.getContext("2d");
    chart = new Chart(ctx, {
      type: "line",
      data: { datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "nearest",
          axis: "x",
          intersect: false,
        },
        scales: {
          x: {
            type: "linear",
            title: { display: true, text: "Year", color: "#94a3b8", font: { family: "Inter", size: 12 } },
            ticks: {
              color: "#64748b",
              font: { family: "Inter" },
              stepSize: 5,
              callback: v => Math.round(v),
            },
            grid: { color: "rgba(0,0,0,0.05)" },
          },
          y: {
            title: { display: true, text: "TFR (children per woman)", color: "#94a3b8", font: { family: "Inter", size: 12 } },
            ticks: {
              color: "#64748b",
              font: { family: "Inter" },
            },
            grid: { color: "rgba(0,0,0,0.05)" },
            suggestedMin: 0,
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#ffffff",
            titleColor: "#1e293b",
            bodyColor: "#475569",
            borderColor: "rgba(0,0,0,0.1)",
            borderWidth: 1,
            cornerRadius: 10,
            titleFont: { family: "Inter", weight: "600" },
            bodyFont: { family: "Inter" },
            padding: 12,
            boxShadow: "0 4px 16px rgba(0,0,0,0.1)",
            callbacks: {
              title: (items) => `Year ${items[0].parsed.x}`,
              label: (item) => {
                const v = item.parsed.y;
                if (v == null) return null;
                if (item.dataset.label.startsWith("_")) return null;
                if (item.dataset.label.includes("upper")) return null;
                // Add source/method data to historical points
                if (item.dataset.originalData) {
                  const pt = item.dataset.originalData[item.dataIndex];
                  if (pt.source || pt.method) {
                    return ` ${pt.source || 'Unk'} (${pt.method || 'Unk'}): ${v.toFixed(3)}`;
                  }
                }
                return ` ${item.dataset.label}: ${v.toFixed(3)}`;
              },
            },
          },
          zoom: {
            pan: { enabled: true, mode: "xy" },
            zoom: {
              wheel: { enabled: true },
              pinch: { enabled: true },
              mode: "xy",
            },
          },
        },
        animation: { duration: 500, easing: "easeOutCubic" },
      },
    });
  }

  function updateChart() {
    if (!chart || !selectedCountryId) return;
    const cid = String(selectedCountryId);
    const datasets = [];

    // 1) Historical Median
    if (DATA.historical_mean && DATA.historical_mean[cid]) {
      const meanData = DATA.historical_mean[cid];
      if (meanData.length > 0) {
        datasets.push({
          label: "Historical Median",
          data: meanData.map(d => ({ x: d.year, y: d.tfr })),
          borderColor: HISTORICAL_MEAN_COLOR,
          backgroundColor: HISTORICAL_MEAN_COLOR,
          borderWidth: 2.5,
          pointRadius: 0,
          pointHitRadius: 4,
          order: 10,
        });
      }
    }

    // 2) Historical Points
    const usedStylesThisCountry = new Set();
    if (showHistPoints && DATA.historical_points && DATA.historical_points[cid]) {
      const pointsData = DATA.historical_points[cid];

      // Group points by source+method so each combo becomes a separate dataset (better legend & tooltips)
      const grouped = {};
      pointsData.forEach(p => {
        const style = getStyleForMethod(p.source, p.method);
        usedStylesThisCountry.add(style.key);
        if (!grouped[style.key]) {
          grouped[style.key] = { label: `${p.source} - ${p.method}`, style, points: [] };
        }
        grouped[style.key].points.push(p);
      });

      for (const grp of Object.values(grouped)) {
        datasets.push({
          label: grp.label,
          originalData: grp.points, // Keep for tooltips
          data: grp.points.map(d => ({ x: d.year, y: d.tfr })),
          borderColor: grp.style.color,
          backgroundColor: grp.style.color,
          pointStyle: grp.style.shape,
          borderWidth: 0,
          showLine: false,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointHitRadius: 8,
          pointBorderColor: grp.style.color,
          pointBorderWidth: 2,
          order: 5,
        });
      }
    }

    // 3) Forecast predictions
    for (const model of enabledModels) {
      const fcData = DATA.forecast?.[model]?.[cid];
      if (!fcData || fcData.length === 0) continue;

      const filtered = fcData.filter(d => d.year <= horizonYear);
      if (filtered.length === 0) continue;

      const color = MODEL_COLORS[model] || { line: "#888", fill: "rgba(136,136,136,0.1)" };
      const label = MODEL_LABELS[model] || model;

      datasets.push({
        label: `${label}`,
        data: filtered.map(d => ({ x: d.year, y: d.median })),
        borderColor: color.line,
        backgroundColor: "transparent",
        borderWidth: 2.5,
        pointRadius: 0,
        pointHitRadius: 6,
        order: 3,
        borderDash: [5, 5],
      });

      const hasBands = filtered.some(d => d.lower != null && d.upper != null);
      if (hasBands) {
        datasets.push({
          label: `${label} upper`,
          data: filtered.map(d => ({ x: d.year, y: d.upper })),
          borderColor: "transparent", backgroundColor: "transparent",
          borderWidth: 0, pointRadius: 0, fill: false, order: 4,
        });
        datasets.push({
          label: `_${label}_lower_fc`,
          data: filtered.map(d => ({ x: d.year, y: d.lower })),
          borderColor: "transparent", backgroundColor: color.fill,
          borderWidth: 0, pointRadius: 0, fill: "-1", order: 4,
        });
      }
    }

    let minX = Infinity;
    let maxX = -Infinity;
    datasets.forEach(ds => {
      ds.data.forEach(p => {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
      });
    });
    if (minX !== Infinity) {
      chart.options.scales.x.min = minX - 1;
      chart.options.scales.x.max = maxX + 1;
    } else {
      chart.options.scales.x.min = undefined;
      chart.options.scales.x.max = undefined;
    }

    chart.data.datasets = datasets;
    chart.update();
    updateLegend(usedStylesThisCountry);
  }

  function updateLegend(usedStylesThisCountry = new Set()) {
    const cid = String(selectedCountryId);
    const items = [];

    if (DATA.historical_mean && DATA.historical_mean[cid]?.length > 0) {
      items.push(`
        <div class="legend-item">
          <span class="legend-swatch" style="background:${HISTORICAL_MEAN_COLOR}"></span>
          Historical Median
        </div>
      `);
    }

    if (showHistPoints && usedStylesThisCountry.size > 0) {
      for (const key of usedStylesThisCountry) {
        const st = methodStyleMap.get(key);
        if (st) {
          const parts = key.split('-');
          items.push(`
            <div class="legend-item" title="${parts[0]} - ${parts[1]}">
              <span style="display:inline-block;width:12px;height:12px;background:${st.color};border-radius:50%;"></span>
              ${parts[0]} - ${parts[1]}
            </div>
          `);
        }
      }
    }

    for (const model of enabledModels) {
      const color = MODEL_COLORS[model]?.line || "#888";
      const label = MODEL_LABELS[model] || model;
      const fcData = DATA.forecast?.[model]?.[cid];
      if (!fcData || fcData.length === 0) continue;
      items.push(`
        <div class="legend-item">
          <span class="legend-swatch" style="background:${color}"></span>
          ${label}
        </div>
      `);
    }
    $chartLegend.innerHTML = items.join("");
  }

  // ── Choropleth Map ──
  function initMap() {
    map = L.map('choropleth-map', {
      zoomControl: false,
      attributionControl: false
    }).setView([20, 0], 2);
    L.control.zoom({ position: 'topright' }).addTo(map);

    // Basic light tiles
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png', {
      subdomains: 'abcd',
      maxZoom: 10
    }).addTo(map);

    geojsonLayer = L.geoJson(worldGeoJson, {
      style: getMapStyle,
      onEachFeature: (feature, layer) => {
        layer.on({
          mouseover: (e) => {
            const layer = e.target;
            layer.setStyle({ weight: 2, color: '#666', fillOpacity: 0.9 });
            if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) layer.bringToFront();

            const cInfo = findCountryByISO(feature.id);
            let name = cInfo ? cInfo.name : feature.properties.name;
            let tooltipHTML = `<div style="font-family: Inter, sans-serif;"><b>${name} (${horizonYear})</b><div style="margin-top:4px; font-size:12px;">`;
            if (cInfo) {
              let hasData = false;
              for (const model of enabledModels) {
                const fcData = DATA.forecast?.[model]?.[String(cInfo.id)];
                if (fcData) {
                  const point = fcData.find(d => d.year === horizonYear);
                  if (point) {
                    tooltipHTML += `<div><span style="display:inline-block;width:8px;height:8px;background:${MODEL_COLORS[model]?.line||'#000'};margin-right:4px;"></span>${MODEL_LABELS[model] || model}: ${point.median.toFixed(3)}</div>`;
                    hasData = true;
                  }
                }
              }
              if (!hasData) tooltipHTML += `<div style="color:#666">No projection for ${horizonYear}</div>`;
            } else {
              tooltipHTML += `<div style="color:#666">No data</div>`;
            }
            tooltipHTML += `</div></div>`;
            
            layer.bindTooltip(tooltipHTML, { direction: 'top', sticky: true, opacity: 0.95 }).openTooltip();
          },
          mouseout: (e) => { 
            geojsonLayer.resetStyle(e.target); 
            e.target.closeTooltip();
          },
          click: (e) => {
            // If they click a country on the map, we select it, then switch back to Chart view
            const iso = feature.id;
            const cInfo = findCountryByISO(iso);
            if (cInfo) {
              selectCountry(cInfo.id);
              switchTab('chart');
            }
          }
        });
      }
    }).addTo(map);
  }

  function findCountryByISO(iso3) {
    // We only have names and IDs in `DATA.countries`, no ISO codes.
    // Try matching on name as a best effort, or fallback to custom lookup.
    // GitHub world.geo.json uses id for ISO3 (e.g., "USA", "ARG").
    // We will do a simple mapping by comparing feature.properties.name with our dataset name.
    if (!iso3) return null;
    const feats = worldGeoJson.features.filter(f => f.id === iso3);
    if (!feats || feats.length === 0) return null;
    const name = feats[0].properties.name.toLowerCase();

    // Heuristic string match
    return DATA.countries.find(c => {
      const cn = c.name.toLowerCase();
      return cn === name || cn.includes(name) || name.includes(cn);
    });
  }

  function getMapStyle(feature) {
    const cInfo = findCountryByISO(feature.id);
    let fillColor = "#eceff1"; // default grey
    if (cInfo) {
      // Find the NeuralTFR prediction for the `horizonYear` limit... or specifically at `horizonYear`?
      // Usually a choropleth shows the value AT the selected year.
      let val = null;

      // Use the first enabled model, normally NeuralTFR
      const activeModel = Array.from(enabledModels)[0] || "NeuralTFR";
      const fcData = DATA.forecast?.[activeModel]?.[String(cInfo.id)];

      if (fcData) {
        const point = fcData.find(d => d.year === horizonYear);
        if (point) val = point.median;
      }

      if (val !== null) {
        fillColor = getColorForValue(val);
      }
    }
    return {
      fillColor: fillColor,
      weight: 1,
      opacity: 1,
      color: 'white',
      fillOpacity: 0.7
    };
  }

  function getColorForValue(v) {
    v = Math.max(1.0, Math.min(5.0, v));
    const factor = (v - 1.0) / (5.0 - 1.0);
    return interpolateColor('#bae6fd', '#1e3a8a', factor);
  }

  function updateMap() {
    if (activeTab === "map" && geojsonLayer) {
      geojsonLayer.setStyle(getMapStyle);
      const activeModel = Array.from(enabledModels)[0] || "NeuralTFR";

      // Update map legend
      $mapLegend.innerHTML = `
        <div style="font-size:0.85rem;margin-bottom:0.8rem;font-weight:600;color:var(--text-primary)">
          ${MODEL_LABELS[activeModel] || activeModel} in ${horizonYear}
        </div>
        <div style="display:flex; align-items:center; width: 100%; gap: 8px;">
          <span style="font-size:0.75rem;">1.0</span>
          <div style="flex:1; height: 12px; border-radius: 6px; background: linear-gradient(to right, #bae6fd, #1e3a8a);"></div>
          <span style="font-size:0.75rem;">5.0+</span>
        </div>
      `;
    }
  }

  // ── Start ──
  init();
})();
