let STATE = {
  token: localStorage.getItem("loco_session_token") || "",
  user: null,
  categories: [],
  samples: [],
  selectedFile: null,
  selectedSamplePath: null,
  lastInspectionResult: null,
};

function getAuthHeaders() {
  const headers = {};
  if (STATE.token) {
    headers["Authorization"] = `Bearer ${STATE.token}`;
    headers["X-Session-Token"] = STATE.token;
  }
  return headers;
}

async function initApp() {
  setupDragAndDrop();
  await Promise.all([checkAuthSession(), loadMetadata(), loadHistory()]);

  const urlParams = new URLSearchParams(window.location.search);
  const view = urlParams.get("view");
  if (view === "auth") {
    setLoggedOutUser();
    openAuthModal("login");
  } else if (view === "dashboard") {
    await handleDemoLogin();
    closeAuthModal();
  } else if (view === "inspection") {
    await handleDemoLogin();
    closeAuthModal();
    const sample = STATE.samples.find(s => s.defect_hint && (s.defect_hint.includes("anomaly") || s.defect_hint.includes("structural"))) || STATE.samples[1] || STATE.samples[0];
    if (sample) {
      selectSample(sample.rel_path, sample.category, sample.defect_hint);
      await runInspection();
    }
  } else if (view === "history") {
    await handleDemoLogin();
    closeAuthModal();
    switchTab("history");
    await loadHistory();
  } else if (view === "analytics" || view === "benchmark") {
    await handleDemoLogin();
    closeAuthModal();
    switchTab("analytics");
  }
}

async function checkAuthSession() {
  try {
    const res = await fetch("/api/auth/me", {
      headers: getAuthHeaders(),
    });
    const data = await res.json();
    if (data.ok && data.authenticated && data.user) {
      setLoggedInUser(data.user, STATE.token);
    } else {
      setLoggedOutUser();
      openAuthModal("login");
    }
  } catch (err) {
    console.warn("Session check error:", err);
    openAuthModal("login");
  }
}

function setLoggedInUser(user, token) {
  STATE.user = user;
  if (token) {
    STATE.token = token;
    localStorage.setItem("loco_session_token", token);
  }
  document.getElementById("userLoggedInBox").style.display = "flex";
  document.getElementById("userLoggedOutBox").style.display = "none";
  const displayName = user.full_name || user.username || "Inspector";
  document.getElementById("userNameDisplay").textContent = displayName;
  document.getElementById("userRoleDisplay").textContent = `${user.role || "Quality Inspector"} (${user.username})`;
  document.getElementById("userAvatar").textContent = displayName.charAt(0).toUpperCase();
}

function setLoggedOutUser() {
  STATE.user = null;
  STATE.token = "";
  localStorage.removeItem("loco_session_token");
  document.getElementById("userLoggedInBox").style.display = "none";
  document.getElementById("userLoggedOutBox").style.display = "flex";
}

function openAuthModal(tab = "login") {
  clearAuthAlert();
  switchAuthTab(tab);
  document.getElementById("authModalBackdrop").classList.add("open");
}

function closeAuthModal() {
  document.getElementById("authModalBackdrop").classList.remove("open");
}

function switchAuthTab(tab) {
  clearAuthAlert();
  const isLogin = tab === "login";
  document.getElementById("tabBtnLogin").classList.toggle("active", isLogin);
  document.getElementById("tabBtnSignup").classList.toggle("active", !isLogin);
  document.getElementById("loginForm").style.display = isLogin ? "block" : "none";
  document.getElementById("signupForm").style.display = isLogin ? "none" : "block";
}

function showAuthAlert(message, type = "error") {
  const el = document.getElementById("authAlertBox");
  el.textContent = message;
  el.className = `alert-box ${type}`;
}

function clearAuthAlert() {
  const el = document.getElementById("authAlertBox");
  el.textContent = "";
  el.className = "alert-box";
}

async function submitLogin(e) {
  e.preventDefault();
  clearAuthAlert();
  const identifier = document.getElementById("loginIdentifier").value.trim();
  const password = document.getElementById("loginPassword").value.trim();
  const btn = document.getElementById("loginSubmitBtn");

  if (!identifier || !password) {
    showAuthAlert("Please enter both your username/email and password.", "error");
    return;
  }

  btn.disabled = true;
  btn.textContent = "Signing in...";
  try {
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier, password }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      showAuthAlert(data.message || "Login failed. Please check your credentials.", "error");
      return;
    }
    setLoggedInUser(data.user, data.token);
    showAuthAlert("Logged in successfully!", "success");
    await loadHistory();
    setTimeout(() => closeAuthModal(), 450);
  } catch (err) {
    showAuthAlert(`Network error while logging in: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "🔑 Log In to Inspector";
  }
}

async function submitSignup(e) {
  e.preventDefault();
  clearAuthAlert();
  const full_name = document.getElementById("signupFullName").value.trim();
  const username = document.getElementById("signupUsername").value.trim();
  const role = document.getElementById("signupRole").value;
  const email = document.getElementById("signupEmail").value.trim();
  const password = document.getElementById("signupPassword").value.trim();
  const confirm_password = document.getElementById("signupConfirmPassword").value.trim();
  const btn = document.getElementById("signupSubmitBtn");

  if (!username && !email) {
    showAuthAlert("Please enter a username and email address.", "error");
    return;
  }
  if (password.length < 4) {
    showAuthAlert("Password must be at least 4 characters long.", "error");
    return;
  }
  if (confirm_password && password !== confirm_password) {
    showAuthAlert("Passwords do not match. Please verify both password fields.", "error");
    return;
  }

  btn.disabled = true;
  btn.textContent = "Creating account...";
  try {
    const res = await fetch("/api/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        full_name,
        username,
        role,
        email,
        password,
        confirm_password,
      }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      showAuthAlert(data.message || "Could not create account.", "error");
      return;
    }
    setLoggedInUser(data.user, data.token);
    showAuthAlert("Account created and signed in!", "success");
    await loadHistory();
    setTimeout(() => closeAuthModal(), 500);
  } catch (err) {
    showAuthAlert(`Network error during sign up: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "✨ Create Account & Sign In";
  }
}

async function handleDemoLogin() {
  clearAuthAlert();
  try {
    const res = await fetch("/api/auth/demo", { method: "POST" });
    const data = await res.json();
    if (data.ok && data.user) {
      setLoggedInUser(data.user, data.token);
      showAuthAlert("Signed in as Demo Inspector!", "success");
      await loadHistory();
      setTimeout(() => closeAuthModal(), 350);
    }
  } catch (err) {
    showAuthAlert(`Demo login error: ${err.message}`, "error");
  }
}

async function handleLogout() {
  try {
    await fetch("/api/auth/logout", {
      method: "POST",
      headers: getAuthHeaders(),
    });
  } catch (err) {
    console.warn(err);
  }
  setLoggedOutUser();
}

/* Metadata & Samples */
async function loadMetadata() {
  try {
    const res = await fetch("/api/meta");
    const data = await res.json();
    if (data.ok) {
      STATE.categories = data.categories || [];
      STATE.samples = data.samples || [];
      const devText = (data.device || "cuda").toUpperCase();
      document.getElementById("deviceText").textContent =
        devText.includes("CUDA") ? "CUDA GPU Active" : "CPU Active";
      renderSamples();
    }
  } catch (err) {
    console.error("Failed to load metadata:", err);
  }
}

function onCategoryFilterChange() {
  const cat = document.getElementById("categorySelect").value;
  if (cat !== "auto") {
    document.getElementById("sampleCategoryFilter").value = cat;
    renderSamples();
  }
  if (STATE.selectedFile || STATE.selectedSamplePath) {
    runInspection();
  }
}

function onModeChange() {
  if (STATE.selectedFile || STATE.selectedSamplePath) {
    runInspection();
  }
}

function renderSamples() {
  const filterCat = document.getElementById("sampleCategoryFilter").value;
  const grid = document.getElementById("samplesGrid");
  const filtered = STATE.samples.filter((s) => s.category === filterCat);

  grid.innerHTML = filtered
    .map((s) => {
      const isSelected = STATE.selectedSamplePath === s.rel_path;
      const tagClass = s.is_faulty ? "faulty" : "good";
      const tagLabel = s.is_faulty
        ? s.subtype === "logical_anomalies"
          ? "LOGICAL FAULT"
          : "STRUCTURAL FAULT"
        : "NORMAL (GOOD)";
      const imgUrl = `/api/sample-image?rel_path=${encodeURIComponent(s.rel_path)}`;
      return `
        <div class="sample-card ${isSelected ? "active" : ""}" onclick="selectSample('${s.rel_path}', '${s.category}', '${s.defect_hint.replace(/'/g, "\\'")}')">
          <img src="${imgUrl}" alt="${s.filename}" loading="lazy" />
          <span class="sample-tag ${tagClass}">${tagLabel}</span>
          <div class="sample-caption" title="${s.defect_hint}">${s.defect_hint}</div>
        </div>
      `;
    })
    .join("");
}

function selectSample(relPath, category, hint) {
  STATE.selectedFile = null;
  STATE.selectedSamplePath = relPath;
  document.getElementById("fileInput").value = "";

  const previewBox = document.getElementById("uploadPreviewBox");
  const previewImg = document.getElementById("uploadPreviewImg");
  const previewName = document.getElementById("uploadPreviewName");
  const previewSub = document.getElementById("uploadPreviewSub");

  previewImg.src = `/api/sample-image?rel_path=${encodeURIComponent(relPath)}`;
  previewName.textContent = relPath;
  previewSub.textContent = `Dataset Sample • ${hint}`;
  previewBox.classList.add("visible");

  renderSamples();
  runInspection();
}

function triggerSampleByType(subtype) {
  const filterCat = document.getElementById("sampleCategoryFilter").value || "juice_bottle";
  const match = STATE.samples.find(
    (s) => s.category === filterCat && s.subtype === subtype
  );
  if (match) {
    selectSample(match.rel_path, match.category, match.defect_hint);
  }
}

/* File Upload & Drag-and-Drop */
function setupDragAndDrop() {
  const dz = document.getElementById("dropzone");
  ["dragenter", "dragover"].forEach((evtName) => {
    dz.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dz.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evtName) => {
    dz.addEventListener(evtName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dz.classList.remove("dragover");
    });
  });
  dz.addEventListener("drop", (e) => {
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      setUploadedFile(files[0]);
    }
  });
}

function handleFileSelect(e) {
  const files = e.target.files;
  if (files && files.length > 0) {
    setUploadedFile(files[0]);
  }
}

function setUploadedFile(file) {
  hideInspectError();
  STATE.selectedFile = file;
  STATE.selectedSamplePath = null;
  renderSamples();

  const reader = new FileReader();
  reader.onload = (ev) => {
    document.getElementById("uploadPreviewImg").src = ev.target.result;
  };
  reader.readAsDataURL(file);

  const kb = (file.size / 1024).toFixed(1);
  document.getElementById("uploadPreviewName").textContent = file.name;
  document.getElementById("uploadPreviewSub").textContent = `${kb} KB • Uploaded File`;
  document.getElementById("uploadPreviewBox").classList.add("visible");

  runInspection();
}

function clearSelectedFile() {
  STATE.selectedFile = null;
  STATE.selectedSamplePath = null;
  document.getElementById("fileInput").value = "";
  document.getElementById("uploadPreviewBox").classList.remove("visible");
  renderSamples();
}

function showInspectError(msg) {
  const el = document.getElementById("inspectErrorAlert");
  el.textContent = msg;
  el.className = "alert-box error";
}

function hideInspectError() {
  const el = document.getElementById("inspectErrorAlert");
  el.textContent = "";
  el.className = "alert-box";
}

/* Run Defect Inspection */
async function runInspection() {
  hideInspectError();

  if (!STATE.selectedFile && !STATE.selectedSamplePath) {
    showInspectError("Please upload an image file or click one of the Quick-Test Reference Samples below.");
    return;
  }

  const btn = document.getElementById("inspectBtn");
  const category = document.getElementById("categorySelect").value;
  const operatingMode = document.getElementById("modeSelect").value;

  btn.disabled = true;
  btn.innerHTML = `<span>⏳</span> Running Neural Patch Extraction &amp; Scoring...`;

  try {
    const formData = new FormData();
    formData.append("category", category);
    formData.append("operating_mode", operatingMode);
    if (STATE.selectedFile) {
      formData.append("file", STATE.selectedFile);
    } else if (STATE.selectedSamplePath) {
      formData.append("sample_rel_path", STATE.selectedSamplePath);
    }

    const res = await fetch("/api/inspect", {
      method: "POST",
      headers: getAuthHeaders(),
      body: formData,
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      showInspectError(data.message || "Inspection failed. Please verify the uploaded image.");
      return;
    }

    STATE.lastInspectionResult = data;
    renderInspectionResult(data);
    loadHistory();
  } catch (err) {
    showInspectError(`Network error during inspection: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span>🔍</span> Inspect Object for Faults &amp; Anomalies`;
  }
}

function renderInspectionResult(res) {
  document.getElementById("emptyResultState").style.display = "none";
  document.getElementById("populatedResultState").style.display = "block";

  document.getElementById("resultMetaBadge").textContent =
    `${res.category_icon} ${res.category_name} • File: ${res.filename} • ${res.elapsed_ms} ms`;

  // Auto-switched category banner
  const switchNotice = document.getElementById("autoSwitchNotice");
  if (res.category_auto_switched) {
    switchNotice.style.display = "block";
    switchNotice.innerHTML = `✨ <strong>Auto-Detected Product Category:</strong> Automatically matched uploaded image to <strong>${res.category_icon} ${res.category_name}</strong> for accurate defect evaluation.`;
  } else {
    switchNotice.style.display = "none";
  }

  // Verdict Banner
  const banner = document.getElementById("verdictBanner");
  const isFaulty = Boolean(res.is_faulty);
  banner.className = `verdict-banner ${isFaulty ? "faulty" : "normal"}`;
  document.getElementById("verdictIcon").textContent = isFaulty ? "🚨" : "✅";
  document.getElementById("verdictTitle").textContent = isFaulty
    ? "FAULTY OBJECT — DEFECT DETECTED (ANOMALOUS)"
    : "PASS — OBJECT IS NORMAL (DEFECT-FREE)";
  document.getElementById("verdictSubtitle").textContent =
    `${res.category_icon} ${res.category_name} • Classification: ${res.subtype_label}`;
  document.getElementById("verdictExplanation").textContent = res.explanation;

  // Specific Defect Badges
  const badgesWrap = document.getElementById("defectBadgesWrap");
  if (res.detected_defects && res.detected_defects.length > 0) {
    badgesWrap.innerHTML = res.detected_defects
      .map((d) => `<span class="badge badge-fail">⚠️ Fault: ${d}</span>`)
      .join("");
  } else if (isFaulty) {
    badgesWrap.innerHTML = `<span class="badge badge-fail">⚠️ Visual Feature Anomaly (${res.peak_region})</span>`;
  } else {
    badgesWrap.innerHTML = `<span class="badge badge-pass">✓ All Quality Checks Passed</span>`;
  }

  // KPIs
  document.getElementById("kpiScore").textContent = Number(res.score).toFixed(4);
  document.getElementById("kpiScore").style.color = isFaulty ? "#f87171" : "#34d399";
  const zSign = res.z_score >= 0 ? "+" : "";
  document.getElementById("kpiZScore").textContent = `${zSign}${res.z_score}σ from normal mean (${res.val_mean})`;

  document.getElementById("kpiThreshold").textContent = Number(res.threshold).toFixed(4);
  document.getElementById("kpiModeNote").textContent = res.operating_mode_label;

  document.getElementById("kpiSeverity").textContent = res.severity;
  document.getElementById("kpiSubtype").textContent = isFaulty ? "Requires Rejection / Review" : "Approved for Line";

  document.getElementById("kpiRegion").textContent = res.peak_region;
  document.getElementById("kpiHotspot").textContent = `Hotspot coverage: ${res.hotspot_pct}%`;

  // Gauge Bar positioning
  const minG = Math.max(15.0, res.val_mean - 2.5 * res.val_std);
  const maxG = res.val_mean + 6.5 * res.val_std;
  const thrPct = Math.min(96, Math.max(4, ((res.threshold - minG) / (maxG - minG)) * 100));
  const scorePct = Math.min(98, Math.max(2, ((res.score - minG) / (maxG - minG)) * 100));

  document.getElementById("gaugeLeftLabel").textContent = `Normal Mean (μ): ${res.val_mean}`;
  document.getElementById("gaugeCenterLabel").textContent = `Cutoff: ${res.threshold} (Score: ${res.score})`;
  document.getElementById("gaugeRightLabel").textContent = `Critical Zone (${(res.val_mean + 5 * res.val_std).toFixed(1)}+)`;
  document.getElementById("gaugeThresholdLine").style.left = `${thrPct}%`;
  document.getElementById("gaugeScoreMarker").style.left = `${scorePct}%`;

  // 4 Visual Images
  document.getElementById("imgOriginal").src = res.images.original;
  document.getElementById("imgLocalized").src = res.images.localized;
  document.getElementById("imgHeatmap").src = res.images.heatmap;
  document.getElementById("imgOverlay").src = res.images.overlay;

  // Operating Points Table
  const opBody = document.getElementById("operatingPointsBody");
  opBody.innerHTML = (res.operating_points || [])
    .map((op) => {
      const mSign = op.margin >= 0 ? "+" : "";
      return `
        <tr class="${op.is_active ? "active-row" : ""}">
          <td><strong>${op.label}</strong> ${op.is_active ? "👈" : ""}</td>
          <td>${op.threshold.toFixed(3)}</td>
          <td>${mSign}${op.margin.toFixed(3)}</td>
          <td>
            <span class="badge ${op.is_faulty ? "badge-fail" : "badge-pass"}">
              ${op.decision}
            </span>
          </td>
        </tr>
      `;
    })
    .join("");

  // Rules & Defects Checklist
  document.getElementById("rulesCardTitle").textContent =
    `📋 ${res.category_icon} ${res.category_name} Quality Specification`;
  document.getElementById("normalRulesList").innerHTML = (res.normal_rules || [])
    .map((r) => `<li>${r}</li>`)
    .join("");
  const allDefects = [
    ...((res.common_defects && res.common_defects.logical) || []).map((d) => `[Logical] ${d}`),
    ...((res.common_defects && res.common_defects.structural) || []).map((d) => `[Structural] ${d}`),
  ];
  document.getElementById("defectRulesList").innerHTML = allDefects
    .map((d) => `<li>${d}</li>`)
    .join("");
}

/* History Log */
async function loadHistory() {
  try {
    const res = await fetch("/api/history", { headers: getAuthHeaders() });
    const data = await res.json();
    const tbody = document.getElementById("historyTableBody");
    if (!data.ok || !data.history || data.history.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted);">No inspections recorded yet. Upload or select a sample image to begin.</td></tr>`;
      return;
    }
    tbody.innerHTML = data.history
      .map((item) => {
        const isFaulty = item.decision === "ANOMALOUS";
        const ts = new Date(item.created_at).toLocaleString();
        return `
          <tr>
            <td>#${item.id}</td>
            <td style="white-space: nowrap;">${ts}</td>
            <td><strong>${item.filename}</strong></td>
            <td>${item.category}</td>
            <td>
              <span class="badge ${isFaulty ? "badge-fail" : "badge-pass"}">
                ${isFaulty ? "FAULTY" : "NORMAL"}
              </span>
            </td>
            <td>${Number(item.score).toFixed(3)} / ${Number(item.threshold).toFixed(3)}</td>
            <td>${item.severity}</td>
            <td>${item.peak_region}</td>
            <td style="max-width: 340px; font-size: 0.78rem; color: var(--text-secondary);">${item.explanation}</td>
          </tr>
        `;
      })
      .join("");
  } catch (err) {
    console.warn("History load error:", err);
  }
}

/* Tab Switching & Lightbox */
function switchTab(tabId) {
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabId}`);
  });
  if (tabId === "history") {
    loadHistory();
  }
}

function openLightbox(src) {
  if (!src) return;
  document.getElementById("lightboxImg").src = src;
  document.getElementById("lightbox").classList.add("open");
}

function closeLightbox() {
  document.getElementById("lightbox").classList.remove("open");
}

window.addEventListener("DOMContentLoaded", initApp);
