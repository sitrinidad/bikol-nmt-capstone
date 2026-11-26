/* Bikolé frontend — talks to the FastAPI middleware with the Fetch API. No fake values: every number shown comes from the API. */
(() => {
  "use strict";

  // Same origin by default (FastAPI serves this page). Override with ?api=http://host:port if the page is hosted elsewhere.
  const API_BASE = (new URLSearchParams(location.search).get("api") || window.BIKOLE_API_BASE || "").replace(/\/$/, "");
  const api = (path) => `${API_BASE}/api/v1${path}`;
  const MAX_CHARS = 1000;
  const HISTORY_KEY = "bikole-history-v1";
  const HISTORY_LIMIT = 20;
  const THEME_KEY = "bikole-theme";

  const FALLBACK_LANGUAGES = [
    { code: "eng", name: "English", label: "English" },
    { code: "bcl_South", name: "Virac", label: "Virac (Southern Bikol)" },
    { code: "bcl_North", name: "Pandan", label: "Pandan (Northern Bikol)" },
  ];

  const $ = (id) => document.getElementById(id);
  const el = {
    srcLang: $("source-lang"), tgtLang: $("target-lang"),
    srcText: $("source-text"), tgtText: $("target-text"),
    count: $("char-count"), clear: $("clear-btn"), copy: $("copy-btn"), swap: $("swap-btn"),
    translate: $("translate-btn"), modelState: $("model-state"),
    direction: $("direction-note"),
    conf: $("confidence-value"), confBar: $("confidence-bar"), latency: $("latency-value"),
    badge: $("status-badge"), statusHint: $("status-hint"),
    themeToggle: $("theme-toggle"), toasts: $("toasts"),
    historyList: $("history-list"), historyEmpty: $("history-empty"), clearHistory: $("clear-history"),
    openFb: $("open-feedback"), dialog: $("feedback-dialog"), form: $("feedback-form"),
    fbSrcText: $("fb-source-text"), fbSrcLang: $("fb-source-lang"), fbTgtLang: $("fb-target-lang"),
    fbGenerated: $("fb-generated"), fbCorrected: $("fb-corrected"), fbExplanation: $("fb-explanation"),
    fbMessage: $("fb-message"), fbSubmit: $("fb-submit"),
  };

  const state = {
    languages: FALLBACK_LANGUAGES,
    directions: null,          // [[src, tgt], ...] from the model; null until known
    busy: false,
    last: null,                // last translation shown {source_text, source_lang, target_lang, translated_text}
    modelReady: false,
    history: [],
  };

  const icons = () => window.lucide && window.lucide.createIcons();
  const labelOf = (code) => (state.languages.find((l) => l.code === code) || {}).label || code;
  const nameOf = (code) => (state.languages.find((l) => l.code === code) || {}).name || code;
  const supported = (s, t) => s !== t && (!state.directions || state.directions.some(([a, b]) => a === s && b === t));
  const fmtConfidence = (c) => (typeof c === "number" ? `${Math.round(c * 1000) / 10}%` : "—");
  const fmtLatency = (ms) => (typeof ms === "number" ? `${ms.toLocaleString()} ms` : "—");

  function storageGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
  function storageSet(key, value) { try { localStorage.setItem(key, value); return true; } catch { return false; } }

  function iconEl(name) { const i = document.createElement("i"); i.setAttribute("data-lucide", name); return i; }

  function setIcon(button, name) {
    const holder = button.querySelector(".btn__icon") || button;
    const old = holder.querySelector("svg, i");
    if (old) old.replaceWith(iconEl(name)); else holder.prepend(iconEl(name));
    icons();
  }

  // ------------------------------------------------------------ notifications
  /** kind: "error" | "success" | "info". Errors stay longer and use role=alert. */
  function notify(message, kind = "info", title) {
    const toast = document.createElement("div");
    toast.className = `toast toast--${kind}`;
    toast.setAttribute("role", kind === "error" ? "alert" : "status");
    const icon = iconEl(kind === "error" ? "circle-alert" : kind === "success" ? "circle-check" : "info");
    icon.classList.add("toast__icon");
    const body = document.createElement("div");
    body.className = "toast__body";
    if (title) { const t = document.createElement("p"); t.className = "toast__title"; t.textContent = title; body.appendChild(t); }
    const msg = document.createElement("p"); msg.textContent = message; body.appendChild(msg);
    const close = document.createElement("button");
    close.type = "button"; close.className = "btn btn--plain toast__close"; close.setAttribute("aria-label", "Dismiss notification");
    close.appendChild(iconEl("x"));
    toast.append(icon, body, close);

    const remove = () => {
      if (!toast.isConnected) return;
      toast.classList.add("is-leaving");
      setTimeout(() => toast.remove(), 200);
    };
    close.addEventListener("click", remove);
    el.toasts.appendChild(toast);
    while (el.toasts.children.length > 3) el.toasts.firstElementChild.remove();
    icons();
    setTimeout(remove, kind === "error" ? 8000 : 3500);
  }

  // -------------------------------------------------------------------- theme
  function applyTheme(theme) {
    const dark = theme === "dark";
    if (dark) document.documentElement.dataset.theme = "dark"; else delete document.documentElement.dataset.theme;
    const label = dark ? "Switch to light mode" : "Switch to dark mode";
    el.themeToggle.setAttribute("aria-pressed", String(dark));
    el.themeToggle.setAttribute("aria-label", label);
    el.themeToggle.title = label;
    setIcon(el.themeToggle, dark ? "sun" : "moon");
  }
  el.themeToggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    storageSet(THEME_KEY, next);
    applyTheme(next);
  });

  // ---------------------------------------------------------------- dropdowns
  function fillSelect(select, selected) {
    select.innerHTML = "";
    for (const l of state.languages) {
      const o = document.createElement("option");
      o.value = l.code; o.textContent = l.label;
      if (l.code === selected) o.selected = true;
      select.appendChild(o);
    }
  }

  /** Block same-language pairs and directions the model was not trained on. */
  function syncOptions(changed) {
    let s = el.srcLang.value, t = el.tgtLang.value;
    if (!supported(s, t)) {
      if (changed === "target") {
        const alt = state.languages.map((l) => l.code).find((c) => supported(c, t));
        if (alt) s = alt;
      } else {
        const alt = state.languages.map((l) => l.code).find((c) => supported(s, c));
        if (alt) t = alt;
      }
      el.srcLang.value = s; el.tgtLang.value = t;
    }
    for (const o of el.tgtLang.options) o.disabled = !supported(s, o.value);
    for (const o of el.srcLang.options) o.disabled = !supported(o.value, t);
    el.direction.textContent = `${nameOf(s)} → ${nameOf(t)}`;
    el.swap.disabled = !supported(t, s);
  }

  // ------------------------------------------------------------------- status
  function setModelState(kind, text) {
    el.modelState.dataset.state = kind;
    el.modelState.querySelector(".model-state__text").textContent = text;
    state.modelReady = kind === "ready";
  }

  function setStatus(kind, hint) {
    const labels = { idle: "Waiting", processing: "Processing", success: "Success", failed: "Failed" };
    el.badge.className = `badge badge--${kind}`;
    el.badge.textContent = labels[kind];
    el.statusHint.textContent = hint;
    bump(el.badge.parentElement);
  }

  function bump(node) {
    node.classList.remove("is-updated");
    void node.offsetWidth;
    node.classList.add("is-updated");
  }

  function setMetrics(confidence, latency) {
    el.conf.textContent = fmtConfidence(confidence);
    el.confBar.style.width = typeof confidence === "number" ? `${Math.max(0, Math.min(100, confidence * 100))}%` : "0";
    el.latency.textContent = fmtLatency(latency);
    bump(el.conf); bump(el.latency);
  }

  function resetMetrics() {
    setMetrics(null, null);
    setStatus("idle", "Translate something to begin");
  }

  function updateButtons() {
    const empty = el.srcText.value.trim().length === 0;
    el.translate.disabled = empty || state.busy;
    el.clear.disabled = el.srcText.value.length === 0 && el.tgtText.classList.contains("is-empty");
  }

  let limitWarned = false;
  function updateCounter() {
    const n = el.srcText.value.length;
    el.count.textContent = `${n.toLocaleString()} / ${MAX_CHARS.toLocaleString()} characters`;
    el.count.classList.toggle("is-near", n > MAX_CHARS * 0.9);
    if (n >= MAX_CHARS && !limitWarned) {
      limitWarned = true;
      notify(`The source text can be at most ${MAX_CHARS.toLocaleString()} characters.`, "info", "Character limit reached");
    }
    if (n < MAX_CHARS) limitWarned = false;
  }

  function setOutput(text) {
    if (text) {
      el.tgtText.textContent = text;
      el.tgtText.classList.remove("is-empty");
      el.tgtText.classList.remove("is-fresh"); void el.tgtText.offsetWidth; el.tgtText.classList.add("is-fresh");
      el.copy.disabled = false;
    } else {
      el.tgtText.textContent = "Your translation will appear here.";
      el.tgtText.classList.add("is-empty");
      el.copy.disabled = true;
    }
  }

  // ------------------------------------------------------------------ network
  async function postJSON(path, body) {
    let res;
    try {
      res = await fetch(api(path), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    } catch {
      const err = new Error("Cannot reach the translation service. Make sure the backend is running (uvicorn backend.main:app).");
      err.title = "Connection failed";
      throw err;
    }
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON */ }
    if (!res.ok || !data || data.status !== "success") {
      const err = new Error((data && (data.error || data.detail)) || `Request failed (HTTP ${res.status}).`);
      err.title = res.status === 422 || res.status === 400 ? "Invalid input" : res.status === 503 ? "Model not ready" : "Request failed";
      throw err;
    }
    return data;
  }

  async function loadLanguages() {
    try {
      const res = await fetch(api("/languages"));
      if (!res.ok) throw new Error();
      const data = await res.json();
      if (Array.isArray(data.languages) && data.languages.length) state.languages = data.languages;
      if (Array.isArray(data.directions)) state.directions = data.directions;
    } catch { /* keep fallback list; translate will report the real error */ }
    const s = el.srcLang.value || "eng", t = el.tgtLang.value || "bcl_South";
    fillSelect(el.srcLang, s); fillSelect(el.tgtLang, t);
    syncOptions("source");
  }

  let healthTimer = null;
  let lastHealth = null;
  async function pollHealth() {
    clearTimeout(healthTimer);
    let next = null;
    try {
      const res = await fetch(api("/health"));
      if (!res.ok) throw new Error();
      const { model } = await res.json();
      next = model.state;
      if (model.state === "ready") {
        setModelState("ready", `Model ready · ${model.device === "cuda" ? "GPU" : "CPU"}`);
        if (lastHealth && lastHealth !== "ready") notify("The translation model is ready.", "success");
        lastHealth = next;
        loadLanguages();   // directions come from the trained model's metadata
        return;
      }
      if (model.state === "loading") setModelState("loading", "Loading the translation model… (first start can take a few minutes)");
      else if (model.state === "error") {
        setModelState("error", "Model failed to load — see the terminal for details");
        if (lastHealth !== "error") notify(model.error || "The model could not be loaded.", "error", "Model failed to load");
        lastHealth = next;
        return;
      } else setModelState("offline", "Model is not loaded");
    } catch {
      next = "offline";
      setModelState("offline", "Translation service offline — start the backend to translate");
    }
    lastHealth = next;
    healthTimer = setTimeout(pollHealth, 3000);
  }

  // ---------------------------------------------------------------- translate
  async function translate() {
    const source_text = el.srcText.value.trim();
    const source_lang = el.srcLang.value, target_lang = el.tgtLang.value;
    if (state.busy) return;
    if (!source_text) { notify("Type or paste some text first.", "error", "Nothing to translate"); return; }
    if (source_lang === target_lang) { notify("Please choose two different languages.", "error", "Invalid languages"); return; }

    state.busy = true; updateButtons();
    el.translate.classList.add("is-loading");
    el.translate.querySelector(".btn__label").textContent = "Translating…";
    setIcon(el.translate, "loader-circle");
    setStatus("processing", `${nameOf(source_lang)} → ${nameOf(target_lang)}`);

    try {
      const data = await postJSON("/translate", { source_text, source_lang, target_lang });
      setOutput(data.translated_text);
      setMetrics(data.confidence_score, data.inference_latency_ms);
      setStatus("success", `${nameOf(source_lang)} → ${nameOf(target_lang)}`);
      state.last = { source_text, source_lang, target_lang, translated_text: data.translated_text };
      addHistory({ ...state.last, confidence_score: data.confidence_score, inference_latency_ms: data.inference_latency_ms });
    } catch (err) {
      setMetrics(null, null);
      setStatus("failed", "No translation was produced");
      notify(err.message, "error", err.title || "Translation failed");
      if (/loading|Cannot reach/.test(err.message)) pollHealth();
    } finally {
      state.busy = false;
      el.translate.classList.remove("is-loading");
      el.translate.querySelector(".btn__label").textContent = "Translate";
      setIcon(el.translate, "sparkles");
      updateButtons();
    }
  }

  // --------------------------------------------------------------------- copy
  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text; document.body.appendChild(ta); ta.select();
      const ok = document.execCommand("copy"); ta.remove();
      if (!ok) throw new Error("copy failed");
    }
  }

  async function copyOutput() {
    const text = el.tgtText.classList.contains("is-empty") ? "" : el.tgtText.textContent;
    if (!text) return;
    try { await copyText(text); } catch { notify("Your browser blocked copying. Select the text and copy it manually.", "error", "Copy failed"); return; }
    const label = el.copy.querySelector("span");
    label.textContent = "Copied";
    setIcon(el.copy, "check");
    setTimeout(() => { label.textContent = "Copy"; setIcon(el.copy, "copy"); }, 1600);
  }

  // ------------------------------------------------------------------ history
  function loadHistory() {
    try {
      const parsed = JSON.parse(storageGet(HISTORY_KEY) || "[]");
      state.history = Array.isArray(parsed) ? parsed.filter((h) => h && h.source_text && h.translated_text).slice(0, HISTORY_LIMIT) : [];
    } catch { state.history = []; }
  }

  function saveHistory() {
    if (!storageSet(HISTORY_KEY, JSON.stringify(state.history))) {
      notify("History could not be saved in this browser (storage is full or blocked).", "error", "History not saved");
    }
  }

  function addHistory(entry) {
    // the same text in the same direction moves to the top instead of being listed twice
    state.history = state.history.filter((h) => !(h.source_text === entry.source_text && h.source_lang === entry.source_lang && h.target_lang === entry.target_lang));
    state.history.unshift({ id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, created_at: new Date().toISOString(), ...entry });
    state.history = state.history.slice(0, HISTORY_LIMIT);
    saveHistory(); renderHistory();
  }

  function relativeTime(iso) {
    const diff = (Date.now() - new Date(iso).getTime()) / 1000;
    if (!isFinite(diff)) return "";
    if (diff < 60) return "just now";
    if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
    return new Date(iso).toLocaleDateString();
  }

  function actionButton(icon, label, onClick) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "btn btn--ghost btn--sm-ghost";
    const span = document.createElement("span"); span.textContent = label;
    b.append(iconEl(icon), span);
    b.addEventListener("click", onClick);
    return b;
  }

  function renderHistory() {
    el.historyList.innerHTML = "";
    el.historyEmpty.hidden = state.history.length > 0;
    el.clearHistory.disabled = state.history.length === 0;
    for (const h of state.history) {
      const li = document.createElement("li");
      li.className = "history-item";

      const meta = document.createElement("div"); meta.className = "history-item__meta";
      const dir = document.createElement("span"); dir.className = "history-item__dir"; dir.textContent = `${nameOf(h.source_lang)} → ${nameOf(h.target_lang)}`;
      const time = document.createElement("time"); time.dateTime = h.created_at; time.textContent = relativeTime(h.created_at);
      meta.append(dir, time);

      const src = document.createElement("p"); src.className = "history-item__src"; src.textContent = h.source_text; src.title = h.source_text;
      const tgt = document.createElement("p"); tgt.className = "history-item__tgt"; tgt.textContent = h.translated_text; tgt.title = h.translated_text;
      const stats = document.createElement("p"); stats.className = "history-item__stats";
      stats.textContent = `Confidence ${fmtConfidence(h.confidence_score)} · ${fmtLatency(h.inference_latency_ms)}`;

      const actions = document.createElement("div"); actions.className = "history-item__actions";
      actions.append(
        actionButton("corner-up-left", "Reuse", () => reuseHistory(h)),
        actionButton("copy", "Copy", async () => {
          try { await copyText(h.translated_text); notify("Translation copied.", "success"); }
          catch { notify("Your browser blocked copying.", "error", "Copy failed"); }
        }),
        actionButton("trash-2", "Delete", () => {
          state.history = state.history.filter((x) => x.id !== h.id);
          saveHistory(); renderHistory();
        }),
      );
      li.append(meta, src, tgt, stats, actions);
      el.historyList.appendChild(li);
    }
    icons();
  }

  function reuseHistory(h) {
    if (!supported(h.source_lang, h.target_lang)) {
      notify("The current model does not support this language direction.", "error", "Cannot reuse");
      return;
    }
    el.srcLang.value = h.source_lang; el.tgtLang.value = h.target_lang;
    syncOptions("source");
    el.srcText.value = h.source_text;
    setOutput(h.translated_text);
    // values recorded from the original model response, labelled as such
    setMetrics(h.confidence_score, h.inference_latency_ms);
    setStatus("success", `From history · ${relativeTime(h.created_at)}`);
    state.last = { source_text: h.source_text, source_lang: h.source_lang, target_lang: h.target_lang, translated_text: h.translated_text };
    updateCounter(); updateButtons();
    document.getElementById("workspace").scrollIntoView({ behavior: "smooth", block: "start" });
    el.srcText.focus({ preventScroll: true });
  }

  el.clearHistory.addEventListener("click", () => {
    if (!state.history.length) return;
    if (!window.confirm("Delete all saved translations from this browser?")) return;
    state.history = []; saveHistory(); renderHistory();
    notify("Translation history cleared.", "success");
  });

  // ----------------------------------------------------------------- feedback
  function openFeedback() {
    fillSelect(el.fbSrcLang, state.last ? state.last.source_lang : el.srcLang.value);
    fillSelect(el.fbTgtLang, state.last ? state.last.target_lang : el.tgtLang.value);
    el.fbSrcText.value = state.last ? state.last.source_text : el.srcText.value.trim();
    el.fbGenerated.value = state.last ? state.last.translated_text : "";
    el.fbCorrected.value = ""; el.fbExplanation.value = "";
    clearFieldErrors(); setFbMessage("", "");
    el.dialog.showModal();
    (el.fbSrcText.value && el.fbGenerated.value ? el.fbCorrected : el.fbSrcText).focus();
  }

  function setFbMessage(text, kind) {
    el.fbMessage.textContent = text;
    el.fbMessage.className = `alert ${kind ? `alert--${kind}` : ""}`;
    el.fbMessage.hidden = !text;
  }

  function clearFieldErrors() {
    el.form.querySelectorAll(".field.has-error").forEach((f) => f.classList.remove("has-error"));
    el.form.querySelectorAll(".field-error").forEach((e) => e.remove());
  }

  function fieldError(input, msg) {
    const field = input.closest(".field");
    field.classList.add("has-error");
    const p = document.createElement("p");
    p.className = "field-error"; p.textContent = msg;
    field.appendChild(p);
  }

  function validateFeedback() {
    clearFieldErrors();
    let ok = true;
    const required = [[el.fbSrcText, "Enter the source text."], [el.fbGenerated, "Enter the generated translation."], [el.fbCorrected, "Enter the corrected translation."]];
    for (const [input, msg] of required) if (!input.value.trim()) { fieldError(input, msg); ok = false; }
    if (el.fbSrcLang.value === el.fbTgtLang.value) { fieldError(el.fbTgtLang, "Choose a language different from the source."); ok = false; }
    if (ok && el.fbCorrected.value.trim() === el.fbGenerated.value.trim()) { fieldError(el.fbCorrected, "The correction is identical to the generated translation."); ok = false; }
    return ok;
  }

  async function submitFeedback(e) {
    e.preventDefault();
    setFbMessage("", "");
    if (!validateFeedback()) { setFbMessage("Please fix the highlighted fields.", "error"); return; }
    el.fbSubmit.disabled = true;
    el.fbSubmit.classList.add("is-loading");
    setIcon(el.fbSubmit, "loader-circle");
    try {
      const data = await postJSON("/feedback", {
        source_text: el.fbSrcText.value.trim(),
        source_lang: el.fbSrcLang.value,
        target_lang: el.fbTgtLang.value,
        generated_translation: el.fbGenerated.value.trim(),
        corrected_translation: el.fbCorrected.value.trim(),
        explanation: el.fbExplanation.value.trim() || null,
      });
      setFbMessage(`${data.message} (Report #${data.feedback_id})`, "success");
      notify("Thank you! Your correction was saved.", "success", "Feedback sent");
      el.fbCorrected.value = ""; el.fbExplanation.value = "";
    } catch (err) {
      setFbMessage(err.message, "error");
    } finally {
      el.fbSubmit.disabled = false;
      el.fbSubmit.classList.remove("is-loading");
      setIcon(el.fbSubmit, "send");
    }
  }

  // ------------------------------------------------------------------- wiring
  el.srcText.addEventListener("input", () => { updateCounter(); updateButtons(); });
  el.srcText.addEventListener("paste", (e) => {
    const pasted = (e.clipboardData || window.clipboardData)?.getData("text") || "";
    const selected = el.srcText.selectionEnd - el.srcText.selectionStart;
    if (el.srcText.value.length - selected + pasted.length > MAX_CHARS) {
      notify(`Only the first ${MAX_CHARS.toLocaleString()} characters were kept.`, "info", "Text shortened");
      limitWarned = true;
    }
  });
  el.srcText.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") translate(); });
  el.srcLang.addEventListener("change", () => syncOptions("source"));
  el.tgtLang.addEventListener("change", () => syncOptions("target"));
  el.translate.addEventListener("click", translate);
  el.copy.addEventListener("click", copyOutput);
  el.clear.addEventListener("click", () => {
    el.srcText.value = ""; setOutput(""); state.last = null;
    resetMetrics(); updateCounter(); updateButtons(); el.srcText.focus();
  });
  el.swap.addEventListener("click", () => {
    const s = el.srcLang.value, t = el.tgtLang.value;
    if (!supported(t, s)) return;
    el.srcLang.value = t; el.tgtLang.value = s;
    if (!el.tgtText.classList.contains("is-empty")) { el.srcText.value = el.tgtText.textContent; setOutput(""); state.last = null; resetMetrics(); }
    syncOptions("source"); updateCounter(); updateButtons();
  });
  el.openFb.addEventListener("click", openFeedback);
  el.form.addEventListener("submit", submitFeedback);
  el.dialog.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", () => el.dialog.close()));
  el.dialog.addEventListener("click", (e) => { if (e.target === el.dialog) el.dialog.close(); });

  // init
  applyTheme(storageGet(THEME_KEY) === "dark" ? "dark" : "light");
  fillSelect(el.srcLang, "eng"); fillSelect(el.tgtLang, "bcl_South");
  syncOptions("source"); updateCounter(); updateButtons(); setOutput(""); icons();
  loadHistory(); renderHistory();
  loadLanguages();
  pollHealth();
  setInterval(() => { if (state.history.length) renderHistory(); }, 60000);   // keep "x min ago" fresh
})();
