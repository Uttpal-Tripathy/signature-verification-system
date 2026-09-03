/**
 * SIGNUM Live Monitor — connects to /ws/alerts and renders every verification
 * event (from ANY client hitting /api/verify, not just this page) as a live,
 * severity-colored feed, with running stat counters and optional sound/desktop
 * notifications for flagged (review/forged) events. This is what makes alerts
 * a system-wide, real-time feature rather than a per-request response field.
 */
(() => {
  const $ = (id) => document.getElementById(id);

  const MAX_FEED_ITEMS = 200;
  const stats = { total: 0, genuine: 0, review: 0, forged: 0 };

  let socket = null;
  let reconnectDelayMs = 1000;
  const MAX_RECONNECT_DELAY_MS = 15000;

  function setWsStatus(online) {
    const dot = $("wsStatusDot");
    const text = $("wsStatusText");
    if (!dot || !text) return;
    dot.className = "status-dot " + (online ? "online" : "offline");
    text.textContent = online ? "LIVE" : "RECONNECTING…";
  }

  function wsUrl() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    // Browsers' native WebSocket API can't set custom headers on the handshake, so
    // the API key travels as a query param here instead of the X-API-Key header
    // the REST calls below use (api/app.py's /ws/alerts checks it the same way).
    const key = window.SIGNUM_API_KEY || "";
    const qs = key ? `?api_key=${encodeURIComponent(key)}` : "";
    return `${proto}//${window.location.host}/ws/alerts${qs}`;
  }

  function connect() {
    socket = new WebSocket(wsUrl());
    socket.addEventListener("open", () => {
      setWsStatus(true);
      reconnectDelayMs = 1000;
    });
    socket.addEventListener("message", (event) => {
      try {
        handleAlert(JSON.parse(event.data));
      } catch {
        // malformed frame -- ignore rather than crash the feed
      }
    });
    socket.addEventListener("close", scheduleReconnect);
    socket.addEventListener("error", () => socket.close());
  }

  function scheduleReconnect() {
    setWsStatus(false);
    setTimeout(connect, reconnectDelayMs);
    reconnectDelayMs = Math.min(reconnectDelayMs * 1.6, MAX_RECONNECT_DELAY_MS);
  }

  function handleAlert(alert) {
    updateStats(alert);
    renderAlert(alert);
    if (alert.severity === "critical" || alert.severity === "warning") {
      if ($("toggleAlertSound")?.checked) playBeep(alert.severity);
      if ($("toggleAlertNotify")?.checked) notifyDesktop(alert);
    }
  }

  function updateStats(alert) {
    stats.total++;
    if (alert.decision === "Genuine" && alert.severity === "info") stats.genuine++;
    else if (alert.severity === "critical") stats.forged++;
    else stats.review++;
    $("statTotal").textContent = stats.total;
    $("statGenuine").textContent = stats.genuine;
    $("statReview").textContent = stats.review;
    $("statForged").textContent = stats.forged;
  }

  function renderAlert(alert) {
    const feed = $("alertFeed");
    const placeholder = $("alertFeedPlaceholder");
    if (placeholder) placeholder.hidden = true;

    const el = document.createElement("div");
    el.className = `alert-item alert-item--${alert.severity}`;
    const time = new Date(alert.timestamp * 1000).toTimeString().slice(0, 8);
    const who = alert.user_id ? ` · ${escapeHtml(alert.user_id)}` : "";
    el.innerHTML = `
      <span class="alert-item__dot"></span>
      <span class="alert-item__decision">${escapeHtml(alert.decision.toUpperCase())}</span>
      <span class="alert-item__message">${escapeHtml(alert.message)}${who}</span>
      <span class="alert-item__time">${time}</span>
    `;
    feed.appendChild(el);

    while (feed.children.length > MAX_FEED_ITEMS + 1) {
      // +1 accounts for the placeholder node, which stays in the DOM (hidden)
      const oldest = feed.querySelector(".alert-item");
      if (oldest) oldest.remove(); else break;
    }
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // ---------------------------------------------------------------- sound
  let audioCtx = null;
  function playBeep(severity) {
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.frequency.value = severity === "critical" ? 880 : 600;
      osc.type = "square";
      gain.gain.setValueAtTime(0.08, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.25);
      osc.connect(gain).connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.25);
      if (severity === "critical") {
        setTimeout(() => playBeep("_second"), 180);
      }
    } catch {
      // Web Audio unavailable/blocked (autoplay policy before user interaction) -- silent no-op
    }
  }

  // ---------------------------------------------------------------- desktop notifications
  function notifyDesktop(alert) {
    if (!("Notification" in window)) return;
    if (Notification.permission === "granted") {
      new Notification(`SIGNUM: ${alert.decision}`, { body: alert.message });
    } else if (Notification.permission !== "denied") {
      Notification.requestPermission();
    }
  }

  $("toggleAlertNotify")?.addEventListener("change", (e) => {
    if (e.target.checked && "Notification" in window && Notification.permission === "default") {
      Notification.requestPermission();
    }
  });

  // ---------------------------------------------------------------- backfill via REST
  // Covers the gap between page load and the WebSocket handshake completing, and
  // gives a working feed even if WebSocket is blocked by a restrictive proxy.
  async function backfill() {
    try {
      const resp = await fetch("/api/alerts/recent?limit=50", { headers: { "X-API-Key": window.SIGNUM_API_KEY || "" } });
      if (!resp.ok) return;
      const alerts = await resp.json();
      for (const alert of alerts) handleAlert(alert);
    } catch {
      // backend not reachable yet -- the WebSocket connection attempt will retry independently
    }
  }

  backfill();
  connect();
})();
