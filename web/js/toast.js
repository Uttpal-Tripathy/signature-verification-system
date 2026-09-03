/**
 * SIGNUM Toast Notifications — animated, auto-dismissing alert banners.
 * Fires for the current user's own verification result (app.js calls this
 * immediately from the /api/verify response, using the alert_severity/
 * alert_message the backend's classify_alert() already computed — see
 * src/sigverify/alerts/broker.py). Deliberately NOT also fired from the
 * Live Monitor's WebSocket feed (monitor.js) for the same event, which would
 * double-toast the very alert this tab just triggered; monitor.js's own
 * sound/desktop-notification handling covers alerts from *other* clients.
 */
(() => {
  function ensureContainer() {
    let el = document.getElementById("toastContainer");
    if (!el) {
      el = document.createElement("div");
      el.id = "toastContainer";
      el.className = "toast-container";
      document.body.appendChild(el);
    }
    return el;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str ?? "";
    return div.innerHTML;
  }

  const ICONS = { info: "✓", warning: "⚠", critical: "⛔" };

  function showToast({ severity = "info", title = "", message = "", durationMs = 6000 } = {}) {
    const container = ensureContainer();
    const el = document.createElement("div");
    el.className = `toast toast--${severity}`;
    el.setAttribute("role", "status");
    el.innerHTML = `
      <span class="toast__icon">${ICONS[severity] || ICONS.info}</span>
      <span class="toast__body">
        <span class="toast__title">${escapeHtml(title)}</span>
        <span class="toast__message">${escapeHtml(message)}</span>
      </span>
      <button class="toast__close" type="button" aria-label="Dismiss">&times;</button>
    `;
    container.appendChild(el);
    // Two rAFs so the initial (pre-transition) state actually paints before
    // the "in" class is added -- a single rAF can still coalesce with the
    // append on some browsers and skip the slide-in transition entirely.
    requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add("toast--in")));

    let dismissed = false;
    const dismiss = () => {
      if (dismissed) return;
      dismissed = true;
      el.classList.remove("toast--in");
      el.classList.add("toast--out");
      setTimeout(() => el.remove(), 350);
    };
    el.querySelector(".toast__close").addEventListener("click", dismiss);
    if (durationMs > 0) setTimeout(dismiss, durationMs);
    return dismiss;
  }

  window.SignumToast = { showToast };
})();
