/**
 * SignaturePad — captures a real-time dual-modality signature sample from a
 * single signing action: the rendered canvas (static image) AND the raw
 * pointer-event stroke sequence (x, y, pressure, tilt, timestamp — the
 * dynamic modality), matching the two input types the backend pipeline
 * expects (sigverify.preprocessing.image_preprocess / stroke_preprocess).
 *
 * Uses the Pointer Events API so pressure/tilt come through natively for
 * stylus input, while still working with mouse/touch (pressure defaults to
 * 0.5 while a button/contact is active, per spec).
 */
class SignaturePad {
  constructor(canvas, { strokeColor = "#00fff2", glow = true, onChange = null } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.strokeColor = strokeColor;
    this.glow = glow;
    this.onChange = onChange;

    this.points = [];          // flat list across all strokes, chronological
    this.strokes = [];         // array of point-index ranges, for redraw
    this._currentStroke = null;
    this._drawing = false;
    this._startTime = null;
    this._lastPoint = null;

    this._resizeToDisplaySize();
    window.addEventListener("resize", () => this._resizeToDisplaySize());

    canvas.addEventListener("pointerdown", (e) => this._onPointerDown(e));
    canvas.addEventListener("pointermove", (e) => this._onPointerMove(e));
    canvas.addEventListener("pointerup", (e) => this._onPointerUp(e));
    canvas.addEventListener("pointercancel", (e) => this._onPointerUp(e));
    canvas.addEventListener("pointerleave", (e) => { if (this._drawing) this._onPointerUp(e); });
  }

  _resizeToDisplaySize() {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const targetW = Math.round(rect.width * dpr);
    const targetH = Math.round(rect.height * dpr);
    const oldW = this.canvas.width;
    const oldH = this.canvas.height;
    if (oldW !== targetW || oldH !== targetH) {
      // Recorded points are in the OLD canvas's pixel space. Without rescaling
      // them, a mid-session resize (responsive layout, window resize) would
      // redraw the signature shifted/distorted relative to the new box.
      if (oldW > 0 && oldH > 0 && this.points.length > 0) {
        const scaleX = targetW / oldW;
        const scaleY = targetH / oldH;
        for (const p of this.points) {
          p.x *= scaleX;
          p.y *= scaleY;
        }
      }
      this.canvas.width = targetW;
      this.canvas.height = targetH;
      this._redraw();
    }
  }

  _localPoint(e) {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    return {
      x: (e.clientX - rect.left) * dpr,
      y: (e.clientY - rect.top) * dpr,
      pressure: e.pressure && e.pressure > 0 ? e.pressure : 0.5,
      tiltX: e.tiltX || 0,
      tiltY: e.tiltY || 0,
      pointerType: e.pointerType || "mouse",
      t: performance.now(),
    };
  }

  _onPointerDown(e) {
    e.preventDefault();
    this.canvas.setPointerCapture(e.pointerId);
    this._drawing = true;
    if (this._startTime === null) this._startTime = performance.now();

    const p = this._localPoint(e);
    this._currentStroke = { start: this.points.length, end: this.points.length };
    this.points.push(p);
    this._lastPoint = p;
    this._paintDot(p);
    this._emitChange();
  }

  _onPointerMove(e) {
    if (!this._drawing) return;
    e.preventDefault();
    const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (const evt of events) {
      const p = this._localPoint(evt);
      this.points.push(p);
      this._currentStroke.end = this.points.length - 1;
      this._paintSegment(this._lastPoint, p);
      this._lastPoint = p;
    }
    this._emitChange();
  }

  _onPointerUp(e) {
    if (!this._drawing) return;
    this._drawing = false;
    if (this._currentStroke) this.strokes.push(this._currentStroke);
    this._currentStroke = null;
    this._emitChange();
  }

  _paintDot(p) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.fillStyle = this.strokeColor;
    if (this.glow) { ctx.shadowColor = this.strokeColor; ctx.shadowBlur = 8; }
    ctx.arc(p.x, p.y, 1.6 + p.pressure * 1.4, 0, Math.PI * 2);
    ctx.fill();
  }

  _paintSegment(a, b) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.strokeStyle = this.strokeColor;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 1.4 + b.pressure * 2.6;
    if (this.glow) { ctx.shadowColor = this.strokeColor; ctx.shadowBlur = 9; }
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  _redraw() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    for (const stroke of this.strokes) {
      for (let i = stroke.start; i < stroke.end; i++) {
        this._paintSegment(this.points[i], this.points[i + 1]);
      }
    }
  }

  clear() {
    this.points = [];
    this.strokes = [];
    this._currentStroke = null;
    this._startTime = null;
    this._lastPoint = null;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this._emitChange();
  }

  isEmpty() {
    return this.points.length === 0;
  }

  /** Telemetry summary for the live HUD strip. */
  getTelemetry() {
    const n = this.points.length;
    if (n === 0) {
      return { points: 0, durationSec: 0, avgPressure: null, avgVelocity: null, inputType: "—" };
    }
    const duration = (this.points[n - 1].t - this.points[0].t) / 1000;
    const avgPressure = this.points.reduce((s, p) => s + p.pressure, 0) / n;

    let velocitySum = 0;
    let velocityCount = 0;
    for (let i = 1; i < n; i++) {
      const dt = (this.points[i].t - this.points[i - 1].t) / 1000;
      if (dt <= 0) continue;
      const dx = this.points[i].x - this.points[i - 1].x;
      const dy = this.points[i].y - this.points[i - 1].y;
      velocitySum += Math.sqrt(dx * dx + dy * dy) / dt;
      velocityCount++;
    }
    const avgVelocity = velocityCount > 0 ? velocitySum / velocityCount : null;
    const inputType = this.points[n - 1].pointerType;

    return { points: n, durationSec: duration, avgPressure, avgVelocity, inputType };
  }

  /** Raw stroke dict matching sigverify.preprocessing.stroke_preprocess's expected input. */
  getStrokeData() {
    const t0 = this.points.length ? this.points[0].t : 0;
    return {
      x: this.points.map((p) => p.x),
      y: this.points.map((p) => p.y),
      pressure: this.points.map((p) => p.pressure),
      tilt_x: this.points.map((p) => p.tiltX),
      tilt_y: this.points.map((p) => p.tiltY),
      timestamp: this.points.map((p) => (p.t - t0) / 1000),
    };
  }

  /** Renders the current canvas to a PNG Blob (the static-image modality). */
  toBlob() {
    return new Promise((resolve) => this.canvas.toBlob(resolve, "image/png"));
  }

  _emitChange() {
    if (this.onChange) this.onChange(this.getTelemetry());
  }
}
