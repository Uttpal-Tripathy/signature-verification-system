/**
 * SIGNUM Animation Engine — a small, dependency-free HTML5 Canvas 2D animation
 * system. One shared requestAnimationFrame loop drives every visual effect in
 * the console (instead of each effect running its own loop): the ambient
 * particle-network background, the scan-sweep overlay while a verification is
 * in flight, and the decision-reveal burst (a particle effect timed to the
 * accept/review/reject result). Respects `prefers-reduced-motion` — when set,
 * `addLayer` is a no-op and the console falls back to the static CSS
 * background/badge styling already in theme.css.
 */
(() => {
  class AnimationEngine {
    constructor() {
      this.layers = new Map(); // name -> layer with .update(dt) and optional .finished/.destroy()
      this._running = false;
      this._lastTs = 0;
      this._reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    }

    addLayer(name, layer) {
      if (this._reducedMotion) return null;
      this.removeLayer(name);
      this.layers.set(name, layer);
      if (!this._running) this._start();
      return layer;
    }

    removeLayer(name) {
      const layer = this.layers.get(name);
      if (layer?.destroy) layer.destroy();
      this.layers.delete(name);
    }

    _start() {
      this._running = true;
      this._lastTs = performance.now();
      const loop = (ts) => {
        if (this.layers.size === 0) {
          this._running = false;
          return;
        }
        const dt = Math.min((ts - this._lastTs) / 1000, 0.05);
        this._lastTs = ts;
        const finished = [];
        for (const [name, layer] of this.layers) {
          layer.update(dt);
          if (layer.finished) finished.push(name);
        }
        for (const name of finished) this.removeLayer(name);
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }
  }

  function fitCanvasToElement(canvas, el) {
    const rect = el.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, rect.width * dpr);
    canvas.height = Math.max(1, rect.height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: rect.width, h: rect.height };
  }

  // ---------------------------------------------------------------- ambient background
  class ParticleNetworkLayer {
    constructor(canvas, { count = 55, color = "0,255,242", linkDistance = 130, speed = 16 } = {}) {
      this.canvas = canvas;
      this.color = color;
      this.linkDistance = linkDistance;
      const fit = fitCanvasToElement(canvas, document.documentElement);
      this.ctx = fit.ctx;
      this.w = fit.w;
      this.h = fit.h;
      this._resizeHandler = () => {
        const r = fitCanvasToElement(canvas, document.documentElement);
        this.ctx = r.ctx; this.w = r.w; this.h = r.h;
      };
      window.addEventListener("resize", this._resizeHandler);

      this.particles = Array.from({ length: count }, () => ({
        x: Math.random() * this.w,
        y: Math.random() * this.h,
        vx: (Math.random() - 0.5) * speed,
        vy: (Math.random() - 0.5) * speed,
        r: 1 + Math.random() * 1.4,
      }));
    }

    update(dt) {
      const { ctx, w, h, particles } = this;
      ctx.clearRect(0, 0, w, h);

      for (const p of particles) {
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        if (p.x <= 0 || p.x >= w) p.vx *= -1;
        if (p.y <= 0 || p.y >= h) p.vy *= -1;
        p.x = Math.max(0, Math.min(w, p.x));
        p.y = Math.max(0, Math.min(h, p.y));
      }

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const a = particles[i], b = particles[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < this.linkDistance) {
            ctx.strokeStyle = `rgba(${this.color},${(1 - dist / this.linkDistance) * 0.22})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      ctx.shadowColor = `rgba(${this.color},0.8)`;
      ctx.shadowBlur = 4;
      ctx.fillStyle = `rgba(${this.color},0.55)`;
      for (const p of particles) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.shadowBlur = 0;
    }

    destroy() {
      window.removeEventListener("resize", this._resizeHandler);
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  // ---------------------------------------------------------------- verification-in-flight sweep
  class ScanSweepLayer {
    constructor(canvas, { color = "0,255,242", cycleSeconds = 1.4 } = {}) {
      this.canvas = canvas;
      this.color = color;
      this.cycleSeconds = cycleSeconds;
      this.t = 0;
      const fit = fitCanvasToElement(canvas, canvas.parentElement);
      this.ctx = fit.ctx; this.w = fit.w; this.h = fit.h;
      this._resizeHandler = () => {
        const r = fitCanvasToElement(canvas, canvas.parentElement);
        this.ctx = r.ctx; this.w = r.w; this.h = r.h;
      };
      window.addEventListener("resize", this._resizeHandler);
    }

    update(dt) {
      this.t += dt;
      const { ctx, w, h } = this;
      ctx.clearRect(0, 0, w, h);
      const progress = (this.t % this.cycleSeconds) / this.cycleSeconds;
      const x = progress * w;

      const grad = ctx.createLinearGradient(x - 70, 0, x + 70, 0);
      grad.addColorStop(0, `rgba(${this.color},0)`);
      grad.addColorStop(0.5, `rgba(${this.color},0.28)`);
      grad.addColorStop(1, `rgba(${this.color},0)`);
      ctx.fillStyle = grad;
      ctx.fillRect(x - 70, 0, 140, h);

      ctx.strokeStyle = `rgba(${this.color},0.95)`;
      ctx.lineWidth = 2;
      ctx.shadowColor = `rgba(${this.color},1)`;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    destroy() {
      window.removeEventListener("resize", this._resizeHandler);
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  // ---------------------------------------------------------------- decision-reveal burst
  class BurstLayer {
    constructor(canvas, { color = "57,255,136", count = 60 } = {}) {
      this.canvas = canvas;
      this.color = color;
      const fit = fitCanvasToElement(canvas, canvas.parentElement);
      this.ctx = fit.ctx; this.w = fit.w; this.h = fit.h;
      const cx = this.w / 2, cy = this.h / 2;
      this.particles = Array.from({ length: count }, () => {
        const angle = Math.random() * Math.PI * 2;
        const speed = 50 + Math.random() * 200;
        return { x: cx, y: cy, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed, life: 1, r: 1.5 + Math.random() * 2 };
      });
    }

    update(dt) {
      const { ctx, w, h } = this;
      ctx.clearRect(0, 0, w, h);
      let alive = false;
      ctx.fillStyle = `rgb(${this.color})`;
      ctx.shadowColor = `rgb(${this.color})`;
      ctx.shadowBlur = 6;
      for (const p of this.particles) {
        if (p.life <= 0) continue;
        alive = true;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.vy += 110 * dt;
        p.life -= dt * 0.9;
        ctx.globalAlpha = Math.max(0, p.life);
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      ctx.shadowBlur = 0;
      this.finished = !alive;
    }

    destroy() {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  window.SignumFX = { AnimationEngine, ParticleNetworkLayer, ScanSweepLayer, BurstLayer };
})();
