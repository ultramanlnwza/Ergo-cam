// session.js - alert rule + session statistics. Pure logic (no browser APIs),
// so it can be unit-tested in Node. See tests/run-tests.mjs.
//
// ALERT RULE (configurable prototype rule, NOT a validated clinical cut-off):
//   - A frame is "above threshold" when it is valid AND neck > threshold (strictly greater).
//   - The consecutive timer grows only while frames are valid and above threshold.
//   - The timer resets to 0 when neck <= threshold, or the measurement is unreliable.
//   - One alert fires when the timer first reaches `duration` seconds. It is then
//     "latched": no further alert fires until the timer has been reset.
//
// Time is accumulated in integer-ish milliseconds to avoid floating point drift.

const MAX_DT_MS = 500; // cap a single frame gap (e.g. tab stall) so it cannot inflate statistics

export class Session {
  /**
   * @param settings object with numeric `threshold` (deg) and `duration` (s).
   *                 Held by reference, so changing it takes effect immediately.
   * @param opts     { alpha: smoothing 0..1 (1 = no smoothing), sampleEvery: seconds between chart samples }
   */
  constructor(settings, opts = {}) {
    this.settings = settings;
    this.alpha = opts.alpha ?? 0.4;
    this.sampleEvery = opts.sampleEvery ?? 0.5;
    this.start(0);
  }

  start(nowMs, wallMs = Date.now()) {
    this.startMs = nowMs;
    this.endMs = null;
    this.wallStart = wallMs;
    this.lastMs = null;
    this.consecutiveMs = 0;
    this.latched = false;
    this.totalAboveMs = 0;
    this.alerts = [];
    this.samples = [];
    this.lastSampleT = -Infinity;
    this.frames = 0;
    this.validFrames = 0;
    this.sumNeck = 0;
    this.sumTrunk = 0;
    this.maxNeck = null;
    this.smNeck = null;
    this.smTrunk = null;
  }

  /** Call when tracking was interrupted (e.g. camera switch): breaks continuity. */
  markGap() {
    this.lastMs = null;
    this.consecutiveMs = 0;
    this.latched = false;
    this.smNeck = null;
    this.smTrunk = null;
  }

  /**
   * Process one measurement.
   * @param m      { valid, neck, trunk } from computeMeasurements()
   * @param nowMs  monotonic time in ms (performance.now())
   */
  frame(m, nowMs, wallMs = Date.now()) {
    let dtMs = this.lastMs == null ? 0 : nowMs - this.lastMs;
    this.lastMs = nowMs;
    dtMs = Math.min(Math.max(dtMs, 0), MAX_DT_MS);
    const t = (nowMs - this.startMs) / 1000;
    this.frames++;

    const valid = !!(m && m.valid && Number.isFinite(m.neck) && Number.isFinite(m.trunk));
    let neck = null;
    let trunk = null;
    let above = false;
    let alertTriggered = false;

    if (valid) {
      const a = this.alpha;
      this.smNeck = this.smNeck == null ? m.neck : a * m.neck + (1 - a) * this.smNeck;
      this.smTrunk = this.smTrunk == null ? m.trunk : a * m.trunk + (1 - a) * this.smTrunk;
      neck = this.smNeck;
      trunk = this.smTrunk;

      this.validFrames++;
      this.sumNeck += neck;
      this.sumTrunk += trunk;
      if (this.maxNeck == null || neck > this.maxNeck) this.maxNeck = neck;

      above = neck > this.settings.threshold;
      if (above) {
        this.consecutiveMs += dtMs;
        this.totalAboveMs += dtMs;
        if (!this.latched && this.consecutiveMs >= this.settings.duration * 1000) {
          this.latched = true;
          alertTriggered = true;
          this.alerts.push({ n: this.alerts.length + 1, elapsedSec: t, wallMs, angle: neck });
        }
      } else {
        this.consecutiveMs = 0;
        this.latched = false;
      }
    } else {
      // Unreliable measurement: reset timer and smoothing, do not count anything.
      this.consecutiveMs = 0;
      this.latched = false;
      this.smNeck = null;
      this.smTrunk = null;
    }

    if (t - this.lastSampleT >= this.sampleEvery) {
      this.lastSampleT = t;
      this.samples.push({ t, neck, trunk, valid });
    }

    return {
      valid,
      neck,
      trunk,
      above,
      consecutiveSec: this.consecutiveMs / 1000,
      alertTriggered,
      alertActive: this.latched,
      elapsedSec: t,
      totalAboveSec: this.totalAboveMs / 1000,
      alertCount: this.alerts.length,
    };
  }

  stop(nowMs) {
    this.endMs = nowMs;
  }

  summary() {
    const end = this.endMs ?? this.lastMs ?? this.startMs;
    const durationSec = Math.max(0, (end - this.startMs) / 1000);
    const n = this.validFrames;
    return {
      startedAt: this.wallStart,
      durationSec,
      alertCount: this.alerts.length,
      alerts: this.alerts.slice(),
      timeAboveSec: this.totalAboveMs / 1000,
      avgNeck: n ? this.sumNeck / n : null,
      avgTrunk: n ? this.sumTrunk / n : null,
      maxNeck: this.maxNeck,
      validPct: this.frames ? (100 * n) / this.frames : 0,
      samples: this.samples.slice(),
      threshold: this.settings.threshold,
      requiredDuration: this.settings.duration,
    };
  }
}
