// angles.js - pure math, no browser APIs (so it can be unit-tested in Node).
//
// DEFINITIONS (important for the judges / documentation)
// -------------------------------------------------------
// MediaPipe gives landmarks in NORMALISED image coordinates:
//   x: 0 (left edge) -> 1 (right edge)
//   y: 0 (TOP edge)  -> 1 (bottom edge)   <-- y grows DOWNWARD
// Because the frame is not square, we first scale x by (width/height) so that
// one unit in x equals one unit in y (otherwise angles are distorted).
// We then flip y so "up" is positive:  up = -(y2 - y1).
//
// Reference axis: the TRUE VERTICAL (straight up in the image).
//
// Estimated Neck Inclination  = angle between vertical and the vector
//     shoulder -> ear    (ear = upper point, shoulder = lower point)
// Estimated Trunk Flexion     = angle between vertical and the vector
//     hip -> shoulder
//
// Sign convention: the person's FACING direction is "forward".
//   positive angle = upper point is forward of the lower point (leaning forward)
//   negative angle = upper point is behind the lower point (leaning back)
// Facing direction comes from nose x vs ear x (nose is in front of the ear).
//
// NOTE: "neck inclination" here is a 2D camera-plane estimate, NOT a clinically
// validated cervical angle (e.g. not the craniovertebral angle).

export const VIS_MIN = 0.5;        // minimum landmark visibility to trust
export const NOSE_VIS_MIN = 0.3;   // nose only used to decide facing direction
export const FRONT_VIEW_RATIO = 0.6; // shoulder-width / torso-length above this = not a side view

export const IDX = {
  nose: 0,
  ear: { left: 7, right: 8 },
  shoulder: { left: 11, right: 12 },
  hip: { left: 23, right: 24 },
};

const vis = (p) => (p && typeof p.visibility === "number" ? p.visibility : 0);

/** Signed angle (degrees) from vertical. dx, dyUp in same units. forward = +1 / -1 */
export function angleFromVertical(dx, dyUp, forward) {
  return (Math.atan2(forward * dx, dyUp) * 180) / Math.PI;
}

/** Pick the body side (left/right) whose ear+shoulder+hip are most visible. */
export function pickSide(lm) {
  const score = (s) =>
    (vis(lm[IDX.ear[s]]) + vis(lm[IDX.shoulder[s]]) + vis(lm[IDX.hip[s]])) / 3;
  const l = score("left");
  const r = score("right");
  return l >= r ? { side: "left", score: l } : { side: "right", score: r };
}

/**
 * @param lm     array of 33 landmarks {x,y,visibility}
 * @param aspect video width / video height
 * @returns {valid, reason, neck, trunk, side, facing}
 */
export function computeMeasurements(lm, aspect) {
  if (!lm || lm.length < 33) return { valid: false, reason: "No person detected" };

  const { side } = pickSide(lm);
  const ear = lm[IDX.ear[side]];
  const sh = lm[IDX.shoulder[side]];
  const hip = lm[IDX.hip[side]];
  const nose = lm[IDX.nose];

  if (vis(ear) < VIS_MIN || vis(sh) < VIS_MIN || vis(hip) < VIS_MIN) {
    return { valid: false, reason: "Ear, shoulder and hip must all be visible from the side" };
  }
  if (vis(nose) < NOSE_VIS_MIN) {
    return { valid: false, reason: "Face direction unclear - turn to a side view" };
  }

  const P = (p) => ({ x: p.x * aspect, y: p.y });
  const e = P(ear), s = P(sh), h = P(hip);

  // Side-view check: if BOTH shoulders are clearly visible and far apart
  // horizontally, the person is facing the camera instead of sideways.
  const lS = lm[IDX.shoulder.left], rS = lm[IDX.shoulder.right];
  if (vis(lS) >= VIS_MIN && vis(rS) >= VIS_MIN) {
    const width = Math.hypot((lS.x - rS.x) * aspect, lS.y - rS.y);
    const torso = Math.hypot(s.x - h.x, s.y - h.y);
    if (torso > 0 && width / torso > FRONT_VIEW_RATIO) {
      return { valid: false, reason: "Turn so the camera sees you from the side" };
    }
  }

  const facing = nose.x < ear.x ? -1 : 1; // image x of nose relative to ear

  const neck = angleFromVertical(e.x - s.x, -(e.y - s.y), facing);
  const trunk = angleFromVertical(s.x - h.x, -(s.y - h.y), facing);

  if (!Number.isFinite(neck) || !Number.isFinite(trunk)) {
    return { valid: false, reason: "Measurement unavailable" };
  }
  return { valid: true, neck, trunk, side, facing };
}
