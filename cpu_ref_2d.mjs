// CPU reference of the same SPH formulation as in the shader.
// Purpose: check stability and parameter scaling before going to the GPU.
const PI = Math.PI;

//
// Two uses:
//   node cpu_ref.mjs        -> energy table (see below)
//   fluid.html?verify       -> imports makeSim and compares step by step
//                              with the GPU (for that: fixed dt, setState, state)
export function makeSim({ N, aspect = 1.78, H = 10, pressure, nearPressure, viscosity, gravity = 9, symmetric = true, restitution = 0.2, dt: dtFixed }) {
  const bx = H * aspect / 2, by = H / 2;
  const blockW = (2 * bx) * 0.5, blockH = H * 0.62;
  const cols = Math.max(1, Math.round(Math.sqrt(N * blockW / blockH)));
  const rows = Math.ceil(N / cols);
  const s = Math.min(blockW / cols, blockH / rows);
  const h = s * 2.8, mass = 1;

  let rho0 = 0;
  const R = Math.ceil(h / s);
  for (let y = -R; y <= R; y++) for (let x = -R; x <= R; x++) {
    const d = Math.hypot(x * s, y * s);
    if (d < h) rho0 += mass * (h - d) ** 2 * 6 / (PI * h ** 4);
  }

  // scale-invariant conversion: slider value -> uniform
  const kP = symmetric ? pressure * h / 2 : pressure * h / rho0;
  const kN = symmetric ? nearPressure * h / 2 : nearPressure * h / rho0;
  const kV = viscosity * s * s;

  const px = new Float64Array(N), py = new Float64Array(N);
  const vx = new Float64Array(N), vy = new Float64Array(N);
  const qx = new Float64Array(N), qy = new Float64Array(N);
  const den = new Float64Array(N), nea = new Float64Array(N);

  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5);
  for (let i = 0; i < N; i++) {
    px[i] = -blockW / 2 + ((i % cols) + 0.5) * s + rnd() * s * 0.25;
    py[i] = -blockH / 2 + (Math.floor(i / cols) + 0.5) * s + rnd() * s * 0.25;
  }

  const gw = Math.max(1, Math.ceil(2 * bx / h)), gh = Math.max(1, Math.ceil(2 * by / h));
  const head = new Int32Array(gw * gh), next = new Int32Array(N);

  const W2 = (d) => d >= h ? 0 : (h - d) ** 2 * 6 / (PI * h ** 4);
  const dW2 = (d) => d >= h ? 0 : -(h - d) * 12 / (PI * h ** 4);
  const W3 = (d) => d >= h ? 0 : (h - d) ** 3 * 10 / (PI * h ** 5);
  const dW3 = (d) => d >= h ? 0 : -((h - d) ** 2) * 30 / (PI * h ** 5);
  const Wv = (d) => d >= h ? 0 : (h * h - d * d) ** 3 * 4 / (PI * h ** 8);

    const r = s * 0.42, maxSpeed = 30;
  const c = Math.sqrt(Math.max(kP, 1e-9));          // speed of sound
  const dt = dtFixed ?? Math.min(1 / 120, 0.25 * h / (c + 4));  // CFL


  function step() {
    for (let i = 0; i < N; i++) {
      vy[i] -= gravity * dt;
      qx[i] = px[i] + vx[i] * dt; qy[i] = py[i] + vy[i] * dt;
    }
    head.fill(-1);
    for (let i = 0; i < N; i++) {
      const cx = Math.floor((qx[i] + bx) / h), cy = Math.floor((qy[i] + by) / h);
      if (cx < 0 || cy < 0 || cx >= gw || cy >= gh) continue;
      const c = cy * gw + cx; next[i] = head[c]; head[c] = i;
    }
    const each = (i, fn) => {
      const cx = Math.floor((qx[i] + bx) / h), cy = Math.floor((qy[i] + by) / h);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const X = cx + dx, Y = cy + dy;
        if (X < 0 || Y < 0 || X >= gw || Y >= gh) continue;
        for (let j = head[Y * gw + X]; j !== -1; j = next[j]) fn(j);
      }
    };
    for (let i = 0; i < N; i++) {
      let D = 0, Nd = 0;
      each(i, (j) => { const d = Math.hypot(qx[i] - qx[j], qy[i] - qy[j]); D += mass * W2(d); Nd += mass * W3(d); });
      den[i] = D; nea[i] = Nd;
    }
    for (let i = 0; i < N; i++) {
      if (den[i] <= 1e-9) continue;
      const pi = (den[i] - rho0) * kP, ni = nea[i] * kN;
      let ax = 0, ay = 0, sx = 0, sy = 0;
      each(i, (j) => {
        if (j === i || den[j] <= 1e-9) return;
        const ox = qx[j] - qx[i], oy = qy[j] - qy[i], d = Math.hypot(ox, oy);
        if (d >= h) return;
        const ux = d > 1e-9 ? ox / d : 0, uy = d > 1e-9 ? oy / d : 1;
        const pj = (den[j] - rho0) * kP, nj = nea[j] * kN;
        let f;
        if (symmetric) {
          // momentum-conserving: the term is antisymmetric in i and j
          f = dW2(d) * mass * (pi / (den[i] * den[i]) + pj / (den[j] * den[j]))
            + dW3(d) * mass * (ni / Math.max(nea[i] * nea[i], 1e-9) + nj / Math.max(nea[j] * nea[j], 1e-9));
        } else {
          f = dW2(d) * (pi + pj) * 0.5 * mass / den[j]
            + dW3(d) * (ni + nj) * 0.5 * mass / Math.max(nea[j], 1e-9);
        }
        ax += ux * f; ay += uy * f;
        const w = Wv(d); sx += (vx[j] - vx[i]) * w; sy += (vy[j] - vy[i]) * w;
      });
      vx[i] += (ax + sx * kV) * dt; vy[i] += (ay + sy * kV) * dt;
      const sp2 = Math.hypot(vx[i], vy[i]);
      if (sp2 > maxSpeed) { vx[i] *= maxSpeed / sp2; vy[i] *= maxSpeed / sp2; }
    }
    for (let i = 0; i < N; i++) {
      px[i] += vx[i] * dt; py[i] += vy[i] * dt;
      if (px[i] < -bx + r) { px[i] = -bx + r; vx[i] *= -restitution; }
      if (px[i] >  bx - r) { px[i] =  bx - r; vx[i] *= -restitution; }
      if (py[i] < -by + r) { py[i] = -by + r; vy[i] *= -restitution; }
      if (py[i] >  by - r) { py[i] =  by - r; vy[i] *= -restitution; }
    }
  }

  return {
    step, rho0, h, s, dt,
    // set state from outside: pos and vel as [x0, y0, x1, y1, ...]
    setState(pos, vel) {
      for (let i = 0; i < N; i++) {
        px[i] = pos[2 * i]; py[i] = pos[2 * i + 1];
        vx[i] = vel[2 * i]; vy[i] = vel[2 * i + 1];
      }
    },
    state: { px, py, vx, vy, den, nea },
    ke() { let e = 0; for (let i = 0; i < N; i++) e += 0.5 * (vx[i]*vx[i] + vy[i]*vy[i]); return e / N; },
    stats() {
      let nan = 0, maxV = 0, sum = 0, cnt = 0, clamped = 0;
      for (let i = 0; i < N; i++) {
        if (!Number.isFinite(px[i]) || !Number.isFinite(vx[i])) { nan++; continue; }
        const v = Math.hypot(vx[i], vy[i]);
        if (v > maxV) maxV = v;
        if (v > 29.9) clamped++;
        if (den[i] > rho0 * 0.5) { sum += den[i]; cnt++; }   // interior only
      }
      return { nan, maxV, meanRho: cnt ? sum / cnt / rho0 : 0, clamped };
    },
  };
}



// The table only runs when called directly with node, not when imported in the browser.
const runAsScript = typeof process !== 'undefined' && process.argv?.[1]
  && import.meta.url === new URL(process.argv[1], 'file://').href;

if (runAsScript) {
  const SIM_SECONDS = 6;
  console.log('What takes the energy out?  (symmetric, pressure 2000, gravity 8, N=1500)\n');
  console.log('  visc  wall   near |  t=1s     t=2s     t=4s     t=6s  | rho/rho0');
  console.log('  ' + '-'.repeat(60));

  for (const cfg of [
    { viscosity:  8, restitution: 0.2, nearPressure: 8 },
    { viscosity: 30, restitution: 0.2, nearPressure: 8 },
    { viscosity: 30, restitution: 0.0, nearPressure: 8 },
    { viscosity: 30, restitution: 0.0, nearPressure: 0 },
  ]) {
    const sim = makeSim({ N: 1500, pressure: 2000, gravity: 8, symmetric: true, ...cfg });
    const total = Math.ceil(SIM_SECONDS / sim.dt);
    const marks = [1, 2, 4, 6].map(t => Math.round(t / sim.dt));
    const out = [];
    for (let k = 1; k <= total; k++) { sim.step(); if (marks.includes(k)) out.push(sim.ke()); }
    console.log('  ' + String(cfg.viscosity).padStart(4) + '  ' + cfg.restitution.toFixed(1).padStart(4) +
      '  ' + String(cfg.nearPressure).padStart(5) + ' |' +
      out.map(v => v.toFixed(3).padStart(7)).join('  ') + '  | ' + sim.stats().meanRho.toFixed(3));
  }
}
