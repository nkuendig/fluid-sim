// CPU reference of the same SPH formulation as in the shader, in 3D (since M2).
// Purpose: correctness oracle for fluid.html and sandbox for parameters.
// The 2D version is kept unchanged in cpu_ref_2d.mjs (for fluid_2d.html).
//
// Two uses:
//   node cpu_ref.mjs        -> lets the river fill up and prints a table
//   fluid.html?verify       -> imports makeSim and compares step by step
//                              with the GPU (for that: fixed dt, setState, state)
const PI = Math.PI;

// The scene: a grey box with a sloped trough, three stones, an inlet and an edge.
// The same numbers are in fluid.html; verify mode passes in its own.
export const SCENE = {
  boundsMin: [-5.4, -2.4, -1.7],      // grid = the space the particles live in
  boundsMax: [ 5.4,  1.4,  1.7],
  slope: 0.1,                         // slope of the terrain plane (tangent of the angle)
  origin: [-5.0, 0.5],                // point (x, y) on the terrain plane at the upper end
  troughRadius: 1.6,                  // the trough is a cylinder cut into the terrain
  troughCenter: 1.0,                  // height of the cylinder axis above the terrain plane
  blockMin: [-5.3, -2.2, -1.6],       // the block the terrain is clipped to
  blockMax: [ 4.0,  3.0,  1.6],
  wallMin: [-5.3, -2.2, -1.6],        // back wall behind the inlet
  wallMax: [-5.0,  1.1,  1.6],
  stones: [                           // spheres: x, y, z, radius
    [-2.0, -0.25, -0.25, 0.42],
    [ 0.3, -0.45,  0.40, 0.34],
    [ 2.2, -0.68, -0.30, 0.38],
  ],
  inlet: { x: -5.0, yMin: 0.08, yMax: 0.48, zMin: -0.6, zMax: 0.6 },
  volume: 4.5,                        // nominal volume: it determines the particle spacing
  sdfEps: 0.01,                       // step size for the gradient
};

// Signed distance field of the scene: distance to the nearest solid surface,
// positive in free space, negative inside the solid.
export function makeSdf(sc) {
  const a = Math.atan(sc.slope), ca = Math.cos(a), sa = Math.sin(a);
  const box = (x, y, z, lo, hi) => {
    const qx = Math.abs(x - 0.5 * (lo[0] + hi[0])) - 0.5 * (hi[0] - lo[0]);
    const qy = Math.abs(y - 0.5 * (lo[1] + hi[1])) - 0.5 * (hi[1] - lo[1]);
    const qz = Math.abs(z - 0.5 * (lo[2] + hi[2])) - 0.5 * (hi[2] - lo[2]);
    return Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0))
         + Math.min(Math.max(qx, qy, qz), 0);
  };
  const sdf = (x, y, z) => {
    // height above the sloped terrain plane
    const hgt = (x - sc.origin[0]) * sa + (y - sc.origin[1]) * ca;
    // terrain minus the trough cylinder, clipped to the block
    const trough = sc.troughRadius - Math.hypot(hgt - sc.troughCenter, z);
    let d = Math.max(hgt, trough, box(x, y, z, sc.blockMin, sc.blockMax));
    d = Math.min(d, box(x, y, z, sc.wallMin, sc.wallMax));
    for (const [sx, sy, sz, sr] of sc.stones) {
      d = Math.min(d, Math.hypot(x - sx, y - sy, z - sz) - sr);
    }
    return d;
  };
  // Normal = direction in which the distance grows fastest
  const normal = (x, y, z) => {
    const e = sc.sdfEps;
    const gx = sdf(x + e, y, z) - sdf(x - e, y, z);
    const gy = sdf(x, y + e, z) - sdf(x, y - e, z);
    const gz = sdf(x, y, z + e) - sdf(x, y, z - e);
    const l = Math.hypot(gx, gy, gz);
    return l < 1e-8 ? [0, 1, 0] : [gx / l, gy / l, gz / l];
  };
  return { sdf, normal, dir: [ca, -sa, 0] };
}

// Rest density: sum the kernel over a perfect cubic lattice.
export function restDensity(h, s, mass) {
  let sum = 0;
  const R = Math.ceil(h / s);
  for (let z = -R; z <= R; z++) for (let y = -R; y <= R; y++) for (let x = -R; x <= R; x++) {
    const d = Math.hypot(x, y, z) * s;
    if (d < h) sum += mass * (h - d) ** 2 * 15 / (2 * PI * h ** 5);
  }
  return sum;
}

// Everything that follows from particle count and scene.
export function layout(N, sc = SCENE, hRatio = 2) {
  const s = Math.cbrt(sc.volume / N);
  const h = s * hRatio, mass = 1;
  const grid = [0, 1, 2].map((k) => Math.max(1, Math.ceil((sc.boundsMax[k] - sc.boundsMin[k]) / h)));
  const nz = Math.max(1, Math.floor((sc.inlet.zMax - sc.inlet.zMin) / s));
  const ny = Math.max(1, Math.floor((sc.inlet.yMax - sc.inlet.yMin) / s));
  return {
    s, h, mass, grid, rho0: restDensity(h, s, mass), radius: 0.5 * s,
    cellCount: grid[0] * grid[1] * grid[2],
    inlet: {
      nz, ny, layer: nz * ny,
      y0: 0.5 * (sc.inlet.yMin + sc.inlet.yMax) - 0.5 * ny * s,
      z0: 0.5 * (sc.inlet.zMin + sc.inlet.zMax) - 0.5 * nz * s,
    },
    park: sc.boundsMin.map((v) => v - 100),   // parking spot, well outside the grid
  };
}

// When is the next layer at the inlet due? Per step the water advances by
// inflow*dt; once it is one particle spacing further, a layer follows.
export function makeEmitter(spacing, layer) {
  let acc = spacing, parity = 0;
  return (dt, inflow) => {
    acc += inflow * dt;
    if (acc < spacing) return { count: 0, along: 0, stagger: 0 };
    acc = Math.min(acc - spacing, spacing);
    parity ^= 1;
    return { count: layer, along: acc, stagger: parity ? 0.25 : -0.25 };
  };
}

export function makeSim({ N, scene = SCENE, hRatio = 2, pressure, nearPressure, viscosity,
                          gravity = 8, restitution = 0.15, inflow = 2, inPlace = false, dt: dtFixed }) {
  const L = layout(N, scene, hRatio);
  const { s, h, mass, rho0, grid: [gx, gy, gz], radius: r, park } = L;
  const [x0, y0, z0] = scene.boundsMin;
  const { sdf, normal, dir } = makeSdf(scene);

  // scale-invariant conversion: slider value -> uniform
  const kP = pressure * h / 2;
  const kN = nearPressure * h / 2;
  const kV = viscosity * s * s * s;      // 3D: kernel sum ~ 1/s³

  const px = new Float64Array(N), py = new Float64Array(N), pz = new Float64Array(N);
  const vx = new Float64Array(N), vy = new Float64Array(N), vz = new Float64Array(N);
  const qx = new Float64Array(N), qy = new Float64Array(N), qz = new Float64Array(N);
  const den = new Float64Array(N), nea = new Float64Array(N);
  // Default (inPlace = false): for all particles the viscosity reads the
  // velocities from before the force step (ux0, uy0, uz0). inPlace = true is
  // the old behaviour with the race: already updated values count too.
  const ux0 = inPlace ? vx : new Float64Array(N), uy0 = inPlace ? vy : new Float64Array(N);
  const uz0 = inPlace ? vz : new Float64Array(N);
  const cell = new Int32Array(N);        // cell of the prediction, -1 = pool
  const head = new Int32Array(gx * gy * gz), next = new Int32Array(N);

  for (let i = 0; i < N; i++) { px[i] = park[0]; py[i] = park[1]; pz[i] = park[2]; }

  // 3D kernels, normalised: integral over the ball = 1
  const W2 = (d) => d >= h ? 0 : (h - d) ** 2 * 15 / (2 * PI * h ** 5);
  const dW2 = (d) => d >= h ? 0 : -(h - d) * 15 / (PI * h ** 5);
  const W3 = (d) => d >= h ? 0 : (h - d) ** 3 * 15 / (PI * h ** 6);
  const dW3 = (d) => d >= h ? 0 : -((h - d) ** 2) * 45 / (PI * h ** 6);
  const Wv = (d) => d >= h ? 0 : (h * h - d * d) ** 3 * 315 / (64 * PI * h ** 9);

  const maxSpeed = 30;
  const c = Math.sqrt(Math.max(kP, 1e-9));          // speed of sound
  const dt = dtFixed ?? Math.min(1 / 120, 0.25 * h / (c + 4));  // CFL

  const cellOf = (x, y, z) => {
    const cx = Math.floor((x - x0) / h), cy = Math.floor((y - y0) / h), cz = Math.floor((z - z0) / h);
    if (cx < 0 || cy < 0 || cz < 0 || cx >= gx || cy >= gy || cz >= gz) return -1;
    return (cz * gy + cy) * gx + cx;
  };

  // emit: { count, along, stagger } as from makeEmitter.
  // poolRank: rank of each particle in the pool (taken from the GPU, because
  // there the order within the bin is random). Without it, index order applies.
  function step(emit = { count: 0, along: 0, stagger: 0 }, poolRank = null) {
    // 1. external forces and prediction. Parked particles stay where they are.
    for (let i = 0; i < N; i++) {
      if (cellOf(px[i], py[i], pz[i]) < 0) { qx[i] = px[i]; qy[i] = py[i]; qz[i] = pz[i]; continue; }
      vy[i] -= gravity * dt;
      qx[i] = px[i] + vx[i] * dt; qy[i] = py[i] + vy[i] * dt; qz[i] = pz[i] + vz[i] * dt;
    }
    // 2. grid. Whoever lands outside belongs to the pool.
    head.fill(-1);
    for (let i = 0; i < N; i++) {
      const ci = cell[i] = cellOf(qx[i], qy[i], qz[i]);
      if (ci >= 0) { next[i] = head[ci]; head[ci] = i; }
    }
    const each = (i, fn) => {
      const cx = Math.floor((qx[i] - x0) / h), cy = Math.floor((qy[i] - y0) / h), cz = Math.floor((qz[i] - z0) / h);
      for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const X = cx + dx, Y = cy + dy, Z = cz + dz;
        if (X < 0 || Y < 0 || Z < 0 || X >= gx || Y >= gy || Z >= gz) continue;
        for (let j = head[(Z * gy + Y) * gx + X]; j !== -1; j = next[j]) fn(j);
      }
    };
    // 3. density
    for (let i = 0; i < N; i++) {
      let D = 0, Nd = 0;
      if (cell[i] >= 0) each(i, (j) => {
        const d = Math.hypot(qx[i] - qx[j], qy[i] - qy[j], qz[i] - qz[j]);
        D += mass * W2(d); Nd += mass * W3(d);
      });
      den[i] = D; nea[i] = Nd;
    }
    // 4. forces
    if (!inPlace) { ux0.set(vx); uy0.set(vy); uz0.set(vz); }
    for (let i = 0; i < N; i++) {
      if (cell[i] < 0 || den[i] <= 1e-9) continue;
      const pi = (den[i] - rho0) * kP, ni = nea[i] * kN;
      let ax = 0, ay = 0, az = 0, sx = 0, sy = 0, sz = 0;
      each(i, (j) => {
        if (j === i || den[j] <= 1e-9) return;
        const ox = qx[j] - qx[i], oy = qy[j] - qy[i], oz = qz[j] - qz[i], d = Math.hypot(ox, oy, oz);
        if (d >= h) return;
        const ux = d > 1e-9 ? ox / d : 0, uy = d > 1e-9 ? oy / d : 1, uz = d > 1e-9 ? oz / d : 0;
        const pj = (den[j] - rho0) * kP, nj = nea[j] * kN;
        // momentum-conserving: the term is antisymmetric in i and j
        const f = dW2(d) * mass * (pi / (den[i] * den[i]) + pj / (den[j] * den[j]))
                + dW3(d) * mass * (ni / Math.max(nea[i] * nea[i], 1e-9) + nj / Math.max(nea[j] * nea[j], 1e-9));
        ax += ux * f; ay += uy * f; az += uz * f;
        const w = Wv(d);
        sx += (ux0[j] - ux0[i]) * w; sy += (uy0[j] - uy0[i]) * w; sz += (uz0[j] - uz0[i]) * w;
      });
      vx[i] += (ax + sx * kV) * dt; vy[i] += (ay + sy * kV) * dt; vz[i] += (az + sz * kV) * dt;
      const sp = Math.hypot(vx[i], vy[i], vz[i]);
      if (sp > maxSpeed) { vx[i] *= maxSpeed / sp; vy[i] *= maxSpeed / sp; vz[i] *= maxSpeed / sp; }
    }
    // 5. integration, collision with the river bed, source and drain
    let rank = 0;
    const doPark = (i) => { px[i] = park[0]; py[i] = park[1]; pz[i] = park[2]; vx[i] = vy[i] = vz[i] = 0; };
    for (let i = 0; i < N; i++) {
      if (cell[i] < 0) {
        // pool: the first emit.count fill the lattice at the inlet, the rest stay parked.
        const k = poolRank ? poolRank[i] : rank++;
        if (k >= 0 && k < emit.count) {
          const iz = k % L.inlet.nz, iy = Math.floor(k / L.inlet.nz);
          px[i] = scene.inlet.x + r + emit.along;
          py[i] = L.inlet.y0 + (iy + 0.5 + emit.stagger) * s;
          pz[i] = L.inlet.z0 + (iz + 0.5 + emit.stagger) * s;
          vx[i] = dir[0] * inflow; vy[i] = dir[1] * inflow; vz[i] = dir[2] * inflow;
        } else doPark(i);
        continue;
      }
      px[i] += vx[i] * dt; py[i] += vy[i] * dt; pz[i] += vz[i] * dt;
      // River bed: if the particle is closer than r to the surface, push it out
      // along the normal and reverse the velocity component into the wall.
      const d = sdf(px[i], py[i], pz[i]);
      if (d < r) {
        const [nx, ny, nz] = normal(px[i], py[i], pz[i]);
        px[i] += nx * (r - d); py[i] += ny * (r - d); pz[i] += nz * (r - d);
        const vn = vx[i] * nx + vy[i] * ny + vz[i] * nz;
        if (vn < 0) {
          vx[i] -= (1 + restitution) * vn * nx; vy[i] -= (1 + restitution) * vn * ny; vz[i] -= (1 + restitution) * vn * nz;
        }
      }
      // Drain: whoever has left the grid is parked.
      if (cellOf(px[i], py[i], pz[i]) < 0) doPark(i);
    }
  }

  return {
    step, rho0, h, s, dt, layout: L,
    // set state from outside: pos and vel as [x0, y0, z0, x1, y1, z1, ...]
    setState(pos, vel) {
      for (let i = 0; i < N; i++) {
        px[i] = pos[3 * i]; py[i] = pos[3 * i + 1]; pz[i] = pos[3 * i + 2];
        vx[i] = vel[3 * i]; vy[i] = vel[3 * i + 1]; vz[i] = vel[3 * i + 2];
      }
    },
    state: { px, py, pz, vx, vy, vz, den, nea, cell },
    stats() {
      let nan = 0, maxV = 0, sum = 0, cnt = 0, clamped = 0, active = 0, ke = 0, maxRho = 0;
      for (let i = 0; i < N; i++) {
        if (!Number.isFinite(px[i]) || !Number.isFinite(vx[i])) { nan++; continue; }
        if (cellOf(px[i], py[i], pz[i]) < 0) continue;
        active++;
        const v = Math.hypot(vx[i], vy[i], vz[i]);
        ke += 0.5 * v * v;
        if (v > maxV) maxV = v;
        if (v > 29.9) clamped++;
        if (den[i] > maxRho) maxRho = den[i];
        if (den[i] > rho0 * 0.5) { sum += den[i]; cnt++; }   // interior only
      }
      return { nan, active, maxV, clamped, ke: active ? ke / active : 0,
               meanRho: cnt ? sum / cnt / rho0 : 0, maxRho: maxRho / rho0 };
    },
  };
}

// The table only runs when called directly with node, not when imported in the browser.
const runAsScript = typeof process !== 'undefined' && process.argv?.[1]
  && import.meta.url === new URL(process.argv[1], 'file://').href;

if (runAsScript) {
  const N = parseInt(process.argv[2] ?? '4000', 10);
  const SECONDS = parseFloat(process.argv[3] ?? '10');
  const inflow = parseFloat(process.argv[4] ?? '2');
  const hRatio = parseFloat(process.argv[5] ?? '2');
  const sim = makeSim({ N, hRatio, pressure: 2000, nearPressure: 8, viscosity: 20, gravity: 8, inflow });
  const emitter = makeEmitter(sim.s, sim.layout.inlet.layer);
  console.log(`River filling up: N=${N}, s=${sim.s.toFixed(4)}, h=${sim.h.toFixed(4)}, dt=${sim.dt.toExponential(2)}, `
    + `layer=${sim.layout.inlet.layer}, inflow=${inflow}\n`);
  console.log('   t  active   share  rho/rho0  max rho  max v   E_kin    clamped  NaN');
  console.log('  ' + '-'.repeat(70));
  const perSecond = Math.round(1 / sim.dt);
  for (let k = 1; k <= Math.ceil(SECONDS / sim.dt); k++) {
    sim.step(emitter(sim.dt, inflow));
    if (k % perSecond === 0) {
      const st = sim.stats();
      console.log(`  ${String(k / perSecond).padStart(2)}  ${String(st.active).padStart(6)}  `
        + `${(st.active / N).toFixed(2).padStart(6)}  ${st.meanRho.toFixed(3).padStart(8)}  `
        + `${st.maxRho.toFixed(2).padStart(7)}  ${st.maxV.toFixed(2).padStart(5)}  ${st.ke.toFixed(2).padStart(6)}  `
        + `${String(st.clamped).padStart(9)}  ${String(st.nan).padStart(3)}`);
    }
  }
}
