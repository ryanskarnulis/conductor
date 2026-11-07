/* The Living Web scene (conductor#94, design system #86): Glitch (Warden) at
 * the hub of a seeded orb web, the apps moored at its frame. Ported from the
 * round-2 prototype; procedural Warden from the Glitch lab (#93), whose
 * canonical model is gateway's createGlitch (gateway#12).
 *
 * Decoration only: the canvas is aria-hidden and every fact it shows is also
 * in the panel. It reads getState() each frame (src/features/agent/sceneState.ts)
 * and never writes to the page. Plain JS on purpose (a port of the prototype,
 * typed at its edge by livingWeb.d.ts). */
import * as THREE from 'three';

export function mountLivingWeb({ canvas, panel, device, getState, view, reducedMotion, onReady = () => {}, onPerf = () => {}, onFail = () => {} }) {
const els = { gl: canvas, panel, device };
const LW = { get S() { return getState(); }, now: () => performance.now(), get view() { return view(); }, get rm() { return reducedMotion(); } };
let disposed = false, rafId = 0;
/* ---------------------------------------------------------------- basics */
const $ = id => els[id] || (els[id] = document.createElement('span'));
const V3 = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const damp = (a, b, l, dt) => a + (b - a) * (1 - Math.exp(-l * dt));
const ease = t => (t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
const UPY = V3(0, 1, 0);
const tA = V3(), tB = V3(), tC = V3(), tD = V3(), tQ = new THREE.Quaternion(), tM = new THREE.Matrix4(), tS = V3(1, 1, 1);
const C = hex => new THREE.Color(hex);
const PAL = { accent: '#9be7ff', ok: '#8fdcaa', warn: '#eec98a', danger: '#f28fa4', silk: '#c7d3e4' };
let reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

// canvas: passed in by mountLivingWeb
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: 'high-performance' });
  if (!renderer.capabilities.isWebGL2) throw new Error('no webgl2');
} catch (e) {
  onFail(e);
  return null;
}
renderer.setClearColor(0x05070c, 1);
renderer.info.autoReset = false;
const gpuInfo = (() => {
  const gl = renderer.getContext(); const ext = gl.getExtension('WEBGL_debug_renderer_info');
  return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
})();
const isPhone = matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 700;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x05070c, .045);
const camera = new THREE.PerspectiveCamera(32, 1.6, .1, 120);

/* environment for reflections: a dark room with a soft key and blue rim panels */
{
  const env = new THREE.Scene();
  env.add(new THREE.Mesh(new THREE.BoxGeometry(30, 30, 30), new THREE.MeshBasicMaterial({ color: 0x020306, side: THREE.BackSide })));
  const panel = (w, h, col, x, y, z, ry = 0, rx = 0) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: col, side: THREE.DoubleSide }));
    m.position.set(x, y, z); m.rotation.set(rx, ry, 0); env.add(m);
  };
  panel(10, 3, new THREE.Color(5, 5.4, 6), 0, 9, 2, 0, Math.PI / 2);
  panel(2, 12, new THREE.Color(.4, 1.6, 3.2), -10, 1, 0, Math.PI / 2);
  panel(2, 12, new THREE.Color(.25, 1.1, 2.2), 10, 1, -3, -Math.PI / 2);
  panel(8, 1.2, new THREE.Color(1.2, 1.4, 1.8), 0, -2, 10);
  const pm = new THREE.PMREMGenerator(renderer);
  scene.environment = pm.fromScene(env, .02).texture;
  pm.dispose();
}
scene.add(new THREE.HemisphereLight(0x7fa8d8, 0x020306, .35));
const key = new THREE.DirectionalLight(0xdbe8ff, 2.4); key.position.set(2.5, 3, 6); scene.add(key);
const rim = new THREE.DirectionalLight(0x58b8ff, 3.4); rim.position.set(-4, 5, -5); scene.add(rim);
const hubLight = new THREE.PointLight(0x9be7ff, 4, 6, 2); hubLight.position.set(0, 0, 1.4); scene.add(hubLight);

/* background: deep gradient dome */
const dome = new THREE.Mesh(new THREE.SphereGeometry(60, 32, 16), new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false, fog: false,
  uniforms: { uTint: { value: C('#0e1a33') } },
  vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }',
  fragmentShader: 'uniform vec3 uTint; varying vec3 vP; void main(){ float k = smoothstep(-.2,1.,vP.z*.8+vP.y*.25); vec3 c = mix(vec3(.008,.011,.02), uTint*.55, k); gl_FragColor = vec4(c,1.); }',
}));
scene.add(dome);

/* drifting dust */
const dust = (() => {
  const n = isPhone ? 260 : 520, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { pos[i * 3] = (Math.random() - .5) * 16; pos[i * 3 + 1] = (Math.random() - .5) * 12; pos[i * 3 + 2] = (Math.random() - .5) * 8 - 1; }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const m = new THREE.PointsMaterial({ size: .028, color: new THREE.Color(.55, .8, 1.1), transparent: true, opacity: .55, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false });
  const p = new THREE.Points(g, m); scene.add(p); return p;
})();

/* ---------------------------------------------------------------- ribbons: camera-facing strips for silk */
const ribbonMat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false, side: THREE.DoubleSide, fog: false });
class Ribbon {
  constructor(strips, n, width) {
    this.k = strips; this.n = n; this.w = width;
    const vc = strips * n * 2;
    this.pos = new Float32Array(vc * 3); this.col = new Float32Array(vc * 3);
    const idx = [];
    for (let s = 0; s < strips; s++) for (let i = 0; i < n - 1; i++) { const a = (s * n + i) * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('color', new THREE.BufferAttribute(this.col, 3).setUsage(THREE.DynamicDrawUsage));
    g.setIndex(idx);
    this.mesh = new THREE.Mesh(g, ribbonMat); this.mesh.frustumCulled = false; this.mesh.renderOrder = 2;
  }
  /* pts: array of Vector3 (n), colFn(u, i) -> sets tC */
  set(s, pts, colFn, wFn) {
    const n = this.n, base = s * n * 2;
    for (let i = 0; i < n; i++) {
      const p = pts[i], a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
      tA.subVectors(b, a); tB.subVectors(camera.position, p);
      tD.crossVectors(tA, tB).normalize().multiplyScalar(this.w * (wFn ? wFn(i / (n - 1)) : 1));
      const o = (base + i * 2) * 3;
      this.pos[o] = p.x + tD.x; this.pos[o + 1] = p.y + tD.y; this.pos[o + 2] = p.z + tD.z;
      this.pos[o + 3] = p.x - tD.x; this.pos[o + 4] = p.y - tD.y; this.pos[o + 5] = p.z - tD.z;
      colFn(i / (n - 1), i);
      this.col[o] = this.col[o + 3] = tC.x; this.col[o + 1] = this.col[o + 4] = tC.y; this.col[o + 2] = this.col[o + 5] = tC.z;
    }
  }
  flush() { this.mesh.geometry.attributes.position.needsUpdate = true; this.mesh.geometry.attributes.color.needsUpdate = true; }
}

/* verlet rope: real silk that sags and rings */
class Rope {
  constructor(n) { this.n = n; this.p = Array.from({ length: n }, () => V3()); this.o = Array.from({ length: n }, () => V3()); this.r = Array.from({ length: n }, () => V3()); }
  reset(a, b) { for (let i = 0; i < this.n; i++) { this.p[i].lerpVectors(a, b, i / (this.n - 1)); this.o[i].copy(this.p[i]); } }
  step(dt, a, b, slack, g) {
    const n = this.n; this.p[0].copy(a); this.p[n - 1].copy(b);
    for (let i = 1; i < n - 1; i++) {
      const p = this.p[i], o = this.o[i];
      tA.subVectors(p, o).multiplyScalar(.985); o.copy(p); p.add(tA); p.y -= g * dt * dt;
    }
    const rest = a.distanceTo(b) * slack / (n - 1);
    for (let it = 0; it < 14; it++) {
      for (let i = 0; i < n - 1; i++) {
        const p = this.p[i], q = this.p[i + 1];
        tA.subVectors(q, p); const d = tA.length() || 1e-6; const k = (d - rest) / d * .5;
        if (i > 0) p.addScaledVector(tA, k);
        if (i + 1 < n - 1) q.addScaledVector(tA, -k);
      }
    }
  }
  pluck(amount, dir) { for (let i = 1; i < this.n - 1; i++) { const s = Math.sin(Math.PI * i / (this.n - 1)); this.o[i].addScaledVector(dir, -amount * s); } }
  pointAt(u, out) { const f = clamp(u, 0, 1) * (this.n - 1), i = Math.min(this.n - 2, Math.floor(f)); return out.lerpVectors(this.p[i], this.p[i + 1], f - i); }
}

/* text sprites */
function label(text, scale = 1) {
  const c = document.createElement('canvas'); c.width = 512; c.height = 96;
  const x = c.getContext('2d'); x.font = '600 44px Inter, system-ui, sans-serif'; x.textAlign = 'center'; x.textBaseline = 'middle';
  x.fillStyle = '#c7d3e4'; x.letterSpacing = '10px'; x.fillText(text.toUpperCase(), 256, 50);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: t, transparent: true, depthWrite: false, toneMapped: false, fog: false, color: new THREE.Color(1.2, 1.3, 1.4) }));
  s.scale.set(1.6 * scale, .3 * scale, 1); return s;
}
/* ---------------------------------------------------------------- the web (conductor hub): a seeded orb web, moored at the apps */
const hub = new THREE.Group(); scene.add(hub);
const rnd = (() => { let s = 0x5eed1e; return () => { s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; })();
const WT = Math.PI * 2, WY = .92, D2R = Math.PI / 180, WEB_R = 4.3;
const wrapA = a => ((a % WT) + WT) % WT;
/* frame vertices (degrees, radius); the three apps are where the web is moored */
const FRAME = [[-140, 4.3, 'chess'], [-93, 3.95], [-40, 4.3, 'music'], [8, 4.6], [50, 4.2], [90, 4.3, 'tasks'], [127, 4.25], [170, 4.65]]
  .map(([d, r, app]) => ({ a: d * D2R, r, app, x: Math.cos(d * D2R) * r, y: Math.sin(d * D2R) * r }));
const FSAG = .055;
/* distance from the hub to the sagging frame thread along angle a (web space, y unscaled) */
function frameAt(a) {
  a = wrapA(a);
  for (let i = 0; i < FRAME.length; i++) {
    const A = FRAME[i], B = FRAME[(i + 1) % FRAME.length];
    const a0 = wrapA(A.a); let a1 = wrapA(B.a), aa = a; if (a1 <= a0) a1 += WT; if (aa < a0) aa += WT;
    if (aa < a0 || aa > a1) continue;
    const dx = Math.cos(a), dy = Math.sin(a), ex = B.x - A.x, ey = B.y - A.y;
    const t = (A.y * dx - A.x * dy) / (ex * dy - ey * dx);
    return ((A.x + t * ex) * dx + (A.y + t * ey) * dy) * (1 - FSAG * Math.sin(Math.PI * t));
  }
  return WEB_R;
}
const wp = (a, r, out = V3()) => out.set(Math.cos(a) * r, Math.sin(a) * r * WY, 0);
const ANCHORS = [['tasks', 'PCC'], ['music', 'music'], ['chess', 'chess']].map(([id, name]) => { const f = FRAME.find(f => f.app === id); return { id, name, a: f.a / D2R, p: wp(f.a, f.r) }; });
/* app icons: a white pawn and a green check outlined in black, a black note outlined in white; status shows on the ring around each */
const ICON_MAT = {
  chess: { color: 0xf6f3ec, metalness: 0, roughness: .2, emissive: 0xffffff, ei: .07, outline: 0x000000 },
  music: { color: 0x040405, metalness: .05, roughness: .14, emissive: 0x000000, ei: 0, outline: 0xffffff },
  tasks: { color: 0x1fcf62, metalness: 0, roughness: .3, emissive: 0x22e06a, ei: .55, outline: 0x000000 },
};
function appIcon(id) {
  const M = ICON_MAT[id], body = new THREE.MeshPhysicalMaterial({ color: M.color, metalness: M.metalness, roughness: M.roughness, clearcoat: 1, clearcoatRoughness: .05, emissive: M.emissive, emissiveIntensity: M.ei });
  const g = new THREE.Group(), add = geo => { const x = new THREE.Mesh(geo, body); g.add(x); return x; };
  if (id === 'chess') {                       /* a pawn, turned on a lathe */
    const prof = [[0, 0], [.118, 0], [.12, .022], [.104, .036], [.08, .05], [.066, .07], [.056, .11], [.046, .17], [.042, .195], [.068, .207], [.07, .222], [.04, .232], [.03, .24]].map(([x, y]) => new THREE.Vector2(x, y));
    add(new THREE.LatheGeometry(prof, 40)).position.y = -.17;
    add(new THREE.SphereGeometry(.062, 32, 20)).position.y = .118;
  } else if (id === 'music') {                /* an eighth note, as on a score */
    const head = add(new THREE.SphereGeometry(.07, 32, 20)); head.scale.set(1.35, .92, .8); head.position.set(-.04, -.12, 0); head.rotation.z = .45;
    add(new THREE.CylinderGeometry(.013, .013, .3, 12)).position.set(.042, .025, 0);
    const flag = new THREE.CatmullRomCurve3([V3(.042, .17, 0), V3(.1, .125, 0), V3(.13, .06, 0), V3(.1, -.005, 0)]);
    add(new THREE.TubeGeometry(flag, 32, .015, 8, false));
    g.children.forEach(m => (m.position.x -= .04));
  } else {                                    /* PCC: a check mark, rounded at the ends and the joint */
    const A = V3(-.13, -.02, 0), B = V3(-.045, -.12, 0), C = V3(.145, .1, 0), r = .037;
    [[A, B], [B, C]].forEach(([p, q]) => {
      const len = p.distanceTo(q), cyl = add(new THREE.CylinderGeometry(r, r, len, 20));
      cyl.position.lerpVectors(p, q, .5); cyl.quaternion.setFromUnitVectors(V3(0, 1, 0), tA.subVectors(q, p).normalize());
    });
    [A, B, C].forEach(p => add(new THREE.SphereGeometry(r, 20, 14)).position.copy(p));
  }
  /* a thin outline: each piece again, pushed out along its normals and drawn back-faces only */
  const line = new THREE.ShaderMaterial({ side: THREE.BackSide, uniforms: { c: { value: new THREE.Color(M.outline) } },
    vertexShader: 'void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position + normal * .0075, 1.); }',
    fragmentShader: 'uniform vec3 c; void main(){ gl_FragColor = vec4(c, 1.); }' });
  [...g.children].forEach(m => { const o = new THREE.Mesh(m.geometry, line); o.position.copy(m.position); o.quaternion.copy(m.quaternion); o.scale.copy(m.scale); g.add(o); });
  g.material = body; g.userData.ei = M.ei; g.scale.setScalar(1.6); g.position.z = .24; return g;   /* in front of the silk, so threads never show through */
}
const anchorMats = [];
ANCHORS.forEach(an => {
  const g = new THREE.Group(); g.position.copy(an.p);
  const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(PAL.accent).multiplyScalar(2.2), toneMapped: false });
  const core = appIcon(an.id);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(.34, .014, 8, 64), mat);
  const prog = new THREE.Mesh(new THREE.TorusGeometry(.44, .02, 8, 96, Math.PI * 2), new THREE.MeshBasicMaterial({ color: new THREE.Color(PAL.accent).multiplyScalar(3), toneMapped: false }));
  prog.visible = false; prog.rotation.z = Math.PI / 2;
  g.add(core, ring, prog);
  const l = label(an.name, .8), side = Math.abs(an.p.x) < 1; l.position.set(side ? .82 : 0, side ? 0 : an.p.y > 0 ? .62 : -.62, 0); g.add(l);
  hub.add(g); an.mat = mat; an.ring = ring; an.prog = prog; an.core = core; anchorMats.push(mat);
});
const live = pts => ({ pts, live: pts.map(p => p.clone()) });
/* frame threads: chord between moorings, sagging toward the hub */
const FN = 24, frameLines = FRAME.map((A, i) => {
  const B = FRAME[(i + 1) % FRAME.length], pts = [];
  for (let j = 0; j < FN; j++) { const t = j / (FN - 1), s = 1 - FSAG * Math.sin(Math.PI * t); pts.push(V3((A.x + (B.x - A.x) * t) * s, (A.y + (B.y - A.y) * t) * s * WY, 0)); }
  return live(pts);
});
/* mooring lines: off the frame into the dark, fading out (never a hard edge) */
const MN = 16, moorings = [];
FRAME.forEach((F, i) => {
  const n = F.app || rnd() < .5 ? 2 : 1;
  for (let k = 0; k < n; k++) {
    const a = F.a + (k ? (rnd() < .5 ? -1 : 1) * (.35 + rnd() * .3) : (rnd() - .5) * .25), len = 1.5 + rnd() * 1.1, pts = [];
    for (let j = 0; j < MN; j++) { const u = j / (MN - 1); pts.push(V3(F.x + Math.cos(a) * len * u, (F.y + Math.sin(a) * len * u) * WY, -u * .25)); }
    moorings.push(live(pts));
  }
});
/* radials: hub to frame, irregularly spaced */
const RADIALS = 34, RN = 26, radialBase = [];
for (let s = 0; s < RADIALS; s++) {
  const a = (s + (rnd() - .5) * .55) / RADIALS * WT + .13, len = frameAt(a), pts = [];
  for (let i = 0; i < RN; i++) pts.push(wp(a, .16 + (len - .16) * i / (RN - 1)));
  radialBase.push({ a: wrapA(a), len, ...live(pts) });
}
radialBase.sort((p, q) => p.a - q.a);
/* hub: a tight, messy spiral where he sits */
const HN = 300, hubPts = [];
for (let i = 0; i < HN; i++) { const q = i / (HN - 1), a = q * WT * 7 + .4; hubPts.push(wp(a, .2 + .44 * q + (rnd() - .5) * .035)); }
const hubMesh = live(hubPts);
/* capture spiral: a free zone, then many turns from hub to frame, each segment sagging; spacing widens outward */
const TURNS = isPhone ? 16 : 24, SUB = 3, rin = 1.12, spiralBase = [], spiralMask = [];
const turnJit = Array.from({ length: TURNS + 1 }, () => 1 + (rnd() - .5) * .05);
let brk = 0;
for (let j = 0; j < TURNS; j++) for (let k = 0; k < RADIALS; k++) {
  const R0 = radialBase[k], R1 = radialBase[(k + 1) % RADIALS], a1 = R1.a + (k === RADIALS - 1 ? WT : 0);
  const rAt = (R, kk) => { const q = (j + kk / RADIALS) / TURNS; return rin + (R.len * .95 - rin) * Math.pow(q, 1.12) * turnJit[j]; };
  const p0 = wp(R0.a, rAt(R0, k)), p1 = wp(a1, rAt(R1, k + 1));
  for (let m = 0; m < SUB; m++) {
    const t = m / SUB; spiralBase.push(V3().lerpVectors(p0, p1, t).multiplyScalar(1 - .018 * Math.sin(Math.PI * t)));
    if (brk <= 0 && j > 1 && rnd() < .012) brk = 3 + Math.floor(rnd() * 5);
    spiralMask.push(brk-- > 0 ? 0 : 1);
  }
}
const SPIRAL_N = spiralBase.length, spiralLive = spiralBase.map(p => p.clone());
const webRadials = new Ribbon(RADIALS, RN, .0075), webSpiral = new Ribbon(1, SPIRAL_N, .0042), webHub = new Ribbon(1, HN, .0045);
const webFrame = new Ribbon(frameLines.length, FN, .011), webMoor = new Ribbon(moorings.length, MN, .008);
hub.add(webRadials.mesh, webSpiral.mesh, webHub.mesh, webFrame.mesh, webMoor.mesh);
/* dew: beads on the capture threads that glint as you move */
const dewIdx = []; spiralMask.forEach((m, i) => { if (m && i % SUB === 1 && rnd() < .3) dewIdx.push(i); });
const dew = (() => {
  const n = dewIdx.length, g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3).setUsage(THREE.DynamicDrawUsage));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3).setUsage(THREE.DynamicDrawUsage));
  const c = document.createElement('canvas'); c.width = c.height = 32; const x = c.getContext('2d');
  const gr = x.createRadialGradient(16, 16, 0, 16, 16, 16); gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(.35, 'rgba(255,255,255,.35)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
  x.fillStyle = gr; x.fillRect(0, 0, 32, 32);
  const m = new THREE.PointsMaterial({ size: .07, map: new THREE.CanvasTexture(c), vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false, fog: false });
  const p = new THREE.Points(g, m); p.frustumCulled = false; p.renderOrder = 2; hub.add(p);
  p.userData.ph = dewIdx.map(() => [rnd() * WT, .6 + rnd() * 1.6]); return p;
})();
const hubGlow = (() => {
  const c = document.createElement('canvas'); c.width = c.height = 256; const x = c.getContext('2d');
  const gr = x.createRadialGradient(128, 128, 0, 128, 128, 128); gr.addColorStop(0, 'rgba(120,200,255,.55)'); gr.addColorStop(.35, 'rgba(60,120,220,.16)'); gr.addColorStop(1, 'rgba(0,0,0,0)');
  x.fillStyle = gr; x.fillRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c);
  const m = new THREE.Mesh(new THREE.PlaneGeometry(9, 9), new THREE.MeshBasicMaterial({ map: t, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false, fog: false }));
  m.position.z = -.6; hub.add(m); return m;
})();
/* a pulse ring for receipts / listening */
const pulseRing = new THREE.Mesh(new THREE.TorusGeometry(1, .012, 8, 128), new THREE.MeshBasicMaterial({ color: new THREE.Color(2, 3, 4), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false, fog: false }));
scene.add(pulseRing); pulseRing.visible = false;
/* thinking motes */
const motes = new THREE.InstancedMesh(new THREE.SphereGeometry(.03, 8, 6), new THREE.MeshBasicMaterial({ color: new THREE.Color(1.2, 3, 4.4), toneMapped: false }), 10);
motes.frustumCulled = false; scene.add(motes);
/* ---------------------------------------------------------------- materials per design */
function circuitMat() {
  return new THREE.ShaderMaterial({
    fog: false,
    uniforms: { uColor: { value: C(PAL.accent) }, uTime: { value: 0 }, uSpeed: { value: .35 }, uI: { value: 1 }, uA: { value: 1 }, uBase: { value: .7 }, uPulse: { value: 4 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }',
    fragmentShader: 'uniform vec3 uColor; uniform float uTime,uSpeed,uI,uA,uBase,uPulse; varying vec2 vUv; void main(){ float d = fract(vUv.x*3. - uTime*uSpeed); float p = smoothstep(0.,.03,d)*(1.-smoothstep(.03,.26,d)); gl_FragColor = vec4(uColor*(uBase + uPulse*p)*uI, uA); }',
  });
}
/* box-project UVs per triangle, so the texture has no pinch at a sphere's poles */
function boxUV(geo, k) {
  const g = geo.toNonIndexed(), p = g.attributes.position, uv = g.attributes.uv, a = V3(), b = V3(), c = V3(), n = V3();
  for (let i = 0; i < p.count; i += 3) {
    a.fromBufferAttribute(p, i); b.fromBufferAttribute(p, i + 1); c.fromBufferAttribute(p, i + 2);
    n.subVectors(c, b).cross(tA.subVectors(a, b)); const ax = Math.abs(n.x), ay = Math.abs(n.y), az = Math.abs(n.z);
    for (let j = 0; j < 3; j++) { const q = [a, b, c][j]; const [u, v] = ax >= ay && ax >= az ? [q.y, q.z] : ay >= az ? [q.x, q.z] : [q.x, q.y]; uv.setXY(i + j, u * k, v * k); }
  }
  return g;
}
let _grain = null;
function grainTexture() {
  if (_grain) return _grain;
  /* an engineered finish rather than dirt: a fine, regular carbon-fibre twill. It changes only sheen and relief, never colour,
     so up close it reads as a technical weave and from a distance as clean matte */
  const N = 256, T = 16, c = document.createElement('canvas'); c.width = c.height = N; const x = c.getContext('2d'), im = x.createImageData(N, N);
  for (let y = 0; y < N; y++) for (let xx = 0; xx < N; xx++) {
    const cx = Math.floor(xx / T), cy = Math.floor(y / T), u = (xx % T) / T, v = (y % T) / T;
    const warp = ((cx + cy) & 1) === 0;              /* alternating tows, offset each row: a 2x2 twill */
    const across = warp ? u : v, along = warp ? v : u;
    const ridge = Math.sin(Math.PI * across), fibre = .5 + .5 * Math.sin(along * Math.PI * 6 + (warp ? 0 : 1.3));
    const val = 150 + ridge * 70 + fibre * 12;
    const i = (y * N + xx) * 4; im.data[i] = im.data[i + 1] = im.data[i + 2] = val; im.data[i + 3] = 255;
  }
  x.putImageData(im, 0, 0);
  _grain = new THREE.CanvasTexture(c); _grain.wrapS = _grain.wrapT = THREE.RepeatWrapping; _grain.repeat.set(10, 10); _grain.anisotropy = 8;
  return _grain;
}
function makeMats(accent, exact, finish) {
  /* exact: lights write alpha .5 so the grade shows them at their true colour, with a soft halo instead of a hot bloom */
  const L = exact ? { toneMapped: false, opacity: .5, transparent: true, blending: THREE.NoBlending, depthWrite: true } : { toneMapped: false };
  const M = {
    shell: new THREE.MeshPhysicalMaterial({ color: 0x0a0d14, metalness: .62, roughness: .3, clearcoat: 1, clearcoatRoughness: .08, envMapIntensity: 1.3 }),
    plate: new THREE.MeshPhysicalMaterial({ color: 0x141b28, metalness: .82, roughness: .2, clearcoat: .7, clearcoatRoughness: .15, envMapIntensity: 1.5 }),
    joint: new THREE.MeshStandardMaterial({ color: 0x040509, metalness: .35, roughness: .5 }),
    glass: new THREE.MeshPhysicalMaterial({ color: 0x010204, metalness: .1, roughness: .03, clearcoat: 1, clearcoatRoughness: 0, envMapIntensity: 2.2 }),
    visorGlass: new THREE.MeshPhysicalMaterial({ color: 0x000000, metalness: 0, roughness: .02, clearcoat: 1, transparent: true, opacity: .32, envMapIntensity: 2.5, depthWrite: false }),
    glow: new THREE.MeshBasicMaterial({ color: C(accent).multiplyScalar(2.4), ...L }),
    glowSoft: new THREE.MeshBasicMaterial({ color: C(accent).multiplyScalar(1.1), ...L }),
    eye: new THREE.MeshBasicMaterial({ color: C(accent).multiplyScalar(2.2), ...L }),
    eyeRing: new THREE.MeshBasicMaterial({ color: C(accent).multiplyScalar(1.25), ...L }),
    /* iris materials write alpha .5, which tells the grade to show them at their exact colour, untouched by the tone curve */
    eyeDim: new THREE.MeshBasicMaterial({ color: C('#00c3ff'), toneMapped: false, opacity: .5, transparent: true, blending: THREE.NoBlending, depthWrite: true }),
    irisMid: new THREE.MeshBasicMaterial({ color: C('#06defd'), toneMapped: false, opacity: .5, transparent: true, blending: THREE.NoBlending, depthWrite: true }),
    irisIn: new THREE.MeshBasicMaterial({ color: C('#2efaff'), toneMapped: false, opacity: .5, transparent: true, blending: THREE.NoBlending, depthWrite: true }),
    catch: new THREE.MeshBasicMaterial({ color: new THREE.Color(1.6, 1.75, 1.9), toneMapped: false }),
    eyeGlass: new THREE.MeshPhysicalMaterial({ color: 0x000000, metalness: 0, roughness: .05, clearcoat: 1, clearcoatRoughness: .05, transparent: true, opacity: .1, envMapIntensity: .9, depthWrite: false }),
    pupilDark: new THREE.MeshBasicMaterial({ color: 0x020409 }),
    shellIn: new THREE.MeshPhysicalMaterial({ color: 0x0a0d14, metalness: .62, roughness: .3, clearcoat: 1, clearcoatRoughness: .08, envMapIntensity: 1.3, side: THREE.DoubleSide }),
    pupil: new THREE.MeshBasicMaterial({ color: new THREE.Color(5, 6, 7), toneMapped: false }),
    circuit: circuitMat(),
  };
  if (finish === 'matte') {
    /* Warden: dark charcoal with a soft matte finish instead of glossy black */
    /* a faint grain in the roughness and surface, so it reads as a slightly rough cast finish */
    const grain = grainTexture();
    const matte = (hex, rough) => ({ color: C(hex), metalness: .2, roughness: rough, roughnessMap: grain, bumpMap: grain, bumpScale: .5, clearcoat: .05, clearcoatRoughness: .7, envMapIntensity: .65 });
    M.shell.setValues(matte('#1f2226', .78)); M.shellIn.setValues(matte('#1f2226', .78));
    M.plate.setValues(matte('#272a2f', .72));
    M.joint.setValues({ color: C('#1f2125'), metalness: .2, roughness: .7 });
  }
  return M;
}
const unitCyl = (r0, r1, seg = 12) => { const g = new THREE.CylinderGeometry(r1, r0, 1, seg, 1); g.translate(0, .5, 0); return g; };
/* a light band inlaid flush with a segment's surface */
const band = (r, h) => new THREE.CylinderGeometry(r, r, h, 28, 1, true);
const ringY = (r, t, arc = Math.PI * 2) => { const g = new THREE.TorusGeometry(r, t, 8, 40, arc); g.rotateX(Math.PI / 2); return g; };
const cone = (r, h, seg = 10) => { const g = new THREE.ConeGeometry(r, h, seg); g.translate(0, h / 2, 0); return g; };
const mesh = (g, m, parent, p, s, r) => { const o = new THREE.Mesh(g, m); if (p) o.position.set(...p); if (s) o.scale.set(...s); if (r) o.rotation.set(...r); if (parent) parent.add(o); return o; };

function makeEye(parent, r, M, pos, tilt = [0, 0, 0], kind = 'ring') {
  const g = new THREE.Group(); g.position.set(...pos); g.rotation.set(...tilt); parent.add(g);
  mesh(new THREE.SphereGeometry(r, 28, 18), M.glass, g);
  const iris = new THREE.Group(); g.add(iris);
  if (kind === 'ring') {
    mesh(new THREE.TorusGeometry(r * .56, r * .1, 8, 40), M.eye, iris, [0, 0, r * .84]);
    mesh(new THREE.SphereGeometry(r * .24, 14, 10), M.pupil, iris, [0, 0, r * .93], [1, 1, .35]);
  } else if (kind === 'lens') {
    mesh(new THREE.TorusGeometry(r * .78, r * .045, 8, 64), M.eye, iris, [0, 0, r * .64]);
    mesh(new THREE.TorusGeometry(r * .5, r * .06, 8, 64), M.eye, iris, [0, 0, r * .87]);
    const ap = new THREE.Group(); ap.position.z = r * .9; iris.add(ap);
    for (let i = 0; i < 6; i++) { const b = mesh(new THREE.BoxGeometry(r * .36, r * .07, r * .02), M.plate, ap); const a = i / 6 * Math.PI * 2; b.position.set(Math.cos(a) * r * .3, Math.sin(a) * r * .3, 0); b.rotation.z = a + 1.1; }
    mesh(new THREE.SphereGeometry(r * .14, 14, 10), M.pupil, iris, [0, 0, r * .95], [1, 1, .4]);
    g.userData.aperture = ap;
  } else {
    mesh(new THREE.SphereGeometry(r * .5, 12, 8), M.eye, iris, [0, 0, r * .62], [1, 1, .5]);
  }
  return { g, iris, rest: g.quaternion.clone(), r };
}

/* a flush eye: every layer is a decal projected onto the head ellipsoid, stacked a hair apart.
   The pupil (with its glints) slides across the iris to look; a shell-coloured lid rolls down from the top. */
function makeFlushEye(parent, M, H, guess, opt) {
  const Cc = V3(...H.c), Rr = V3(...H.r), pA = V3(), pB = V3(), pC = V3(), pD = V3();
  const proj = (p, eps, out) => {
    pA.subVectors(p, Cc); const k = 1 / Math.sqrt((pA.x / Rr.x) ** 2 + (pA.y / Rr.y) ** 2 + (pA.z / Rr.z) ** 2);
    out.copy(Cc).addScaledVector(pA, k);
    pB.set((out.x - Cc.x) / (Rr.x * Rr.x), (out.y - Cc.y) / (Rr.y * Rr.y), (out.z - Cc.z) / (Rr.z * Rr.z)).normalize();
    return out.addScaledVector(pB, eps);
  };
  const P0 = proj(guess.clone(), 0, V3());
  const n = V3((P0.x - Cc.x) / (Rr.x * Rr.x), (P0.y - Cc.y) / (Rr.y * Rr.y), (P0.z - Cc.z) / (Rr.z * Rr.z)).normalize();
  const v0 = V3(0, 1, 0).addScaledVector(n, -n.y).normalize(), u0 = V3().crossVectors(v0, n).normalize();
  const t = opt.tilt, U = u0.clone().multiplyScalar(Math.cos(t)).addScaledVector(v0, Math.sin(t)), Vv = u0.clone().multiplyScalar(-Math.sin(t)).addScaledVector(v0, Math.cos(t));
  const A = opt.a, B = opt.b, RINGS = 7, SEGS = 56;
  const anchor = new THREE.Object3D(); anchor.position.copy(P0); parent.add(anchor);
  const make = (mat, eps) => {
    const g = new THREE.BufferGeometry(), pos = new Float32Array((RINGS + 1) * SEGS * 3), idx = [];
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
    for (let i = 0; i < RINGS; i++) for (let j = 0; j < SEGS; j++) { const a = i * SEGS + j, b = i * SEGS + (j + 1) % SEGS, c = (i + 1) * SEGS + j, d = (i + 1) * SEGS + (j + 1) % SEGS; idx.push(a, c, b, b, c, d); }
    g.setIndex(idx); const m = new THREE.Mesh(g, mat); m.userData.eps = eps; m.renderOrder = 4; parent.add(m); return m;
  };
  const fill = (m, fn) => {
    const pos = m.geometry.attributes.position;
    for (let i = 0; i <= RINGS; i++) for (let j = 0; j < SEGS; j++) {
      const xy = fn(i / RINGS, j / SEGS * Math.PI * 2);
      proj(pC.copy(P0).addScaledVector(U, xy[0]).addScaledVector(Vv, xy[1]), m.userData.eps, pD);
      pos.setXYZ(i * SEGS + j, pD.x, pD.y, pD.z);
    }
    pos.needsUpdate = true; if (!m.material.isMeshBasicMaterial) m.geometry.computeVertexNormals(); m.geometry.computeBoundingSphere();
  };
  const ell = (r0, r1) => (rho, th) => { const r = r0 + (r1 - r0) * rho; return [Math.cos(th) * A * r, Math.sin(th) * B * r]; };
  const disc = (cx, cy, r0, r1) => (rho, th) => { const r = r0 + (r1 - r0) * rho; return [cx + Math.cos(th) * A * r, cy + Math.sin(th) * B * r]; };
  /* static layers: dark bezel, thin glowing rim, iris */
  fill(make(M.plate, .0012), ell(1, 1.1));
  fill(make(M.eyeRing, .0018), ell(.93, 1));
  fill(make(M.eyeDim, .0018), ell(0, .95));
  /* moving layers */
  const mid = make(M.irisMid, .0024), inn = make(M.irisIn, .0028), pup = make(M.pupilDark, .0032);
  const g1 = make(M.catch, .0036), g2 = make(M.catch, .0036), lid = make(M.shell, .0044), low = make(M.shell, .0044);
  /* lids close along the head's horizontal, not the tilted eye's axis, so a half-closed lid never droops into a frown */
  const mUp = [Math.sin(t), Math.cos(t)];
  const lidFn = (k, dir) => { const mx = mUp[0] * dir, my = mUp[1] * dir, Ae = A * 1.01, Be = B * 1.01;
    const hmax = Math.sqrt((Ae * mx) ** 2 + (Be * my) ** 2), hl = hmax * (1 - 2 * k), wx = my, wy = -mx;
    const qa = (wx / Ae) ** 2 + (wy / Be) ** 2, qb = 2 * (mx * hl * wx / Ae ** 2 + my * hl * wy / Be ** 2), qc = (mx * hl / Ae) ** 2 + (my * hl / Be) ** 2 - 1;
    const disc = Math.max(0, qb * qb - 4 * qa * qc), s1 = (-qb - Math.sqrt(disc)) / (2 * qa), s2 = (-qb + Math.sqrt(disc)) / (2 * qa);
    return (rho, th) => { let x = Math.cos(th) * Ae * rho, y = Math.sin(th) * Be * rho; const h = x * mx + y * my;
      if (h < hl) { let sl = x * wx + y * wy; sl = clamp(sl, s1, s2); x = mx * hl + wx * sl; y = my * hl + wy * sl; } return [x, y]; }; };
  let last = '';
  const update = (gx, gy, lidK, pupilS = 1, ringS = 1, lowK = 0) => {
    const key = [gx, gy, lidK, pupilS, ringS, lowK].map(v => v.toFixed(3)).join();
    if (key === last) return; last = key;
    const cx = gx * .2 * A, cy = gy * .2 * B, ps = .423 * pupilS;
    fill(mid, disc(cx, cy, .554 * ringS, .698 * ringS));
    fill(inn, disc(cx, cy, ps, .554 * ringS));
    fill(pup, disc(cx, cy, 0, ps));
    fill(g1, disc(cx + .203 * Math.cos(2.36) * A, cy + .203 * Math.sin(2.36) * B, 0, .12));
    fill(g2, disc(cx + .238 * Math.cos(-.75) * A, cy + .238 * Math.sin(-.75) * B, 0, .055));
    fill(lid, lidFn(lidK, 1));
    fill(low, lidFn(lowK, -1));
  };
  update(0, 0, 0);
  return { g: anchor, flush: true, update, P0, n, U, V: Vv, rest: anchor.quaternion.clone(), r: B, iris: anchor };
}

function makeBigEye(parent, r, M, pos, normal, opt = {}) {
  const o = new THREE.Group(); o.position.copy(pos); o.quaternion.setFromUnitVectors(V3(0, 0, 1), normal.clone().normalize()); parent.add(o);
  /* the shape group makes the eye oval and leans its long axis */
  const g = new THREE.Group(); g.rotation.z = opt.tilt || 0; g.scale.set(opt.sx || 1, opt.sy || 1, 1); o.add(g);
  const R = r * 1.2, back = -R * .93, th = Math.asin(Math.min(.98, r / R)) + .02;
  const cap = (rad, t0, t1, mat, seg = 48) => { const geo = new THREE.SphereGeometry(rad, seg, 24, 0, Math.PI * 2, t0, t1); geo.rotateX(Math.PI / 2); return new THREE.Mesh(geo, mat); };
  /* bezel: a dark ring that sits in the shell, with a thin glowing rim inside it */
  /* a socket wall from the bezel back into the shell, so the eyeball never shows where the head curves away */
  { const wall = new THREE.CylinderGeometry(r * 1.0, r * 1.0, r * .7, 56, 1, true); wall.rotateX(Math.PI / 2); const w = new THREE.Mesh(wall, M.shellIn); w.position.z = -r * .42; g.add(w); }
  mesh(new THREE.TorusGeometry(r * 1.0, r * .045, 12, 72), M.plate, g, [0, 0, -r * .07]);
  mesh(new THREE.TorusGeometry(r * .965, r * .016, 8, 72), M.eyeRing, g, [0, 0, -r * .03]);
  /* the eyeball turns inside the socket */
  const pivot = new THREE.Group(); pivot.position.z = back; g.add(pivot);
  /* the eyeball itself stays seated; only the iris and pupil roll across it, so nothing leaves the socket */
  { const ball = cap(R, 0, th, M.eyeDim); ball.position.z = back; g.add(ball); }
  const ring = new THREE.Group(); pivot.add(ring);
  /* the part that moves is solid black: one pupil, no iris rings */
  const pupil = new THREE.Group(); pivot.add(pupil);
  pupil.add(cap(R * 1.004, 0, .36, M.pupilDark, 40));
  pupil.add(cap(R * 1.003, .36, .12, M.irisIn, 40));
  pupil.add(cap(R * 1.002, .48, .14, M.irisMid, 40));
  /* glints live on the pupil itself and travel with it */
  [[.17, 2.36, .12], [.2, -.75, .055]].forEach(([th2, ph, sz]) => {
    const d = R * 1.008, gl = mesh(new THREE.SphereGeometry(r * sz, 16, 10), M.catch, pupil, [d * Math.sin(th2) * Math.cos(ph), d * Math.sin(th2) * Math.sin(ph), d * Math.cos(th2)], [1, 1, .3]);
    gl.quaternion.setFromUnitVectors(V3(0, 0, 1), gl.position.clone().normalize());
  });
  /* eyelid: a shell hemisphere that rolls down over the eyeball */
  const lid = new THREE.Group(); lid.position.z = back; g.add(lid);
  mesh(new THREE.SphereGeometry(R * 1.05, 48, 16, 0, Math.PI * 2, 0, Math.PI / 2), M.shell, lid);
  lid.rotation.x = -1.05;
  return { g: o, shape: g, pivot, ring, pupil, lid, rest: o.quaternion.clone(), r, iris: ring };
}

/* a pincer: a thick base on the outside of the face that curves in to a fang, whose tip is itself the light */
function taperTube(curve, u0, u1, r0, r1, mat, seg = 24) {
  const pts = []; for (let i = 0; i <= seg; i++) pts.push(curve.getPoint(u0 + (u1 - u0) * i / seg));
  const path = new THREE.CatmullRomCurve3(pts);
  const geo = new THREE.TubeGeometry(path, seg, 1, 14, false);
  const pos = geo.attributes.position, rs = 15;
  for (let i = 0; i <= seg; i++) {
    const c = path.getPointAt(i / seg), k = i / seg, rad = r0 + (r1 - r0) * Math.pow(k, 1.25);
    for (let j = 0; j < rs; j++) { const idx = i * rs + j; tA.fromBufferAttribute(pos, idx).sub(c).multiplyScalar(rad).add(c); pos.setXYZ(idx, tA.x, tA.y, tA.z); }
  }
  geo.computeVertexNormals();
  return new THREE.Mesh(geo, mat);
}
function buildPincer(f, s, M) {
  const curve = new THREE.CatmullRomCurve3([V3(0, 0, 0), V3(s * .008, -.062, .012), V3(-s * .018, -.112, .034), V3(-s * .075, -.138, .062)]);
  mesh(new THREE.SphereGeometry(.044, 20, 14), M.plate, f, null, [1, .9, 1]);
  const R0 = .04, split = .62, Rs = R0 + (.006 - R0) * Math.pow(split, 1.25);
  f.add(taperTube(curve, 0, split, R0, Rs, M.shell));
  const ring = new THREE.Mesh(new THREE.TorusGeometry(Rs * 1.08, .005, 8, 32), M.glowSoft);
  ring.position.copy(curve.getPoint(split)); tB.copy(curve.getTangent(split)); ring.quaternion.setFromUnitVectors(V3(0, 0, 1), tB); f.add(ring);
  f.add(taperTube(curve, split, 1, Rs, .004, M.glow, 16));
}

/* a circuit trace: straight runs between corners, laid on the abdomen's surface; corners get solder pads */
function circuitTrace(parent, pts, rad, thick, M) {
  /* inlaid, not raised: a flat strip lying on the shell's surface, with flat pads at the ends */
  const dir = ([a, b]) => V3(Math.sin(b), Math.cos(a) * Math.cos(b), Math.sin(a) * Math.cos(b)).normalize();
  const line = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const d0 = dir(pts[i]), d1 = dir(pts[i + 1]), n = 12;
    for (let k = 0; k < n; k++) line.push(d0.clone().lerp(d1, k / n).normalize());
  }
  line.push(dir(pts[pts.length - 1]));
  const N = line.length, pos = new Float32Array(N * 6), uv = new Float32Array(N * 4), idx = [], w = thick * 1.25;
  let len = 0; const cum = [0];
  for (let i = 1; i < N; i++) { len += line[i].distanceTo(line[i - 1]); cum.push(len); }
  for (let i = 0; i < N; i++) {
    const p = line[i], tg = line[Math.min(N - 1, i + 1)].clone().sub(line[Math.max(0, i - 1)]).normalize();
    const side = V3().crossVectors(p, tg).normalize().multiplyScalar(w / rad);
    const a = p.clone().add(side).normalize().multiplyScalar(rad), b = p.clone().sub(side).normalize().multiplyScalar(rad);
    pos.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6); uv.set([cum[i] / len, 0, cum[i] / len, 1], i * 4);
    if (i < N - 1) { const q = i * 2; idx.push(q, q + 1, q + 2, q + 1, q + 3, q + 2); }
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('uv', new THREE.BufferAttribute(uv, 2)); g.setIndex(idx);
  const strip = new THREE.Mesh(g, M.circuit); strip.material.side = THREE.DoubleSide; parent.add(strip);
  [0, pts.length - 1].forEach(i => {
    const d = dir(pts[i]), pad = mesh(new THREE.CircleGeometry(thick * 2.3, 20), M.glow, parent);
    pad.position.copy(d).multiplyScalar(rad + .0004); pad.quaternion.setFromUnitVectors(V3(0, 0, 1), d);
  });
}

/* --- design 1: Warden. Armored classic, the closest heir to today's mascot. */
function buildWarden(M) {
  const g = new THREE.Group(), P = {};
  const head = mesh(boxUV(new THREE.SphereGeometry(.42, 56, 36), 1.6), M.shell, g, [0, 0, .3], [1, .68, 1.08]);
  /* no lights on the head: the eyes carry the face */
  const abd = mesh(boxUV(new THREE.SphereGeometry(.62, 64, 40), 1.1), M.shell, g, [0, .13, -.64], [1, .84, 1.22]);
  /* jagged circuit traces over the back: two mirrored runs that climb from the waist, kink outward and race down the flanks,
     plus a notched spine, with solder pads at the corners */
  [-1, 1].forEach(s => {
    circuitTrace(abd, [[1.3, .1], [.82, .13], [.6, .36], [.18, .4], [-.12, .66], [-.62, .7], [-.9, .5], [-1.32, .44]].map(([a, b]) => [a, s * b]), .6215, .0105, M);
    circuitTrace(abd, [[.6, .36], [.5, .58], [.2, .82]].map(([a, b]) => [a, s * b]), .6215, .007, M);
    circuitTrace(abd, [[-.62, .7], [-.5, .98], [-.3, 1.12]].map(([a, b]) => [a, s * b]), .6215, .007, M);
  });
  /* the spine: two small jogs, and two short 45° stubs that end in pads, one off each side */
  circuitTrace(abd, [[1.05, 0], [.45, 0], [.32, .07], [-.18, .07], [-.3, 0], [-.62, 0], [-.74, -.07], [-1.15, -.07]], .6215, .0105, M);
  circuitTrace(abd, [[.1, .07], [-.09, .26], [-.32, .26]], .6215, .0092, M);
  circuitTrace(abd, [[-.46, 0], [-.65, -.21], [-.88, -.21]], .6215, .0092, M);
  mesh(new THREE.TorusGeometry(.5, .022, 8, 96), M.glowSoft, abd, [0, -.38, 0], [1, 1, 1], [Math.PI / 2, 0, 0]);
  mesh(unitCyl(.13, .12, 16), M.joint, g, [0, .05, -.02], [1, .22, 1], [Math.PI / 2, 0, 0]);
  /* spinnerets */
  mesh(cone(.07, .14), M.plate, g, [0, .06, -1.36], null, [-Math.PI / 2, 0, 0]);
  mesh(new THREE.TorusGeometry(.065, .014, 8, 32), M.glow, g, [0, .06, -1.38]);
  P.spinRear = V3(0, .06, -1.48); P.attachTop = V3(0, .66, -.72);
  /* chelicerae + palps */
  P.fangs = [-1, 1].map(s => { const f = new THREE.Group(); f.position.set(s * .15, -.17, .62); g.add(f); buildPincer(f, s, M); return f; });
  P.palps = [];
  /* eyes */
  /* two big eyes set flush into the head: an eyeball sunk behind the shell, so only a shallow glass dome shows inside a bezel */
  /* eyes are painted onto the head's own curved surface, so they sit perfectly flush; 10% smaller than before */
  P.eyes = [-1, 1].map(s => makeFlushEye(g, M, { c: [0, 0, .3], r: [.42, .286, .454] }, V3(s * .185, .125, .655), { tilt: -s * .42, a: .153 * .82, b: .153 * 1.18 }));
  P.bodyMeshes = [head, abd]; P.abd = abd;
  P.colliders = [{ c: V3(0, 0, .3), r: V3(.42, .286, .454) }, { c: V3(0, .13, -.64), r: V3(.62, .52, .756) }, ...P.eyes.map(e => ({ c: e.g.position.clone(), r: V3(.15, .15, .15) }))];
  return { group: g, P };
}
/* iris colours measured from the reference art: outer, mid, and next to the pupil */
const IRIS = [['eyeDim', '#00c3ff'], ['irisMid', '#06defd'], ['irisIn', '#2efaff']];
function legParts(style, M) {
  if (style === 'armored') return [
    /* thick, short segments in the reference's style; elbows are capped at the leg's own radius, so no balls */
    { seg: 'coxa', geo: unitCyl(.096, .088, 20), mat: M.plate, len: true },
    { seg: 'femur', at: 0, geo: new THREE.SphereGeometry(.088, 20, 14), mat: M.plate },
    { seg: 'femur', geo: unitCyl(.088, .078, 20), mat: M.shell, len: true },
    { seg: 'femur', at: .52, geo: band(.0838, .028), mat: M.glow },
    { seg: 'tibia', at: 0, geo: new THREE.SphereGeometry(.0785, 20, 14), mat: M.shell },
    { seg: 'tibia', geo: unitCyl(.078, .05, 20), mat: M.shell, len: true },
    { seg: 'tibia', at: .1, geo: band(.0757, .026), mat: M.glow },
    { seg: 'tibia', at: .62, geo: band(.0612, .022), mat: M.glowSoft },
    { seg: 'tibia', at: 1, geo: cone(.05, .2, 16), mat: M.glow },
  ];
  if (style === 'needle') return [
    { seg: 'coxa', geo: unitCyl(.045, .04), mat: M.plate, len: true },
    { seg: 'femur', at: 0, geo: new THREE.SphereGeometry(.052, 14, 10), mat: M.joint },
    { seg: 'femur', geo: unitCyl(.032, .027), mat: M.shell, len: true },
    { seg: 'tibia', at: 0, geo: new THREE.SphereGeometry(.046, 14, 10), mat: M.joint },
    { seg: 'tibia', at: 0, geo: new THREE.SphereGeometry(.024, 10, 8), mat: M.glow },
    { seg: 'tibia', geo: unitCyl(.027, .007), mat: M.shell, len: true },
    { seg: 'tibia', at: .1, geo: unitCyl(.037, .033).translate(0, -.5, 0).scale(1, .22, 1), mat: M.plate },
    { seg: 'tibia', at: .42, geo: ringY(.026, .006), mat: M.glow },
    { seg: 'tibia', at: 1, geo: cone(.01, .22, 6), mat: M.glow },
  ];
  return [
    { seg: 'coxa', at: 0, geo: new THREE.SphereGeometry(.105, 20, 14), mat: M.plate },
    { seg: 'coxa', geo: unitCyl(.09, .086), mat: M.shell, len: true },
    { seg: 'femur', geo: unitCyl(.086, .076), mat: M.shell, len: true },
    { seg: 'tibia', at: 0, geo: new THREE.SphereGeometry(.104, 20, 14), mat: M.plate },
    { seg: 'tibia', at: 0, geo: ringY(.104, .016), mat: M.glow },
    { seg: 'tibia', geo: unitCyl(.076, .062), mat: M.shell, len: true },
    { seg: 'tibia', at: 1, geo: new THREE.SphereGeometry(.082, 16, 12), mat: M.plate },
    { seg: 'tibia', at: 1, geo: unitCyl(.07, .07, 20).translate(0, -.5, 0).scale(1, .03, 1), mat: M.glow },
  ];
}

/* 2-bone IK: knee bends toward the pole */
function ik2(root, target, l1, l2, pole, knee, foot) {
  const dir = tA.subVectors(target, root); let d = dir.length(); dir.divideScalar(d || 1);
  d = clamp(d, Math.abs(l1 - l2) + 1e-3, l1 + l2 - 1e-3);
  const ca = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), sa = Math.sqrt(Math.max(0, 1 - ca * ca));
  const n = tB.copy(pole).addScaledVector(dir, -pole.dot(dir)).normalize();
  knee.copy(root).addScaledVector(dir, ca * l1).addScaledVector(n, sa * l1);
  foot.copy(root).addScaledVector(dir, d);
}
function snapToWeb(p, out) {
  const a = wrapA(Math.atan2(p.y / WY, p.x)); let best = radialBase[0], bd = 9;
  for (const r of radialBase) { const d = Math.abs(((a - r.a) % WT + WT * 1.5) % WT - Math.PI); if (d < bd) { bd = d; best = r; } }
  const r = clamp(Math.hypot(p.x, p.y / WY), .5, best.len * .95);
  return out.set(Math.cos(best.a) * r, Math.sin(best.a) * r * WY, 0);
}

/* ---------------------------------------------------------------- the spider rig */
class Spider {
  constructor(d) {
    this.d = d; this.M = makeMats(d.accent, d.exactLights, d.finish); this.accent = C(d.accent);
    const b = d.build(this.M); this.body = b.group; this.P = b.P; scene.add(this.body);
    const L = d.leg; this.L = L;
    this.parts = legParts(L.style, this.M).map(pt => { const im = new THREE.InstancedMesh(pt.geo, pt.mat, 8); im.instanceMatrix.setUsage(THREE.DynamicDrawUsage); im.frustumCulled = false; scene.add(im); return { ...pt, im }; });
    this.legs = [];
    for (let pair = 0; pair < 4; pair++) for (const s of [1, -1]) {
      const a = L.angles[pair] * Math.PI / 180;
      const out = V3(s * Math.sin(a), 0, Math.cos(a));
      let hip = V3(s * Math.sin(a) * L.hipR, -.02, Math.cos(a) * L.hipR * .9 + L.headZ);
      if (L.surfaceHips) { /* seat the hip on the head's lower flank, below and behind the eyes */
        const H = L.surfaceHips, d = V3(s * Math.sin(a), H.y, Math.cos(a));
        const k = 1 / Math.sqrt((d.x / H.r[0]) ** 2 + (d.y / H.r[1]) ** 2 + (d.z / H.r[2]) ** 2);
        hip = V3(...H.c).addScaledVector(d, k * H.inset);
      }
      const coxa = hip.clone().addScaledVector(out, L.coxa).add(V3(0, .05, 0));
      const rest = hip.clone().addScaledVector(out, L.reach).add(V3(0, -L.height, 0));
      this.legs.push({ pair, s, out, hip, coxa, rest, foot: V3(), from: V3(), to: V3(), t: 1, stepping: false, group: (pair + (s > 0 ? 0 : 1)) % 2, w: { hip: V3(), coxa: V3(), knee: V3(), foot: V3() }, override: null, nudge: V3() });
    }
    this.pos = V3(0, 0, L.height); this.q = new THREE.Quaternion(); this.heading = 0; this.look = 0; this.roll = 0;
    this.blink = 0; this.nextBlink = 2; this.nextShift = 1.5; this.spinW = V3(); this.attachW = V3(); this.hop = 0;
    this.pend = { x: 0, z: 0, vx: 0, vz: 0, L: 3.9, vL: 0 };
    this.visible = true; this.lookTarget = V3(0, 0, 6);
  }
  /* legs never pass through the body or the eyes: push the knee out of every collider, and if the upper leg's
     midpoint is still inside one, push the knee out further along that collider's normal */
  keepOffBody(l) {
    const B = this.body, m = .085;
    const out = (p) => { let moved = false; for (const k of this.P.colliders) { tC.subVectors(p, k.c); tC.set(tC.x / (k.r.x + m), tC.y / (k.r.y + m), tC.z / (k.r.z + m)); const len = tC.length(); if (len < 1) { tC.multiplyScalar(1 / Math.max(len, 1e-4)); p.set(k.c.x + tC.x * (k.r.x + m), k.c.y + tC.y * (k.r.y + m), k.c.z + tC.z * (k.r.z + m)); moved = true; } } return moved; };
    const knee = B.worldToLocal(l.w.knee.clone()), cx = B.worldToLocal(l.w.coxa.clone());
    for (let it = 0; it < 3; it++) {
      out(knee);
      const mid = cx.clone().lerp(knee, .5), m0 = mid.clone();
      if (!out(mid)) break;
      knee.addScaledVector(mid.sub(m0), 2);
    }
    /* the lower leg too: keep the foot out, and lift it if the shin's midpoint dips into the body */
    const foot = B.worldToLocal(l.w.foot.clone());
    for (let it = 0; it < 3; it++) {
      out(foot);
      const mid = knee.clone().lerp(foot, .5), m0 = mid.clone();
      if (!out(mid)) break;
      foot.addScaledVector(mid.sub(m0), 2);
    }
    l.w.knee.copy(B.localToWorld(knee)); l.w.foot.copy(B.localToWorld(foot));
  }
  setVisible(v) { this.visible = v; this.body.visible = v; this.parts.forEach(p => (p.im.visible = v)); }
  countTris() { let t = 0; this.body.traverse(o => { if (o.isMesh) { const g = o.geometry; t += (g.index ? g.index.count : g.attributes.position.count) / 3; } }); this.parts.forEach(p => { const g = p.geo; t += (g.index ? g.index.count : g.attributes.position.count) / 3 * 8; }); return Math.round(t); }
  plant() { this.legs.forEach(l => { this.restWorld(l, l.foot); l.t = 1; l.stepping = false; }); }
  restWorld(l, out) { out.copy(l.rest).add(l.nudge); this.body.localToWorld(out); if (ctx.stage === 'hub') snapToWeb(out, out); return out; }

  update(c) {
    const dt = c.dt, t = c.t, st = c.st, S = c.state, mo = c.motion;
    const B = this.body;
    /* ---- body pose */
    let fwd = V3(), up = V3();
    if (c.stage === 'hub') {
      let h = Math.PI, look = .15, roll = 0, lift = 0;
      const sway = Math.sin(t * .4) * .12 * mo;
      if (S === 'hubIdle') { h = Math.PI + sway - c.mouse.x * .25; look = .25 + .15 * Math.sin(t * .6) * mo; }
      if (S === 'spin' || S === 'receipt' || S === 'job') { const an = ANCHORS[c.strand.anchor]; h = Math.atan2(an.p.x, -an.p.y); look = .1; }
      if (S === 'listen') { look = .55; lift = .04 * c.voice; }
      if (S === 'think') { h = Math.PI + Math.sin(t * .9) * .25 * mo; look = .2; }
      if (S === 'speak') { look = .75; h = Math.PI - c.mouse.x * .2; lift = .05 * c.voice; }
      if (S === 'wait') { look = .7; roll = .28; }
      if (S === 'error') { h = mo && Math.random() < .08 ? Math.PI + (Math.random() - .5) * .4 : this.heading; look = .3; }
      if (S === 'clickHop') { const k = clamp(st / 1.1, 0, 1); h = Math.PI + ease(k) * Math.PI * 2 * (mo ? 1 : 0); lift = Math.sin(Math.PI * clamp(st / .7, 0, 1)) * .9 * mo; look = .6; }
      if (c.hover) look = Math.max(look, .6);
      const unwrap = h - this.heading; this.heading += (Math.abs(unwrap) > Math.PI && S !== 'clickHop' ? unwrap - Math.sign(unwrap) * Math.PI * 2 : unwrap) * (1 - Math.exp(-(S === 'clickHop' ? 30 : 4) * dt));
      if (S === 'clickHop' && st > 1.1) this.heading = Math.PI;
      this.look = damp(this.look, look, 3.5, dt); this.roll = damp(this.roll, roll, 4, dt);
      const f2 = V3(-Math.sin(this.heading), Math.cos(this.heading), 0);
      const kk = this.look * .95;
      fwd.copy(f2).multiplyScalar(Math.cos(kk)).add(V3(0, 0, Math.sin(kk))).normalize();
      up.set(0, 0, 1).addScaledVector(fwd, -fwd.z).normalize();
      up.applyAxisAngle(fwd, this.roll);
      const breath = Math.sin(t * 1.6) * .015 * mo;
      const jitter = S === 'error' && mo && Math.random() < .15 ? V3((Math.random() - .5) * .08, (Math.random() - .5) * .08, 0) : V3();
      tA.set(0, 0, this.L.height + breath + lift).add(jitter);
      this.pos.lerp(tA, 1 - Math.exp(-10 * dt));
    } else {
      /* hanging: a damped pendulum on a thread whose length can change (the drop) */
      const p = this.pend;
      let targetL = 3.9;
      if (S === 'clickDrop') { const u = st; targetL = u < 1.4 ? 7.2 : 3.9; if (!mo) targetL = 3.9; }
      p.vL += ((targetL - p.L) * (S === 'clickDrop' && st < 1.4 ? 60 : 14) - p.vL * (S === 'clickDrop' && st < 1.4 ? 5 : 6)) * dt; p.L += p.vL * dt;
      const g = 9.8, k = g / p.L;
      const fx = (Math.sin(t * .7) * .25 + Math.sin(t * 1.9) * .08) * mo + (c.hover ? c.mouse.x * .6 : 0);
      p.vx += (-k * p.x - p.vx * .9 + fx * .35) * dt; p.x += p.vx * dt;
      p.vz += (-k * p.z - p.vz * .9 + Math.sin(t * .5) * .06 * mo) * dt; p.z += p.vz * dt;
      const look = c.hover || S === 'hover' ? .55 : (S === 'clickDrop' ? .7 : .25);
      this.look = damp(this.look, look, 4, dt);
      fwd.set(Math.sin(p.x * .6) + c.mouse.x * .3 * this.look, -.55 + this.look * .45, .85).normalize();
      up.set(-p.x * .4, 1, .5 - this.look * .3).addScaledVector(fwd, 0).normalize();
      up.addScaledVector(fwd, -up.dot(fwd)).normalize();
      this.attachW.set(TOP.x + Math.sin(p.x) * p.L, TOP.y - p.L * Math.cos(p.x) * Math.cos(p.z), TOP.z + Math.sin(p.z) * p.L);
    }
    const x = V3().crossVectors(up, fwd).normalize();
    tM.makeBasis(x, up, fwd); tQ.setFromRotationMatrix(tM);
    this.q.slerp(tQ, 1 - Math.exp(-(c.stage === 'hub' ? 9 : 6) * dt)); if (c.snap) this.q.copy(tQ);
    B.quaternion.copy(this.q);
    if (c.stage === 'hang') { tA.copy(this.P.attachTop).applyQuaternion(this.q); this.pos.subVectors(this.attachW, tA); }
    if (c.snap && c.stage === 'hub') this.pos.set(0, 0, this.L.height);
    B.position.copy(this.pos);
    const br = 1 + Math.sin(t * 1.6) * .02 * mo; this.P.abd.scale.y = this.P.abd.userData.sy ?? (this.P.abd.userData.sy = this.P.abd.scale.y); this.P.abd.scale.y = this.P.abd.userData.sy * br;
    B.updateMatrixWorld(true);
    this.spinW.copy(this.P.spinRear); B.localToWorld(this.spinW);
    this.attachPoint = this.P.attachTop.clone(); B.localToWorld(this.attachPoint);

    /* ---- colour: accent, amber while waiting, rose flicker on error, green flash on receipt */
    const target = S === 'wait' ? C(PAL.warn) : S === 'error' ? C(PAL.danger) : this.accent.clone();
    if (c.flash > 0) target.lerp(C(PAL.ok), c.flash);
    let inten = 1;
    if (S === 'listen') inten = .75 + c.voice * .9;
    if (S === 'speak') inten = .8 + c.voice * 1.1;
    if (S === 'think') inten = .8 + .3 * Math.sin(t * 4);
    if (S === 'error') inten = mo ? (Math.sin(t * 2.6 * Math.PI) > -.2 ? 1.1 : .25) : .9;  /* ~2.6 flashes a second, under the 3 Hz limit */
    if (c.hover || S === 'hover') inten *= 1.25;
    const LX = this.d.exactLights ? [1, .7, 1, .85] : [2.4, 1.1, 2.2, .95];
    this.M.glow.color.copy(target).multiplyScalar(LX[0] * inten);
    this.M.glowSoft.color.copy(target).multiplyScalar(LX[1] * inten);
    this.M.eye.color.copy(target).multiplyScalar(LX[2] * inten * (S === 'speak' ? 1.2 : 1));
    this.M.eyeRing.color.copy(target).multiplyScalar(LX[3] * inten);
    if (this.d.exactLights) { const cu = this.M.circuit.uniforms; cu.uA.value = .5; cu.uBase.value = .75; cu.uPulse.value = .6; }
    /* iris colours measured from the reference art (#00c3ff outer, #06defd mid, #2efaff by the pupil), pre-compensated for the tone curve */
    { const off = S === 'wait' || S === 'error', k = Math.min(1, inten);
      IRIS.forEach(([mat, hex]) => { const m = this.M[mat]; m.color.set(hex); if (off) m.color.lerp(target, .8); m.color.multiplyScalar(k); }); }
    this.M.circuit.uniforms.uColor.value.copy(target);
    this.M.circuit.uniforms.uI.value = inten;
    this.M.circuit.uniforms.uTime.value = t;
    this.M.circuit.uniforms.uSpeed.value = S === 'think' ? 1.6 : S === 'job' ? .9 : S === 'error' ? 0 : .3;

    /* ---- eyes */
    this.blink -= dt; this.nextBlink -= dt;
    if (this.nextBlink < 0 && mo) { this.blink = .14; this.nextBlink = 2 + Math.random() * 3.5; }
    const lookAt = c.eyeTarget;
    this.P.eyes.forEach((e, i) => {
      if (e.flush) {
        if (S === 'think' && mo) tA.copy(this.pos).add(V3(Math.cos(t * 2.2) * 2, Math.sin(t * 2.2) * 1.5 + 1, 2)); else tA.copy(lookAt);
        this.body.worldToLocal(tA); tA.sub(e.P0);
        const dn = Math.max(.25, tA.dot(e.n));
        e.gx = damp(e.gx || 0, clamp(tA.dot(e.U) / dn * 1.6, -1, 1), 12, dt); e.gy = damp(e.gy || 0, clamp(tA.dot(e.V) / dn * 1.6, -1, 1), 12, dt);
        /* open by default; a blink closes fully; thinking narrows a little; happy moments squint up from below */
        const lidWant = this.blink > 0 ? 1 : S === 'think' ? .26 : S === 'error' ? (mo && Math.random() < .2 ? .6 : .2) : 0;
        const lowWant = (c.hover || S === 'hover' || S === 'clickHop' || S === 'clickDrop') ? .26 : S === 'receipt' && c.flash > .3 ? .2 : 0;
        e.lidK = damp(e.lidK ?? 0, lidWant, this.blink > 0 ? 40 : 10, dt); e.lowK = damp(e.lowK ?? 0, lowWant, 10, dt);
        const pupilS = S === 'listen' || c.hover || S === 'hover' ? 1.2 : S === 'think' ? .82 : 1;
        const ringS = S === 'speak' ? 1 + c.voice * .1 : 1;
        e.update(e.gx, e.gy, clamp(e.lidK, 0, 1), pupilS, ringS, clamp(e.lowK, 0, 1));
        return;
      }
      if (e.pivot) {
        if (S === 'think' && mo) tA.copy(this.pos).add(V3(Math.cos(t * 2.2) * 2, Math.sin(t * 2.2) * 1.5 + 1, 2)); else tA.copy(lookAt);
        e.g.worldToLocal(tA);
        const yaw = clamp(Math.atan2(tA.x, Math.max(.2, tA.z)), -.34, .34), pitch = clamp(-Math.atan2(tA.y, Math.max(.2, tA.z)), -.26, .26);
        e.pivot.rotation.y = damp(e.pivot.rotation.y, yaw, 12, dt); e.pivot.rotation.x = damp(e.pivot.rotation.x, pitch, 12, dt);
        const lidWant = this.blink > 0 ? 1 : S === 'think' ? .42 : S === 'error' ? (mo && Math.random() < .2 ? .7 : .3) : S === 'wait' ? .22 : (c.hover || S === 'hover' || S === 'clickHop' || S === 'clickDrop') ? -.15 : .12;
        e.lidK = damp(e.lidK ?? 0, lidWant, this.blink > 0 ? 40 : 10, dt);
        e.lid.rotation.x = -1.05 + (Math.PI / 2 + 1.05) * clamp(e.lidK, -.1, 1);
        const pulse = S === 'speak' ? 1 + c.voice * .12 : S === 'listen' ? 1.04 + c.voice * .06 : (c.hover || S === 'hover' ? 1.06 : 1);
        e.ring.scale.setScalar(pulse); e.pupil.scale.setScalar(S === 'listen' || c.hover || S === 'hover' ? 1.25 : S === 'think' ? .8 : 1);
        return;
      }
      const save = e.g.quaternion.clone();
      if (S === 'think' && mo) { tA.copy(this.pos).add(V3(Math.cos(t * 2.2 + i) * 2, Math.sin(t * 2.2) * 1.5 + 1, 2)); e.g.lookAt(tA); }
      else e.g.lookAt(lookAt);
      const want = e.g.quaternion.clone(); e.g.quaternion.copy(save);
      const ang = e.rest.angleTo(want), lim = .55;
      if (ang > lim) want.copy(e.rest).slerp(want, lim / ang);
      e.g.quaternion.slerp(want, 1 - Math.exp(-12 * dt));
      const bl = this.blink > 0 ? .12 : 1;
      const pulse = S === 'speak' ? 1 + c.voice * .35 : S === 'listen' ? 1.1 + c.voice * .15 : (c.hover || S === 'hover' ? 1.18 : 1);
      e.iris.scale.set(pulse, pulse * bl, pulse);
      if (e.g.userData.aperture) e.g.userData.aperture.rotation.z += dt * (S === 'think' ? 4 : S === 'listen' ? 1.5 : .3) * (mo || .0);
    });
    /* ---- parts */
    const talk = S === 'speak' ? c.voice : S === 'listen' ? .15 : 0;
    this.P.fangs.forEach((f, i) => { f.rotation.x = -talk * .5 + (S === 'clickHop' || S === 'clickDrop' ? -.3 : 0); f.rotation.z = (i ? 1 : -1) * talk * .3; });
    this.P.palps.forEach((f, i) => { f.rotation.x = Math.sin(t * 9 + i * 1.7) * .35 * (talk + (S === 'think' ? .2 : 0)) * (mo || 0); });
    if (this.P.rings) this.P.rings.forEach((r, i) => { r.rotation.z += dt * (S === 'think' ? 3.5 : S === 'job' ? 1.6 : .35) * (i % 2 ? -1 : 1) * (1 + i * .3) * (mo || 0); });
    if (this.P.visor) drawVisor(this, c);

    /* ---- legs */
    this.updateLegs(c);
  }

  updateLegs(c) {
    const dt = c.dt, t = c.t, S = c.state, B = this.body, mo = c.motion, L = this.L;
    const upW = V3(0, 1, 0).applyQuaternion(this.q), fwdW = V3(0, 0, 1).applyQuaternion(this.q);
    let anyA = false, anyB = false;
    this.legs.forEach(l => { if (l.stepping) (l.group ? (anyB = true) : (anyA = true)); });
    this.nextShift -= dt;
    this.legs.forEach((l, i) => {
      l.w.hip.copy(l.hip); B.localToWorld(l.w.hip);
      l.w.coxa.copy(l.coxa); B.localToWorld(l.w.coxa);
      const outW = l.out.clone().applyQuaternion(this.q);
      let target = null;
      if (c.stage === 'hang') {
        if (l.pair === (this.P.colliders ? 3 : 0)) {
          /* a pair grips the thread above (Warden uses his hind legs, so nothing crosses his face) */
          const ap = this.attachPoint;
          tA.subVectors(thread.p[Math.max(0, thread.n - 6)], ap).normalize();
          target = ap.clone().addScaledVector(tA, .85 + (l.s > 0 ? 0 : .18)).addScaledVector(V3(1, 0, 0).applyQuaternion(this.q), l.s * .03);
        } else {
          const spread = S === 'clickDrop' && c.st < 1.6 ? 1.35 : 1;
          const loc = l.rest.clone(); loc.x *= spread; loc.y = -.5 + .55 * (spread - 1) + Math.sin(t * 1.3 + i) * .07 * mo; loc.z += Math.cos(t * 1.1 + i * 1.3) * .06 * mo;
          if ((c.hover || S === 'hover') && l.pair === 1 && l.s < 0) { loc.set(-1.0, .55 + Math.sin(t * 9) * .12 * (mo || 0), .9 + Math.cos(t * 9) * .1 * (mo || 0)); }
          if (S === 'clickDrop' && c.st < 1.6 && mo) { loc.y += Math.sin(t * 22 + i) * .12; }
          target = B.localToWorld(loc);
        }
        l.foot.lerp(target, c.snap ? 1 : 1 - Math.exp(-16 * dt));
      } else {
        /* the hub: feet stand on real radials and step when the body moves away from them */
        let ov = null;
        if (S === 'listen' && l.pair === 0) { ov = this.restWorld(l, V3()).addScaledVector(fwdW, -.15); ov.z += .25 + c.voice * .08; }
        if (S === 'wait' && l.pair === 0 && l.s > 0) { ov = this.restWorld(l, V3()); ov.z += .3 * Math.abs(Math.sin(t * 4.2)) * (mo ? 1 : .6) + (mo ? 0 : .1); }
        if (S === 'job' && l.pair === 3 && l.s < 0) { ov = strand.rope.pointAt(.07, V3()); }
        if (S === 'spin' && l.pair === 3 && strand.mode === 'shoot') { ov = this.spinW.clone().addScaledVector(outW, .25).addScaledVector(fwdW, -.2 - .2 * Math.sin(t * 16) * mo); ov.z += .1; }
        if (S === 'error' && mo) { ov = this.restWorld(l, V3()); if (Math.random() < .3) l.nudge.set((Math.random() - .5) * .3, 0, (Math.random() - .5) * .3); ov.z += Math.random() < .2 ? .15 : 0; }
        if (S === 'clickHop' && c.st < .9) { ov = B.localToWorld(l.rest.clone().multiplyScalar(.78).add(V3(0, .25, 0))); }
        if (ov) { l.foot.lerp(ov, 1 - Math.exp(-14 * dt)); l.stepping = false; l.t = 1; l.held = true; }
        else {
          if (l.held) { l.held = false; l.from.copy(l.foot); this.restWorld(l, l.to); l.t = 0; l.stepping = true; }
          const want = this.restWorld(l, tD);
          const dist = l.foot.distanceTo(want);
          const other = l.group ? anyA : anyB;
          if (!l.stepping && (dist > .32 && !other || dist > .8 || c.snap)) {
            if (c.snap) { l.foot.copy(want); }
            else { l.from.copy(l.foot); l.to.copy(want); l.t = 0; l.stepping = true; if (l.group) anyB = true; else anyA = true; }
          }
          if (l.stepping) {
            l.t = Math.min(1, l.t + dt / .15);
            const e = ease(l.t);
            l.foot.lerpVectors(l.from, l.to, e); l.foot.z += Math.sin(Math.PI * l.t) * .24;
            if (l.t >= 1) { l.stepping = false; c.onStep && c.onStep(l.foot); }
          }
        }
      }
      if (c.stage === 'hub' && S === 'hubIdle' && !(c.recall && c.recall.state === 'run') && this.nextShift < 0 && mo && i === Math.floor((t * 3.7) % 8)) {
        l.nudge.set((Math.random() - .5) * .25, 0, (Math.random() - .5) * .25); this.nextShift = 1.2 + Math.random() * 2.2;
      }
      if (S !== 'error' && S !== 'hubIdle') l.nudge.multiplyScalar(.96);
      /* IK, knees up and out */
      const pole = upW.clone().multiplyScalar(1).addScaledVector(outW, .35);
      if (c.stage === 'hang' && l.pair === (this.P.colliders ? 3 : 0)) pole.addScaledVector(fwdW, this.P.colliders ? -.6 : .6);
      ik2(l.w.coxa, l.foot, L.l1, L.l2, pole, l.w.knee, l.w.foot);
      if (this.P.colliders) this.keepOffBody(l);
    });
    /* write instance matrices */
    this.parts.forEach(pt => {
      this.legs.forEach((l, i) => {
        let a, b;
        if (pt.seg === 'coxa') { a = l.w.hip; b = l.w.coxa; } else if (pt.seg === 'femur') { a = l.w.coxa; b = l.w.knee; } else { a = l.w.knee; b = l.w.foot; }
        tA.subVectors(b, a); const len = tA.length(); tA.divideScalar(len || 1);
        tQ.setFromUnitVectors(UPY, tA);
        if (pt.len) { tB.copy(a); tS.set(1, len, 1); } else { tB.copy(a).addScaledVector(tA, len * pt.at); tS.set(1, 1, 1); }
        tM.compose(tB, tQ, tS); pt.im.setMatrixAt(i, tM);
      });
      pt.im.instanceMatrix.needsUpdate = true;
    });
  }
}
/* ---------------------------------------------------------------- post: HDR bloom, glitch, grade */
const post = (() => {
  const rtOpt = { type: THREE.HalfFloatType, depthBuffer: false };
  const sceneRT = new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType, samples: isPhone ? 0 : 4 });
  const hA = new THREE.WebGLRenderTarget(4, 4, rtOpt), hB = new THREE.WebGLRenderTarget(4, 4, rtOpt);
  const qA = new THREE.WebGLRenderTarget(4, 4, rtOpt), qB = new THREE.WebGLRenderTarget(4, 4, rtOpt);
  const outRT = new THREE.WebGLRenderTarget(4, 4, { depthBuffer: false });
  const quadScene = new THREE.Scene(), quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2)); quad.frustumCulled = false; quadScene.add(quad);
  const vs = 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy,0.,1.); }';
  const bright = new THREE.ShaderMaterial({ uniforms: { t: { value: null }, th: { value: 1.0 } }, vertexShader: vs, depthTest: false,
    fragmentShader: 'uniform sampler2D t; uniform float th; varying vec2 vUv; void main(){ vec4 s = texture2D(t,vUv); vec3 c = s.rgb; float l = max(c.r,max(c.g,c.b)); float mk = clamp(1. - abs(s.a - .5)*2., 0., 1.); vec3 hot = c*smoothstep(th,th+.9,l); vec3 soft = c*smoothstep(.3,1.,l)*.42; gl_FragColor = vec4(mix(hot, soft, mk),1.); }' });
  const blur = new THREE.ShaderMaterial({ uniforms: { t: { value: null }, dir: { value: new THREE.Vector2() } }, vertexShader: vs, depthTest: false,
    fragmentShader: 'uniform sampler2D t; uniform vec2 dir; varying vec2 vUv; void main(){ vec3 c = texture2D(t,vUv).rgb*.2270; c += (texture2D(t,vUv+dir*1.3846).rgb+texture2D(t,vUv-dir*1.3846).rgb)*.3162; c += (texture2D(t,vUv+dir*3.2308).rgb+texture2D(t,vUv-dir*3.2308).rgb)*.0703; gl_FragColor = vec4(c,1.); }' });
  const comp = new THREE.ShaderMaterial({
    uniforms: { tS: { value: null }, tA: { value: null }, tB: { value: null }, uBloom: { value: 1 }, uGlitch: { value: 0 }, uTime: { value: 0 }, uFade: { value: 0 }, uStill: { value: 0 }, uRes: { value: new THREE.Vector2() }, uMotion: { value: 1 } },
    vertexShader: vs, depthTest: false,
    fragmentShader: `uniform sampler2D tS,tA,tB; uniform float uBloom,uGlitch,uTime,uFade,uStill,uMotion; uniform vec2 uRes; varying vec2 vUv;
float h(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
vec3 aces(vec3 x){ return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.); }
void main(){
  vec2 uv = vUv; float g = uGlitch; float tk = floor(uTime*8.)*uMotion;
  if(g > 0.){ float band = floor(uv.y*26.); float r = h(vec2(band, tk)); if(r > 1. - .22*g) uv.x += (h(vec2(band, tk+3.)) - .5)*.09*g*uMotion; }
  vec2 dc = uv - .5; float ca = .0016 + .01*g;
  vec4 s = texture2D(tS, uv);
  vec3 col = vec3(texture2D(tS, uv + dc*ca + vec2(.007*g,0.)).r, s.g, texture2D(tS, uv - dc*ca - vec2(.007*g,0.)).b);
  vec3 bl = texture2D(tA, uv).rgb*.32 + texture2D(tB, uv).rgb*.62;
  if(g > 0.){ bl = vec3(texture2D(tB, uv + vec2(.012*g,0.)).r, bl.g, texture2D(tB, uv - vec2(.012*g,0.)).b); }
  col += bl*uBloom;
  float ek = clamp(1. - abs(s.a - .5)*2., 0., 1.);
  vec3 raw = clamp(s.rgb, 0., 1.);
  raw = mix(raw*12.92, 1.055*pow(raw, vec3(1./2.4)) - .055, step(.0031308, raw));
  col = aces(col*1.08);
  col = pow(col, vec3(1./2.2));
  col = mix(col, raw, ek);
  if(uStill < .5){
    col *= 1. - g*.22*step(.5, fract(uv.y*uRes.y*.25));
    float v = smoothstep(1.2, .3, length(dc*vec2(1.4,1.))); col *= mix(mix(.5, 1., v), 1., ek);
    col += (h(uv*uRes + fract(uTime)*91.) - .5)*.022*(1. - ek);
    col *= 1. - uFade;
    gl_FragColor = vec4(col, 1.);
  } else {
    float a = clamp(min(1., s.a*2.) + dot(bl, vec3(.3,.5,.2))*1.6, 0., 1.);
    gl_FragColor = vec4(col, a);
  }
}` });
  function size(w, h) {
    sceneRT.setSize(w, h); hA.setSize(w >> 1, h >> 1); hB.setSize(w >> 1, h >> 1); qA.setSize(w >> 2, h >> 2); qB.setSize(w >> 2, h >> 2);
    comp.uniforms.uRes.value.set(w, h);
  }
  function pass(mat, target) { quad.material = mat; renderer.setRenderTarget(target); renderer.render(quadScene, quadCam); }
  function run(target, w, h) {
    bright.uniforms.t.value = sceneRT.texture; pass(bright, hA);
    blur.uniforms.t.value = hA.texture; blur.uniforms.dir.value.set(1 / (w >> 1), 0); pass(blur, hB);
    blur.uniforms.t.value = hB.texture; blur.uniforms.dir.value.set(0, 1 / (h >> 1)); pass(blur, hA);
    blur.uniforms.t.value = hA.texture; blur.uniforms.dir.value.set(2 / (w >> 2), 0); pass(blur, qA);
    blur.uniforms.t.value = qA.texture; blur.uniforms.dir.value.set(0, 2 / (h >> 2)); pass(blur, qB);
    comp.uniforms.tS.value = sceneRT.texture; comp.uniforms.tA.value = hA.texture; comp.uniforms.tB.value = qB.texture;
    pass(comp, target);
  }
  return { sceneRT, outRT, comp, size, run };
})();
let ringT = -1, ringCol = C(PAL.accent), ringAt = V3();
function ringPulse(at, col) { ringT = 0; ringCol.set(col); ringAt.copy(at); }
let recallPulsed = -1;
/* recall: light runs inward along the web to the hub; when found, one green pulse lands on him */
function recallAdd(u, c) {
  const R = c.recall; if (!R) return 0;
  if (R.state === 'run') { const d = fr(u * 2.2 + c.t * .8) - .5; return (1.5 * Math.exp(-d * d * 60) + .25) * (1.1 - u * .6) * Math.min(1, R.age / 300 + (c.motion ? 0 : 1)); }
  const a = c.motion ? R.age : 1e6; if (a > 1500) return 0;
  const front = 1 - a / 700; return 3 * Math.exp(-((u - front) ** 2) * 40) + .6 * Math.max(0, 1 - a / 1500);
}
function updateWeb(c) {
  const RC = c.recall && (c.recall.state === 'run' || (c.motion && c.recall.age < 1500)), RT = c.recall && c.recall.state === 'done' ? [.56, .86, .66] : [.32, .78, 1.25];
  if (c.recall && c.recall.state === 'done' && c.recall.since !== recallPulsed && c.recall.age < 1500) { recallPulsed = c.recall.since; if (c.motion) { ringPulse(active.pos, PAL.ok); c.flash = 1; } }
  const S = c.state, tr = (S === 'listen' ? .05 * c.voice : S === 'error' ? .02 : .004) * (c.motion || (S === 'listen' ? .0 : 0));
  const spot = active.pos;
  const err = () => (S === 'error' ? .6 + .4 * Math.random() : 1);
  const tint = (u, f) => { if (RC) { const a = recallAdd(u, c) * f; tC.x += RT[0] * a; tC.y += RT[1] * a; tC.z += RT[2] * a; } return tC; };
  radialBase.forEach((r, s) => {
    for (let i = 0; i < RN; i++) { const p = r.pts[i], q = r.live[i]; const u = i / (RN - 1); q.copy(p); q.z = Math.sin(u * Math.PI * 3 + c.t * 14 + s) * tr * Math.sin(Math.PI * u); }
    webRadials.set(s, r.live, u => { const k = (.4 + .3 * Math.exp(-u * 3)) * err(); tC.set(.35 * k, .6 * k, .9 * k); return tint(u, 1); });
  });
  webRadials.flush();
  for (let i = 0; i < SPIRAL_N; i++) { spiralLive[i].copy(spiralBase[i]); spiralLive[i].z = Math.sin(i * .11 + c.t * 12) * tr * .6; }
  webSpiral.set(0, spiralLive, (u, i) => { const k = (.3 + .16 * Math.exp(-u * 2.5)) * spiralMask[i]; tC.set(.32 * k, .55 * k, .85 * k); return tint(u, .8 * spiralMask[i]); }); webSpiral.flush();
  webHub.set(0, hubMesh.pts, () => { const k = .55 * err(); tC.set(.4 * k, .65 * k, .95 * k); return tint(.02, .6); }); webHub.flush();
  frameLines.forEach((f, s) => {
    for (let i = 0; i < FN; i++) { const u = i / (FN - 1); f.live[i].copy(f.pts[i]); f.live[i].z = Math.sin(u * Math.PI * 2 + c.t * 10 + s) * tr * .8 * Math.sin(Math.PI * u); }
    webFrame.set(s, f.live, () => { const k = .78 * err(); return tC.set(.36 * k, .6 * k, .92 * k); });
  });
  webFrame.flush();
  moorings.forEach((m, s) => webMoor.set(s, m.pts, u => { const k = .75 * Math.pow(1 - u, 2.4); return tC.set(.34 * k, .56 * k, .88 * k); }));
  webMoor.flush();
  { const P = dew.geometry.attributes.position.array, C = dew.geometry.attributes.color.array, ph = dew.userData.ph;
    dewIdx.forEach((si, n) => {
      const p = spiralLive[si]; P[n * 3] = p.x; P[n * 3 + 1] = p.y; P[n * 3 + 2] = p.z + .004;
      const [o, w] = ph[n], gl = c.motion ? Math.pow(Math.max(0, Math.sin(c.tReal * w + o + par.x * 3 + p.x * .6)), 24) : 0, k = .22 + 2.4 * gl;
      C[n * 3] = .55 * k; C[n * 3 + 1] = .8 * k; C[n * 3 + 2] = 1.1 * k;
    });
    dew.geometry.attributes.position.needsUpdate = true; dew.geometry.attributes.color.needsUpdate = true; }
  hubGlow.material.opacity = (S === 'wait' ? .6 : S === 'error' ? .5 : .85) + .15 * Math.sin(c.t * 1.6) * c.motion + (c.recall && c.recall.state === 'run' ? .35 : 0);
  hubGlow.material.color.set(S === 'wait' ? PAL.warn : S === 'error' ? PAL.danger : '#ffffff');
  hubLight.color.set(S === 'wait' ? PAL.warn : S === 'error' ? PAL.danger : PAL.accent); hubLight.intensity = 3 + (S === 'speak' || S === 'listen' ? c.voice * 4 : 0);
  /* listening rings */
  if (S === 'listen' && c.motion && ringT < 0 && c.voice > .85) ringPulse(spot, PAL.accent);
  if (ringT >= 0) {
    ringT += c.dt; const k = ringT / 1.1; pulseRing.visible = true;
    pulseRing.position.copy(ringAt); pulseRing.position.z = .05; pulseRing.scale.setScalar(.4 + k * 3.2);
    pulseRing.material.color.copy(ringCol).multiplyScalar(3 * (1 - k)); pulseRing.material.opacity = 1 - k;
    if (k >= 1) { ringT = -1; pulseRing.visible = false; }
  }
  /* thinking motes orbit the body */
  motes.visible = S === 'think';
  if (motes.visible) for (let i = 0; i < 10; i++) { const a = c.t * (1.4 + i * .07) * (c.motion || .0) + i * .63; tB.set(Math.cos(a) * (1.1 + .2 * (i % 3)), Math.sin(a) * (1.1 + .2 * (i % 3)), .5 + .3 * Math.sin(a * 2 + i)).add(spot); tM.makeTranslation(tB.x, tB.y, tB.z); motes.setMatrixAt(i, tM); }
  motes.instanceMatrix.needsUpdate = true;
}

/* ================================================================ Living Web round 2: the scene, driven by window.LW */

const DESIGN = { key: 'warden', name: 'Warden', accent: '#00c3ff', exactLights: true, finish: 'matte', build: buildWarden,
  leg: { style: 'armored', l1: .72, l2: .84, coxa: .12, hipR: .35, headZ: .3, reach: 1.22, angles: [58, 86, 114, 142], height: .36, scale: 1,
    surfaceHips: { c: [0, 0, .3], r: [.42, .286, .454], y: -.28, inset: .9 } } };
const APPS = ['tasks', 'music', 'chess'], AIDX = { tasks: 0, music: 1, chess: 2 };
const ctx = { t: 0, tReal: 0, dt: 0, st: 0, state: 'hubIdle', stage: 'hub', motion: LW.rm ? 0 : 1, mouse: new THREE.Vector2(), eyeTarget: V3(0, 0, 8), eyeLocal: new THREE.Vector2(), voice: 0, hover: false, flash: 0, snap: true, strand: null, jobProgress: 0, recall: null };
/* the focus strand the rig reads: which anchor he turns to, and which rope a hind leg rests on during a job */
const strand = { anchor: 1, mode: 'none', rope: null };
ctx.strand = strand;
const active = new Spider(DESIGN);
let reducedNow = LW.rm;

/* ---------------------------------------------------------------- one strand per app, its look a function of ledger state */
const COL = { blue: [.32, .78, 1.25], green: [.56, .86, .66], amber: [.93, .79, .54], rose: [.95, .56, .64], silk: [.5, .62, .8] };
const fr = x => ((x % 1) + 1) % 1;
class Strand {
  constructor(app) {
    this.app = app; this.an = ANCHORS[AIDX[app]]; this.rope = new Rope(30); this.rib = new Ribbon(1, 30, .016);
    scene.add(this.rib.mesh); this.rib.mesh.visible = false; this.tip = V3(); this.key = ''; this.prog = -1; this.arrived = false;
  }
  anchor(state, age, c) {
    const an = this.an, mo = c.motion, T = c.tReal;
    let col = PAL.accent, k = 1.1, ring = 1, em = null;
    if (state === 'queued') k = 1.4;
    if (state === 'run') { k = 2.6; ring = 1 + .09 * Math.sin(T * 6) * mo; }
    if (state === 'job') k = 2.4;
    if (state === 'wait') { const p = Math.sin(T * Math.PI * 2 / 1.6) * mo; col = PAL.warn; k = 2.2 + .8 * p; ring = 1 + .1 * p; }
    if (state === 'done') { if (age >= 180) { col = PAL.ok; k = 3; em = PAL.ok; } else k = 2.6; }
    if (state === 'failed') { col = PAL.danger; k = 2.6; em = PAL.danger; }
    an.mat.color.set(col).multiplyScalar(k); an.ring.scale.setScalar(ring);
    const m = an.core.material; m.emissiveIntensity = an.core.userData.ei * (state === 'done' && age >= 180 ? 1.6 : state === 'failed' ? .5 : 1);
    an.core.rotation.y = Math.sin(T * .55 + an.p.x) * .38 * mo; an.core.rotation.x = .38;   /* a slow turn, tipped toward the camera */
    const job = LW.S.job; an.prog.visible = state === 'job' && !!job;
    if (an.prog.visible) { const p = job.n / job.of; if (p !== this.prog) { this.prog = p; an.prog.geometry.dispose(); an.prog.geometry = new THREE.TorusGeometry(.44, .02, 8, 96, Math.PI * 2 * p); } }
  }
  update(c, info, clock) {
    const sp = active, R = this.rope, mo = c.motion;
    const state = info ? info.state : 'none';
    let age = info ? clock - info.since : 0;
    this.anchor(state, age, c);
    if (state === 'none') { this.rib.mesh.visible = false; this.key = ''; return; }
    if (!mo) age = 1e6;                     /* reduced motion: every transition is already finished */
    let reach = .8;
    if (state === 'queued') reach = .8 * clamp((age - (info.k || 0) * 300) / 300, 0, 1);
    if (state === 'done') reach = .8 + .2 * clamp(age / 180, 0, 1);
    reach = Math.max(reach, .002);
    this.tip.lerpVectors(sp.spinW, this.an.p, reach);
    const key = state + info.since;
    if (this.key === '') R.reset(sp.spinW, this.tip);
    if (key !== this.key) { if (state === 'failed' && mo) R.pluck(.14, V3(0, 0, 1)); this.key = key; this.arrived = false; }
    if (state === 'done' && !this.arrived && age >= 1180) {
      this.arrived = true;
      if (age < 2600) { R.pluck(.06, V3(0, 0, 1)); ringPulse(sp.pos, PAL.ok); c.flash = 1; }
    }
    R.step(c.dt, sp.spinW, this.tip, state === 'failed' ? 1.12 : state === 'queued' ? 1.03 : 1.015, 2.4);
    for (let i = 0; i < R.n; i++) {
      R.r[i].copy(R.p[i]);
      if (state === 'job') { const u = i / (R.n - 1); R.r[i].z += Math.sin(Math.PI * u * 5) * Math.sin(c.t * 38) * .018 * Math.sin(Math.PI * u) * mo; }
    }
    const T = c.t, pk = 1 - clamp((age - 180) / 1000, 0, 1);
    const col = (a, k) => tC.set(a[0] * k, a[1] * k, a[2] * k);
    const fn = u => {
      const g = u * reach;
      if (state === 'queued') return col(COL.silk, fr(g * 14) < .4 ? 2.2 : .08);
      if (state === 'run') { let k = 1 + 2.4 * (fr(g * 9 - T * 1.6) < .35); if (age < 900) k += 5 * Math.exp(-((g - age / 900 * .8) ** 2) * 500); return col(COL.blue, k); }
      if (state === 'job') { const s = .5 + .5 * Math.sin(g * 40 - T * 9); return col(COL.blue, 1 + 1.5 * s * s); }
      if (state === 'wait') return col(COL.amber, .7 + 2 * (fr(g * 7 - T * .7) < .4));
      if (state === 'done') {
        if (age < 1180) { const e = Math.exp(-((g - pk) ** 2) * 700) * 5; return g > pk ? col(COL.green, 2 + e) : col(COL.blue, 1 + 2 * (fr(g * 9 - T * 1.6) < .35) + e); }
        return col(COL.green, 1.3 + 1.2 * Math.exp(-(age - 1180) / 700));
      }
      if (u > .78 && u < .88) return tC.set(0, 0, 0);       /* failed: the strand frays */
      const fl = age < 1400 ? (Math.sin(age / 1000 * 2.6 * Math.PI * 2) > -.2 ? 1 : .35) : 1;
      return col(COL.rose, 1.6 * fl);
    };
    this.rib.mesh.visible = true;
    this.rib.set(0, R.r, fn, u => (.7 + .5 * Math.sin(Math.PI * u)) * (state === 'queued' ? .8 : 1) * (state === 'failed' && u > .88 ? .55 : 1));
    this.rib.flush();
  }
}
const strands = Object.fromEntries(APPS.map(a => [a, new Strand(a)]));

/* ---------------------------------------------------------------- what Warden does, from the same state */
let hopUntil = -1;
function spiderState(S, clock) {
  if (performance.now() < hopUntil) return { state: 'clickHop' };
  if (S.lastFail && clock - S.lastFail.t < 1500 && S.steps[Object.keys(S.steps).find(k => S.steps[k].state === 'failed')]) return { state: 'error', focus: S.lastFail.app };
  if (S.pres === 'listen' || S.pres === 'think' || S.pres === 'speak') return { state: S.pres };
  if (S.pres === 'wait') return { state: 'wait', focus: S.confirm ? S.confirm.app : null };
  if (S.pres === 'work') {
    let f = S.lastStart, mode = 'run';
    if (S.lastReceipt && f && S.lastReceipt.t > f.t) f = S.lastReceipt; else if (f && clock - f.t < 900) mode = 'shoot';
    if (f && f.app === 'conductor') return { state: 'hubIdle' };   /* recall is his own work: he stays at the hub, still */
    return f ? { state: 'spin', focus: f.app, mode } : { state: 'hubIdle' };
  }
  return S.job ? { state: 'job', focus: S.job.app, mode: 'hum' } : { state: 'hubIdle' };
}
function voiceEnv(t) { const g = Math.sin(t * 1.25) > -.35 ? 1 : 0; return clamp((.45 + .55 * Math.sin(t * 29)) * (.55 + .45 * Math.sin(t * 6.7 + 1)) * g, 0, 1); }

/* ---------------------------------------------------------------- framing: the web fits the area beside (or above) the panel */
let W = 2, H = 2, DPR = 1, panelPx = 0, panelNow = -1, dist = 14;
const viewport = { fullW: 2, fullH: 2, regW: 2, regH: 2 };
function measurePanel() {
  const r = canvas.getBoundingClientRect(), pr = $('panel').getBoundingClientRect();
  panelPx = (LW.view === 'phone' ? Math.max(0, r.bottom - pr.top + 8) : Math.max(0, r.right - pr.left + 8)) * DPR;
}
function resize() {
  const r = canvas.getBoundingClientRect();
  DPR = Math.min(devicePixelRatio || 1, isPhone ? 1.5 : 2);
  W = Math.max(2, Math.round(r.width * DPR)); H = Math.max(2, Math.round(r.height * DPR));
  renderer.setSize(W, H, false); post.size(W, H); measurePanel(); stillFrames = 30;
}
function applyView(dt) {
  panelNow = panelNow < 0 || ctx.snap ? panelPx : damp(panelNow, panelPx, 5, dt);
  const P = Math.round(panelNow), v = viewport;
  if (LW.view === 'phone') { v.fullW = W; v.fullH = H + P; v.regW = W; v.regH = Math.max(H * .3, H - P); camera.setViewOffset(v.fullW, v.fullH, 0, P, W, H); }
  else { v.fullW = W + P; v.fullH = H; v.regW = Math.max(W * .3, W - P); v.regH = H; camera.setViewOffset(v.fullW, v.fullH, P, 0, W, H); }
  camera.fov = 32; camera.aspect = v.fullW / v.fullH; camera.updateProjectionMatrix();
}
const ro1 = new ResizeObserver(resize); ro1.observe(canvas);
const ro2 = new ResizeObserver(() => { measurePanel(); stillFrames = 30; }); ro2.observe($('panel'));

const CAMDIR = V3(0, -4.25, 10.2).normalize(), CAMTGT = V3(0, .05, 0), AX = V3(1, 0, 0), tgt = V3();
const par = new THREE.Vector2();
function frameCamera(dt, S) {
  const v = viewport, fit = LW.view === 'phone' ? 4.5 : 4.85;
  let d0 = fit * v.fullH / (Math.tan(camera.fov * Math.PI / 360) * Math.min(v.regW, v.regH));
  if (S.pres === 'listen' || S.pres === 'speak') d0 *= .92; else if (S.pres === 'think') d0 *= 1.03;
  dist = ctx.snap ? d0 : damp(dist, d0, 1.3, dt);
  par.lerp(ctx.mouse, 1 - Math.exp(-3 * dt));
  const mo = ctx.motion, az = (Math.sin(ctx.tReal * .07) * .1 + par.x * .2) * mo, el = (Math.sin(ctx.tReal * .05) * .05 + par.y * .1) * mo;
  tA.copy(CAMDIR).applyAxisAngle(UPY, az).applyAxisAngle(AX, -el);
  tgt.copy(CAMTGT); if (LW.view === 'phone') tgt.y += .75;   /* phone: leave room above the web for the HUD */
  camera.position.copy(tgt).addScaledVector(tA, dist); camera.up.set(0, 1, 0); camera.lookAt(tgt);
  scene.fog.density = .5 / dist;
}

/* ---------------------------------------------------------------- input: cursor or tilt for parallax; click him for a hop */
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
$('device').addEventListener('pointermove', e => {
  const r = canvas.getBoundingClientRect(); ndc.set((e.clientX - r.left) / r.width * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ctx.mouse.copy(ndc); canvas.style.cursor = hit() ? 'pointer' : '';
});
$('device').addEventListener('pointerleave', () => ctx.mouse.set(0, 0));
const onTilt = e => { if (e.gamma == null) return; ctx.mouse.set(clamp(e.gamma / 30, -1, 1), clamp((e.beta - 45) / 40, -1, 1)); };
addEventListener('deviceorientation', onTilt);
function hit() { ray.setFromCamera(ndc, camera); return ray.intersectObjects(active.P.bodyMeshes, false).length > 0; }
canvas.addEventListener('click', () => { if (hit()) hopUntil = performance.now() + 1700; });

/* ---------------------------------------------------------------- the loop */
let last = performance.now(), stillFrames = 60, lastKey = '', fadeNow = 0, ready = false;
const costs = []; let fpsAcc = 0, fpsN = 0, frameNo = 0;
const fmt = n => n.toLocaleString('en-US');
function frame(now) {
  if (disposed) return;
  rafId = requestAnimationFrame(frame);
  const S = LW.S, clock = LW.now();
  const rmNow = LW.rm; if (rmNow !== reducedNow) { reducedNow = rmNow; ctx.snap = true; }
  ctx.motion = rmNow ? 0 : 1;
  const key = S.idx + LW.view + ctx.motion + (performance.now() < hopUntil);
  if (key !== lastKey) { lastKey = key; stillFrames = 60; }
  const rawDt = (now - last) / 1000, dt = Math.min(.05, rawDt); last = now;
  if (!ctx.motion && stillFrames <= 0) return;           /* reduced motion: a still scene, redrawn only when the ledger changes */
  stillFrames--;
  const t0 = performance.now();
  ctx.dt = dt; ctx.tReal += dt; ctx.t += dt * ctx.motion;
  const ss = spiderState(S, clock);
  if (ss.state !== ctx.state) { ctx.state = ss.state; ctx.st = 0; } else ctx.st += dt;
  if (ss.focus) strand.anchor = AIDX[ss.focus];
  strand.mode = ss.mode || 'run'; strand.rope = strands[ss.focus || 'music'].rope;
  ctx.voice = ctx.state === 'speak' || ctx.state === 'listen' ? (ctx.motion ? voiceEnv(ctx.tReal) : .5) : 0;
  ctx.jobProgress = S.job ? S.job.n / S.job.of : 0;
  ctx.recall = S.recall ? { state: S.recall.state, age: clock - S.recall.since, since: S.recall.since } : null;
  applyView(dt); frameCamera(dt, S);
  /* eyes: the cursor, projected in front of him; while he works, the app he works on */
  ray.setFromCamera(ctx.mouse, camera);
  ctx.eyeTarget.copy(ray.ray.origin).addScaledVector(ray.ray.direction, camera.position.distanceTo(active.pos) * .55);
  if (ss.focus && (ctx.state === 'spin' || ctx.state === 'job')) ctx.eyeTarget.lerp(ANCHORS[strand.anchor].p.clone().setZ(1), .6);
  active.update(ctx);
  APPS.forEach(a => strands[a].update(ctx, S.strands[a], clock));
  updateWeb(ctx);
  ctx.flash = Math.max(0, ctx.flash - dt * .8);
  dust.rotation.y += dt * .01 * ctx.motion; dust.position.y = Math.sin(ctx.tReal * .1) * .2 * ctx.motion;
  const comp = post.comp.uniforms;
  comp.uGlitch.value = ctx.state === 'error' ? (ctx.motion ? .5 + .4 * (Math.sin(ctx.tReal * 2.6 * Math.PI) > .4 ? 1 : 0) : .3) : 0;
  fadeNow = ctx.snap ? (S.screen ? 0 : .95) : damp(fadeNow, S.screen ? 0 : .95, 2.5, dt);
  comp.uTime.value = ctx.tReal; comp.uFade.value = fadeNow; comp.uStill.value = 0; comp.uMotion.value = ctx.motion;
  renderer.info.reset();
  renderer.setRenderTarget(post.sceneRT); renderer.render(scene, camera);
  const tris = renderer.info.render.triangles;
  post.run(null, W, H);
  const calls = renderer.info.render.calls;
  ctx.snap = false;
  if (++frameNo % 20 === 0) { renderer.getContext().finish(); costs.push(performance.now() - t0); if (costs.length > 60) costs.shift(); }
  if (!ready) { ready = true; canvas.classList.add('on'); onReady(Math.round(performance.now())); }
  fpsAcc += rawDt; fpsN++;
  if (fpsAcc > .5) {
    const fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0;
    const cs = costs.slice().sort((a, b) => a - b), med = cs.length ? cs[cs.length >> 1] : 0;
    onPerf({ fps, cost: med, tris, calls, W, H, state: ctx.state, label: `${fps.toFixed(0)} fps · ${med.toFixed(1)} ms · ${fmt(tris)} tris · ${calls} calls` });
  }
}
resize();
rafId = requestAnimationFrame(t => { last = t; rafId = requestAnimationFrame(frame); });
return {
  relayout() { requestAnimationFrame(measurePanel); stillFrames = 30; },
  dispose() {
    disposed = true; cancelAnimationFrame(rafId); ro1.disconnect(); ro2.disconnect(); removeEventListener('deviceorientation', onTilt);
    scene.traverse(o => { if (o.geometry) o.geometry.dispose(); const m = o.material; if (m) (Array.isArray(m) ? m : [m]).forEach(x => { if (x.map) x.map.dispose(); x.dispose(); }); });
    renderer.dispose();
  },
};

}
