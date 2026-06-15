const BlinnPhongPatina = {
    lightSource: { x: 0.3, y: -0.4, z: 0.86 },
    ambientStrength: 0.25,
    diffuseStrength: 0.7,
    specularStrength: 0.35,
    shininess: 48.0,
    patina: {
        baseColor: [0.58, 0.80, 0.56],
        patinaColor: [0.80, 0.55, 0.25],
        highlightColor: [1.0, 0.95, 0.85],
        transitionSharpness: 2.5
    },

    normalize(v) {
        const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) + 1e-12;
        return [v[0]/len, v[1]/len, v[2]/len];
    },

    dot(a, b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    },

    reflect(v, n) {
        const d = 2 * this.dot(v, n);
        return [v[0] - d*n[0], v[1] - d*n[1], v[2] - d*n[2]];
    },

    computeNormal(x, y, shapeFunc) {
        const eps = 0.005;
        const h0 = shapeFunc(x, y);
        const hx = shapeFunc(x + eps, y);
        const hy = shapeFunc(x, y + eps);
        const nx = (h0 - hx) / eps;
        const ny = (h0 - hy) / eps;
        const nz = 1.0;
        return this.normalize([nx, ny, nz]);
    },

    jadeBiShape(x, y) {
        const r = Math.sqrt(x*x + y*y);
        if (r < 0.12 || r > 0.9) return 0;
        const profile = Math.exp(-Math.pow((r - 0.5) / 0.35, 2));
        return profile * 0.18;
    },

    jadeCongShape(x, y) {
        const ax = Math.abs(x), ay = Math.abs(y);
        const r = Math.sqrt(x*x + y*y);
        if (r > 0.9) return 0;
        if (r < 0.18) return -0.02;
        const edgeSharp = Math.max(0, Math.pow(Math.max(ax, ay), 12)) * 0.1;
        const radial = Math.exp(-Math.pow(r / 0.8, 4));
        return radial * 0.15 + edgeSharp * 0.05;
    },

    jadeZhuShape(x, y) {
        const r = Math.sqrt(x*x + y*y);
        if (r > 0.98) return 0;
        const sphere = Math.sqrt(Math.max(0, 1.0 - r*r));
        return sphere * 0.25;
    },

    jadeGuanShape(x, y) {
        const ax = Math.abs(x), ay = Math.abs(y);
        if (ax > 0.4 || ay > 0.9) return 0;
        const profile = Math.exp(-Math.pow(ax / 0.4, 8));
        const taper = 1.0 - Math.pow(ay / 0.9, 4) * 0.3;
        return profile * 0.2 * taper;
    },

    jadeGeneralShape(x, y) {
        const r = Math.sqrt(x*x + y*y);
        if (r > 0.95) return 0;
        return Math.exp(-Math.pow(r / 0.7, 3)) * 0.2;
    },

    getShapeFunction(jadeType) {
        const map = {
            '玉璧': this.jadeBiShape.bind(this),
            '玉琮': this.jadeCongShape.bind(this),
            '玉珠': this.jadeZhuShape.bind(this),
            '玉管': this.jadeGuanShape.bind(this)
        };
        return map[jadeType] || this.jadeGeneralShape.bind(this);
    },

    computeColor(x, y, patinaCoverage, shapeFunc, light, polishLevel) {
        const h = shapeFunc(x, y);
        if (h <= 0) return null;

        const normal = this.computeNormal(x, y, shapeFunc);
        const viewDir = this.normalize([-x * 0.5, -y * 0.5, 1.0]);
        const halfDir = this.normalize([
            light[0] + viewDir[0],
            light[1] + viewDir[1],
            light[2] + viewDir[2]
        ]);

        const ambient = this.ambientStrength;

        const diffDot = Math.max(0, this.dot(normal, light));
        const diffuse = this.diffuseStrength * diffDot;

        const specDot = Math.max(0, this.dot(normal, halfDir));
        const effectiveShininess = this.shininess * (0.3 + polishLevel * 0.7);
        const specular = this.specularStrength * Math.pow(specDot, effectiveShininess) * polishLevel;

        const r = Math.sqrt(x*x + y*y);
        const radialPatina = 1.0 - Math.exp(-Math.pow(r / (0.3 + patinaCoverage * 0.5), this.patina.transitionSharpness));
        const localPatina = Math.max(0, patinaCoverage * radialPatina - 0.2 * h);

        const bc = this.patina.baseColor;
        const pc = this.patina.patinaColor;
        const hc = this.patina.highlightColor;

        let base = [
            bc[0] * (1 - localPatina) + pc[0] * localPatina,
            bc[1] * (1 - localPatina) + pc[1] * localPatina,
            bc[2] * (1 - localPatina) + pc[2] * localPatina
        ];

        const lit = [
            base[0] * (ambient + diffuse) + hc[0] * specular,
            base[1] * (ambient + diffuse) + hc[1] * specular,
            base[2] * (ambient + diffuse) + hc[2] * specular
        ];

        return [
            Math.min(1, lit[0]),
            Math.min(1, lit[1]),
            Math.min(1, lit[2])
        ];
    },

    drawPatinaSurface(canvas, jadeType, options = {}) {
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        const patinaCoverage = options.patinaCoverage ?? 0.5;
        const polishLevel = options.polishLevel ?? 0.5;
        const lightAngle = options.lightAngle ?? 0;

        const light = this.normalize([
            Math.sin(lightAngle) * 0.6,
            -0.4,
            Math.cos(lightAngle) * 0.6
        ]);

        const shapeFunc = this.getShapeFunction(jadeType);

        const bgGradient = ctx.createRadialGradient(w/2, h/2, 0, w/2, h/2, w/2);
        bgGradient.addColorStop(0, '#1e2030');
        bgGradient.addColorStop(1, '#0f1018');
        ctx.fillStyle = bgGradient;
        ctx.fillRect(0, 0, w, h);

        const imgData = ctx.getImageData(0, 0, w, h);
        const data = imgData.data;
        const cx = w / 2, cy = h / 2;
        const scale = Math.min(w, h) / 2.1;

        for (let py = 0; py < h; py++) {
            for (let px = 0; px < w; px++) {
                const nx = (px - cx) / scale;
                const ny = (py - cy) / scale;
                const color = this.computeColor(
                    nx, ny, patinaCoverage, shapeFunc, light, polishLevel
                );
                if (color) {
                    const idx = (py * w + px) * 4;
                    data[idx] = Math.floor(color[0] * 255);
                    data[idx + 1] = Math.floor(color[1] * 255);
                    data[idx + 2] = Math.floor(color[2] * 255);
                    data[idx + 3] = 255;
                }
            }
        }
        ctx.putImageData(imgData, 0, 0);
    }
};

const VirtualPlay = {
    state: {
        polishLevel: 0.3,
        patinaCoverage: 0.6,
        playCount: 0,
        wearMap: null,
        lightAngle: 0
    },

    init(canvas, jadeType, options = {}) {
        this.canvas = canvas;
        this.jadeType = jadeType;
        this.state.polishLevel = options.initialPolish ?? 0.3;
        this.state.patinaCoverage = options.initialPatina ?? 0.6;
        this.state.playCount = 0;
        this.state.lightAngle = 0;

        const w = canvas.width;
        const h = canvas.height;
        this.state.wearMap = new Float32Array(w * h).fill(0);
        this._bindEvents();
        this.render();
    },

    _bindEvents() {
        if (this._bound) return;
        this._bound = true;

        const canvas = this.canvas;
        let isDrawing = false;
        let lastX = 0, lastY = 0;

        const getPos = (e) => {
            const rect = canvas.getBoundingClientRect();
            const cx = e.clientX ?? (e.touches && e.touches[0].clientX);
            const cy = e.clientY ?? (e.touches && e.touches[0].clientY);
            return {
                x: (cx - rect.left) * (canvas.width / rect.width),
                y: (cy - rect.top) * (canvas.height / rect.height)
            };
        };

        const applyWear = (x, y, strength = 1.0) => {
            const w = canvas.width;
            const h = canvas.height;
            const wearMap = this.state.wearMap;
            const radius = Math.max(15, Math.min(w, h) * 0.08);
            const radiusSq = radius * radius;

            for (let dy = -radius; dy <= radius; dy++) {
                for (let dx = -radius; dx <= radius; dx++) {
                    const distSq = dx * dx + dy * dy;
                    if (distSq > radiusSq) continue;
                    const px = Math.floor(x + dx);
                    const py = Math.floor(y + dy);
                    if (px < 0 || px >= w || py < 0 || py >= h) continue;
                    const falloff = 1.0 - distSq / radiusSq;
                    const idx = py * w + px;
                    wearMap[idx] = Math.min(1.0, wearMap[idx] + falloff * 0.02 * strength);
                }
            }

            this.state.playCount++;
            this.state.polishLevel = Math.min(1.0, this.state.polishLevel + 0.0008);
            this.state.patinaCoverage = Math.max(0.0, this.state.patinaCoverage - 0.0005);
        };

        const onStart = (e) => {
            e.preventDefault();
            isDrawing = true;
            const p = getPos(e);
            lastX = p.x; lastY = p.y;
            applyWear(p.x, p.y, 1.2);
            this.render();
            this._emitPlayEvent(p.x, p.y);
        };

        const onMove = (e) => {
            if (!isDrawing) return;
            e.preventDefault();
            const p = getPos(e);

            const steps = Math.ceil(Math.hypot(p.x - lastX, p.y - lastY) / 5);
            for (let i = 0; i < steps; i++) {
                const t = i / Math.max(1, steps);
                applyWear(
                    lastX + (p.x - lastX) * t,
                    lastY + (p.y - lastY) * t,
                    0.6
                );
            }
            lastX = p.x; lastY = p.y;
            this.render();
        };

        const onEnd = () => {
            isDrawing = false;
        };

        canvas.addEventListener('mousedown', onStart);
        canvas.addEventListener('mousemove', onMove);
        canvas.addEventListener('mouseup', onEnd);
        canvas.addEventListener('mouseleave', onEnd);
        canvas.addEventListener('touchstart', onStart, { passive: false });
        canvas.addEventListener('touchmove', onMove, { passive: false });
        canvas.addEventListener('touchend', onEnd);

        let animTime = 0;
        this._lightAnim = setInterval(() => {
            animTime += 0.03;
            this.state.lightAngle = Math.sin(animTime) * 0.4;
            if (isDrawing) this.render();
        }, 50);
    },

    render() {
        const canvas = this.canvas;
        const w = canvas.width;
        const h = canvas.height;
        const ctx = canvas.getContext('2d');

        BlinnPhongPatina.drawPatinaSurface(canvas, this.jadeType, {
            patinaCoverage: this.state.patinaCoverage,
            polishLevel: this.state.polishLevel,
            lightAngle: this.state.lightAngle
        });

        if (this.state.wearMap) {
            const imgData = ctx.getImageData(0, 0, w, h);
            const data = imgData.data;
            const wear = this.state.wearMap;

            for (let i = 0; i < wear.length; i++) {
                if (wear[i] <= 0) continue;
                const idx = i * 4;
                const alpha = wear[i] * 0.25;
                data[idx] = Math.min(255, data[idx] + Math.floor(alpha * 200));
                data[idx + 1] = Math.min(255, data[idx + 1] + Math.floor(alpha * 180));
                data[idx + 2] = Math.min(255, data[idx + 2] + Math.floor(alpha * 140));
            }
            ctx.putImageData(imgData, 0, 0);

            const sum = wear.reduce((a, b) => a + b, 0);
            const avgWear = sum / wear.length;

            ctx.save();
            ctx.globalAlpha = 0.6;
            const grd = ctx.createRadialGradient(w*0.3, h*0.25, 0, w*0.3, h*0.25, w*0.15);
            grd.addColorStop(0, 'rgba(255, 240, 220, ' + (0.15 + avgWear * 0.5) + ')');
            grd.addColorStop(1, 'rgba(255, 240, 220, 0)');
            ctx.fillStyle = grd;
            ctx.beginPath();
            ctx.arc(w*0.3, h*0.25, w*0.15, 0, Math.PI*2);
            ctx.fill();
            ctx.restore();
        }
    },

    getStats() {
        const wear = this.state.wearMap;
        let avgWear = 0, maxWear = 0, wearArea = 0;
        if (wear) {
            for (let i = 0; i < wear.length; i++) {
                avgWear += wear[i];
                if (wear[i] > maxWear) maxWear = wear[i];
                if (wear[i] > 0.1) wearArea++;
            }
            avgWear /= wear.length;
            wearArea = wearArea / wear.length;
        }
        return {
            playCount: this.state.playCount,
            polishLevel: this.state.polishLevel,
            patinaCoverage: this.state.patinaCoverage,
            averageWear: avgWear,
            maxWear: maxWear,
            wearAreaRatio: wearArea
        };
    },

    reset() {
        this.state.polishLevel = 0.3;
        this.state.patinaCoverage = 0.6;
        this.state.playCount = 0;
        if (this.state.wearMap) this.state.wearMap.fill(0);
        this.render();
    },

    autoPlay(seconds = 5) {
        const canvas = this.canvas;
        const w = canvas.width;
        const h = canvas.height;
        const cx = w / 2, cy = h / 2;
        let t = 0;
        const startTime = Date.now();
        const totalMs = seconds * 1000;

        const interval = setInterval(() => {
            const elapsed = Date.now() - startTime;
            if (elapsed > totalMs) {
                clearInterval(interval);
                return;
            }
            t += 0.15;
            const rx = cx + Math.cos(t * 1.7) * w * 0.25 * (1 - elapsed / totalMs * 0.3);
            const ry = cy + Math.sin(t * 2.3) * h * 0.25 * (1 - elapsed / totalMs * 0.3);

            if (this.state.wearMap) {
                const radius = 18;
                for (let dy = -radius; dy <= radius; dy++) {
                    for (let dx = -radius; dx <= radius; dx++) {
                        const distSq = dx*dx + dy*dy;
                        if (distSq > radius*radius) continue;
                        const px = Math.floor(rx + dx), py = Math.floor(ry + dy);
                        if (px < 0 || px >= w || py < 0 || py >= h) continue;
                        const falloff = 1 - distSq / (radius*radius);
                        const idx = py * w + px;
                        this.state.wearMap[idx] = Math.min(1.0, this.state.wearMap[idx] + falloff * 0.015);
                    }
                }
                this.state.playCount++;
                this.state.polishLevel = Math.min(1.0, this.state.polishLevel + 0.0005);
                this.state.patinaCoverage = Math.max(0.0, this.state.patinaCoverage - 0.0003);
            }
            this.render();
        }, 40);
    },

    _emitPlayEvent(x, y) {
        if (typeof CustomEvent !== 'undefined') {
            const ev = new CustomEvent('jade-play', {
                detail: { x, y, stats: this.getStats() }
            });
            this.canvas.dispatchEvent(ev);
        }
    },

    destroy() {
        if (this._lightAnim) clearInterval(this._lightAnim);
        this._bound = false;
    }
};
