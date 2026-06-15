const DensityMap = {
    _contourCache: new Map(),
    _cacheStats: { hits: 0, misses: 0, evictions: 0 },
    _maxCacheSize: 20,

    draw(canvas, densityMap, options = {}) {
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);

        const gridSize = densityMap.length || 100;
        const cellW = w / gridSize;
        const cellH = h / gridSize;

        const colorScheme = options.colorScheme || 'heat';

        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                const value = densityMap[i][j] || 0;
                const color = this.getColor(value, colorScheme);
                ctx.fillStyle = color;
                ctx.fillRect(j * cellW, i * cellH, cellW + 0.5, cellH + 0.5);
            }
        }

        if (options.contour) {
            this.drawContours(ctx, densityMap, w, h, options);
        }
    },

    getColor(value, scheme = 'heat') {
        const v = Math.max(0, Math.min(255, value));
        const t = v / 255;

        switch (scheme) {
            case 'heat':
                return this.heatColor(t);
            case 'viridis':
                return this.viridisColor(t);
            case 'seismic':
                return this.seismicColor(t);
            default:
                return this.heatColor(t);
        }
    },

    heatColor(t) {
        const r = Math.floor(255 * Math.min(1, t * 2));
        const g = Math.floor(255 * Math.min(1, Math.max(0, t * 2 - 0.5)));
        const b = Math.floor(255 * Math.max(0, t * 2 - 1));
        const a = 0.1 + t * 0.8;
        return `rgba(${r}, ${g}, ${b}, ${a})`;
    },

    viridisColor(t) {
        const colors = [
            [68, 1, 84],
            [72, 40, 120],
            [62, 74, 137],
            [49, 104, 142],
            [38, 130, 142],
            [31, 158, 137],
            [53, 183, 121],
            [109, 205, 89],
            [180, 222, 44],
            [253, 231, 37]
        ];
        
        const idx = t * (colors.length - 1);
        const i = Math.floor(idx);
        const f = idx - i;
        
        if (i >= colors.length - 1) {
            const [r, g, b] = colors[colors.length - 1];
            return `rgba(${r}, ${g}, ${b}, 0.8)`;
        }
        
        const [r1, g1, b1] = colors[i];
        const [r2, g2, b2] = colors[i + 1];
        const r = Math.floor(r1 + (r2 - r1) * f);
        const g = Math.floor(g1 + (g2 - g1) * f);
        const b = Math.floor(b1 + (b2 - b1) * f);
        
        return `rgba(${r}, ${g}, ${b}, 0.8)`;
    },

    seismicColor(t) {
        if (t < 0.5) {
            const t2 = t * 2;
            const r = Math.floor(30 + t2 * 70);
            const g = Math.floor(50 + t2 * 150);
            const b = Math.floor(150 + t2 * 105);
            return `rgba(${r}, ${g}, ${b}, 0.7)`;
        } else {
            const t2 = (t - 0.5) * 2;
            const r = Math.floor(200 + t2 * 55);
            const g = Math.floor(200 - t2 * 170);
            const b = Math.floor(50 - t2 * 40);
            return `rgba(${r}, ${g}, ${b}, 0.7)`;
        }
    },

    _hashDensityMap(densityMap) {
        if (!densityMap || !densityMap.length) return 'empty';

        const gridSize = densityMap.length;
        let hash = gridSize.toString() + 'x' + (densityMap[0]?.length || 0);

        const step = Math.max(1, Math.floor(gridSize / 16));
        for (let i = 0; i < gridSize; i += step) {
            for (let j = 0; j < gridSize; j += step) {
                const v = densityMap[i]?.[j] || 0;
                hash += '_' + Math.round(v);
            }
        }

        let hashNum = 0;
        for (let i = 0; i < hash.length; i++) {
            hashNum = ((hashNum << 5) - hashNum + hash.charCodeAt(i)) | 0;
        }
        return hash + '_' + Math.abs(hashNum).toString(36);
    },

    _evictOldCache() {
        if (this._contourCache.size <= this._maxCacheSize) return;

        let oldestKey = null;
        let oldestTime = Infinity;
        for (const [key, value] of this._contourCache.entries()) {
            if (value._lastUsed < oldestTime) {
                oldestTime = value._lastUsed;
                oldestKey = key;
            }
        }
        if (oldestKey) {
            this._contourCache.delete(oldestKey);
            this._cacheStats.evictions++;
        }
    },

    drawContours(ctx, densityMap, w, h, options = {}) {
        const gridSize = densityMap.length;
        const levels = options.contourLevels || [50, 100, 150, 200];
        const style = options.contourStyle || {
            strokeStyle: 'rgba(255, 255, 255, 0.3)',
            lineWidth: 1
        };

        const cacheKey = `${this._hashDensityMap(densityMap)}_${w}x${h}_${levels.join(',')}`;
        const cached = this._contourCache.get(cacheKey);

        if (cached && cached.segments) {
            this._cacheStats.hits++;
            cached._lastUsed = Date.now();

            ctx.strokeStyle = style.strokeStyle;
            ctx.lineWidth = style.lineWidth;

            for (const seg of cached.segments) {
                ctx.beginPath();
                ctx.moveTo(seg[0], seg[1]);
                ctx.lineTo(seg[2], seg[3]);
                ctx.stroke();
            }

            if (this._cacheStats.hits % 100 === 0) {
                const total = this._cacheStats.hits + this._cacheStats.misses;
                const hitRate = ((this._cacheStats.hits / total) * 100).toFixed(1);
                console.debug(`[density.js] 等值线缓存命中率: ${hitRate}% (hits=${this._cacheStats.hits}, misses=${this._cacheStats.misses})`);
            }
            return;
        }

        this._cacheStats.misses++;
        const segments = [];

        ctx.strokeStyle = style.strokeStyle;
        ctx.lineWidth = style.lineWidth;

        for (let i = 0; i < gridSize - 1; i++) {
            for (let j = 0; j < gridSize - 1; j++) {
                const v00 = densityMap[i][j] || 0;
                const v10 = densityMap[i][j + 1] || 0;
                const v01 = densityMap[i + 1][j] || 0;
                const v11 = densityMap[i + 1][j + 1] || 0;

                for (const level of levels) {
                    const x = j * (w / gridSize);
                    const y = i * (h / gridSize);
                    const cellW = w / gridSize;
                    const cellH = h / gridSize;

                    const edges = [];

                    if ((v00 - level) * (v10 - level) < 0) {
                        const t = (level - v00) / (v10 - v00);
                        edges.push([x + t * cellW, y]);
                    }
                    if ((v10 - level) * (v11 - level) < 0) {
                        const t = (level - v10) / (v11 - v10);
                        edges.push([x + cellW, y + t * cellH]);
                    }
                    if ((v01 - level) * (v11 - level) < 0) {
                        const t = (level - v01) / (v11 - v01);
                        edges.push([x + t * cellW, y + cellH]);
                    }
                    if ((v00 - level) * (v01 - level) < 0) {
                        const t = (level - v00) / (v01 - v00);
                        edges.push([x, y + t * cellH]);
                    }

                    if (edges.length >= 2) {
                        const seg = [edges[0][0], edges[0][1], edges[1][0], edges[1][1]];
                        segments.push(seg);

                        ctx.beginPath();
                        ctx.moveTo(seg[0], seg[1]);
                        ctx.lineTo(seg[2], seg[3]);
                        ctx.stroke();
                    }
                }
            }
        }

        this._contourCache.set(cacheKey, {
            segments: segments,
            gridSize: gridSize,
            levels: levels,
            w: w,
            h: h,
            _created: Date.now(),
            _lastUsed: Date.now()
        });

        this._evictOldCache();
    },

    _clearContourCache() {
        this._contourCache.clear();
        this._cacheStats = { hits: 0, misses: 0, evictions: 0 };
        console.debug('[density.js] 等值线缓存已清空');
    },

    _getCacheStats() {
        const total = this._cacheStats.hits + this._cacheStats.misses;
        return {
            size: this._contourCache.size,
            maxSize: this._maxCacheSize,
            hits: this._cacheStats.hits,
            misses: this._cacheStats.misses,
            evictions: this._cacheStats.evictions,
            hitRate: total > 0 ? (this._cacheStats.hits / total * 100).toFixed(1) + '%' : 'N/A'
        };
    },

    drawLegend(canvas, colorScheme = 'heat') {
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        const gradient = ctx.createLinearGradient(0, h, 0, 0);
        for (let i = 0; i <= 10; i++) {
            const t = i / 10;
            const color = this.getColor(t * 255, colorScheme);
            gradient.addColorStop(t, color);
        }

        ctx.fillStyle = gradient;
        ctx.fillRect(10, 10, 20, h - 20);

        ctx.fillStyle = '#fff';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('高', 35, 15);
        ctx.fillText('低', 35, h - 5);
    },

    drawOverlay(canvas, jadeType, densityMap, options = {}) {
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        ctx.clearRect(0, 0, w, h);

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = w;
        tempCanvas.height = h;
        const tempCtx = tempCanvas.getContext('2d');

        this.draw(tempCanvas, densityMap, options);

        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = w;
        maskCanvas.height = h;
        const maskCtx = maskCanvas.getContext('2d');
        
        const drawFunc = JadeCanvas.jadeTypes[jadeType] || JadeCanvas.jadeTypes['玉璧'];
        
        maskCtx.fillStyle = '#000';
        maskCtx.fillRect(0, 0, w, h);
        
        maskCtx.globalCompositeOperation = 'destination-out';
        drawFunc(maskCtx, w, h);
        
        const resultCtx = ctx;
        resultCtx.save();
        
        resultCtx.drawImage(tempCanvas, 0, 0);
        
        resultCtx.globalCompositeOperation = 'destination-in';
        resultCtx.drawImage(maskCanvas, 0, 0);
        
        resultCtx.restore();
        
        resultCtx.globalCompositeOperation = 'destination-over';
        
        JadeCanvas.drawJade(canvas, jadeType);
        
        resultCtx.globalCompositeOperation = 'source-over';
    }
};
