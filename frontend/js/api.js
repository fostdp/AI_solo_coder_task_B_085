const API = {
    async getArtifacts(page = 1, pageSize = 20, culture = '', keyword = '') {
        const params = new URLSearchParams({
            page,
            page_size: pageSize
        });
        if (culture) params.append('culture', culture);
        if (keyword) params.append('keyword', keyword);
        
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/?${params}`);
        return response.json();
    },

    async getArtifact(artifactId) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/`);
        return response.json();
    },

    async getRamanSpectrum(artifactId) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/raman/`);
        return response.json();
    },

    async getXRFSpectrum(artifactId) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/xrf/`);
        return response.json();
    },

    async getDiffusionResult(artifactId) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/diffusion/`);
        return response.json();
    },

    async runDiffusion(artifactId, data = {}) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/diffusion/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return response.json();
    },

    async getAnomalyResult(artifactId) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/anomaly/`);
        return response.json();
    },

    async runAnomalyDetection(artifactId) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/anomaly/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        return response.json();
    },

    async getDensityMap(artifactId) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/density-map/`);
        return response.json();
    },

    async getAlerts(status = '', type = '', limit = 50) {
        const params = new URLSearchParams({ limit });
        if (status) params.append('status', status);
        if (type) params.append('type', type);
        
        const response = await fetch(`${CONFIG.API_BASE_URL}/alerts/?${params}`);
        return response.json();
    },

    async acknowledgeAlert(alertId) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/alerts/${alertId}/acknowledge/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        return response.json();
    },

    async getStatsSummary() {
        const response = await fetch(`${CONFIG.API_BASE_URL}/stats/summary/`);
        return response.json();
    },

    async startSimulator(interval = 30) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/simulator/start/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ interval })
        });
        return response.json();
    },

    async stopSimulator() {
        const response = await fetch(`${CONFIG.API_BASE_URL}/simulator/stop/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        return response.json();
    },

    async getProvenance(artifactId) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/provenance/`);
        return response.json();
    },

    async runProvenance(artifactId, data = {}) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/provenance/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return response.json();
    },

    async getPHInversion(artifactId) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/ph-inversion/`);
        return response.json();
    },

    async runPHInversion(artifactId, data = {}) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/ph-inversion/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return response.json();
    },

    async getForgeryProcess(artifactId) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/forgery-process/`);
        return response.json();
    },

    async runForgeryProcess(artifactId, data = {}) {
        const response = await fetch(`${CONFIG.API_BASE_URL}/artifacts/${artifactId}/forgery-process/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return response.json();
    }
};
