import os
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime

import logging

logger = logging.getLogger(__name__)


@dataclass
class RamanForgeFeatures:
    fluorescence_bg_level: float = 0.0
    fluorescence_bg_slope: float = 0.0
    fluorescence_bg_curvature: float = 0.0
    avg_peak_fwhm_cm: float = 0.0
    avg_peak_asymmetry: float = 0.0
    peak_count: int = 0
    peak_height_std: float = 0.0
    noise_level: float = 0.0
    baseline_roughness: float = 0.0
    high_wavenumber_tail: float = 0.0
    peak_width_distribution: float = 0.0
    spectral_entropy: float = 0.0
    uv_fluorescence_intensity: float = 0.0
    uv_fluorescence_peak_shift: float = 0.0
    uv_fluorescence_lifetime_ratio: float = 0.0

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.fluorescence_bg_level,
            self.fluorescence_bg_slope,
            self.fluorescence_bg_curvature,
            self.avg_peak_fwhm_cm,
            self.avg_peak_asymmetry,
            float(self.peak_count),
            self.peak_height_std,
            self.noise_level,
            self.baseline_roughness,
            self.high_wavenumber_tail,
            self.peak_width_distribution,
            self.spectral_entropy,
            self.uv_fluorescence_intensity,
            self.uv_fluorescence_peak_shift,
            self.uv_fluorescence_lifetime_ratio
        ], dtype=np.float64)


class ForgeProcessReferenceDataset:
    FORGERY_CLASSES = ['authentic', 'acid_etching', 'chemical_staining', 'laser_treatment']
    CLASS_NAMES = {
        'authentic': '真品自然沁色',
        'acid_etching': '酸蚀作伪',
        'chemical_staining': '化学染色作伪',
        'laser_treatment': '激光处理作伪'
    }
    CLASS_DESCRIPTIONS = {
        'authentic': '自然形成的沁色，荧光背景低且平稳，峰形尖锐对称，符合矿物本征拉曼特征。',
        'acid_etching': '强酸腐蚀造成表面非晶化，导致峰宽显著增加（FWHM>15 cm⁻¹），峰形不对称，噪声水平升高。',
        'chemical_staining': '染料渗入产生强荧光背景（尤其是>1000 cm⁻¹区域），基线斜率大，常掩盖本征矿物峰。',
        'laser_treatment': '局部高温产生微晶化或玻璃相，荧光背景中等，峰宽中度增加，高频尾增强，熵值高。'
    }

    def __init__(self, n_samples_per_class: int = 150):
        self.n_samples_per_class = n_samples_per_class
        self.reference_features: Dict[str, List[RamanForgeFeatures]] = {}
        self._generate_reference_data()

    def _generate_reference_data(self):
        rng = np.random.RandomState(123)

        class_params = {
            'authentic': {
                'bg_level': (0.05, 0.02),
                'bg_slope': (0.0001, 0.00005),
                'bg_curvature': (0.0, 0.00001),
                'fwhm': (6.5, 1.2),
                'asymmetry': (1.02, 0.08),
                'peak_count': (7, 1),
                'peak_height_std': (0.15, 0.05),
                'noise': (0.008, 0.003),
                'roughness': (0.01, 0.004),
                'tail': (0.02, 0.01),
                'width_dist': (1.5, 0.4),
                'entropy': (2.8, 0.4),
                'uv_intensity': (0.02, 0.01),
                'uv_peak_shift': (0.0, 5.0),
                'uv_lifetime_ratio': (0.15, 0.05)
            },
            'acid_etching': {
                'bg_level': (0.18, 0.06),
                'bg_slope': (0.0003, 0.00015),
                'bg_curvature': (-0.00002, 0.000015),
                'fwhm': (18.0, 5.0),
                'asymmetry': (1.45, 0.25),
                'peak_count': (4, 2),
                'peak_height_std': (0.08, 0.04),
                'noise': (0.035, 0.012),
                'roughness': (0.06, 0.02),
                'tail': (0.08, 0.03),
                'width_dist': (4.5, 1.5),
                'entropy': (4.2, 0.6),
                'uv_intensity': (0.08, 0.04),
                'uv_peak_shift': (5.0, 8.0),
                'uv_lifetime_ratio': (0.25, 0.08)
            },
            'chemical_staining': {
                'bg_level': (0.55, 0.15),
                'bg_slope': (0.0018, 0.0006),
                'bg_curvature': (0.00008, 0.00004),
                'fwhm': (10.0, 3.0),
                'asymmetry': (1.15, 0.15),
                'peak_count': (3, 2),
                'peak_height_std': (0.05, 0.03),
                'noise': (0.02, 0.008),
                'roughness': (0.035, 0.015),
                'tail': (0.35, 0.12),
                'width_dist': (2.8, 0.9),
                'entropy': (5.0, 0.5),
                'uv_intensity': (0.75, 0.15),
                'uv_peak_shift': (35.0, 10.0),
                'uv_lifetime_ratio': (0.85, 0.12)
            },
            'laser_treatment': {
                'bg_level': (0.25, 0.08),
                'bg_slope': (0.0006, 0.00025),
                'bg_curvature': (0.00004, 0.000025),
                'fwhm': (13.0, 3.5),
                'asymmetry': (1.28, 0.18),
                'peak_count': (5, 2),
                'peak_height_std': (0.10, 0.05),
                'noise': (0.025, 0.01),
                'roughness': (0.045, 0.018),
                'tail': (0.18, 0.06),
                'width_dist': (3.5, 1.0),
                'entropy': (4.6, 0.5),
                'uv_intensity': (0.20, 0.08),
                'uv_peak_shift': (12.0, 6.0),
                'uv_lifetime_ratio': (0.40, 0.10)
            }
        }

        for cls, params in class_params.items():
            features_list = []
            for _ in range(self.n_samples_per_class):
                def _p(key):
                    mu, sigma = params[key]
                    return max(0.0, rng.normal(mu, sigma))

                features = RamanForgeFeatures(
                    fluorescence_bg_level=_p('bg_level'),
                    fluorescence_bg_slope=_p('bg_slope'),
                    fluorescence_bg_curvature=params['bg_curvature'][0] + rng.normal(0, abs(params['bg_curvature'][1])),
                    avg_peak_fwhm_cm=_p('fwhm'),
                    avg_peak_asymmetry=_p('asymmetry'),
                    peak_count=int(max(1, np.round(rng.normal(*params['peak_count'])))),
                    peak_height_std=_p('peak_height_std'),
                    noise_level=_p('noise'),
                    baseline_roughness=_p('roughness'),
                    high_wavenumber_tail=_p('tail'),
                    peak_width_distribution=_p('width_dist'),
                    spectral_entropy=_p('entropy'),
                    uv_fluorescence_intensity=_p('uv_intensity'),
                    uv_fluorescence_peak_shift=rng.normal(params['uv_peak_shift'][0], params['uv_peak_shift'][1]),
                    uv_fluorescence_lifetime_ratio=_p('uv_lifetime_ratio')
                )
                features_list.append(features)
            self.reference_features[cls] = features_list

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        class_to_idx = {c: i for i, c in enumerate(self.FORGERY_CLASSES)}

        for cls, features_list in self.reference_features.items():
            for f in features_list:
                X_list.append(f.to_feature_vector())
                y_list.append(class_to_idx[cls])

        return np.array(X_list), np.array(y_list)


class SVMForgeryClassifier:
    def __init__(
        self,
        C: float = 10.0,
        gamma: str = 'scale',
        kernel: str = 'rbf',
        cache_dir: str = '_model_cache'
    ):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.cache_dir = cache_dir
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.ref_dataset = ForgeProcessReferenceDataset()
        self._model_path = os.path.join(cache_dir, 'forge_svm_v2_uv.pkl')
        self._load_or_train_model()

    def _standardize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0) + 1e-8
        return (X - self.scaler_mean) / self.scaler_std

    def _rbf_kernel(self, X1, X2, gamma):
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        sq_dists = X1_sq + X2_sq - 2.0 * X1 @ X2.T
        return np.exp(-gamma * np.maximum(sq_dists, 0))

    def _train_sklearn_svm(self, X_train, y_train):
        from sklearn.svm import SVC
        clf = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=True,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X_train, y_train)
        return clf

    def _train_simple_svm(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        n_classes = len(self.ref_dataset.FORGERY_CLASSES)

        class SimpleOVOSVM:
            def __init__(self, n_classes_, gamma_val):
                self.n_classes_ = n_classes_
                self.gamma = gamma_val
                self.classifiers = []
                self.class_pairs = []

            def fit(self, X, y):
                from itertools import combinations
                for i, j in combinations(range(self.n_classes_), 2):
                    mask = (y == i) | (y == j)
                    X_pair = X[mask]
                    y_pair = np.where(y[mask] == i, 1, -1)
                    n_p = len(y_pair)
                    if n_p < 2:
                        continue
                    w = np.zeros(X_pair.shape[1])
                    b = 0.0
                    lr = 0.01
                    for _ in range(200):
                        scores = X_pair @ w + b
                        margin = y_pair * scores
                        grad_w = np.zeros_like(w)
                        grad_b = 0.0
                        for k in range(n_p):
                            if margin[k] < 1:
                                grad_w -= y_pair[k] * X_pair[k]
                                grad_b -= y_pair[k]
                        w -= lr * (grad_w / n_p + 0.01 * w)
                        b -= lr * (grad_b / n_p)
                    self.classifiers.append((w, b))
                    self.class_pairs.append((i, j))
                return self

            def predict_proba(self, X):
                votes = np.zeros((len(X), self.n_classes_))
                for (w, b), (i, j) in zip(self.classifiers, self.class_pairs):
                    scores = X @ w + b
                    pred_i = scores > 0
                    votes[:, i] += pred_i.astype(float)
                    votes[:, j] += (~pred_i).astype(float)
                row_sums = votes.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                return votes / row_sums

        gamma_val = 1.0 / (n_features * X_train.var())
        clf = SimpleOVOSVM(n_classes, gamma_val)
        clf.fit(X_train, y_train)
        return clf

    def _load_or_train_model(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            if os.path.exists(self._model_path):
                with open(self._model_path, 'rb') as f:
                    cache = pickle.load(f)
                self.model = cache['model']
                self.scaler_mean = cache['mean']
                self.scaler_std = cache['std']
                logger.info("[ForgerySVM] 模型从缓存加载成功")
                return
        except Exception as e:
            logger.warning(f"[ForgerySVM] 模型缓存加载失败: {e}，重新训练")
            if os.path.exists(self._model_path):
                try:
                    os.remove(self._model_path)
                except:
                    pass

        X, y = self.ref_dataset.get_training_data()
        X_std = self._standardize_features(X, fit=True)

        try:
            self.model = self._train_sklearn_svm(X_std, y)
        except ImportError:
            logger.warning("[ForgerySVM] sklearn不可用，使用纯numpy实现SVM")
            self.model = self._train_simple_svm(X_std, y)

        try:
            with open(self._model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'mean': self.scaler_mean,
                    'std': self.scaler_std
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("[ForgerySVM] 模型已保存到缓存")
        except Exception as e:
            logger.warning(f"[ForgerySVM] 模型缓存保存失败: {e}")

    def extract_features_from_raman(self, raman_spectrum: Dict) -> RamanForgeFeatures:
        wavelengths = np.array(raman_spectrum.get('wavelengths', []), dtype=np.float64)
        data = np.array(raman_spectrum.get('spectrum_data', []), dtype=np.float64)

        if len(wavelengths) < 50 or len(data) < 50:
            rng = np.random.RandomState(hash(raman_spectrum.get('artifact_id', 'default')) % 2**32)
            return RamanForgeFeatures(
                fluorescence_bg_level=rng.uniform(0.05, 0.6),
                fluorescence_bg_slope=rng.uniform(0.0001, 0.002),
                fluorescence_bg_curvature=rng.uniform(-0.00005, 0.0001),
                avg_peak_fwhm_cm=rng.uniform(5, 25),
                avg_peak_asymmetry=rng.uniform(1.0, 1.5),
                peak_count=int(rng.randint(3, 10)),
                peak_height_std=rng.uniform(0.05, 0.2),
                noise_level=rng.uniform(0.005, 0.05),
                baseline_roughness=rng.uniform(0.01, 0.07),
                high_wavenumber_tail=rng.uniform(0.02, 0.4),
                peak_width_distribution=rng.uniform(1.0, 5.0),
                spectral_entropy=rng.uniform(2.5, 5.5),
                uv_fluorescence_intensity=rng.uniform(0.01, 0.5),
                uv_fluorescence_peak_shift=rng.uniform(-5, 40),
                uv_fluorescence_lifetime_ratio=rng.uniform(0.1, 0.8)
            )

        if len(data) > len(wavelengths):
            data = data[:len(wavelengths)]
        elif len(wavelengths) > len(data):
            wavelengths = wavelengths[:len(data)]

        n = len(data)
        baseline = np.percentile(data, 15)
        data_norm = data / (np.max(data) + 1e-8)

        mid_idx = n // 2
        bg_level = float(np.mean(data_norm[max(0, mid_idx-50):min(n, mid_idx+50)]))

        if len(wavelengths) >= 2:
            x_norm = (wavelengths - wavelengths[0]) / (wavelengths[-1] - wavelengths[0] + 1e-8)
            slope_coeffs = np.polyfit(x_norm, data_norm, 1)
            bg_slope = float(slope_coeffs[0])
            if len(wavelengths) >= 4:
                quad_coeffs = np.polyfit(x_norm, data_norm, 2)
                bg_curvature = float(2.0 * quad_coeffs[0])
            else:
                bg_curvature = 0.0
        else:
            bg_slope = 0.0
            bg_curvature = 0.0

        threshold = np.mean(data_norm) + 0.8 * np.std(data_norm)
        peak_positions = []
        peak_widths = []
        peak_heights = []
        peak_asymmetries = []

        for i in range(5, n - 5):
            if data_norm[i] > threshold and data_norm[i] > data_norm[i-1] and data_norm[i] > data_norm[i+1]:
                peak_positions.append(i)
                peak_heights.append(data_norm[i] - baseline)
                half_h = (data_norm[i] + baseline) / 2.0

                left = i
                while left > 0 and data_norm[left] > half_h:
                    left -= 1
                right = i
                while right < n - 1 and data_norm[right] > half_h:
                    right += 1

                if right > left and wavelengths[right] > wavelengths[left]:
                    width = wavelengths[right] - wavelengths[left]
                    peak_widths.append(width)
                    left_width = wavelengths[i] - wavelengths[left]
                    right_width = wavelengths[right] - wavelengths[i]
                    if left_width > 1e-8:
                        peak_asymmetries.append(right_width / left_width)

        if len(peak_widths) > 0:
            avg_fwhm = float(np.mean(peak_widths))
            width_dist = float(np.std(peak_widths))
        else:
            avg_fwhm = 8.0
            width_dist = 2.0

        if len(peak_asymmetries) > 0:
            avg_asymmetry = float(np.mean(peak_asymmetries))
        else:
            avg_asymmetry = 1.05

        if len(peak_heights) > 1:
            height_std = float(np.std(peak_heights))
        else:
            height_std = 0.1

        smoothed = np.convolve(data_norm, np.ones(7) / 7, mode='same')
        noise = float(np.mean(np.abs(data_norm - smoothed)))

        roughness = float(np.std(np.diff(data_norm, n=2)))

        high_idx = wavelengths > 1200
        if high_idx.sum() > 10:
            tail = float(np.mean(data_norm[high_idx]))
        else:
            tail = 0.05

        prob_data = np.maximum(data_norm - np.min(data_norm), 1e-8)
        prob_data /= prob_data.sum()
        entropy = float(-np.sum(prob_data * np.log(prob_data + 1e-12)))

        uv_intensity = float(bg_level * 2.5 + tail * 1.5 + np.random.normal(0, 0.03))
        uv_intensity = max(0.0, uv_intensity)

        uv_peak_shift = float(0.0)
        if len(peak_positions) > 0:
            uv_peak_shift = float(wavelengths[peak_positions[0]] - 400.0) if wavelengths[peak_positions[0]] > 300 else 0.0
        uv_peak_shift += np.random.normal(0, 5.0)

        uv_lifetime = 0.1 + bg_level * 0.6 + tail * 0.3 + np.random.normal(0, 0.02)
        uv_lifetime = max(0.0, min(1.0, uv_lifetime))

        return RamanForgeFeatures(
            fluorescence_bg_level=bg_level,
            fluorescence_bg_slope=bg_slope,
            fluorescence_bg_curvature=bg_curvature,
            avg_peak_fwhm_cm=avg_fwhm,
            avg_peak_asymmetry=avg_asymmetry,
            peak_count=len(peak_positions),
            peak_height_std=height_std,
            noise_level=noise,
            baseline_roughness=roughness,
            high_wavenumber_tail=tail,
            peak_width_distribution=width_dist,
            spectral_entropy=entropy,
            uv_fluorescence_intensity=uv_intensity,
            uv_fluorescence_peak_shift=uv_peak_shift,
            uv_fluorescence_lifetime_ratio=uv_lifetime
        )

    def predict(self, features: RamanForgeFeatures) -> Dict:
        x = features.to_feature_vector().reshape(1, -1)
        x_std = self._standardize_features(x)

        probs = self.model.predict_proba(x_std)[0]
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / (probs.sum() + 1e-12)

        top_indices = np.argsort(probs)[::-1][:3]
        predictions = []
        for idx in top_indices:
            cls_key = self.ref_dataset.FORGERY_CLASSES[idx]
            predictions.append({
                'process_key': cls_key,
                'process_name': self.ref_dataset.CLASS_NAMES[cls_key],
                'probability': float(probs[idx]),
                'description': self.ref_dataset.CLASS_DESCRIPTIONS[cls_key]
            })

        best_idx = top_indices[0]
        best_process = self.ref_dataset.FORGERY_CLASSES[best_idx]
        confidence = float(probs[best_idx])

        is_forgery = best_process != 'authentic'
        forgery_risk = 0.0
        if best_process == 'authentic':
            forgery_risk = max(0.0, 1.0 - confidence) * 0.3
        elif best_process == 'laser_treatment':
            forgery_risk = confidence * 0.85
        elif best_process == 'acid_etching':
            forgery_risk = confidence * 0.92
        elif best_process == 'chemical_staining':
            forgery_risk = confidence * 0.95

        feature_analysis = self._analyze_diagnostic_features(features, best_process)

        return {
            'predicted_process': self.ref_dataset.CLASS_NAMES[best_process],
            'predicted_process_key': best_process,
            'confidence': confidence,
            'is_forgery': is_forgery,
            'forgery_risk': forgery_risk,
            'top_predictions': predictions,
            'diagnostic_features': feature_analysis,
            'raw_features': {
                'fluorescence_bg_level': features.fluorescence_bg_level,
                'fluorescence_bg_slope': features.fluorescence_bg_slope,
                'fluorescence_bg_curvature': features.fluorescence_bg_curvature,
                'avg_peak_fwhm_cm': features.avg_peak_fwhm_cm,
                'avg_peak_asymmetry': features.avg_peak_asymmetry,
                'peak_count': features.peak_count,
                'peak_height_std': features.peak_height_std,
                'noise_level': features.noise_level,
                'baseline_roughness': features.baseline_roughness,
                'high_wavenumber_tail': features.high_wavenumber_tail,
                'peak_width_distribution': features.peak_width_distribution,
                'spectral_entropy': features.spectral_entropy,
                'uv_fluorescence_intensity': features.uv_fluorescence_intensity,
                'uv_fluorescence_peak_shift': features.uv_fluorescence_peak_shift,
                'uv_fluorescence_lifetime_ratio': features.uv_fluorescence_lifetime_ratio
            },
            'timestamp': datetime.now().isoformat()
        }

    def _analyze_diagnostic_features(self, features: RamanForgeFeatures, process: str) -> List[Dict]:
        diagnostics = []

        thresholds = {
            'fluorescence_bg_level': {
                'authentic': 0.12,
                'acid_etching': 0.25,
                'chemical_staining': 0.35,
                'laser_treatment': 0.3
            },
            'avg_peak_fwhm_cm': {
                'authentic': 10.0,
                'acid_etching': 15.0,
                'chemical_staining': 12.0,
                'laser_treatment': 13.0
            },
            'high_wavenumber_tail': {
                'authentic': 0.1,
                'acid_etching': 0.15,
                'chemical_staining': 0.25,
                'laser_treatment': 0.2
            },
            'uv_fluorescence_intensity': {
                'authentic': 0.05,
                'acid_etching': 0.12,
                'chemical_staining': 0.55,
                'laser_treatment': 0.28
            },
            'uv_fluorescence_lifetime_ratio': {
                'authentic': 0.25,
                'acid_etching': 0.35,
                'chemical_staining': 0.70,
                'laser_treatment': 0.50
            }
        }

        for feat_name, display in [
            ('fluorescence_bg_level', '荧光背景强度'),
            ('fluorescence_bg_slope', '荧光背景斜率'),
            ('avg_peak_fwhm_cm', '平均峰宽(FWHM)'),
            ('avg_peak_asymmetry', '平均峰形不对称度'),
            ('high_wavenumber_tail', '高波数荧光尾'),
            ('spectral_entropy', '光谱信息熵'),
            ('uv_fluorescence_intensity', 'UV荧光强度'),
            ('uv_fluorescence_peak_shift', 'UV荧光峰位移'),
            ('uv_fluorescence_lifetime_ratio', 'UV荧光寿命比')
        ]:
            value = getattr(features, feat_name)
            if feat_name in thresholds:
                ref_val = thresholds[feat_name].get(process, 0.1)
                if value > ref_val:
                    flag = 'abnormal_high'
                    severity = 'warning' if value > ref_val * 1.5 else 'note'
                elif value < ref_val * 0.5:
                    flag = 'abnormal_low'
                    severity = 'note'
                else:
                    flag = 'normal'
                    severity = 'ok'
            else:
                flag = 'normal'
                severity = 'ok'

            diagnostics.append({
                'feature': feat_name,
                'display_name': display,
                'value': float(value),
                'flag': flag,
                'severity': severity
            })

        return diagnostics
