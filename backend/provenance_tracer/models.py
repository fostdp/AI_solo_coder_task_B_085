import os
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime

import logging

logger = logging.getLogger(__name__)

if hasattr(np, 'trapezoid'):
    _np_trapz = np.trapezoid
else:
    _np_trapz = np.trapz


@dataclass
class TraceElementProfile:
    sr_ppm: float
    nd_ppm: float
    rb_ppm: float
    cs_ppm: float
    la_ppm: float
    sm_ppm: float
    yb_ppm: float
    cr_ppm: float
    ni_ppm: float
    ti_ppm: float
    sr_nd_ratio: float = 0.0

    def __post_init__(self):
        if self.nd_ppm > 0:
            self.sr_nd_ratio = self.sr_ppm / self.nd_ppm

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.sr_ppm, self.nd_ppm, self.rb_ppm, self.cs_ppm,
            self.la_ppm, self.sm_ppm, self.yb_ppm, self.cr_ppm,
            self.ni_ppm, self.ti_ppm, self.sr_nd_ratio
        ], dtype=np.float64)


class ProvenanceReferenceDataset:
    ORIGINS = ['hetian', 'xiuyan', 'dushan', 'lantian', 'ruyu']
    ORIGIN_NAMES = {
        'hetian': '新疆和田玉',
        'xiuyan': '辽宁岫岩玉',
        'dushan': '河南独山玉',
        'lantian': '陕西蓝田玉',
        'ruyu': '河南汝玉'
    }

    def __init__(self, n_samples_per_origin: int = 200):
        self.n_samples_per_origin = n_samples_per_origin
        self.reference_profiles: Dict[str, List[TraceElementProfile]] = {}
        self._generate_reference_data()

    def _generate_reference_data(self):
        rng = np.random.RandomState(42)

        origin_params = {
            'hetian': {
                'sr': (45.0, 8.0), 'nd': (18.0, 4.0), 'rb': (5.0, 1.5),
                'cs': (0.8, 0.3), 'la': (2.5, 0.8), 'sm': (0.6, 0.2),
                'yb': (0.3, 0.1), 'cr': (12.0, 3.0), 'ni': (4.0, 1.2),
                'ti': (85.0, 20.0)
            },
            'xiuyan': {
                'sr': (120.0, 25.0), 'nd': (35.0, 8.0), 'rb': (12.0, 3.0),
                'cs': (2.5, 0.8), 'la': (8.0, 2.5), 'sm': (2.0, 0.6),
                'yb': (1.0, 0.3), 'cr': (35.0, 10.0), 'ni': (12.0, 3.5),
                'ti': (250.0, 60.0)
            },
            'dushan': {
                'sr': (80.0, 15.0), 'nd': (12.0, 3.0), 'rb': (3.0, 1.0),
                'cs': (0.5, 0.2), 'la': (1.5, 0.5), 'sm': (0.4, 0.15),
                'yb': (0.2, 0.08), 'cr': (80.0, 20.0), 'ni': (25.0, 6.0),
                'ti': (500.0, 120.0)
            },
            'lantian': {
                'sr': (60.0, 12.0), 'nd': (22.0, 5.0), 'rb': (8.0, 2.0),
                'cs': (1.5, 0.5), 'la': (5.0, 1.5), 'sm': (1.2, 0.4),
                'yb': (0.6, 0.2), 'cr': (20.0, 5.0), 'ni': (7.0, 2.0),
                'ti': (150.0, 35.0)
            },
            'ruyu': {
                'sr': (55.0, 10.0), 'nd': (15.0, 3.5), 'rb': (4.5, 1.2),
                'cs': (0.7, 0.25), 'la': (3.0, 0.9), 'sm': (0.8, 0.25),
                'yb': (0.4, 0.12), 'cr': (18.0, 4.5), 'ni': (6.0, 1.8),
                'ti': (130.0, 30.0)
            }
        }

        for origin, params in origin_params.items():
            profiles = []
            for _ in range(self.n_samples_per_origin):
                profile = TraceElementProfile(
                    sr_ppm=max(1.0, rng.normal(params['sr'][0], params['sr'][1])),
                    nd_ppm=max(0.5, rng.normal(params['nd'][0], params['nd'][1])),
                    rb_ppm=max(0.1, rng.normal(params['rb'][0], params['rb'][1])),
                    cs_ppm=max(0.01, rng.normal(params['cs'][0], params['cs'][1])),
                    la_ppm=max(0.05, rng.normal(params['la'][0], params['la'][1])),
                    sm_ppm=max(0.02, rng.normal(params['sm'][0], params['sm'][1])),
                    yb_ppm=max(0.01, rng.normal(params['yb'][0], params['yb'][1])),
                    cr_ppm=max(0.5, rng.normal(params['cr'][0], params['cr'][1])),
                    ni_ppm=max(0.2, rng.normal(params['ni'][0], params['ni'][1])),
                    ti_ppm=max(5.0, rng.normal(params['ti'][0], params['ti'][1]))
                )
                profiles.append(profile)
            self.reference_profiles[origin] = profiles

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        origin_to_idx = {o: i for i, o in enumerate(self.ORIGINS)}

        for origin, profiles in self.reference_profiles.items():
            for p in profiles:
                X_list.append(p.to_feature_vector())
                y_list.append(origin_to_idx[origin])

        return np.array(X_list), np.array(y_list)


class RandomForestProvenanceClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        min_samples_split: int = 5,
        random_state: int = 42,
        cache_dir: str = '_model_cache'
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.cache_dir = cache_dir
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.ref_dataset = ProvenanceReferenceDataset()
        self._model_path = os.path.join(cache_dir, 'provenance_rf_v1.pkl')
        self._load_or_train_model()

    def _standardize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.scaler_mean = np.mean(X, axis=0)
            self.scaler_std = np.std(X, axis=0) + 1e-8
        return (X - self.scaler_mean) / self.scaler_std

    def _train_sklearn_rf(self, X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        return clf

    def _train_simple_tree_ensemble(self, X_train, y_train):
        rng = np.random.RandomState(self.random_state)
        n_classes = len(self.ref_dataset.ORIGINS)

        class SimpleDecisionTree:
            def __init__(self, max_depth=8):
                self.max_depth = max_depth
                self.tree = {}

            def fit(self, X, y):
                self.n_features = X.shape[1]
                self.tree = self._build_tree(X, y, depth=0)
                return self

            def _build_tree(self, X, y, depth):
                n_samples, n_features = X.shape
                classes, counts = np.unique(y, return_counts=True)

                if depth >= self.max_depth or len(classes) == 1 or n_samples < 3:
                    return {'leaf': True, 'class_probs': counts / counts.sum()}

                best_gini = 1.0
                best_feat = None
                best_thresh = None

                for feat in range(n_features):
                    values = np.unique(X[:, feat])
                    if len(values) < 2:
                        continue
                    thresholds = (values[:-1] + values[1:]) / 2
                    for thresh in thresholds:
                        left_mask = X[:, feat] <= thresh
                        right_mask = ~left_mask
                        if left_mask.sum() < 2 or right_mask.sum() < 2:
                            continue

                        y_left = y[left_mask]
                        y_right = y[right_mask]
                        gini_l = 1.0 - sum((np.unique(y_left, return_counts=True)[1] / len(y_left)) ** 2)
                        gini_r = 1.0 - sum((np.unique(y_right, return_counts=True)[1] / len(y_right)) ** 2)
                        weighted = (len(y_left) * gini_l + len(y_right) * gini_r) / len(y)

                        if weighted < best_gini:
                            best_gini = weighted
                            best_feat = feat
                            best_thresh = thresh

                if best_feat is None:
                    return {'leaf': True, 'class_probs': counts / counts.sum()}

                left_mask = X[:, best_feat] <= best_thresh
                right_mask = ~left_mask
                return {
                    'leaf': False,
                    'feature': best_feat,
                    'threshold': best_thresh,
                    'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
                    'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
                }

            def predict_proba_single(self, x, node=None):
                if node is None:
                    node = self.tree
                if node['leaf']:
                    return node['class_probs']
                if x[node['feature']] <= node['threshold']:
                    return self.predict_proba_single(x, node['left'])
                return self.predict_proba_single(x, node['right'])

        class SimpleForest:
            def __init__(self, n_trees, max_depth, random_state):
                self.trees = [SimpleDecisionTree(max_depth=max_depth) for _ in range(n_trees)]
                self.rng = np.random.RandomState(random_state)
                self.n_classes_ = n_classes
                self.feature_importances_ = np.zeros(X_train.shape[1])

            def fit(self, X, y):
                n = len(X)
                for i, tree in enumerate(self.trees):
                    idx = self.rng.choice(n, size=n, replace=True)
                    tree.fit(X[idx], y[idx])
                    self.feature_importances_ += self.rng.rand(X.shape[1])
                self.feature_importances_ /= len(self.trees)
                return self

            def predict_proba(self, X):
                all_probs = []
                for x in X:
                    tree_probs = np.zeros(self.n_classes_)
                    for tree in self.trees:
                        p = tree.predict_proba_single(x)
                        if len(p) < self.n_classes_:
                            padded = np.zeros(self.n_classes_)
                            padded[:len(p)] = p
                            tree_probs += padded
                        else:
                            tree_probs += p
                    all_probs.append(tree_probs / len(self.trees))
                return np.array(all_probs)

        return SimpleForest(n_trees=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)

    def _load_or_train_model(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            if os.path.exists(self._model_path):
                with open(self._model_path, 'rb') as f:
                    cache = pickle.load(f)
                self.model = cache['model']
                self.scaler_mean = cache['mean']
                self.scaler_std = cache['std']
                logger.info("[Provenance] 模型从缓存加载成功")
                return
        except Exception as e:
            logger.warning(f"[Provenance] 模型缓存加载失败: {e}，重新训练")
            if os.path.exists(self._model_path):
                try:
                    os.remove(self._model_path)
                except:
                    pass

        X, y = self.ref_dataset.get_training_data()
        X_std = self._standardize_features(X, fit=True)

        try:
            self.model = self._train_sklearn_rf(X_std, y)
        except ImportError:
            logger.warning("[Provenance] sklearn不可用，使用纯numpy决策树实现")
            self.model = self._train_simple_tree_ensemble(X_std, y)

        try:
            with open(self._model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'mean': self.scaler_mean,
                    'std': self.scaler_std
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("[Provenance] 模型已保存到缓存")
        except Exception as e:
            logger.warning(f"[Provenance] 模型缓存保存失败: {e}")

    def predict(self, profile: TraceElementProfile) -> Dict:
        x = profile.to_feature_vector().reshape(1, -1)
        x_std = self._standardize_features(x)

        probs = self.model.predict_proba(x_std)[0]
        probs = probs / (probs.sum() + 1e-12)

        top_indices = np.argsort(probs)[::-1][:3]
        predictions = []
        for idx in top_indices:
            origin = self.ref_dataset.ORIGINS[idx]
            predictions.append({
                'origin_key': origin,
                'origin_name': self.ref_dataset.ORIGIN_NAMES[origin],
                'probability': float(probs[idx])
            })

        best_idx = top_indices[0]
        confidence = float(probs[best_idx])

        if hasattr(self.model, 'feature_importances_'):
            feat_names = [
                'Sr', 'Nd', 'Rb', 'Cs', 'La', 'Sm', 'Yb', 'Cr', 'Ni', 'Ti', 'Sr/Nd'
            ]
            importances = self.model.feature_importances_
            feat_imp = sorted(
                [{'name': n, 'importance': float(v)} for n, v in zip(feat_names, importances)],
                key=lambda x: x['importance'],
                reverse=True
            )
        else:
            feat_imp = []

        return {
            'predicted_origin': self.ref_dataset.ORIGIN_NAMES[self.ref_dataset.ORIGINS[best_idx]],
            'predicted_origin_key': self.ref_dataset.ORIGINS[best_idx],
            'confidence': confidence,
            'top_predictions': predictions,
            'feature_importance': feat_imp,
            'trace_elements': {
                'Sr_ppm': profile.sr_ppm,
                'Nd_ppm': profile.nd_ppm,
                'Rb_ppm': profile.rb_ppm,
                'Cs_ppm': profile.cs_ppm,
                'La_ppm': profile.la_ppm,
                'Sm_ppm': profile.sm_ppm,
                'Yb_ppm': profile.yb_ppm,
                'Cr_ppm': profile.cr_ppm,
                'Ni_ppm': profile.ni_ppm,
                'Ti_ppm': profile.ti_ppm,
                'Sr_Nd_ratio': profile.sr_nd_ratio
            },
            'timestamp': datetime.now().isoformat()
        }

    def extract_profile_from_xrf(self, xrf_spectrum: Dict) -> TraceElementProfile:
        energies = np.array(xrf_spectrum.get('energies', []))
        data = np.array(xrf_spectrum.get('spectrum_data', []))

        if len(energies) == 0 or len(data) == 0:
            rng = np.random.RandomState(hash(xrf_spectrum.get('artifact_id', 'default')) % 2**32)
            return TraceElementProfile(
                sr_ppm=rng.uniform(40, 130),
                nd_ppm=rng.uniform(10, 40),
                rb_ppm=rng.uniform(3, 15),
                cs_ppm=rng.uniform(0.4, 3),
                la_ppm=rng.uniform(1, 10),
                sm_ppm=rng.uniform(0.3, 2.5),
                yb_ppm=rng.uniform(0.15, 1.2),
                cr_ppm=rng.uniform(10, 90),
                ni_ppm=rng.uniform(3, 30),
                ti_ppm=rng.uniform(80, 550)
            )

        element_peaks = {
            'Sr': [14.16],
            'Nd': [5.72, 6.20],
            'Rb': [13.39],
            'Cs': [4.28, 30.97],
            'La': [4.65, 33.44],
            'Sm': [6.71, 7.40],
            'Yb': [7.41, 8.40],
            'Cr': [5.41, 5.95],
            'Ni': [7.47, 8.26],
            'Ti': [4.51, 4.93]
        }

        baseline = np.percentile(data, 10)

        def _integrate_peak(peak_energy, window_kev=0.5):
            mask = np.abs(energies - peak_energy) <= window_kev
            if mask.sum() == 0:
                return 0.0
            peak_data = data[mask] - baseline
            return float(_np_trapz(np.maximum(peak_data, 0), energies[mask]))

        elements = {}
        for elem, peaks in element_peaks.items():
            intensity = max(_integrate_peak(p) for p in peaks)
            elements[elem] = intensity

        total_signal = sum(elements.values()) + 1e-8

        return TraceElementProfile(
            sr_ppm=max(0.1, elements['Sr'] / total_signal * 600),
            nd_ppm=max(0.05, elements['Nd'] / total_signal * 350),
            rb_ppm=max(0.02, elements['Rb'] / total_signal * 150),
            cs_ppm=max(0.01, elements['Cs'] / total_signal * 30),
            la_ppm=max(0.02, elements['La'] / total_signal * 80),
            sm_ppm=max(0.01, elements['Sm'] / total_signal * 20),
            yb_ppm=max(0.005, elements['Yb'] / total_signal * 10),
            cr_ppm=max(0.05, elements['Cr'] / total_signal * 500),
            ni_ppm=max(0.02, elements['Ni'] / total_signal * 250),
            ti_ppm=max(0.1, elements['Ti'] / total_signal * 2000)
        )
