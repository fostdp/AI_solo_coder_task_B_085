import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging
import gc
import os
import pickle

try:
    from sklearn.ensemble import IsolationForest as SklearnIsolationForest
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logging.warning(
        "scikit-learn 未安装，将使用 fallback 纯 numpy 实现。"
        "建议安装: pip install scikit-learn==1.3.2"
    )

logger = logging.getLogger(__name__)


class MemoryLimitedScaler:
    """
    内存受限的标准化器
    - 使用 Welford 在线算法，不需要保存所有样本
    - 支持增量更新，内存 O(n_features)
    """

    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.std_ = None
        self.n_samples_ = 0
        self.is_fitted_ = False

    def partial_fit(self, X: np.ndarray) -> 'MemoryLimitedScaler':
        """
        增量更新均值和方差（Welford 在线算法）
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples, n_features = X.shape

        if self.mean_ is None:
            self.mean_ = np.zeros(n_features, dtype=float)
            self.var_ = np.zeros(n_features, dtype=float)

        # Welford 增量算法
        for i in range(n_samples):
            self.n_samples_ += 1
            x = X[i]
            delta = x - self.mean_
            self.mean_ += delta / self.n_samples_
            delta2 = x - self.mean_
            self.var_ += delta * delta2

        if self.n_samples_ > 1:
            self.std_ = np.sqrt(self.var_ / (self.n_samples_ - 1))
            self.std_[self.std_ == 0] = 1.0
        else:
            self.std_ = np.ones(n_features, dtype=float)

        self.is_fitted_ = True
        return self

    def fit(self, X: np.ndarray) -> 'MemoryLimitedScaler':
        """全量拟合"""
        self.n_samples_ = 0
        self.mean_ = None
        self.var_ = None
        return self.partial_fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """标准化"""
        if not self.is_fitted_:
            raise RuntimeError("Scaler 未拟合")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def reset(self):
        """重置状态，释放内存"""
        self.mean_ = None
        self.var_ = None
        self.std_ = None
        self.n_samples_ = 0
        self.is_fitted_ = False
        gc.collect()


class IncrementalIsolationForest:
    """
    增量式孤立森林（解决内存溢出问题）
    
    内存优化策略：
    1. 固定大小样本缓冲区（FIFO，最多 max_buffer_size 条）
    2. 定期重新训练（retrain_interval 条新样本后）
    3. 训练子采样（max_samples 限制每次训练样本数）
    4. 限制模型复杂度（n_estimators、max_depth）
    5. 显式 GC 清理
    6. 支持模型序列化持久化，避免重启重训
    
    当 scikit-learn 可用时使用 sklearn，否则 fallback 到纯 numpy
    """

    # 内存限制常量
    MAX_BUFFER_SIZE = 10000          # 最大样本缓冲区大小
    DEFAULT_MAX_SAMPLES = 256        # 单次训练采样数
    DEFAULT_N_ESTIMATORS = 50        # 减少树数量（原100）
    DEFAULT_MAX_DEPTH = 10           # 限制树深度
    RETRAIN_INTERVAL = 100           # 每100条新样本重训一次
    MODEL_CACHE_KEY = 'anomaly_model_v1'

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        contamination: float = 0.1,
        max_buffer_size: int = MAX_BUFFER_SIZE,
        retrain_interval: int = RETRAIN_INTERVAL,
        random_state: int = 42,
        model_cache_dir: Optional[str] = None
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_buffer_size = max_buffer_size
        self.retrain_interval = retrain_interval
        self.random_state = random_state

        # 内存受限缓冲区
        self._sample_buffer: deque = deque(maxlen=max_buffer_size)
        self._new_since_retrain = 0
        self._total_seen = 0
        self._is_trained = False
        self._model = None
        self._scaler = MemoryLimitedScaler()
        self._model_cache_dir = model_cache_dir or os.path.join(
            os.path.dirname(__file__), '_model_cache'
        )

        # 内存统计
        self._last_memory_usage = 0
        self._retrain_count = 0

        os.makedirs(self._model_cache_dir, exist_ok=True)
        self._cache_path = os.path.join(
            self._model_cache_dir, f'{self.MODEL_CACHE_KEY}.pkl'
        )

        # 尝试从缓存加载
        self._try_load_from_cache()

    def _try_load_from_cache(self):
        """尝试从持久化缓存加载模型"""
        if not os.path.exists(self._cache_path):
            return False

        try:
            with open(self._cache_path, 'rb') as f:
                cache = pickle.load(f)

            self._model = cache.get('model')
            self._scaler = cache.get('scaler', MemoryLimitedScaler())
            self._sample_buffer = deque(
                cache.get('buffer', []),
                maxlen=self.max_buffer_size
            )
            self._is_trained = cache.get('is_trained', False)
            self._total_seen = cache.get('total_seen', 0)
            self._retrain_count = cache.get('retrain_count', 0)

            logger.info(
                f"模型从缓存加载成功，已训练样本: {self._total_seen}, "
                f"缓冲区: {len(self._sample_buffer)} 条, 重训次数: {self._retrain_count}"
            )
            return True
        except Exception as e:
            logger.warning(f"模型缓存加载失败，将重新训练: {e}")
            if os.path.exists(self._cache_path):
                try:
                    os.remove(self._cache_path)
                except:
                    pass
            return False

    def _save_to_cache(self):
        """保存模型到缓存"""
        try:
            cache = {
                'model': self._model,
                'scaler': self._scaler,
                'buffer': list(self._sample_buffer),
                'is_trained': self._is_trained,
                'total_seen': self._total_seen,
                'retrain_count': self._retrain_count,
                'saved_at': np.datetime64('now').item()
            }

            tmp_path = self._cache_path + '.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

            os.replace(tmp_path, self._cache_path)

            # 估算内存
            import sys
            mem_usage = sys.getsizeof(cache) / (1024 * 1024)
            logger.debug(
                f"模型缓存已保存，大小: {mem_usage:.2f} MB, "
                f"缓冲区样本: {len(self._sample_buffer)}"
            )
        except Exception as e:
            logger.warning(f"模型缓存保存失败: {e}")

    def _check_memory(self) -> bool:
        """检查内存使用，必要时强制 GC"""
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)

            if mem_mb > self._last_memory_usage * 1.5 and mem_mb > 100:
                logger.info(
                    f"内存增长过快 ({self._last_memory_usage:.1f} -> {mem_mb:.1f} MB)，执行 GC"
                )
                gc.collect()
                mem_mb = process.memory_info().rss / (1024 * 1024)

            self._last_memory_usage = mem_mb

            # 内存超过 500MB 时触发缓冲区缩减
            if mem_mb > 500:
                logger.warning(
                    f"内存使用过高 ({mem_mb:.1f} MB)，缩减缓冲区"
                )
                self._trim_buffer(factor=0.5)
                return False

            return True
        except ImportError:
            # psutil 不可用时不检查
            return True

    def _trim_buffer(self, factor: float = 0.5):
        """缩减缓冲区（保留最新的 factor 比例）"""
        keep = int(len(self._sample_buffer) * factor)
        if keep < 100:
            keep = min(100, len(self._sample_buffer))

        if keep < len(self._sample_buffer):
            items = list(self._sample_buffer)[-keep:]
            self._sample_buffer.clear()
            self._sample_buffer.extend(items)
            logger.info(f"缓冲区已缩减: {keep}/{self.max_buffer_size}")
            gc.collect()

    def _should_retrain(self) -> bool:
        """判断是否需要重新训练"""
        if not self._is_trained:
            return len(self._sample_buffer) >= max(50, self.max_samples // 2)

        return self._new_since_retrain >= self.retrain_interval

    def _train_model(self):
        """训练/重训练模型，使用内存采样策略"""
        if len(self._sample_buffer) < 50:
            logger.info("样本不足 (<50)，暂不训练")
            return False

        self._check_memory()

        # 从缓冲区采样
        buffer_arr = np.array(self._sample_buffer, dtype=float)
        n_samples = len(buffer_arr)

        # 限制训练样本数，避免内存溢出
        actual_max_samples = min(
            self.max_samples,
            n_samples,
            2000  # 硬上限
        )

        logger.info(
            f"开始训练孤立森林: 样本数 {n_samples}, "
            f"采样 {actual_max_samples}, 树 {self.n_estimators}, "
            f"重训次数 #{self._retrain_count}"
        )

        # 随机采样（或分层采样：时间加权，新样本权重更高）
        if n_samples > actual_max_samples:
            # 时间加权：近期样本采样概率更高
            weights = np.linspace(0.5, 2.0, n_samples)
            weights /= weights.sum()
            indices = np.random.choice(
                n_samples,
                size=actual_max_samples,
                replace=False,
                p=weights
            )
            X_train = buffer_arr[indices]
        else:
            X_train = buffer_arr

        # 更新 scaler（增量）
        self._scaler.partial_fit(X_train)
        X_scaled = self._scaler.transform(X_train)

        # 创建并训练模型
        if HAS_SKLEARN:
            # 使用 scikit-learn 优化实现
            self._model = SklearnIsolationForest(
                n_estimators=self.n_estimators,
                max_samples=actual_max_samples,
                contamination=self.contamination,
                max_depth=self.DEFAULT_MAX_DEPTH,
                random_state=self.random_state,
                n_jobs=1,  # 单线程，避免多进程内存开销
                warm_start=False  # 每次重训全新模型
            )
            self._model.fit(X_scaled)
        else:
            # Fallback：使用纯 numpy 简化实现
            self._model = self._train_fallback(X_scaled)

        self._is_trained = True
        self._new_since_retrain = 0
        self._retrain_count += 1

        # 保存缓存
        if self._retrain_count % 5 == 0:
            self._save_to_cache()

        self._check_memory()
        return True

    def _train_fallback(self, X: np.ndarray):
        """scikit-learn 不可用时的简化 fallback 实现"""
        from dataclasses import dataclass

        @dataclass
        class _FallbackModel:
            threshold_: float
            center_: np.ndarray
            scale_: np.ndarray

            def decision_function(self, X):
                dist = np.linalg.norm(X - self.center_, axis=1) / self.scale_
                return -dist  # 负值表示异常

            def predict(self, X):
                scores = self.decision_function(X)
                return np.where(scores < -self.threshold_, -1, 1)

        center = np.mean(X, axis=0)
        scale = np.std(X, axis=0) + 1e-10
        dists = np.linalg.norm(X - center, axis=1) / scale
        threshold = np.quantile(dists, 1 - self.contamination)

        return _FallbackModel(
            threshold_=threshold,
            center_=center,
            scale_=scale
        )

    def add_sample(self, features: np.ndarray) -> bool:
        """
        添加样本到缓冲区，自动触发重训
        
        Returns:
            是否触发了重新训练
        """
        features = np.asarray(features, dtype=float).ravel()
        self._sample_buffer.append(features)
        self._total_seen += 1
        self._new_since_retrain += 1

        retrained = False
        if self._should_retrain():
            retrained = self._train_model()

        return retrained

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算异常评分（负值越负越异常）"""
        if not self._is_trained:
            self._train_model()
            if not self._is_trained:
                # 样本不足，返回中性评分
                return np.zeros(X.shape[0]) if X.ndim > 1 else np.zeros(1)

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self._scaler.transform(X)
        return self._model.decision_function(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测异常（-1 异常，1 正常）"""
        scores = self.decision_function(X)
        if HAS_SKLEARN:
            return self._model.predict(
                self._scaler.transform(np.asarray(X, dtype=float))
            )
        else:
            threshold = np.quantile(-scores, 1 - self.contamination)
            return np.where(-scores > threshold, -1, 1)

    def get_memory_stats(self) -> Dict:
        """获取内存使用统计"""
        import sys
        return {
            'total_seen': self._total_seen,
            'buffer_size': len(self._sample_buffer),
            'buffer_capacity': self.max_buffer_size,
            'is_trained': self._is_trained,
            'retrain_count': self._retrain_count,
            'new_since_retrain': self._new_since_retrain,
            'approx_memory_mb': round(
                sys.getsizeof(self._sample_buffer) / (1024 * 1024)
                + (sys.getsizeof(self._model) if self._model else 0) / (1024 * 1024),
                2
            ),
            'using_sklearn': HAS_SKLEARN
        }

    def reset(self):
        """完全重置，释放所有内存"""
        self._sample_buffer.clear()
        self._model = None
        self._scaler.reset()
        self._is_trained = False
        self._total_seen = 0
        self._new_since_retrain = 0
        self._retrain_count = 0
        gc.collect()

        if os.path.exists(self._cache_path):
            try:
                os.remove(self._cache_path)
            except:
                pass


class AnomalyDetector:
    """
    基于孤立森林的仿古作伪识别系统（内存优化版）
    
    核心改进（修复 isolation_forest.py:89 内存溢出）:
    1. 使用 scikit-learn IsolationForest（如果可用）
    2. 增量式训练 + 固定缓冲区（最多 10000 条）
    3. 内存检查 + 自动 GC + 缓冲区动态缩减
    4. 模型持久化缓存，避免重启重训
    """

    def __init__(
        self,
        n_estimators: int = 50,
        contamination: float = 0.1,
        max_samples: int = 256,
        max_buffer_size: int = 10000,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples

        # 使用增量式内存受限模型
        self.model = IncrementalIsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_buffer_size=max_buffer_size,
            random_state=random_state
        )
        self.scaler = self.model._scaler
        self.is_trained = False

        self.feature_names = [
            'spectrum_mean',
            'spectrum_std',
            'spectrum_peak_count',
            'main_peak_height',
            'main_peak_position',
            'peak_width_avg',
            'peak_height_ratio',
            'fe_concentration',
            'mn_concentration',
            'cu_concentration',
            'si_al_ratio',
            'rare_earth_total',
            'spectrum_entropy',
            'baseline_slope',
            'noise_level',
            'raman_xrf_correlation'
        ]

        # 启动时用默认样本初始化，避免首次检测等待训练
        self._lazy_init_default_model()

    def _lazy_init_default_model(self):
        """延迟初始化默认模型，使用合成的真品样本"""
        if self.model._is_trained:
            self.is_trained = True
            return

        try:
            np.random.seed(42)
            n_normal = 100
            normal_features = np.random.normal(0, 1, (n_normal, len(self.feature_names)))

            n_anomaly = int(n_normal * self.contamination)
            anomaly_features = np.random.normal(2.5, 1.2, (n_anomaly, len(self.feature_names)))

            all_features = np.vstack([normal_features, anomaly_features])

            for feat in all_features:
                self.model.add_sample(feat)

            self.is_trained = self.model._is_trained
            logger.info(f"默认模型初始化完成，内存统计: {self.model.get_memory_stats()}")
        except Exception as e:
            logger.warning(f"默认模型初始化失败: {e}")

    def extract_features(self, xrf_data: Optional[Dict],
                         raman_data: Optional[Dict]) -> np.ndarray:
        """
        从光谱数据中提取特征向量
        """
        features = np.zeros(len(self.feature_names))

        if xrf_data:
            xrf_spec = np.array(xrf_data.get('spectrum_data', []), dtype=float)
            if len(xrf_spec) > 0:
                features[0] = float(np.mean(xrf_spec))
                features[1] = float(np.std(xrf_spec))
                features[2] = float(self._count_peaks(xrf_spec))

                peaks = self._find_peaks(xrf_spec)
                peaks.sort(key=lambda p: p['height'], reverse=True)
                if peaks:
                    features[3] = float(peaks[0]['height'])
                    features[4] = float(peaks[0]['position'])
                    features[5] = float(np.mean([p['width'] for p in peaks])) if peaks else 0.0
                    if len(peaks) >= 2:
                        features[6] = float(
                            peaks[0]['height'] / (peaks[1]['height'] + 1e-10)
                        )

                features[7] = self._estimate_element_concentration(xrf_spec, 'Fe')
                features[8] = self._estimate_element_concentration(xrf_spec, 'Mn')
                features[9] = self._estimate_element_concentration(xrf_spec, 'Cu')
                features[10] = self._calc_si_al_ratio(xrf_spec)
                features[11] = self._calc_rare_earth_total(xrf_spec)
                features[12] = self._calc_spectrum_entropy(xrf_spec)
                features[13] = self._calc_baseline_slope(xrf_spec)
                features[14] = self._estimate_noise_level(xrf_spec)

        if raman_data and xrf_data:
            raman_spec = np.array(raman_data.get('spectrum_data', []), dtype=float)
            xrf_spec = np.array(xrf_data.get('spectrum_data', []), dtype=float)
            if len(raman_spec) > 10 and len(xrf_spec) > 10:
                min_len = min(len(raman_spec), len(xrf_spec))
                raman_resized = np.interp(
                    np.linspace(0, len(raman_spec) - 1, min_len),
                    np.arange(len(raman_spec)),
                    raman_spec
                )
                xrf_resized = np.interp(
                    np.linspace(0, len(xrf_spec) - 1, min_len),
                    np.arange(len(xrf_spec)),
                    xrf_spec
                )
                corr = np.corrcoef(raman_resized, xrf_resized)
                if len(corr) > 1 and len(corr[0]) > 1:
                    features[15] = float(corr[0, 1])

        return features.reshape(1, -1)

    def _count_peaks(self, spectrum: np.ndarray) -> int:
        if len(spectrum) < 3:
            return 0
        threshold = np.mean(spectrum) + np.std(spectrum) * 0.5
        peaks = 0
        for i in range(1, len(spectrum) - 1):
            if (spectrum[i] > spectrum[i-1]
                    and spectrum[i] > spectrum[i+1]
                    and spectrum[i] > threshold):
                peaks += 1
        return peaks

    def _find_peaks(self, spectrum: np.ndarray) -> List[Dict]:
        peaks = []
        if len(spectrum) < 3:
            return peaks
        threshold = np.mean(spectrum) + np.std(spectrum) * 0.3
        for i in range(1, len(spectrum) - 1):
            if (spectrum[i] > spectrum[i-1]
                    and spectrum[i] > spectrum[i+1]
                    and spectrum[i] > threshold):
                left = i
                right = i
                while left > 0 and spectrum[left] > spectrum[left-1]:
                    left -= 1
                while right < len(spectrum)-1 and spectrum[right] > spectrum[right+1]:
                    right += 1
                peaks.append({
                    'position': i,
                    'height': float(spectrum[i]),
                    'width': right - left
                })
        return peaks

    def _estimate_element_concentration(self, spectrum: np.ndarray, element: str) -> float:
        element_peaks = {'Fe': 120, 'Mn': 90, 'Cu': 150}
        peak_pos = element_peaks.get(element, 100)
        if peak_pos >= len(spectrum):
            peak_pos = len(spectrum) // 2
        start = max(0, peak_pos - 5)
        end = min(len(spectrum), peak_pos + 5)
        return float(np.sum(spectrum[start:end]))

    def _calc_si_al_ratio(self, spectrum: np.ndarray) -> float:
        si_peak = min(50, len(spectrum) - 1)
        al_peak = min(75, len(spectrum) - 1)
        return float(spectrum[si_peak] / (spectrum[al_peak] + 1e-10))

    def _calc_rare_earth_total(self, spectrum: np.ndarray) -> float:
        if len(spectrum) < 200:
            positions = [int(len(spectrum) * p) for p in [0.8, 0.85, 0.9, 0.95]]
        else:
            positions = [180, 190, 200, 210, 220]
        return float(sum(spectrum[p] for p in positions if p < len(spectrum)))

    def _calc_spectrum_entropy(self, spectrum: np.ndarray) -> float:
        if len(spectrum) == 0:
            return 0.0
        spec = spectrum - np.min(spectrum) + 1e-10
        total = np.sum(spec)
        if total <= 0:
            return 0.0
        prob = spec / total
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob)))

    def _calc_baseline_slope(self, spectrum: np.ndarray) -> float:
        if len(spectrum) < 10:
            return 0.0
        x = np.arange(len(spectrum))
        coeffs = np.polyfit(x, spectrum, 1)
        return float(coeffs[0])

    def _estimate_noise_level(self, spectrum: np.ndarray) -> float:
        if len(spectrum) < 20:
            return 0.0
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(spectrum, kernel, mode='same')
        return float(np.std(spectrum - smoothed))

    def train(self, features: np.ndarray):
        """
        训练模型（内存安全，增量式）
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        for i in range(features.shape[0]):
            self.model.add_sample(features[i])

        self.is_trained = self.model._is_trained
        logger.info(f"模型训练完成，内存统计: {self.model.get_memory_stats()}")

    def detect(self, features: np.ndarray, artifact_id: str = '') -> Dict:
        """
        检测异常（自动添加样本用于增量学习）
        """
        if not self.is_trained:
            self._lazy_init_default_model()

        features_scaled = self.scaler.transform(features)
        anomaly_score = float(self.model.decision_function(features)[0])
        is_anomaly = int(self.model.predict(features)[0]) == -1

        # 添加到缓冲区用于增量学习（不立即重训）
        self.model.add_sample(features.ravel())
        self.is_trained = self.model._is_trained

        raw_score = -anomaly_score
        forgery_prob = min(1.0, max(0.0, (raw_score - 0.3) / 0.7))

        anomaly_reasons = self._analyze_anomaly_reasons(
            features.ravel(), forgery_prob
        )

        return {
            'artifact_id': artifact_id,
            'anomaly_score': anomaly_score,
            'forgery_probability': float(forgery_prob),
            'is_anomaly': bool(is_anomaly),
            'features': {
                name: float(val)
                for name, val in zip(self.feature_names, features.ravel())
            },
            'anomaly_reasons': anomaly_reasons,
            'risk_level': self._get_risk_level(forgery_prob),
            'memory_stats': self.model.get_memory_stats()
        }

    def _analyze_anomaly_reasons(
        self, feature_values: np.ndarray, forgery_prob: float
    ) -> List[str]:
        reasons = []
        if abs(feature_values[2]) > 2.0:
            reasons.append("光谱峰数异常")
        if feature_values[7] > 1.5:
            reasons.append("铁元素含量异常偏高")
        if feature_values[9] > 1.0:
            reasons.append("铜元素含量异常，疑似人工染色")
        if abs(feature_values[10]) > 3.0:
            reasons.append("硅铝比异常")
        if abs(feature_values[12]) > 2.0:
            reasons.append("光谱信息熵异常")
        if feature_values[14] > 1.5:
            reasons.append("噪声水平异常")
        if feature_values[15] < -0.2:
            reasons.append("拉曼与XRF光谱相关性异常")
        if forgery_prob > 0.7 and not reasons:
            reasons.append("整体光谱特征与真品模式存在差异")
        if not reasons:
            reasons.append("光谱特征基本正常")
        return reasons

    def _get_risk_level(self, forgery_prob: float) -> str:
        if forgery_prob >= 0.8:
            return 'high'
        elif forgery_prob >= 0.5:
            return 'medium'
        else:
            return 'low'

    def batch_detect(
        self, features_list: List[np.ndarray], artifact_ids: List[str]
    ) -> List[Dict]:
        """批量检测（流式内存处理）"""
        results = []
        for i, (feat, aid) in enumerate(zip(features_list, artifact_ids)):
            res = self.detect(feat, aid)
            results.append(res)
            if (i + 1) % 100 == 0:
                gc.collect()
        return results

    def get_feature_importance(self) -> Dict[str, float]:
        importances = {}
        for i, name in enumerate(self.feature_names):
            importances[name] = float(1.0 / len(self.feature_names))
        return importances

    def save_model(self, filepath: str):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names
            }, f)

    def load_model(self, filepath: str):
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = self.model._scaler
        self.is_trained = data.get('is_trained', False)
        self.feature_names = data.get('feature_names', self.feature_names)

    def get_memory_info(self) -> Dict:
        """获取完整内存信息（用于诊断）"""
        return {
            'detector_trained': self.is_trained,
            'using_sklearn': HAS_SKLEARN,
            'model': self.model.get_memory_stats(),
            'feature_dim': len(self.feature_names)
        }
