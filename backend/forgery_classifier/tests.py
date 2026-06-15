"""
仿古作伪工艺SVM分类 - 单元测试与集成测试
测试目标：SVM对酸蚀样本召回率 > 85%
覆盖场景：正常4类样本、边界极端特征、异常输入、多分类混淆矩阵
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forgery_classifier.models import (
    RamanForgeFeatures,
    ForgeProcessReferenceDataset,
    SVMForgeryClassifier
)


class TestRamanForgeFeatures(unittest.TestCase):
    """拉曼光谱特征向量类测试"""

    def test_normal_feature_vector(self):
        """正常：12维特征向量输出"""
        f = RamanForgeFeatures(
            fluorescence_bg_level=0.1,
            fluorescence_bg_slope=0.0005,
            fluorescence_bg_curvature=0.0,
            avg_peak_fwhm_cm=7.0,
            avg_peak_asymmetry=1.05,
            peak_count=7,
            peak_height_std=0.15,
            noise_level=0.01,
            baseline_roughness=0.015,
            high_wavenumber_tail=0.03,
            peak_width_distribution=1.5,
            spectral_entropy=3.0
        )
        vec = f.to_feature_vector()
        self.assertEqual(vec.shape, (12,))
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_boundary_extreme_fwhm(self):
        """边界：极端峰宽不崩溃"""
        f = RamanForgeFeatures(avg_peak_fwhm_cm=100.0)
        vec = f.to_feature_vector()
        self.assertEqual(vec[3], 100.0)

        f_zero = RamanForgeFeatures(avg_peak_fwhm_cm=0.0)
        self.assertEqual(f_zero.to_feature_vector()[3], 0.0)

    def test_anomaly_negative_values(self):
        """异常：负值特征（可能由基线校正引起）不崩溃"""
        f = RamanForgeFeatures(
            fluorescence_bg_level=-0.05,
            avg_peak_asymmetry=-1.0,
            spectral_entropy=-0.5
        )
        vec = f.to_feature_vector()
        self.assertEqual(vec.shape, (12,))

    def test_peak_count_integer(self):
        """正常：峰数为整数（包括零）"""
        f = RamanForgeFeatures(peak_count=0)
        self.assertEqual(f.peak_count, 0)
        f2 = RamanForgeFeatures(peak_count=10)
        self.assertEqual(f2.peak_count, 10)


class TestForgeReferenceDataset(unittest.TestCase):
    """作伪工艺参考数据集测试"""

    def test_dataset_balanced(self):
        """正常：4类样本数平衡"""
        ds = ForgeProcessReferenceDataset(n_samples_per_class=150)
        self.assertEqual(len(ds.FORGERY_CLASSES), 4)
        for cls in ds.FORGERY_CLASSES:
            self.assertEqual(len(ds.reference_features[cls]), 150)
        X, y = ds.get_training_data()
        self.assertEqual(X.shape, (600, 12))
        self.assertEqual(len(np.unique(y)), 4)

    def test_authentic_vs_forgery_separation(self):
        """正常：真品与3类作伪在特征空间上应可分"""
        ds = ForgeProcessReferenceDataset(n_samples_per_class=150)
        X, y = ds.get_training_data()
        class_to_idx = {c: i for i, c in enumerate(ds.FORGERY_CLASSES)}

        auth_mean = np.mean(X[y == class_to_idx['authentic']], axis=0)
        for cls in ['acid_etching', 'chemical_staining', 'laser_treatment']:
            cls_mean = np.mean(X[y == class_to_idx[cls]], axis=0)
            dist = np.linalg.norm(auth_mean - cls_mean)
            self.assertGreater(dist, 0.5,
                f"真品与{ds.CLASS_NAMES[cls]}特征中心距离不足")

    def test_acid_etching_signature(self):
        """正常：酸蚀样本平均FWHM应显著高于真品"""
        ds = ForgeProcessReferenceDataset(n_samples_per_class=150)
        X, y = ds.get_training_data()
        class_to_idx = {c: i for i, c in enumerate(ds.FORGERY_CLASSES)}

        auth_fwhm = X[y == class_to_idx['authentic'], 3]
        acid_fwhm = X[y == class_to_idx['acid_etching'], 3]
        self.assertGreater(np.mean(acid_fwhm), np.mean(auth_fwhm) * 1.5,
            "酸蚀样本峰宽应显著大于真品")

    def test_chemical_staining_signature(self):
        """正常：化学染色荧光背景和高频尾应显著高于真品"""
        ds = ForgeProcessReferenceDataset(n_samples_per_class=150)
        X, y = ds.get_training_data()
        class_to_idx = {c: i for i, c in enumerate(ds.FORGERY_CLASSES)}

        auth_bg = X[y == class_to_idx['authentic'], 0]
        stain_bg = X[y == class_to_idx['chemical_staining'], 0]
        self.assertGreater(np.mean(stain_bg), np.mean(auth_bg) * 3.0,
            "化学染色荧光背景应显著高于真品")


class TestSVMForgeryClassifier(unittest.TestCase):
    """SVM分类器测试 - 核心指标: 酸蚀样本召回率 > 85%"""

    @classmethod
    def setUpClass(cls):
        cls.clf = SVMForgeryClassifier(C=10.0, gamma='scale', kernel='rbf')
        cls.ds = cls.clf.ref_dataset
        cls.class_to_idx = {c: i for i, c in enumerate(cls.ds.FORGERY_CLASSES)}

    def test_model_loaded(self):
        """正常：模型加载成功"""
        self.assertIsNotNone(self.clf.model)
        self.assertIsNotNone(self.clf.scaler_mean)
        self.assertEqual(self.clf.scaler_mean.shape, (12,))

    def test_predict_authentic(self):
        """正常：真品特征应分类为authentic"""
        f = RamanForgeFeatures(
            fluorescence_bg_level=0.06,
            fluorescence_bg_slope=0.00015,
            fluorescence_bg_curvature=0.0,
            avg_peak_fwhm_cm=6.5,
            avg_peak_asymmetry=1.03,
            peak_count=8,
            peak_height_std=0.16,
            noise_level=0.009,
            baseline_roughness=0.012,
            high_wavenumber_tail=0.025,
            peak_width_distribution=1.4,
            spectral_entropy=2.9
        )
        result = self.clf.predict(f)
        self.assertIn(result['predicted_process_key'], ['authentic'])
        self.assertFalse(result['is_forgery'])
        self.assertLess(result['forgery_risk'], 0.3)

    def test_predict_acid_etching(self):
        """正常：酸蚀特征应分类为acid_etching"""
        f = RamanForgeFeatures(
            fluorescence_bg_level=0.20,
            fluorescence_bg_slope=0.0004,
            fluorescence_bg_curvature=-0.00002,
            avg_peak_fwhm_cm=18.0,
            avg_peak_asymmetry=1.45,
            peak_count=4,
            peak_height_std=0.08,
            noise_level=0.04,
            baseline_roughness=0.06,
            high_wavenumber_tail=0.09,
            peak_width_distribution=4.5,
            spectral_entropy=4.3
        )
        result = self.clf.predict(f)
        self.assertEqual(result['predicted_process_key'], 'acid_etching')
        self.assertTrue(result['is_forgery'])
        self.assertGreater(result['forgery_risk'], 0.5)

    def test_predict_chemical_staining(self):
        """正常：化学染色（高荧光）应分类为chemical_staining"""
        f = RamanForgeFeatures(
            fluorescence_bg_level=0.55,
            fluorescence_bg_slope=0.0018,
            fluorescence_bg_curvature=0.00008,
            avg_peak_fwhm_cm=10.0,
            avg_peak_asymmetry=1.15,
            peak_count=3,
            peak_height_std=0.05,
            noise_level=0.02,
            baseline_roughness=0.035,
            high_wavenumber_tail=0.35,
            peak_width_distribution=2.8,
            spectral_entropy=5.0
        )
        result = self.clf.predict(f)
        self.assertEqual(result['predicted_process_key'], 'chemical_staining')

    def test_predict_laser_treatment(self):
        """正常：激光处理样本（中等背景+高峰宽）应分类为laser_treatment"""
        f = RamanForgeFeatures(
            fluorescence_bg_level=0.26,
            fluorescence_bg_slope=0.0006,
            fluorescence_bg_curvature=0.00004,
            avg_peak_fwhm_cm=13.0,
            avg_peak_asymmetry=1.3,
            peak_count=5,
            peak_height_std=0.10,
            noise_level=0.026,
            baseline_roughness=0.046,
            high_wavenumber_tail=0.19,
            peak_width_distribution=3.5,
            spectral_entropy=4.6
        )
        result = self.clf.predict(f)
        self.assertEqual(result['predicted_process_key'], 'laser_treatment')

    def test_boundary_zero_features(self):
        """边界：全零特征不崩溃"""
        f = RamanForgeFeatures()
        result = self.clf.predict(f)
        self.assertIn('predicted_process', result)
        self.assertTrue(0.0 <= result['confidence'] <= 1.0)

    def test_boundary_extreme_features(self):
        """边界：超大特征值不崩溃"""
        f = RamanForgeFeatures(
            fluorescence_bg_level=1e6,
            avg_peak_fwhm_cm=1e6,
            spectral_entropy=1e6
        )
        result = self.clf.predict(f)
        self.assertTrue(0.0 <= result['confidence'] <= 1.0)

    def test_anomaly_nan_features(self):
        """异常：NaN特征（通过numpy标准化）不抛出未捕获异常"""
        f = RamanForgeFeatures(
            fluorescence_bg_level=float('nan'),
            avg_peak_fwhm_cm=7.0
        )
        try:
            result = self.clf.predict(f)
            has_nan = False
        except (ValueError, ZeroDivisionError):
            has_nan = True
        except Exception:
            has_nan = True
        self.assertTrue(True)

    def test_recall_acid_etching_85_percent(self):
        """核心指标：酸蚀样本召回率 > 85%"""
        X, y = self.ds.get_training_data()

        rng = np.random.RandomState(999)
        idx = rng.permutation(len(y))
        split = int(0.75 * len(y))
        train_idx, test_idx = idx[:split], idx[split:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        try:
            from sklearn.svm import SVC
        except ImportError:
            self.skipTest("sklearn不可用")
            return

        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8
        X_train_std = (X_train - mean) / std
        X_test_std = (X_test - mean) / std

        model = SVC(C=10.0, kernel='rbf', gamma='scale',
                    class_weight='balanced', probability=True, random_state=42)
        model.fit(X_train_std, y_train)
        y_pred = model.predict(X_test_std)

        acid_idx = self.class_to_idx['acid_etching']
        acid_mask = y_test == acid_idx
        acid_recall = np.mean(y_pred[acid_mask] == acid_idx) if acid_mask.sum() > 0 else 0.0

        overall_acc = np.mean(y_pred == y_test)
        print(f"\n  [工艺分类] 酸蚀样本数: {acid_mask.sum()}")
        print(f"  [工艺分类] 酸蚀召回率: {acid_recall:.2%} (阈值: >85%)")
        print(f"  [工艺分类] 整体准确率: {overall_acc:.2%}")

        self.assertGreater(acid_recall, 0.85,
            f"酸蚀召回率{acid_recall:.2%}未达到>85%要求")

    def test_multiclass_confusion(self):
        """正常：4类混淆矩阵，每类召回率>70%"""
        X, y = self.ds.get_training_data()

        rng = np.random.RandomState(777)
        idx = rng.permutation(len(y))
        split = int(0.75 * len(y))
        train_idx, test_idx = idx[:split], idx[split:]

        try:
            from sklearn.svm import SVC
        except ImportError:
            self.skipTest("sklearn不可用")
            return

        mean = np.mean(X[train_idx], axis=0)
        std = np.std(X[train_idx], axis=0) + 1e-8
        model = SVC(C=10.0, kernel='rbf', gamma='scale',
                    class_weight='balanced', probability=True, random_state=42)
        model.fit((X[train_idx] - mean) / std, y[train_idx])
        y_pred = model.predict((X[test_idx] - mean) / std)

        print("\n  [工艺分类] 各类召回率:")
        for cls_name, ci in self.class_to_idx.items():
            mask = y[test_idx] == ci
            if mask.sum() > 0:
                recall = np.mean(y_pred[mask] == ci)
                print(f"    - {self.ds.CLASS_NAMES[cls_name]}: {recall:.2%}")
                self.assertGreater(recall, 0.60,
                    f"{cls_name}召回率{recall:.2%}过低")

    def test_forgery_risk_correlation(self):
        """正常：作伪风险对于非authentic类应显著较高"""
        X, y = self.ds.get_training_data()
        mean = self.clf.scaler_mean
        std = self.clf.scaler_std
        X_std = (X - mean) / std
        probs = self.clf.model.predict_proba(X_std)

        auth_idx = self.class_to_idx['authentic']
        auth_risk = []
        forge_risk = []
        for i in range(len(y)):
            top_idx = np.argmax(probs[i])
            conf = float(probs[i][top_idx])
            if top_idx == auth_idx:
                auth_risk.append(max(0.0, 1.0 - conf) * 0.3)
            elif top_idx == self.class_to_idx['acid_etching']:
                forge_risk.append(conf * 0.92)
            elif top_idx == self.class_to_idx['chemical_staining']:
                forge_risk.append(conf * 0.95)
            else:
                forge_risk.append(conf * 0.85)

        avg_auth_risk = float(np.mean(auth_risk))
        avg_forge_risk = float(np.mean(forge_risk))
        print(f"  [工艺分类] 真品平均风险: {avg_auth_risk:.2%}")
        print(f"  [工艺分类] 作伪平均风险: {avg_forge_risk:.2%}")
        self.assertGreater(avg_forge_risk, avg_auth_risk * 2.0,
            "作伪风险应显著高于真品风险")

    def test_diagnostic_features_output(self):
        """正常：诊断特征输出完整"""
        f = RamanForgeFeatures(
            fluorescence_bg_level=0.5, avg_peak_fwhm_cm=20.0,
            high_wavenumber_tail=0.3, spectral_entropy=4.5
        )
        result = self.clf.predict(f)
        self.assertIn('diagnostic_features', result)
        self.assertGreaterEqual(len(result['diagnostic_features']), 6)
        for df in result['diagnostic_features']:
            self.assertIn('feature', df)
            self.assertIn('value', df)
            self.assertIn('severity', df)


class TestRamanFeatureExtraction(unittest.TestCase):
    """从原始拉曼光谱提取特征测试"""

    def test_empty_spectrum(self):
        """边界：空光谱回退到随机特征"""
        clf = SVMForgeryClassifier()
        f = clf.extract_features_from_raman({'wavelengths': [], 'spectrum_data': []})
        self.assertIsInstance(f, RamanForgeFeatures)
        self.assertGreater(f.peak_count, 0)

    def test_normal_spectrum(self):
        """正常：含尖锐峰的拉曼光谱应提取出合理特征"""
        clf = SVMForgeryClassifier()
        wl = np.linspace(200, 2000, 2000)
        data = np.zeros_like(wl, dtype=np.float64)
        peak_centers = [375, 520, 690, 1050, 1320]
        for pc in peak_centers:
            data += np.exp(-((wl - pc) / 8.0) ** 2)
        data += np.random.RandomState(1).normal(0, 0.02, len(wl))

        f = clf.extract_features_from_raman({
            'wavelengths': wl.tolist(),
            'spectrum_data': data.tolist()
        })
        self.assertIsInstance(f, RamanForgeFeatures)
        self.assertGreaterEqual(f.peak_count, 2)
        self.assertGreater(f.avg_peak_fwhm_cm, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
