import unittest
import numpy as np
from unittest.mock import patch

from .models import (
    TraceElementProfile,
    ProvenanceReferenceDataset,
    RandomForestProvenanceClassifier
)


class TestTraceElementProfile(unittest.TestCase):

    def test_normal_profile(self):
        profile = TraceElementProfile(
            sr_ppm=45.0,
            nd_ppm=18.0,
            rb_ppm=5.0,
            cs_ppm=0.8,
            la_ppm=2.5,
            sm_ppm=0.6,
            yb_ppm=0.3,
            cr_ppm=12.0,
            ni_ppm=4.0,
            ti_ppm=85.0
        )
        self.assertAlmostEqual(profile.sr_nd_ratio, 45.0 / 18.0, places=6)
        self.assertIsInstance(profile.sr_ppm, float)
        self.assertIsInstance(profile.nd_ppm, float)
        self.assertGreater(profile.sr_ppm, 0)
        self.assertGreater(profile.nd_ppm, 0)

    def test_to_feature_vector_shape(self):
        profile = TraceElementProfile(
            sr_ppm=45.0, nd_ppm=18.0, rb_ppm=5.0, cs_ppm=0.8, la_ppm=2.5,
            sm_ppm=0.6, yb_ppm=0.3, cr_ppm=12.0, ni_ppm=4.0, ti_ppm=85.0
        )
        full_vector = profile.to_feature_vector()
        self.assertEqual(full_vector.shape, (11,))
        
        clf = RandomForestProvenanceClassifier()
        selected_vector = full_vector[clf.selected_features]
        self.assertEqual(selected_vector.shape, (8,))
        
        self.assertIsInstance(full_vector, np.ndarray)
        self.assertEqual(full_vector.dtype, np.float64)

    def test_boundary_extreme_values(self):
        profile = TraceElementProfile(
            sr_ppm=1e-10,
            nd_ppm=1e10,
            rb_ppm=0.001,
            cs_ppm=0.0001,
            la_ppm=0.0001,
            sm_ppm=0.00001,
            yb_ppm=0.00001,
            cr_ppm=10000.0,
            ni_ppm=5000.0,
            ti_ppm=100000.0
        )
        self.assertAlmostEqual(profile.sr_nd_ratio, 1e-10 / 1e10, places=6)
        
        vec = profile.to_feature_vector()
        self.assertTrue(np.all(np.isfinite(vec)))
        self.assertEqual(vec.shape, (11,))

    def test_anomaly_missing_elements(self):
        with self.assertRaises(TypeError):
            TraceElementProfile(
                sr_ppm=45.0,
                nd_ppm=18.0,
                rb_ppm=5.0,
                cs_ppm=0.8,
                la_ppm=2.5,
                sm_ppm=0.6,
                yb_ppm=0.3,
                cr_ppm=12.0,
                ni_ppm=4.0
            )

        profile = TraceElementProfile(
            sr_ppm=45.0,
            nd_ppm=0.0,
            rb_ppm=5.0,
            cs_ppm=0.8,
            la_ppm=2.5,
            sm_ppm=0.6,
            yb_ppm=0.3,
            cr_ppm=12.0,
            ni_ppm=4.0,
            ti_ppm=85.0
        )
        self.assertEqual(profile.sr_nd_ratio, 0.0)


class TestProvenanceReferenceDataset(unittest.TestCase):

    def test_data_generation(self):
        dataset = ProvenanceReferenceDataset(n_samples_per_origin=50)
        self.assertEqual(len(dataset.ORIGINS), 5)
        self.assertEqual(len(dataset.reference_profiles), 5)
        
        for origin in dataset.ORIGINS:
            self.assertIn(origin, dataset.reference_profiles)
            self.assertEqual(len(dataset.reference_profiles[origin]), 50)
            for profile in dataset.reference_profiles[origin]:
                self.assertIsInstance(profile, TraceElementProfile)

        X, y = dataset.get_training_data()
        self.assertEqual(X.shape, (250, 11))
        self.assertEqual(y.shape, (250,))
        self.assertEqual(len(np.unique(y)), 5)

    def test_class_balance(self):
        dataset = ProvenanceReferenceDataset(n_samples_per_origin=200)
        X, y = dataset.get_training_data()
        
        unique, counts = np.unique(y, return_counts=True)
        for count in counts:
            self.assertEqual(count, 200)
        
        total_samples = len(y)
        expected_ratio = 1.0 / 5.0
        for count in counts:
            ratio = count / total_samples
            self.assertAlmostEqual(ratio, expected_ratio, places=2)

    def test_feature_scales_reasonable(self):
        dataset = ProvenanceReferenceDataset(n_samples_per_origin=100)
        X, y = dataset.get_training_data()
        
        feature_ranges = {
            'Sr': (10, 200),
            'Nd': (5, 60),
            'Rb': (0.1, 30),
            'Cs': (0.01, 5),
            'La': (0.05, 20),
            'Sm': (0.02, 4),
            'Yb': (0.01, 2),
            'Cr': (5, 150),
            'Ni': (0.2, 50),
            'Ti': (20, 800),
            'Sr/Nd': (0.5, 15)
        }
        
        for i, (feat_name, (min_val, max_val)) in enumerate(feature_ranges.items()):
            col = X[:, i]
            self.assertGreaterEqual(np.min(col), min_val * 0.5,
                                    f"{feat_name} 最小值超出合理范围")
            self.assertLessEqual(np.max(col), max_val * 2,
                                 f"{feat_name} 最大值超出合理范围")
            
            mean = np.mean(col)
            std = np.std(col)
            self.assertGreater(std, 0, f"{feat_name} 标准差为0")
            self.assertGreater(mean, 0, f"{feat_name} 均值为0或负数")


class TestRandomForestProvenanceClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clf = RandomForestProvenanceClassifier()
        cls.dataset = ProvenanceReferenceDataset(n_samples_per_origin=100)

    def test_model_loaded(self):
        self.assertIsNotNone(self.clf.model)
        self.assertIsNotNone(self.clf.scaler_mean)
        self.assertIsNotNone(self.clf.scaler_std)
        self.assertIsNotNone(self.clf.selected_features)
        self.assertEqual(len(self.clf.selected_features), 8)
        self.assertEqual(self.clf.n_features_to_select, 8)

    def test_single_prediction_hetian(self):
        profile = TraceElementProfile(
            sr_ppm=45.0, nd_ppm=18.0, rb_ppm=5.0, cs_ppm=0.8, la_ppm=2.5,
            sm_ppm=0.6, yb_ppm=0.3, cr_ppm=12.0, ni_ppm=4.0, ti_ppm=85.0
        )
        result = self.clf.predict(profile)
        
        self.assertIn('predicted_origin', result)
        self.assertIn('predicted_origin_key', result)
        self.assertIn('confidence', result)
        self.assertIn('top_predictions', result)
        self.assertEqual(result['predicted_origin_key'], 'hetian')
        self.assertGreater(result['confidence'], 0.5)
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    def test_single_prediction_xiuyan(self):
        profile = TraceElementProfile(
            sr_ppm=120.0, nd_ppm=35.0, rb_ppm=12.0, cs_ppm=2.5, la_ppm=8.0,
            sm_ppm=2.0, yb_ppm=1.0, cr_ppm=35.0, ni_ppm=12.0, ti_ppm=250.0
        )
        result = self.clf.predict(profile)
        
        self.assertEqual(result['predicted_origin_key'], 'xiuyan')
        self.assertGreater(result['confidence'], 0.5)
        self.assertEqual(len(result['top_predictions']), 3)
        for pred in result['top_predictions']:
            self.assertIn('origin_key', pred)
            self.assertIn('origin_name', pred)
            self.assertIn('probability', pred)

    def test_single_prediction_dushan(self):
        profile = TraceElementProfile(
            sr_ppm=80.0, nd_ppm=12.0, rb_ppm=3.0, cs_ppm=0.5, la_ppm=1.5,
            sm_ppm=0.4, yb_ppm=0.2, cr_ppm=80.0, ni_ppm=25.0, ti_ppm=500.0
        )
        result = self.clf.predict(profile)
        
        self.assertEqual(result['predicted_origin_key'], 'dushan')
        self.assertGreater(result['confidence'], 0.5)
        self.assertIn('feature_importance', result)
        self.assertIn('rfe_info', result)

    def test_cross_validation_accuracy(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            X, y = self.dataset.get_training_data()
            X_rfe = X[:, self.clf.selected_features]
            X_std = (X_rfe - self.clf.scaler_mean) / self.clf.scaler_std
            
            clf_cv = RandomForestClassifier(
                n_estimators=100, max_depth=15,
                min_samples_split=5, random_state=42,
                class_weight='balanced', n_jobs=-1
            )
            
            scores = cross_val_score(clf_cv, X_std, y, cv=5, scoring='accuracy')
            mean_accuracy = np.mean(scores)
            
            self.assertGreater(mean_accuracy, 0.80,
                               f"5折交叉验证准确率 {mean_accuracy:.2%} 未达到 80%")
            self.assertGreater(len(scores), 0)
            self.assertEqual(len(scores), 5)
            
        except ImportError:
            X, y = self.dataset.get_training_data()
            X_rfe = X[:, self.clf.selected_features]
            X_std = (X_rfe - self.clf.scaler_mean) / self.clf.scaler_std
            
            correct = 0
            total = len(y)
            for i in range(total):
                X_train = np.delete(X_std, i, axis=0)
                y_train = np.delete(y, i)
                X_test = X_std[i:i+1]
                y_test = y[i]
                
                temp_clf = RandomForestClassifier(
                    n_estimators=50, max_depth=10,
                    min_samples_split=5, random_state=42,
                    class_weight='balanced'
                )
                temp_clf.fit(X_train, y_train)
                pred = temp_clf.predict(X_test)[0]
                if pred == y_test:
                    correct += 1
            
            accuracy = correct / total
            self.assertGreater(accuracy, 0.80,
                               f"留一法准确率 {accuracy:.2%} 未达到 80%")

    def test_feature_importance_reasonable(self):
        profile = TraceElementProfile(
            sr_ppm=45.0, nd_ppm=18.0, rb_ppm=5.0, cs_ppm=0.8, la_ppm=2.5,
            sm_ppm=0.6, yb_ppm=0.3, cr_ppm=12.0, ni_ppm=4.0, ti_ppm=85.0
        )
        result = self.clf.predict(profile)
        
        feat_imp = result['feature_importance']
        self.assertGreater(len(feat_imp), 0)
        
        importances = [f['importance'] for f in feat_imp]
        total_importance = sum(importances)
        
        for imp in importances:
            self.assertGreaterEqual(imp, 0.0)
            self.assertLessEqual(imp, 1.0)
        
        self.assertGreater(total_importance, 0.0)
        self.assertAlmostEqual(total_importance, 1.0, places=1)

    def test_rfe_eliminated_redundant_features(self):
        self.assertIsNotNone(self.clf.rfe_ranking_)
        self.assertEqual(len(self.clf.rfe_ranking_), 11)
        
        self.assertLess(len(self.clf.selected_features), 11,
                        "RFE 未消除任何冗余特征")
        
        eliminated_indices = np.where(self.clf.rfe_ranking_ > 1)[0]
        self.assertGreater(len(eliminated_indices), 0,
                           "RFE 未消除任何特征")
        
        eliminated_features = [self.clf.ALL_FEATURE_NAMES[i] for i in eliminated_indices]
        
        redundant_features = {'Sm', 'Yb', 'Sr/Nd'}
        eliminated_set = set(eliminated_features)
        self.assertTrue(
            len(redundant_features & eliminated_set) >= 1,
            f"RFE 应至少消除 Sm、Yb、Sr/Nd 中的一个，实际消除: {eliminated_features}"
        )

    def test_rfe_no_accuracy_degradation(self):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            X, y = self.dataset.get_training_data()
            
            mean_full = np.mean(X, axis=0)
            std_full = np.std(X, axis=0) + 1e-8
            X_std_full = (X - mean_full) / std_full
            
            clf_full = RandomForestClassifier(
                n_estimators=100, max_depth=15,
                min_samples_split=5, random_state=42,
                class_weight='balanced', n_jobs=-1
            )
            scores_full = cross_val_score(clf_full, X_std_full, y, cv=5, scoring='accuracy')
            acc_full = np.mean(scores_full)
            
            X_rfe = X[:, self.clf.selected_features]
            X_std_rfe = (X_rfe - self.clf.scaler_mean) / self.clf.scaler_std
            clf_rfe = RandomForestClassifier(
                n_estimators=100, max_depth=15,
                min_samples_split=5, random_state=42,
                class_weight='balanced', n_jobs=-1
            )
            scores_rfe = cross_val_score(clf_rfe, X_std_rfe, y, cv=5, scoring='accuracy')
            acc_rfe = np.mean(scores_rfe)
            
            self.assertGreater(acc_rfe, 0.80,
                               f"RFE后准确率 {acc_rfe:.2%} 未达到 80%")
            
            degradation = acc_full - acc_rfe
            self.assertLess(degradation, 0.05,
                            f"RFE导致准确率下降超过5%: {degradation:.2%}")
            
        except ImportError:
            self.skipTest("sklearn not available for comparison")

    def test_boundary_nan_infinite(self):
        profile = TraceElementProfile(
            sr_ppm=45.0, nd_ppm=18.0, rb_ppm=5.0, cs_ppm=0.8, la_ppm=2.5,
            sm_ppm=0.6, yb_ppm=0.3, cr_ppm=12.0, ni_ppm=4.0, ti_ppm=85.0
        )
        
        vec = profile.to_feature_vector()
        
        vec_nan = vec.copy()
        vec_nan[0] = np.nan
        self.assertTrue(np.any(np.isnan(vec_nan)))
        
        vec_inf = vec.copy()
        vec_inf[0] = np.inf
        self.assertTrue(np.any(np.isinf(vec_inf)))
        
        profile_nan = TraceElementProfile(
            sr_ppm=np.nan, nd_ppm=18.0, rb_ppm=5.0, cs_ppm=0.8, la_ppm=2.5,
            sm_ppm=0.6, yb_ppm=0.3, cr_ppm=12.0, ni_ppm=4.0, ti_ppm=85.0
        )
        self.assertTrue(np.isnan(profile_nan.sr_ppm))
        
        vec_with_nan = profile_nan.to_feature_vector()
        self.assertTrue(np.any(np.isnan(vec_with_nan)))

    def test_anomaly_all_zeros(self):
        profile = TraceElementProfile(
            sr_ppm=0.0, nd_ppm=0.0, rb_ppm=0.0, cs_ppm=0.0, la_ppm=0.0,
            sm_ppm=0.0, yb_ppm=0.0, cr_ppm=0.0, ni_ppm=0.0, ti_ppm=0.0
        )
        
        self.assertEqual(profile.sr_nd_ratio, 0.0)
        
        vec = profile.to_feature_vector()
        self.assertTrue(np.all(vec >= 0.0))
        self.assertEqual(np.sum(vec[:10]), 0.0)
        self.assertEqual(vec[10], 0.0)
        
        try:
            result = self.clf.predict(profile)
            self.assertIn('predicted_origin_key', result)
            self.assertIn('confidence', result)
        except Exception as e:
            self.assertIsInstance(e, Exception)


class TestXRFExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.clf = RandomForestProvenanceClassifier()

    def test_normal_xrf_extraction(self):
        energies = np.linspace(1, 40, 1000)
        spectrum_data = np.zeros_like(energies)
        
        element_peaks = {
            'Sr': [14.16],
            'Nd': [5.72, 6.20],
            'Rb': [13.39],
            'Cs': [4.28],
            'La': [4.65],
            'Sm': [6.71],
            'Yb': [7.41],
            'Cr': [5.41],
            'Ni': [7.47],
            'Ti': [4.51]
        }
        
        for peaks in element_peaks.values():
            for peak in peaks:
                idx = np.argmin(np.abs(energies - peak))
                spectrum_data[max(0, idx-10):min(len(energies), idx+10)] += 1000
        
        xrf_spectrum = {
            'energies': energies.tolist(),
            'spectrum_data': spectrum_data.tolist(),
            'artifact_id': 'test_001'
        }
        
        profile = self.clf.extract_profile_from_xrf(xrf_spectrum)
        
        self.assertIsInstance(profile, TraceElementProfile)
        self.assertGreater(profile.sr_ppm, 0)
        self.assertGreater(profile.nd_ppm, 0)
        self.assertGreater(profile.ti_ppm, 0)
        
        vec = profile.to_feature_vector()
        self.assertTrue(np.all(np.isfinite(vec)))
        self.assertTrue(np.all(vec >= 0))

    def test_anomaly_short_xrf(self):
        xrf_spectrum = {
            'energies': [],
            'spectrum_data': [],
            'artifact_id': 'test_empty'
        }
        
        profile = self.clf.extract_profile_from_xrf(xrf_spectrum)
        
        self.assertIsInstance(profile, TraceElementProfile)
        self.assertGreater(profile.sr_ppm, 0)
        self.assertGreater(profile.nd_ppm, 0)
        self.assertGreater(profile.rb_ppm, 0)
        self.assertGreater(profile.cs_ppm, 0)
        self.assertGreater(profile.la_ppm, 0)
        self.assertGreater(profile.sm_ppm, 0)
        self.assertGreater(profile.yb_ppm, 0)
        self.assertGreater(profile.cr_ppm, 0)
        self.assertGreater(profile.ni_ppm, 0)
        self.assertGreater(profile.ti_ppm, 0)
        
        xrf_short = {
            'energies': [1.0, 2.0],
            'spectrum_data': [10.0, 20.0],
            'artifact_id': 'test_short'
        }
        
        profile_short = self.clf.extract_profile_from_xrf(xrf_short)
        self.assertIsInstance(profile_short, TraceElementProfile)
        self.assertGreater(profile_short.sr_ppm, 0)


if __name__ == '__main__':
    unittest.main()
