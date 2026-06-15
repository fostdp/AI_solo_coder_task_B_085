import unittest
import numpy as np
from .models import (
    RamanForgeFeatures,
    ForgeProcessReferenceDataset,
    SVMForgeryClassifier
)


class TestRamanForgeFeatures(unittest.TestCase):

    def test_normal_feature_vector(self):
        features = RamanForgeFeatures(
            fluorescence_bg_level=0.1,
            fluorescence_bg_slope=0.0005,
            fluorescence_bg_curvature=0.00001,
            avg_peak_fwhm_cm=8.0,
            avg_peak_asymmetry=1.05,
            peak_count=7,
            peak_height_std=0.15,
            noise_level=0.01,
            baseline_roughness=0.02,
            high_wavenumber_tail=0.05,
            peak_width_distribution=2.0,
            spectral_entropy=3.0,
            uv_fluorescence_intensity=0.03,
            uv_fluorescence_peak_shift=2.0,
            uv_fluorescence_lifetime_ratio=0.2
        )
        vector = features.to_feature_vector()
        self.assertEqual(vector.shape, (15,))
        self.assertEqual(vector[12], 0.03)
        self.assertEqual(vector[13], 2.0)
        self.assertEqual(vector[14], 0.2)

    def test_boundary_peak_count(self):
        features = RamanForgeFeatures(peak_count=1)
        vector = features.to_feature_vector()
        self.assertEqual(vector[5], 1.0)

        features_high = RamanForgeFeatures(peak_count=15)
        vector_high = features_high.to_feature_vector()
        self.assertEqual(vector_high[5], 15.0)

    def test_anomaly_negative_values(self):
        features = RamanForgeFeatures(
            fluorescence_bg_level=-0.1,
            noise_level=-0.05,
            uv_fluorescence_intensity=-0.2
        )
        vector = features.to_feature_vector()
        self.assertLess(vector[0], 0)
        self.assertLess(vector[7], 0)
        self.assertLess(vector[12], 0)

    def test_anomaly_nan_values(self):
        features = RamanForgeFeatures(
            fluorescence_bg_level=np.nan,
            uv_fluorescence_lifetime_ratio=np.nan
        )
        vector = features.to_feature_vector()
        self.assertTrue(np.isnan(vector[0]))
        self.assertTrue(np.isnan(vector[14]))


class TestForgeReferenceDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = ForgeProcessReferenceDataset(n_samples_per_class=100)

    def test_normal_generation(self):
        X, y = self.dataset.get_training_data()
        self.assertEqual(X.shape[0], 400)
        self.assertEqual(X.shape[1], 15)
        self.assertEqual(y.shape[0], 400)
        self.assertTrue(np.all(np.isin(y, [0, 1, 2, 3])))

    def test_class_balance(self):
        _, y = self.dataset.get_training_data()
        unique, counts = np.unique(y, return_counts=True)
        for count in counts:
            self.assertEqual(count, 100)

    def test_acid_etching_signature(self):
        acid_features = self.dataset.reference_features['acid_etching']
        authentic_features = self.dataset.reference_features['authentic']

        acid_fwhm = np.mean([f.avg_peak_fwhm_cm for f in acid_features])
        authentic_fwhm = np.mean([f.avg_peak_fwhm_cm for f in authentic_features])

        self.assertGreater(acid_fwhm, 15.0)
        self.assertGreater(acid_fwhm, authentic_fwhm * 1.5)

    def test_chemical_staining_signature(self):
        staining_features = self.dataset.reference_features['chemical_staining']
        authentic_features = self.dataset.reference_features['authentic']

        staining_bg = np.mean([f.fluorescence_bg_level for f in staining_features])
        authentic_bg = np.mean([f.fluorescence_bg_level for f in authentic_features])

        self.assertGreater(staining_bg, 0.35)
        self.assertGreater(staining_bg, authentic_bg * 3)

    def test_uv_fluorescence_discriminates_staining(self):
        staining_features = self.dataset.reference_features['chemical_staining']
        authentic_features = self.dataset.reference_features['authentic']

        staining_uv = np.mean([f.uv_fluorescence_intensity for f in staining_features])
        authentic_uv = np.mean([f.uv_fluorescence_intensity for f in authentic_features])

        self.assertGreater(staining_uv, authentic_uv * 5)
        self.assertGreater(staining_uv, 0.5)

    def test_uv_lifetime_discriminates_staining(self):
        staining_features = self.dataset.reference_features['chemical_staining']
        authentic_features = self.dataset.reference_features['authentic']

        staining_lifetime = np.mean([f.uv_fluorescence_lifetime_ratio for f in staining_features])
        authentic_lifetime = np.mean([f.uv_fluorescence_lifetime_ratio for f in authentic_features])

        self.assertGreater(staining_lifetime, 0.7)
        self.assertGreater(staining_lifetime, authentic_lifetime * 3)


class TestSVMForgeryClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classifier = SVMForgeryClassifier()

    def test_model_loaded(self):
        self.assertIsNotNone(self.classifier.model)
        self.assertEqual(self.classifier.scaler_mean.shape, (15,))
        self.assertEqual(self.classifier.scaler_std.shape, (15,))

    def test_predict_authentic(self):
        features = RamanForgeFeatures(
            fluorescence_bg_level=0.05,
            fluorescence_bg_slope=0.0001,
            fluorescence_bg_curvature=0.0,
            avg_peak_fwhm_cm=6.5,
            avg_peak_asymmetry=1.02,
            peak_count=7,
            peak_height_std=0.15,
            noise_level=0.008,
            baseline_roughness=0.01,
            high_wavenumber_tail=0.02,
            peak_width_distribution=1.5,
            spectral_entropy=2.8,
            uv_fluorescence_intensity=0.02,
            uv_fluorescence_peak_shift=0.0,
            uv_fluorescence_lifetime_ratio=0.15
        )
        result = self.classifier.predict(features)
        self.assertEqual(result['predicted_process_key'], 'authentic')
        self.assertFalse(result['is_forgery'])
        self.assertIn('confidence', result)

    def test_predict_acid_etching(self):
        features = RamanForgeFeatures(
            fluorescence_bg_level=0.18,
            fluorescence_bg_slope=0.0003,
            fluorescence_bg_curvature=-0.00002,
            avg_peak_fwhm_cm=18.0,
            avg_peak_asymmetry=1.45,
            peak_count=4,
            peak_height_std=0.08,
            noise_level=0.035,
            baseline_roughness=0.06,
            high_wavenumber_tail=0.08,
            peak_width_distribution=4.5,
            spectral_entropy=4.2,
            uv_fluorescence_intensity=0.08,
            uv_fluorescence_peak_shift=5.0,
            uv_fluorescence_lifetime_ratio=0.25
        )
        result = self.classifier.predict(features)
        self.assertEqual(result['predicted_process_key'], 'acid_etching')
        self.assertTrue(result['is_forgery'])

    def test_predict_chemical_staining(self):
        features = RamanForgeFeatures(
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
            spectral_entropy=5.0,
            uv_fluorescence_intensity=0.75,
            uv_fluorescence_peak_shift=35.0,
            uv_fluorescence_lifetime_ratio=0.85
        )
        result = self.classifier.predict(features)
        self.assertEqual(result['predicted_process_key'], 'chemical_staining')
        self.assertTrue(result['is_forgery'])
        self.assertGreater(result['raw_features']['uv_fluorescence_intensity'], 0.5)

    def test_predict_laser_treatment(self):
        features = RamanForgeFeatures(
            fluorescence_bg_level=0.25,
            fluorescence_bg_slope=0.0006,
            fluorescence_bg_curvature=0.00004,
            avg_peak_fwhm_cm=13.0,
            avg_peak_asymmetry=1.28,
            peak_count=5,
            peak_height_std=0.10,
            noise_level=0.025,
            baseline_roughness=0.045,
            high_wavenumber_tail=0.18,
            peak_width_distribution=3.5,
            spectral_entropy=4.6,
            uv_fluorescence_intensity=0.20,
            uv_fluorescence_peak_shift=12.0,
            uv_fluorescence_lifetime_ratio=0.40
        )
        result = self.classifier.predict(features)
        self.assertEqual(result['predicted_process_key'], 'laser_treatment')
        self.assertTrue(result['is_forgery'])

    def test_recall_acid_etching_85_percent(self):
        dataset = ForgeProcessReferenceDataset(n_samples_per_class=100)
        acid_features = dataset.reference_features['acid_etching']

        correct = 0
        for f in acid_features:
            result = self.classifier.predict(f)
            if result['predicted_process_key'] == 'acid_etching':
                correct += 1

        recall = correct / len(acid_features)
        self.assertGreater(recall, 0.85)

    def test_recall_chemical_staining_80_percent(self):
        dataset = ForgeProcessReferenceDataset(n_samples_per_class=100)
        staining_features = dataset.reference_features['chemical_staining']

        correct = 0
        for f in staining_features:
            result = self.classifier.predict(f)
            if result['predicted_process_key'] == 'chemical_staining':
                correct += 1

        recall = correct / len(staining_features)
        self.assertGreater(recall, 0.80)

    def test_multiclass_confusion(self):
        dataset = ForgeProcessReferenceDataset(n_samples_per_class=50)
        confusion = np.zeros((4, 4), dtype=int)
        class_to_idx = {c: i for i, c in enumerate(['authentic', 'acid_etching', 'chemical_staining', 'laser_treatment'])}

        for cls, features_list in dataset.reference_features.items():
            for f in features_list:
                result = self.classifier.predict(f)
                true_idx = class_to_idx[cls]
                pred_idx = class_to_idx[result['predicted_process_key']]
                confusion[true_idx, pred_idx] += 1

        for i in range(4):
            self.assertGreater(confusion[i, i], np.sum(confusion[i, :]) * 0.7)

    def test_confidence_calibration(self):
        features = RamanForgeFeatures(
            fluorescence_bg_level=0.05,
            fluorescence_bg_slope=0.0001,
            fluorescence_bg_curvature=0.0,
            avg_peak_fwhm_cm=6.5,
            avg_peak_asymmetry=1.02,
            peak_count=7,
            peak_height_std=0.15,
            noise_level=0.008,
            baseline_roughness=0.01,
            high_wavenumber_tail=0.02,
            peak_width_distribution=1.5,
            spectral_entropy=2.8,
            uv_fluorescence_intensity=0.02,
            uv_fluorescence_peak_shift=0.0,
            uv_fluorescence_lifetime_ratio=0.15
        )
        result = self.classifier.predict(features)
        self.assertGreater(result['confidence'], 0.5)
        self.assertLessEqual(result['confidence'], 1.0)

        probs = [p['probability'] for p in result['top_predictions']]
        self.assertAlmostEqual(sum(probs), 1.0, places=2)

    def test_boundary_low_signal(self):
        features = RamanForgeFeatures(
            fluorescence_bg_level=0.001,
            fluorescence_bg_slope=0.0,
            fluorescence_bg_curvature=0.0,
            avg_peak_fwhm_cm=5.0,
            avg_peak_asymmetry=1.0,
            peak_count=1,
            peak_height_std=0.001,
            noise_level=0.001,
            baseline_roughness=0.001,
            high_wavenumber_tail=0.001,
            peak_width_distribution=0.1,
            spectral_entropy=0.1,
            uv_fluorescence_intensity=0.001,
            uv_fluorescence_peak_shift=0.0,
            uv_fluorescence_lifetime_ratio=0.001
        )
        result = self.classifier.predict(features)
        self.assertIn('predicted_process_key', result)
        self.assertIn('confidence', result)

    def test_anomaly_missing_features(self):
        features = RamanForgeFeatures()
        result = self.classifier.predict(features)
        self.assertIn('predicted_process_key', result)
        self.assertIn('is_forgery', result)

    def test_anomaly_all_zeros(self):
        features = RamanForgeFeatures(
            fluorescence_bg_level=0.0,
            fluorescence_bg_slope=0.0,
            fluorescence_bg_curvature=0.0,
            avg_peak_fwhm_cm=0.0,
            avg_peak_asymmetry=0.0,
            peak_count=0,
            peak_height_std=0.0,
            noise_level=0.0,
            baseline_roughness=0.0,
            high_wavenumber_tail=0.0,
            peak_width_distribution=0.0,
            spectral_entropy=0.0,
            uv_fluorescence_intensity=0.0,
            uv_fluorescence_peak_shift=0.0,
            uv_fluorescence_lifetime_ratio=0.0
        )
        result = self.classifier.predict(features)
        self.assertIn('predicted_process_key', result)
        self.assertIn('forgery_risk', result)


class TestRamanFeatureExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.classifier = SVMForgeryClassifier()

    def test_normal_raman_extraction(self):
        wavelengths = np.linspace(200, 1800, 500)
        rng = np.random.RandomState(42)
        spectrum_data = np.zeros_like(wavelengths)

        peak_positions = [400, 700, 1000, 1300, 1600]
        for pos in peak_positions:
            spectrum_data += np.exp(-((wavelengths - pos) ** 2) / (2 * 20 ** 2))

        spectrum_data += rng.normal(0, 0.02, len(wavelengths))
        spectrum_data += 0.05 * (wavelengths / 1800)

        raman_spectrum = {
            'wavelengths': wavelengths.tolist(),
            'spectrum_data': spectrum_data.tolist(),
            'artifact_id': 'test_001'
        }

        features = self.classifier.extract_features_from_raman(raman_spectrum)
        self.assertIsInstance(features, RamanForgeFeatures)
        self.assertGreater(features.peak_count, 0)
        self.assertGreater(features.avg_peak_fwhm_cm, 0)
        self.assertGreaterEqual(features.uv_fluorescence_intensity, 0)

    def test_anomaly_short_spectrum(self):
        wavelengths = np.linspace(200, 300, 30)
        spectrum_data = np.random.rand(30)

        raman_spectrum = {
            'wavelengths': wavelengths.tolist(),
            'spectrum_data': spectrum_data.tolist(),
            'artifact_id': 'test_short'
        }

        features = self.classifier.extract_features_from_raman(raman_spectrum)
        self.assertIsInstance(features, RamanForgeFeatures)
        self.assertGreaterEqual(features.peak_count, 3)
        self.assertLessEqual(features.peak_count, 10)
        self.assertGreater(features.uv_fluorescence_intensity, 0)


if __name__ == '__main__':
    unittest.main()
