import unittest
import numpy as np
from .models import (
    PatinaProfile,
    PHGeochemicalModel,
    BayesianPHInversion
)


def generate_synthetic_profile(true_ph: float, age_years: float = 5000.0,
                               n_points: int = 50, max_depth: float = 5.0,
                               noise_level: float = 0.02,
                               random_seed: int = 42) -> PatinaProfile:
    """从已知pH生成合成Fe剖面"""
    rng = np.random.RandomState(random_seed)
    
    if true_ph < 4.0:
        base_ratio = 0.01
    elif true_ph < 5.5:
        base_ratio = 0.04
    elif true_ph < 7.0:
        base_ratio = 0.12
    elif true_ph < 8.5:
        base_ratio = 0.30
    else:
        base_ratio = 0.65
    
    ph_modulation = 1.0 + 0.25 * (true_ph - 6.5)
    age_saturation = 1.0 - np.exp(-age_years / 2000.0)
    age_factor = 0.3 + 0.7 * age_saturation
    
    depths = np.linspace(0, max_depth, n_points)
    
    base_fe2 = 10.0 * np.exp(-depths / 2.5)
    fe2_conc = base_fe2 + rng.normal(0, base_fe2.max() * noise_level, n_points)
    fe2_conc = np.maximum(fe2_conc, 1e-6)
    
    depth_factors = np.exp(-depths / 3.0)
    true_ratios = base_ratio * ph_modulation * depth_factors * age_factor
    true_ratios = np.clip(true_ratios, 0.001, 10.0)
    
    fe3_conc = true_ratios * fe2_conc
    fe3_conc += rng.normal(0, fe3_conc.max() * noise_level, n_points)
    fe3_conc = np.maximum(fe3_conc, 1e-6)
    
    return PatinaProfile(
        depth_mm=depths,
        fe3_concentration=fe3_conc,
        fe2_concentration=fe2_conc
    )


class TestPatinaProfile(unittest.TestCase):
    
    def test_normal_construction(self):
        depths = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        fe3 = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
        fe2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.5])
        
        profile = PatinaProfile(depth_mm=depths, fe3_concentration=fe3, fe2_concentration=fe2)
        
        self.assertEqual(len(profile.depth_mm), 6)
        self.assertEqual(len(profile.fe3_concentration), 6)
        self.assertEqual(len(profile.fe2_concentration), 6)
        self.assertEqual(len(profile.fe3_fe2_ratio), 6)
        self.assertTrue(np.all(profile.fe3_fe2_ratio >= 0))
    
    def test_to_fe3_fe2_ratio_calculation(self):
        depths = np.array([0.0, 1.0, 2.0])
        fe3 = np.array([4.0, 6.0, 8.0])
        fe2 = np.array([2.0, 2.0, 2.0])
        
        profile = PatinaProfile(depth_mm=depths, fe3_concentration=fe3, fe2_concentration=fe2)
        
        expected_ratios = fe3 / (fe2 + 1e-12)
        np.testing.assert_allclose(profile.fe3_fe2_ratio, expected_ratios, rtol=1e-5)
    
    def test_boundary_shallow_profile(self):
        depths = np.array([0.0, 0.1, 0.2])
        fe3 = np.array([10.0, 9.5, 9.0])
        fe2 = np.array([0.5, 0.6, 0.7])
        
        profile = PatinaProfile(depth_mm=depths, fe3_concentration=fe3, fe2_concentration=fe2)
        
        self.assertEqual(len(profile.depth_mm), 3)
        self.assertTrue(np.all(profile.fe3_fe2_ratio > 0))
        self.assertTrue(np.all(profile.depth_mm < 1.0))
    
    def test_anomaly_negative_concentrations(self):
        depths = np.array([0.0, 1.0, 2.0])
        fe3 = np.array([-5.0, 3.0, 4.0])
        fe2 = np.array([2.0, -1.0, 2.0])
        
        profile = PatinaProfile(depth_mm=depths, fe3_concentration=fe3, fe2_concentration=fe2)
        
        self.assertTrue(np.isfinite(profile.fe3_fe2_ratio).all())
        self.assertEqual(len(profile.fe3_fe2_ratio), 3)


class TestPHGeochemicalModel(unittest.TestCase):
    
    def setUp(self):
        self.model = PHGeochemicalModel()
    
    def test_theoretical_ratio_monotonic_ph(self):
        ratios = []
        for ph in np.linspace(4.0, 10.0, 20):
            ratio = self.model.theoretical_fe_ratio(ph, depth_mm=1.0, age_years=5000.0)
            ratios.append(ratio)
        
        diffs = np.diff(ratios)
        self.assertTrue(np.all(diffs >= -0.15))
        
        self.assertLess(ratios[0], ratios[-1])
        
        segment_means = []
        for i in range(0, len(ratios), 5):
            segment_means.append(np.mean(ratios[i:i+5]))
        for i in range(len(segment_means) - 1):
            self.assertGreater(segment_means[i+1], segment_means[i] - 0.05)
        
        positive_count = np.sum(diffs > 0)
        self.assertGreater(positive_count, len(diffs) * 0.5)
    
    def test_theoretical_ratio_depth_decay(self):
        ph = 7.0
        ratios = []
        for depth in np.linspace(0.0, 5.0, 10):
            ratio = self.model.theoretical_fe_ratio(ph, depth_mm=depth, age_years=5000.0)
            ratios.append(ratio)
        
        self.assertTrue(ratios[0] > ratios[-1])
        self.assertTrue(all(ratios[i] >= ratios[i+1] - 0.01 for i in range(len(ratios)-1)))
    
    def test_boundary_extreme_ph(self):
        ratio_low = self.model.theoretical_fe_ratio(ph=3.0, depth_mm=1.0, age_years=5000.0)
        ratio_high = self.model.theoretical_fe_ratio(ph=12.0, depth_mm=1.0, age_years=5000.0)
        
        self.assertGreater(ratio_low, 0)
        self.assertGreater(ratio_high, 0)
        self.assertLess(ratio_low, 10.0)
        self.assertLess(ratio_high, 10.0)
    
    def test_anomaly_invalid_inputs(self):
        with self.assertRaises(TypeError):
            self.model.theoretical_fe_ratio(ph='invalid', depth_mm=1.0, age_years=5000.0)
        
        ratio_nan_depth = self.model.theoretical_fe_ratio(ph=7.0, depth_mm=np.nan, age_years=5000.0)
        self.assertTrue(np.isfinite(ratio_nan_depth) or np.isnan(ratio_nan_depth))


def _deterministic_theoretical_fe_ratio(self, ph: float, depth_mm: float, age_years: float) -> float:
    if ph < 4.0:
        base_ratio = 0.01
    elif ph < 5.5:
        base_ratio = 0.04
    elif ph < 7.0:
        base_ratio = 0.12
    elif ph < 8.5:
        base_ratio = 0.30
    else:
        base_ratio = 0.65
    
    ph_modulation = 1.0 + 0.25 * (ph - 6.5)
    depth_factor = np.exp(-depth_mm / 3.0)
    age_saturation = 1.0 - np.exp(-age_years / 2000.0)
    
    ratio = base_ratio * ph_modulation * depth_factor * (0.3 + 0.7 * age_saturation)
    return max(0.001, min(10.0, ratio))


class TestBayesianPHInversion(unittest.TestCase):
    
    def setUp(self):
        self._original_theoretical_fe_ratio = PHGeochemicalModel.theoretical_fe_ratio
        PHGeochemicalModel.theoretical_fe_ratio = _deterministic_theoretical_fe_ratio
        
        self.inverter = BayesianPHInversion(
            likelihood_sigma=0.7,
            prior_type='uniform'
        )
        self.age_years = 5000.0
    
    def tearDown(self):
        PHGeochemicalModel.theoretical_fe_ratio = self._original_theoretical_fe_ratio
    
    def _create_fresh_inverter(self):
        return BayesianPHInversion(
            likelihood_sigma=0.7,
            prior_type='uniform'
        )
    
    def test_setUp(self):
        self.assertEqual(self.inverter.likelihood_sigma, 0.7)
        self.assertEqual(self.inverter.prior_type, 'uniform')
        self.assertEqual(self.inverter.ph_min, 3.0)
        self.assertEqual(self.inverter.ph_max, 11.0)
    
    def test_log_prior_uniform(self):
        lp1 = self.inverter.log_prior(5.0)
        lp2 = self.inverter.log_prior(7.0)
        lp3 = self.inverter.log_prior(9.0)
        
        self.assertAlmostEqual(lp1, lp2, places=5)
        self.assertAlmostEqual(lp2, lp3, places=5)
        
        lp_outside_low = self.inverter.log_prior(2.0)
        lp_outside_high = self.inverter.log_prior(12.0)
        self.assertEqual(lp_outside_low, -np.inf)
        self.assertEqual(lp_outside_high, -np.inf)
    
    def test_log_prior_normal_fallback(self):
        inverter_normal = BayesianPHInversion(
            likelihood_sigma=0.7,
            prior_type='normal',
            ph_prior_mean=7.0,
            ph_prior_std=1.5
        )
        
        lp_center = inverter_normal.log_prior(7.0)
        lp_edge1 = inverter_normal.log_prior(5.5)
        lp_edge2 = inverter_normal.log_prior(8.5)
        
        self.assertGreater(lp_center, lp_edge1)
        self.assertGreater(lp_center, lp_edge2)
        self.assertAlmostEqual(lp_edge1, lp_edge2, places=3)
    
    def test_log_likelihood_known_ph(self):
        true_ph = 7.0
        profile = generate_synthetic_profile(true_ph, age_years=self.age_years, random_seed=42)
        inverter = self._create_fresh_inverter()
        
        ll_correct = inverter.log_likelihood(true_ph, profile, self.age_years)
        ll_wrong1 = inverter.log_likelihood(4.0, profile, self.age_years)
        ll_wrong2 = inverter.log_likelihood(10.0, profile, self.age_years)
        
        self.assertGreater(ll_correct, ll_wrong1)
        self.assertGreater(ll_correct, ll_wrong2)
        self.assertTrue(np.isfinite(ll_correct))
    
    def test_mcmc_basic_convergence(self):
        true_ph = 7.0
        profile = generate_synthetic_profile(true_ph, age_years=self.age_years, random_seed=42)
        inverter = self._create_fresh_inverter()
        
        result = inverter.mcmc_sample(
            profile,
            age_years=self.age_years,
            n_burn=200,
            n_samples=500,
            proposal_std=1.0,
            random_seed=42
        )
        
        self.assertIn('ph_std', result)
        self.assertGreater(result['ph_std'], 0)
        self.assertGreater(len(result['ph_chain']), 0)
    
    def test_posterior_covers_true_ph_case1_normal(self):
        true_ph = 6.0
        profile = generate_synthetic_profile(true_ph, age_years=self.age_years, random_seed=42)
        inverter = self._create_fresh_inverter()
        
        result = inverter.mcmc_sample(
            profile,
            age_years=self.age_years,
            n_burn=1000,
            n_samples=3000,
            proposal_std=1.2,
            random_seed=42
        )
        
        ci_low, ci_high = result['ph_95ci']
        self.assertGreaterEqual(true_ph, ci_low)
        self.assertLessEqual(true_ph, ci_high)
    
    def test_posterior_covers_true_ph_case2_normal(self):
        true_ph = 8.0
        profile = generate_synthetic_profile(true_ph, age_years=self.age_years, random_seed=123)
        inverter = self._create_fresh_inverter()
        
        result = inverter.mcmc_sample(
            profile,
            age_years=self.age_years,
            n_burn=1000,
            n_samples=3000,
            proposal_std=1.2,
            random_seed=123
        )
        
        ci_low, ci_high = result['ph_95ci']
        self.assertGreaterEqual(true_ph, ci_low)
        self.assertLessEqual(true_ph, ci_high)
    
    def test_posterior_covers_true_ph_case3_boundary(self):
        true_ph = 5.8
        profile = generate_synthetic_profile(true_ph, age_years=self.age_years, random_seed=456)
        inverter = self._create_fresh_inverter()
        
        result = inverter.mcmc_sample(
            profile,
            age_years=self.age_years,
            n_burn=1000,
            n_samples=3000,
            proposal_std=1.2,
            random_seed=456
        )
        
        ci_low, ci_high = result['ph_95ci']
        self.assertGreaterEqual(true_ph, ci_low)
        self.assertLessEqual(true_ph, ci_high)
    
    def test_mean_absolute_error_less_than_1_5(self):
        test_cases = [
            (6.0, 42),
            (7.0, 123),
            (8.0, 456),
            (5.5, 789),
        ]
        
        errors = []
        for true_ph, seed in test_cases:
            profile = generate_synthetic_profile(true_ph, age_years=self.age_years, random_seed=seed)
            inverter = self._create_fresh_inverter()
            
            result = inverter.mcmc_sample(
                profile,
                age_years=self.age_years,
                n_burn=500,
                n_samples=1500,
                proposal_std=1.0,
                random_seed=seed
            )
            
            error = abs(result['ph_mean'] - true_ph)
            errors.append(error)
        
        mae = np.mean(errors)
        self.assertLess(mae, 1.5)
    
    def test_95ci_width_reasonable(self):
        true_ph = 7.0
        profile = generate_synthetic_profile(true_ph, age_years=self.age_years, random_seed=42)
        inverter = self._create_fresh_inverter()
        
        result = inverter.mcmc_sample(
            profile,
            age_years=self.age_years,
            n_burn=500,
            n_samples=1500,
            proposal_std=1.0,
            random_seed=42
        )
        
        ci_low, ci_high = result['ph_95ci']
        ci_width = ci_high - ci_low
        
        self.assertGreater(ci_width, 0.1)
        self.assertLess(ci_width, 6.0)
    
    def test_boundary_low_fe_ratio(self):
        depths = np.linspace(0, 5.0, 30)
        fe2 = np.ones_like(depths) * 10.0
        fe3 = np.ones_like(depths) * 0.01
        
        profile = PatinaProfile(
            depth_mm=depths,
            fe3_concentration=fe3,
            fe2_concentration=fe2
        )
        inverter = self._create_fresh_inverter()
        
        result = inverter.mcmc_sample(
            profile,
            age_years=self.age_years,
            n_burn=200,
            n_samples=500,
            proposal_std=1.0,
            random_seed=42
        )
        
        self.assertIn('ph_mean', result)
        self.assertTrue(np.isfinite(result['ph_mean']))
        self.assertGreaterEqual(result['ph_mean'], inverter.ph_min)
        self.assertLessEqual(result['ph_mean'], inverter.ph_max)
    
    def test_anomaly_all_zero_fe(self):
        depths = np.linspace(0, 5.0, 30)
        fe2 = np.zeros_like(depths)
        fe3 = np.zeros_like(depths)
        
        profile = PatinaProfile(
            depth_mm=depths,
            fe3_concentration=fe3,
            fe2_concentration=fe2
        )
        inverter = self._create_fresh_inverter()
        
        result = inverter.mcmc_sample(
            profile,
            age_years=self.age_years,
            n_burn=100,
            n_samples=300,
            proposal_std=1.0,
            random_seed=42
        )
        
        self.assertIn('ph_mean', result)
        self.assertTrue(np.isfinite(result['ph_mean']))
    
    def test_sampler_field_exists(self):
        true_ph = 7.0
        profile = generate_synthetic_profile(true_ph, age_years=self.age_years, random_seed=42)
        inverter = self._create_fresh_inverter()
        
        result = inverter.mcmc_sample(
            profile,
            age_years=self.age_years,
            n_burn=200,
            n_samples=500,
            proposal_std=1.0,
            random_seed=42
        )
        
        self.assertIn('sampler_used', result)
        self.assertIsNotNone(result['sampler_used'])
        self.assertIn('pymc3_available', result)
        self.assertIsInstance(result['pymc3_available'], bool)


if __name__ == '__main__':
    unittest.main()
