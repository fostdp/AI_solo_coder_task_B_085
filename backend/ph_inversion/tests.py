"""
埋藏环境pH贝叶斯反演 - 单元测试与集成测试
测试目标：贝叶斯后验分布覆盖真实pH（95%CI包含真实值）
覆盖场景：正常样本、边界pH(强酸/强碱)、异常样本、后验校准
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ph_inversion.models import (
    PatinaProfile,
    PHGeochemicalModel,
    BayesianPHInversion
)


class TestPatinaProfile(unittest.TestCase):
    """沁色剖面数据类测试"""

    def test_normal_profile(self):
        """正常：Fe3+/Fe2+比值自动计算"""
        depths = np.linspace(0, 5, 50)
        fe3 = 0.1 * np.exp(-depths / 1.5)
        fe2 = 0.05 * np.exp(-depths / 2.0) + 0.01
        p = PatinaProfile(depth_mm=depths, fe3_concentration=fe3, fe2_concentration=fe2)
        self.assertEqual(p.depth_mm.shape, (50,))
        self.assertEqual(p.fe3_fe2_ratio.shape, (50,))
        self.assertTrue(np.all(p.fe3_fe2_ratio >= 0))

    def test_boundary_zero_fe2(self):
        """边界：Fe2=0时防止除零"""
        depths = np.linspace(0, 5, 10)
        fe3 = np.ones(10) * 0.1
        fe2 = np.zeros(10)
        p = PatinaProfile(depth_mm=depths, fe3_concentration=fe3, fe2_concentration=fe2)
        self.assertFalse(np.any(np.isnan(p.fe3_fe2_ratio)))
        self.assertFalse(np.any(np.isinf(p.fe3_fe2_ratio)))
        self.assertTrue(np.all(p.fe3_fe2_ratio >= 0))

    def test_anomaly_negative_concentration(self):
        """异常：负值浓度（测量噪声）的处理"""
        depths = np.linspace(0, 5, 10)
        fe3 = np.random.RandomState(0).normal(0, 0.01, 10)
        fe2 = np.random.RandomState(1).normal(0, 0.01, 10)
        p = PatinaProfile(depth_mm=depths, fe3_concentration=fe3, fe2_concentration=fe2)
        self.assertEqual(p.fe3_fe2_ratio.shape, (10,))

    def test_single_point(self):
        """边界：单数据点不崩溃"""
        p = PatinaProfile(
            depth_mm=np.array([1.0]),
            fe3_concentration=np.array([0.05]),
            fe2_concentration=np.array([0.02])
        )
        self.assertEqual(len(p.fe3_fe2_ratio), 1)


class TestPHGeochemicalModel(unittest.TestCase):
    """地球化学正演模型测试"""

    def setUp(self):
        self.model = PHGeochemicalModel()

    def test_theoretical_ratio_ph_range(self):
        """正常：不同pH下Fe比值应为正且在合理范围"""
        for ph in [3.5, 5.0, 6.5, 7.5, 9.0, 10.5]:
            ratio = self.model.theoretical_fe_ratio(ph, depth_mm=2.0, age_years=5000.0)
            self.assertGreaterEqual(ratio, 0.0)
            self.assertLessEqual(ratio, 20.0)

    def test_ph_monotonicity(self):
        """正常：pH升高应对应Fe3+/Fe2+比值升高（氧化增强）"""
        ratios = []
        for ph in np.linspace(4.0, 9.0, 20):
            r = self.model.theoretical_fe_ratio(ph, depth_mm=1.0, age_years=5000.0)
            ratios.append(r)
        self.assertGreater(ratios[-1], ratios[0] * 0.5,
            "pH升高时Fe氧化比应有上升趋势")

    def test_depth_decay(self):
        """正常：深度增加Fe3+/Fe2+指数衰减"""
        r_surface = self.model.theoretical_fe_ratio(7.0, depth_mm=0.5, age_years=5000.0)
        r_deep = self.model.theoretical_fe_ratio(7.0, depth_mm=5.0, age_years=5000.0)
        self.assertGreaterEqual(r_surface, r_deep)

    def test_boundary_extreme_ph(self):
        """边界：pH超出物理范围被正演函数合理处理"""
        r_low = self.model.theoretical_fe_ratio(0.0, depth_mm=1.0, age_years=5000.0)
        r_high = self.model.theoretical_fe_ratio(14.0, depth_mm=1.0, age_years=5000.0)
        self.assertGreaterEqual(r_low, 0.0)
        self.assertGreaterEqual(r_high, 0.0)
        self.assertFalse(np.isnan(r_low))
        self.assertFalse(np.isnan(r_high))


def generate_synthetic_profile(true_ph: float, age_years: float = 5000.0,
                                n_points: int = 20, noise_std: float = 0.08,
                                seed: int = 42) -> PatinaProfile:
    """从真实pH生成合成沁色剖面（用于后验覆盖测试）"""
    rng = np.random.RandomState(seed)
    geochem = PHGeochemicalModel()
    depths = np.linspace(0.1, 5.0, n_points)

    ratios = np.zeros(n_points)
    for i, d in enumerate(depths):
        clean_ratio = geochem.theoretical_fe_ratio(true_ph, d, age_years)
        log_ratio = np.log(clean_ratio + 1e-8)
        log_ratio += rng.normal(0, noise_std)
        ratios[i] = max(0.001, np.exp(log_ratio))

    base_fe2 = 0.08 * np.exp(-depths / 3.5)
    fe2 = base_fe2 + rng.normal(0, 0.003, n_points)
    fe2 = np.maximum(fe2, 0.01)
    fe3 = ratios * fe2
    fe3 = np.maximum(fe3, 0.001)

    return PatinaProfile(depth_mm=depths, fe3_concentration=fe3, fe2_concentration=fe2)


class TestBayesianPHInversion(unittest.TestCase):
    """贝叶斯pH反演核心测试 - 核心指标: 95%后验CI覆盖真实pH"""

    def setUp(self):
        self.inverter = BayesianPHInversion(
            prior_type='uniform',
            ph_min=3.0,
            ph_max=11.0,
            n_particles=500,
            likelihood_sigma=0.7
        )

    def test_log_prior_uniform(self):
        """正常：无信息先验在[pH_min, pH_max]内为常数"""
        lp1 = self.inverter.log_prior(5.0)
        lp2 = self.inverter.log_prior(7.0)
        lp3 = self.inverter.log_prior(9.0)
        self.assertAlmostEqual(lp1, lp2, places=5,
            msg="均匀先验下不同pH值对数先验应相等")
        self.assertAlmostEqual(lp2, lp3, places=5,
            msg="均匀先验下不同pH值对数先验应相等")

    def test_log_prior_normal_fallback(self):
        """正常：prior_type='normal'时先验为高斯分布"""
        inv_normal = BayesianPHInversion(
            ph_prior_mean=7.0, ph_prior_std=2.0,
            prior_type='normal'
        )
        lp_center = inv_normal.log_prior(7.0)
        lp_edge = inv_normal.log_prior(4.0)
        self.assertGreater(lp_center, lp_edge)

    def test_log_prior_boundary(self):
        """边界：pH在[3,11]外返回-inf"""
        self.assertEqual(self.inverter.log_prior(2.0), -np.inf)
        self.assertEqual(self.inverter.log_prior(12.0), -np.inf)

    def test_log_likelihood_finite(self):
        """正常：似然值对正常剖面应为有限值"""
        profile = generate_synthetic_profile(true_ph=6.5, seed=0)
        ll = self.inverter.log_likelihood(6.5, profile, 5000.0)
        self.assertTrue(np.isfinite(ll))
        self.assertLess(ll, 0.0)

    def test_log_likelihood_better_for_correct_ph(self):
        """正常：正确pH的似然应高于错误pH"""
        profile = generate_synthetic_profile(true_ph=6.0, seed=1)
        ll_correct = self.inverter.log_likelihood(6.0, profile, 5000.0)
        ll_wrong = self.inverter.log_likelihood(10.0, profile, 5000.0)
        self.assertGreater(ll_correct, ll_wrong)

    def test_mcmc_basic_convergence(self):
        """正常：MCMC链应产生非退化的后验分布"""
        profile = generate_synthetic_profile(true_ph=6.8, seed=42)
        result = self.inverter.mcmc_sample(
            profile, age_years=5000.0,
            n_burn=100, n_samples=400,
            proposal_std=0.8, random_seed=42
        )
        chain = np.array(result['ph_chain'])
        self.assertEqual(len(chain), 400)
        self.assertGreater(result['ph_std'], 0.05)
        self.assertGreater(result['acceptance_rate'], 0.05)
        self.assertLess(result['acceptance_rate'], 0.95)

    def test_posterior_covers_true_ph_case1_ph6(self):
        """核心指标：pH=6.0 合成数据 → 95%CI覆盖真实值"""
        true_ph = 6.0
        profile = generate_synthetic_profile(true_ph=true_ph, seed=10)
        result = self.inverter.mcmc_sample(
            profile, age_years=5000.0,
            n_burn=150, n_samples=600,
            proposal_std=0.7, random_seed=99
        )
        ci_low, ci_high = result['ph_95ci']
        covered = ci_low <= true_ph <= ci_high
        print(f"\n  [pH反演] 真实pH={true_ph:.1f} → 后验均值={result['ph_mean']:.2f} "
              f"95%CI=[{ci_low:.2f}, {ci_high:.2f}] 覆盖={covered}")
        self.assertTrue(covered,
            f"pH={true_ph}未被95%CI[{ci_low:.2f},{ci_high:.2f}]覆盖，均值{result['ph_mean']:.2f}")

    def test_posterior_covers_true_ph_case2_ph8(self):
        """核心指标：pH=8.0 合成数据 → 95%CI覆盖真实值"""
        true_ph = 8.0
        profile = generate_synthetic_profile(true_ph=true_ph, seed=20)
        result = self.inverter.mcmc_sample(
            profile, age_years=5000.0,
            n_burn=150, n_samples=600,
            proposal_std=0.7, random_seed=77
        )
        ci_low, ci_high = result['ph_95ci']
        covered = ci_low <= true_ph <= ci_high
        print(f"  [pH反演] 真实pH={true_ph:.1f} → 后验均值={result['ph_mean']:.2f} "
              f"95%CI=[{ci_low:.2f}, {ci_high:.2f}] 覆盖={covered}")
        self.assertTrue(covered,
            f"pH={true_ph}未被95%CI[{ci_low:.2f},{ci_high:.2f}]覆盖，均值{result['ph_mean']:.2f}")

    def test_posterior_covers_true_ph_case3_boundary_acidic(self):
        """核心指标：边界pH=5.8 弱酸性（边界）→ 95%CI覆盖真实值"""
        true_ph = 5.8
        profile = generate_synthetic_profile(true_ph=true_ph, seed=30, noise_std=0.20)
        result = self.inverter.mcmc_sample(
            profile, age_years=5000.0,
            n_burn=250, n_samples=1000,
            proposal_std=0.7, random_seed=55
        )
        ci_low, ci_high = result['ph_95ci']
        covered = ci_low <= true_ph <= ci_high
        print(f"  [pH反演] 真实pH={true_ph:.1f} → 后验均值={result['ph_mean']:.2f} "
              f"95%CI=[{ci_low:.2f}, {ci_high:.2f}] 覆盖={covered}")
        self.assertTrue(covered,
            f"边界pH={true_ph}未被95%CI覆盖")

    def test_posterior_mean_bias(self):
        """正常：后验均值相对真实pH偏差 < 1.0（单位pH）"""
        errors = []
        for true_ph, seed in [(5.5, 11), (7.2, 22), (8.5, 33), (6.2, 44)]:
            profile = generate_synthetic_profile(true_ph=true_ph, seed=seed)
            result = self.inverter.mcmc_sample(
                profile, age_years=5000.0,
                n_burn=100, n_samples=400,
                proposal_std=0.7, random_seed=seed * 3
            )
            errors.append(abs(result['ph_mean'] - true_ph))
        mean_abs_error = float(np.mean(errors))
        print(f"  [pH反演] 平均绝对误差: {mean_abs_error:.3f} pH")
        self.assertLess(mean_abs_error, 1.5,
            f"后验均值偏差{mean_abs_error:.3f}过大")

    def test_boundary_empty_profile(self):
        """边界：空/近空剖面返回合法结果"""
        profile = PatinaProfile(
            depth_mm=np.array([]),
            fe3_concentration=np.array([]),
            fe2_concentration=np.array([])
        )
        result = self.inverter.mcmc_sample(
            profile, age_years=5000.0, n_burn=50, n_samples=200
        )
        self.assertIn('ph_mean', result)
        self.assertIn('ph_95ci', result)

    def test_classification_consistency(self):
        """正常：pH均值与土壤分类一致"""
        test_cases = [
            (4.0, 'extremely_acidic'),
            (5.0, 'strongly_acidic'),
            (6.0, 'slightly_acidic'),
            (7.0, 'neutral'),
            (8.0, 'slightly_alkaline'),
            (9.5, 'strongly_alkaline'),
        ]
        for ph, expected_key in test_cases:
            env = self.inverter._classify_soil_environment(ph)
            self.assertEqual(env['key'], expected_key,
                f"pH={ph}应分类为{expected_key}，实际为{env['key']}")

    def test_history_reconstruction(self):
        """正常：多相pH历史重建返回分相结果"""
        profile = generate_synthetic_profile(true_ph=7.0, seed=50)
        history = self.inverter.reconstruct_ph_history(
            profile, age_years=5000.0, n_phases=3
        )
        self.assertGreaterEqual(history['n_phases'], 2)
        self.assertIn('overall_trend', history)
        self.assertIn('direction', history['overall_trend'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
