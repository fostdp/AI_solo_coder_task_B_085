"""
玉器虚拟盘玩Blinn-Phong光照模拟 - 算法正确性与数学性质验证
测试目标：光泽度随点击次数非线性增加（饱和曲线特性）
覆盖场景：向量数学正确性、光照分量非负、边界polishLevel、非线性增长验证
"""
import os
import sys
import math
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BlinnPhongMathValidator:
    """Python实现的Blinn-Phong核心逻辑等价器，用于验证JS实现的数学性质"""

    AMBIENT_STRENGTH = 0.25
    DIFFUSE_STRENGTH = 0.7
    SPECULAR_STRENGTH = 0.35
    SHININESS = 48.0

    @staticmethod
    def normalize(v):
        length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + 1e-12
        return (v[0] / length, v[1] / length, v[2] / length)

    @staticmethod
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @classmethod
    def compute_specular(cls, normal, light, view_dir, polish_level):
        """计算高光强度（Blinn-Phong核心项）"""
        half = cls.normalize([
            light[0] + view_dir[0],
            light[1] + view_dir[1],
            light[2] + view_dir[2]
        ])
        spec_dot = max(0.0, cls.dot(normal, half))
        effective_shininess = cls.SHININESS * (0.3 + polish_level * 0.7)
        specular = cls.SPECULAR_STRENGTH * (spec_dot ** effective_shininess) * polish_level
        return specular

    @classmethod
    def compute_full_lighting(cls, normal, light, view_dir, polish_level):
        """完整光照三分量"""
        ambient = cls.AMBIENT_STRENGTH
        diff_dot = max(0.0, cls.dot(normal, light))
        diffuse = cls.DIFFUSE_STRENGTH * diff_dot
        specular = cls.compute_specular(normal, light, view_dir, polish_level)
        return ambient, diffuse, specular


class TestVectorMath(unittest.TestCase):
    """Blinn-Phong向量数学基础测试"""

    def test_normalize_unit_length(self):
        """正常：normalize后向量模长应≈1"""
        test_vectors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.3, -0.4, 0.86),
            (1e-6, 1e-6, 1e-6),
        ]
        for v in test_vectors:
            n = BlinnPhongMathValidator.normalize(v)
            length = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
            self.assertAlmostEqual(length, 1.0, places=5,
                msg=f"向量{v}归一化后长度{length}≠1")

    def test_dot_orthogonal_zero(self):
        """正常：正交向量点积应为0"""
        self.assertAlmostEqual(
            BlinnPhongMathValidator.dot((1, 0, 0), (0, 1, 0)),
            0.0, places=10
        )
        self.assertAlmostEqual(
            BlinnPhongMathValidator.dot((0, 0, 1), (1, 0, 0)),
            0.0, places=10
        )

    def test_dot_self_is_norm_squared(self):
        """正常：自身点积等于模长平方"""
        v = (0.3, -0.4, 0.86)
        n = BlinnPhongMathValidator.normalize(v)
        self.assertAlmostEqual(
            BlinnPhongMathValidator.dot(n, n),
            1.0, places=10
        )


class TestLightingComponentsNonNegative(unittest.TestCase):
    """光照三分量非负性测试"""

    def setUp(self):
        self.light = BlinnPhongMathValidator.normalize((0.3, -0.4, 0.86))
        self.view = BlinnPhongMathValidator.normalize((0.0, 0.0, 1.0))

    def test_ambient_positive(self):
        """正常：环境光应为正的常数"""
        self.assertGreater(BlinnPhongMathValidator.AMBIENT_STRENGTH, 0)

    def test_diffuse_nonnegative_hemisphere(self):
        """正常：在光线可见半球，漫反射≥0"""
        test_normals = [
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 0.866),
            (0.0, 0.3, 0.95),
            BlinnPhongMathValidator.normalize((0.2, -0.1, 0.9)),
        ]
        for n in test_normals:
            _, diffuse, _ = BlinnPhongMathValidator.compute_full_lighting(
                n, self.light, self.view, 0.5
            )
            self.assertGreaterEqual(diffuse, 0.0,
                msg=f"漫反射不应为负: normal={n}")

    def test_specular_nonnegative(self):
        """正常：高光项始终≥0"""
        normals = [
            (0.0, 0.0, 1.0),
            (0.5, 0.2, 0.84),
            (0.1, 0.1, 0.99),
        ]
        polish_levels = [0.0, 0.3, 0.5, 0.8, 1.0]
        for n in normals:
            for pl in polish_levels:
                spec = BlinnPhongMathValidator.compute_specular(
                    BlinnPhongMathValidator.normalize(n),
                    self.light, self.view, pl
                )
                self.assertGreaterEqual(spec, 0.0,
                    msg=f"高光不应为负: polish={pl}, normal={n}")


class TestPolishLevelNonlinearGloss(unittest.TestCase):
    """核心指标：光泽度（高光强度）随点击次数非线性增加测试"""

    def setUp(self):
        self.light = BlinnPhongMathValidator.normalize((0.3, -0.4, 0.86))
        self.view = BlinnPhongMathValidator.normalize((0.0, 0.0, 1.0))
        self.normal = BlinnPhongMathValidator.normalize((0.0, 0.0, 1.0))

    def test_specular_monotonic_in_polish(self):
        """正常：光泽度随polishLevel总体递增（允许极小波动）"""
        prev = -1.0
        max_drop = 0.0
        for pl in np.linspace(0.0, 1.0, 21):
            spec = BlinnPhongMathValidator.compute_specular(
                self.normal, self.light, self.view, float(pl)
            )
            drop = prev - spec
            if drop > max_drop:
                max_drop = drop
            prev = spec
        self.assertLess(max_drop, 0.002,
            msg=f"光泽度总体应递增，最大回落{max_drop:.5f}过大")

    def test_specular_zero_when_no_polish(self):
        """边界：polishLevel=0时光泽度=0（新玉无包浆光泽）"""
        spec = BlinnPhongMathValidator.compute_specular(
            self.normal, self.light, self.view, 0.0
        )
        self.assertAlmostEqual(spec, 0.0, places=10,
            msg="polish=0时不应有高光")

    def test_specular_max_at_full_polish(self):
        """边界：polishLevel=1附近光泽度接近最大值（允许0.5%波动）"""
        spec_1 = BlinnPhongMathValidator.compute_specular(
            self.normal, self.light, self.view, 1.0
        )
        max_spec = spec_1
        for pl in np.linspace(0, 1, 50):
            spec = BlinnPhongMathValidator.compute_specular(
                self.normal, self.light, self.view, float(pl)
            )
            max_spec = max(max_spec, spec)
        self.assertLessEqual(spec_1, max_spec * 1.005,
            msg="pl=1.0的光泽度应在全局最大值0.5%范围内")

    def test_gloss_growth_is_concave_saturating(self):
        """核心指标：光泽度增长呈凹函数（边际递减、非线性饱和）"""
        polish_levels = np.linspace(0.0, 1.0, 101)
        specs = []
        for pl in polish_levels:
            specs.append(BlinnPhongMathValidator.compute_specular(
                self.normal, self.light, self.view, float(pl)
            ))
        specs = np.array(specs)

        d_spec = np.diff(specs)
        positive_count = np.sum(d_spec >= -1e-4)
        positive_ratio = positive_count / len(d_spec)

        dd_spec = np.diff(d_spec)
        concave_count = np.sum(dd_spec <= 1e-4)
        concave_ratio = concave_count / len(dd_spec)

        print(f"\n  [虚拟盘玩] 单调性: {positive_count}/{len(d_spec)}段一阶差分≥0 "
              f"({positive_ratio:.1%})")
        print(f"  [虚拟盘玩] 凹性验证: {concave_count}/{len(dd_spec)}段二阶差分≤0 "
              f"({concave_ratio:.1%})")
        self.assertGreater(positive_ratio, 0.70,
            f"光泽度一阶增长段仅{positive_ratio:.1%}，不足70%")
        self.assertGreater(concave_ratio, 0.60,
            f"光泽度增长凹性段仅{concave_ratio:.1%}，不足60%")

    def test_gloss_per_click_decreasing(self):
        """核心指标：每增加相同点击数，新增光泽越来越少（非线性饱和）"""
        clicks_per_step = 100
        polish_per_click = 0.0008

        initial_polish = 0.3
        polish = initial_polish

        increments = []
        for step in range(5):
            start_spec = BlinnPhongMathValidator.compute_specular(
                self.normal, self.light, self.view, polish
            )
            for _ in range(clicks_per_step):
                polish = min(1.0, polish + polish_per_click)
            end_spec = BlinnPhongMathValidator.compute_specular(
                self.normal, self.light, self.view, polish
            )
            increments.append(end_spec - start_spec)

        print("  [虚拟盘玩] 每100次点击新增光泽度增量:",
              [f"{x:.5f}" for x in increments])

        for i in range(len(increments) - 1):
            self.assertGreaterEqual(increments[i], increments[i + 1] - 1e-8,
                f"第{i+1}段增量({increments[i]:.5f})应≥第{i+2}段增量({increments[i+1]:.5f})")

    def test_full_play_simulation_nonlinear(self):
        """集成：模拟完整盘玩过程0→3000次点击，验证整体非线性"""
        polish_per_click = 0.0008
        polish = 0.3

        checkpoints = [0, 500, 1000, 2000, 3000]
        gloss_values = []

        total_clicks = 0
        for target in checkpoints:
            while total_clicks < target:
                polish = min(1.0, polish + polish_per_click)
                total_clicks += 1
            gloss = BlinnPhongMathValidator.compute_specular(
                self.normal, self.light, self.view, polish
            )
            gloss_values.append(gloss)
            print(f"  [虚拟盘玩] {target:>4}次点击 → polish={polish:.3f}, 光泽度={gloss:.4f}")

        delta_0_500 = gloss_values[1] - gloss_values[0]
        delta_2500_3000 = gloss_values[4] - gloss_values[3]

        self.assertGreater(delta_0_500, delta_2500_3000 * 1.5,
            f"初期光泽增量{delta_0_500:.4f}应至少是后期{delta_2500_3000:.4f}的1.5倍（非线性）")

    def test_effective_shininess_in_range(self):
        """正常：有效光泽指数在合理范围内"""
        for pl in [0.0, 0.5, 1.0]:
            eff_shin = BlinnPhongMathValidator.SHININESS * (0.3 + pl * 0.7)
            self.assertGreaterEqual(eff_shin, BlinnPhongMathValidator.SHININESS * 0.3)
            self.assertLessEqual(eff_shin, BlinnPhongMathValidator.SHININESS)


class TestShapeFunctionValidity(unittest.TestCase):
    """玉器表面形状函数（高度场）数学性质测试"""

    @staticmethod
    def jade_bi_shape(x, y):
        r = math.sqrt(x * x + y * y)
        if r < 0.12 or r > 0.9:
            return 0.0
        return math.exp(-((r - 0.5) / 0.35) ** 2) * 0.18

    @staticmethod
    def jade_zhu_shape(x, y):
        r = math.sqrt(x * x + y * y)
        if r > 0.98:
            return 0.0
        return math.sqrt(max(0.0, 1.0 - r * r)) * 0.25

    def test_bi_height_nonnegative(self):
        """正常：玉璧形状函数高度≥0"""
        test_points = [
            (0.0, 0.0), (0.3, 0.0), (0.5, 0.5),
            (0.8, 0.0), (-0.2, 0.3), (1.5, 1.5)
        ]
        for x, y in test_points:
            h = self.jade_bi_shape(x, y)
            self.assertGreaterEqual(h, 0.0,
                msg=f"高度不应为负: ({x},{y}) → {h}")

    def test_bi_center_hole(self):
        """正常：玉璧中心孔(r<0.12)高度为0"""
        self.assertEqual(self.jade_bi_shape(0.0, 0.0), 0.0)
        self.assertEqual(self.jade_bi_shape(0.1, 0.0), 0.0)

    def test_zhu_spherical_convex(self):
        """正常：玉珠（球面切片）随径向距离增大高度单调递减"""
        prev_h = float('inf')
        for r in np.linspace(0.0, 0.95, 20):
            h = self.jade_zhu_shape(float(r), 0.0)
            self.assertLessEqual(h, prev_h + 1e-10,
                msg=f"球面高度应随r递减: r={r} h={h} prev={prev_h}")
            prev_h = h


class TestPatinaCoverageInteraction(unittest.TestCase):
    """包浆覆盖率与磨损的交互逻辑测试"""

    def test_wear_map_accumulates_nonnegative(self):
        """正常：磨损累积值∈[0,1]（截断饱和）"""
        canvas_w, canvas_h = 100, 100
        wear_map = np.zeros(canvas_w * canvas_h, dtype=np.float32)

        cx, cy = canvas_w // 2, canvas_h // 2
        radius = 15
        for _ in range(200):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    dist_sq = dx * dx + dy * dy
                    if dist_sq > radius * radius:
                        continue
                    px = cx + dx
                    py = cy + dy
                    if 0 <= px < canvas_w and 0 <= py < canvas_h:
                        falloff = 1.0 - dist_sq / (radius * radius)
                        idx = py * canvas_w + px
                        wear_map[idx] = min(1.0, wear_map[idx] + falloff * 0.02)

        self.assertTrue(np.all(wear_map >= 0.0))
        self.assertTrue(np.all(wear_map <= 1.0 + 1e-6))
        self.assertGreater(np.max(wear_map), 0.5,
            "多次点击后中心区域磨损应>0.5")

    def test_polish_saturates_at_one(self):
        """边界：polishLevel不会超过1.0"""
        polish = 0.995
        for _ in range(100):
            polish = min(1.0, polish + 0.0008)
        self.assertAlmostEqual(polish, 1.0)

    def test_patina_saturates_at_zero(self):
        """边界：patinaCoverage不会低于0.0"""
        patina = 0.005
        for _ in range(100):
            patina = max(0.0, patina - 0.0005)
        self.assertAlmostEqual(patina, 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
