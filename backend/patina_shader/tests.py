import unittest
import math
import os
from .models import (
    PatinaShaderState,
    PatinaShaderRenderer,
    polish_level
)


class BlinnPhongMathValidator:
    @staticmethod
    def validate_polish_level(click_count: int) -> float:
        return 1.0 - math.exp(-click_count / 800.0)

    @staticmethod
    def validate_effective_shininess(polish_level_val: float) -> float:
        return 48.0 * (0.3 + polish_level_val * 0.7)

    @staticmethod
    def validate_specular_strength(polish_level_val: float) -> float:
        return 0.15 + polish_level_val * 0.6

    @staticmethod
    def validate_specular(n_dot_h: float, effective_shininess: float, specular_strength: float) -> float:
        return math.pow(max(n_dot_h, 0.0), effective_shininess) * specular_strength

    @staticmethod
    def validate_normalize(v: tuple) -> tuple:
        length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + 1e-12
        return (v[0] / length, v[1] / length, v[2] / length)

    @staticmethod
    def validate_dot(a: tuple, b: tuple) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class TestVectorMath(unittest.TestCase):
    def setUp(self):
        self.renderer = PatinaShaderRenderer()

    def test_normalize_unit_length(self):
        v = (1.0, 0.0, 0.0)
        result = self.renderer._normalize(v)
        expected = BlinnPhongMathValidator.validate_normalize(v)
        self.assertAlmostEqual(result[0], expected[0], places=10)
        self.assertAlmostEqual(result[1], expected[1], places=10)
        self.assertAlmostEqual(result[2], expected[2], places=10)
        length = math.sqrt(result[0] ** 2 + result[1] ** 2 + result[2] ** 2)
        self.assertAlmostEqual(length, 1.0, places=10)

    def test_normalize_small_vector(self):
        v = (1e-6, 1e-6, 1e-6)
        result = self.renderer._normalize(v)
        expected = BlinnPhongMathValidator.validate_normalize(v)
        self.assertAlmostEqual(result[0], expected[0], places=10)
        self.assertAlmostEqual(result[1], expected[1], places=10)
        self.assertAlmostEqual(result[2], expected[2], places=10)
        length = math.sqrt(result[0] ** 2 + result[1] ** 2 + result[2] ** 2)
        self.assertAlmostEqual(length, 1.0, places=5)

    def test_dot_product_orthogonal(self):
        a = (1.0, 0.0, 0.0)
        b = (0.0, 1.0, 0.0)
        result = self.renderer._dot(a, b)
        expected = BlinnPhongMathValidator.validate_dot(a, b)
        self.assertAlmostEqual(result, expected, places=10)
        self.assertAlmostEqual(result, 0.0, places=10)


class TestLightingComponentsNonNegative(unittest.TestCase):
    def setUp(self):
        self.renderer = PatinaShaderRenderer()
        self.state = PatinaShaderState(polish_level=0.5)

    def test_ambient_non_negative(self):
        normal = (0.0, 0.0, 1.0)
        ambient, _, _ = self.renderer.compute_lighting(normal, self.state)
        self.assertGreaterEqual(ambient, 0.0)

    def test_diffuse_non_negative(self):
        normal = (0.0, 0.0, 1.0)
        _, diffuse, _ = self.renderer.compute_lighting(normal, self.state)
        self.assertGreaterEqual(diffuse, 0.0)

    def test_specular_non_negative(self):
        normal = (0.0, 0.0, 1.0)
        _, _, specular = self.renderer.compute_lighting(normal, self.state)
        self.assertGreaterEqual(specular, 0.0)


class TestPolishLevelNonlinearGloss(unittest.TestCase):
    def setUp(self):
        self.renderer = PatinaShaderRenderer()
        self.validator = BlinnPhongMathValidator()

    def test_polish_level_monotonic(self):
        prev = -1.0
        for clicks in range(0, 500, 10):
            current = polish_level(clicks)
            validated = self.validator.validate_polish_level(clicks)
            self.assertAlmostEqual(current, validated, places=10)
            self.assertGreaterEqual(current, prev)
            prev = current

    def test_polish_level_saturation(self):
        large_clicks = 10000
        result = polish_level(large_clicks)
        self.assertGreater(result, 0.99)
        self.assertLessEqual(result, 1.0)

    def test_effective_shininess_increases(self):
        prev_shininess = 0.0
        for clicks in range(0, 500, 20):
            pl = polish_level(clicks)
            current = self.renderer.compute_effective_shininess(pl)
            validated = self.validator.validate_effective_shininess(pl)
            self.assertAlmostEqual(current, validated, places=10)
            self.assertGreater(current, prev_shininess)
            prev_shininess = current

    def test_specular_strength_increases(self):
        prev_strength = 0.0
        for clicks in range(0, 500, 20):
            pl = polish_level(clicks)
            current = self.renderer.compute_specular_strength(pl)
            validated = self.validator.validate_specular_strength(pl)
            self.assertAlmostEqual(current, validated, places=10)
            self.assertGreater(current, prev_strength)
            prev_strength = current

    def test_overall_monotonic_3000_clicks(self):
        prev_polish = -1.0
        for clicks in range(0, 3001, 1):
            current = polish_level(clicks)
            self.assertGreaterEqual(current, prev_polish)
            prev_polish = current

    def test_concavity(self):
        values = [polish_level(clicks) for clicks in range(0, 300)]
        first_diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
        second_diffs = [first_diffs[i+1] - first_diffs[i] for i in range(len(first_diffs)-1)]
        concave_count = sum(1 for d in second_diffs if d < 0)
        concave_ratio = concave_count / len(second_diffs)
        self.assertGreater(concave_ratio, 0.6)

    def test_increment_decreasing(self):
        for start in range(0, 3000, 100):
            increment_1 = polish_level(start + 100) - polish_level(start)
            increment_2 = polish_level(start + 200) - polish_level(start + 100)
            self.assertLess(increment_2, increment_1)

    def test_apply_click_multiple(self):
        renderer = PatinaShaderRenderer()
        initial_clicks = renderer._state.click_count
        initial_polish = renderer._state.polish_level

        renderer.apply_click(5)
        self.assertEqual(renderer._state.click_count, initial_clicks + 5)
        expected_polish = polish_level(initial_clicks + 5)
        self.assertAlmostEqual(renderer._state.polish_level, expected_polish, places=10)
        self.assertGreater(renderer._state.polish_level, initial_polish)

        renderer.apply_click(10)
        self.assertEqual(renderer._state.click_count, initial_clicks + 15)
        expected_polish = polish_level(initial_clicks + 15)
        self.assertAlmostEqual(renderer._state.polish_level, expected_polish, places=10)


class TestShapeFunctionValidity(unittest.TestCase):
    def setUp(self):
        self.validator = BlinnPhongMathValidator()

    def test_polish_level_at_zero(self):
        result = polish_level(0)
        validated = self.validator.validate_polish_level(0)
        self.assertAlmostEqual(result, validated, places=10)
        self.assertEqual(result, 0.0)

    def test_polish_level_at_infinity_approaches_1(self):
        for clicks, threshold in [(2000, 0.9), (3000, 0.95), (5000, 0.98), (10000, 0.999)]:
            result = polish_level(clicks)
            self.assertGreater(result, threshold)
            self.assertLessEqual(result, 1.0)
        result = polish_level(100000)
        self.assertGreater(result, 0.9999)

    def test_polish_level_derivative_decreasing(self):
        prev_derivative = float('inf')
        for clicks in range(0, 500, 10):
            derivative = (polish_level(clicks + 1) - polish_level(clicks)) / 1.0
            self.assertLess(derivative, prev_derivative)
            prev_derivative = derivative


class TestPatinaCoverageInteraction(unittest.TestCase):
    def setUp(self):
        self.renderer = PatinaShaderRenderer()
        self.normal = (0.0, 0.0, 1.0)

    def test_patina_coverage_blocks_specular(self):
        state_low = PatinaShaderState(polish_level=0.0)
        state_high = PatinaShaderState(polish_level=1.0)
        _, _, spec_low = self.renderer.compute_lighting(self.normal, state_low)
        _, _, spec_high = self.renderer.compute_lighting(self.normal, state_high)
        self.assertGreater(spec_high, spec_low)

    def test_partial_coverage(self):
        state_partial = PatinaShaderState(polish_level=0.5)
        state_zero = PatinaShaderState(polish_level=0.0)
        state_full = PatinaShaderState(polish_level=1.0)
        
        shininess_zero = self.renderer.compute_effective_shininess(state_zero.polish_level)
        shininess_partial = self.renderer.compute_effective_shininess(state_partial.polish_level)
        shininess_full = self.renderer.compute_effective_shininess(state_full.polish_level)
        self.assertGreater(shininess_partial, shininess_zero)
        self.assertGreater(shininess_full, shininess_partial)
        
        strength_zero = self.renderer.compute_specular_strength(state_zero.polish_level)
        strength_partial = self.renderer.compute_specular_strength(state_partial.polish_level)
        strength_full = self.renderer.compute_specular_strength(state_full.polish_level)
        self.assertGreater(strength_partial, strength_zero)
        self.assertGreater(strength_full, strength_partial)
        
        _, _, spec_partial = self.renderer.compute_lighting(self.normal, state_partial)
        _, _, spec_zero = self.renderer.compute_lighting(self.normal, state_zero)
        self.assertGreater(spec_partial, spec_zero)

    def test_zero_coverage_no_effect(self):
        state = PatinaShaderState(polish_level=0.0)
        ambient, diffuse, specular = self.renderer.compute_lighting(self.normal, state)
        self.assertEqual(ambient, 0.25)
        self.assertGreaterEqual(diffuse, 0.0)
        self.assertGreaterEqual(specular, 0.0)
        effective_shininess = self.renderer.compute_effective_shininess(0.0)
        self.assertEqual(effective_shininess, 48.0 * 0.3)
        spec_strength = self.renderer.compute_specular_strength(0.0)
        self.assertEqual(spec_strength, 0.15)


class TestShaderFileIntegrity(unittest.TestCase):
    def setUp(self):
        self.renderer = PatinaShaderRenderer()
        self.shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')

    def test_vertex_shader_exists(self):
        vertex_path = os.path.join(self.shader_dir, 'vertex_shader.glsl')
        self.assertTrue(os.path.exists(vertex_path))
        with open(vertex_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertGreater(len(content), 0)

    def test_fragment_shader_contains_blinn_phong(self):
        fragment_path = os.path.join(self.shader_dir, 'fragment_shader.glsl')
        self.assertTrue(os.path.exists(fragment_path))
        with open(fragment_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn('effectiveShininess', content)
        self.assertIn('48.0 * (0.3 + uPolishLevel * 0.7)', content)
        self.assertIn('0.15 + uPolishLevel * 0.6', content)
        self.assertIn('pow(specDot, effectiveShininess)', content)


if __name__ == '__main__':
    unittest.main()
