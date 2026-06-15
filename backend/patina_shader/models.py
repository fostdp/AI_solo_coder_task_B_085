import math
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import logging

logger = logging.getLogger(__name__)


@dataclass
class PatinaShaderState:
    polish_level: float = 0.0
    click_count: int = 0
    light_dir: Tuple[float, float, float] = field(default_factory=lambda: (0.3, -0.4, 0.86))
    view_dir: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 1.0))
    base_color: Tuple[float, float, float] = field(default_factory=lambda: (0.7, 0.55, 0.35))

    def __post_init__(self):
        self.polish_level = max(0.0, min(1.0, self.polish_level))
        self.click_count = max(0, self.click_count)


def polish_level(click_count: int) -> float:
    return 1.0 - math.exp(-click_count / 800.0)


class PatinaShaderRenderer:
    SHININESS_BASE = 48.0
    AMBIENT_STRENGTH = 0.25
    DIFFUSE_STRENGTH = 0.6
    SPECULAR_BASE = 0.15
    SPECULAR_SCALE = 0.6

    def __init__(self, shader_dir: Optional[str] = None):
        if shader_dir is None:
            shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')
        self.shader_dir = shader_dir
        self._vertex_source: Optional[str] = None
        self._fragment_source: Optional[str] = None
        self._state = PatinaShaderState()

    def _normalize(self, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + 1e-12
        return (v[0] / length, v[1] / length, v[2] / length)

    def _dot(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def load_shaders(self) -> Tuple[str, str]:
        vertex_path = os.path.join(self.shader_dir, 'vertex_shader.glsl')
        fragment_path = os.path.join(self.shader_dir, 'fragment_shader.glsl')

        with open(vertex_path, 'r', encoding='utf-8') as f:
            self._vertex_source = f.read()

        with open(fragment_path, 'r', encoding='utf-8') as f:
            self._fragment_source = f.read()

        return self._vertex_source, self._fragment_source

    def compute_effective_shininess(self, polish_level: float) -> float:
        return self.SHININESS_BASE * (0.3 + polish_level * 0.7)

    def compute_specular_strength(self, polish_level: float) -> float:
        return self.SPECULAR_BASE + polish_level * self.SPECULAR_SCALE

    def compute_lighting(
        self,
        normal: Tuple[float, float, float],
        state: Optional[PatinaShaderState] = None
    ) -> Tuple[float, float, float]:
        if state is None:
            state = self._state

        n = self._normalize(normal)
        l = self._normalize(state.light_dir)
        v = self._normalize(state.view_dir)

        h = self._normalize((
            l[0] + v[0],
            l[1] + v[1],
            l[2] + v[2]
        ))

        ambient = self.AMBIENT_STRENGTH
        diffuse = max(self._dot(n, l), 0.0) * self.DIFFUSE_STRENGTH

        spec_dot = max(self._dot(n, h), 0.0)
        effective_shininess = self.compute_effective_shininess(state.polish_level)
        spec_strength = self.compute_specular_strength(state.polish_level)
        specular = math.pow(spec_dot, effective_shininess) * spec_strength

        return ambient, diffuse, specular

    def apply_click(self, clicks: int = 1) -> PatinaShaderState:
        self._state.click_count += clicks
        self._state.polish_level = polish_level(self._state.click_count)
        return self._state

    def get_uniforms(self, state: Optional[PatinaShaderState] = None) -> dict:
        if state is None:
            state = self._state

        return {
            'uPolishLevel': state.polish_level,
            'uLightDir': list(self._normalize(state.light_dir)),
            'uViewDir': list(self._normalize(state.view_dir)),
            'uBaseColor': list(state.base_color),
            'uAmbientStrength': self.AMBIENT_STRENGTH,
            'uDiffuseStrength': self.DIFFUSE_STRENGTH,
            'uSpecularBase': self.SPECULAR_BASE,
            'uSpecularScale': self.SPECULAR_SCALE,
            'uShininessBase': self.SHININESS_BASE
        }

    def get_shader_sources(self) -> dict:
        if self._vertex_source is None or self._fragment_source is None:
            self.load_shaders()
        return {
            'vertex_shader': self._vertex_source,
            'fragment_shader': self._fragment_source
        }
