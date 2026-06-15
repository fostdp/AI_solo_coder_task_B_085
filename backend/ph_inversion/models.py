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
class PatinaProfile:
    depth_mm: np.ndarray
    fe3_concentration: np.ndarray
    fe2_concentration: np.ndarray
    fe3_fe2_ratio: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        self.depth_mm = np.asarray(self.depth_mm, dtype=np.float64)
        self.fe3_concentration = np.asarray(self.fe3_concentration, dtype=np.float64)
        self.fe2_concentration = np.asarray(self.fe2_concentration, dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.fe3_fe2_ratio = np.where(
                self.fe2_concentration > 1e-12,
                self.fe3_concentration / (self.fe2_concentration + 1e-12),
                0.0
            )


class PHGeochemicalModel:
    def __init__(self):
        self.k_oxidation = 2.5e-4
        self.eh_ph_slope = -0.059
        self.fe3_solubility_ksp = 2.79e-39
        self.fe2_solubility_ksp = 4.87e-17
        self.temperature_c = 20.0

    def theoretical_fe_ratio(self, ph: float, depth_mm: float, age_years: float) -> float:
        t_seconds = max(age_years * 365.25 * 24 * 3600, 1.0)
        eh = 0.8 - self.eh_ph_slope * ph

        if ph < 4.0:
            base_ratio = 0.005
        elif ph < 5.5:
            base_ratio = 0.02
        elif ph < 7.0:
            base_ratio = 0.08
        elif ph < 8.5:
            base_ratio = 0.25
        else:
            base_ratio = 0.55

        oxidation_rate = self.k_oxidation * np.exp(0.12 * (ph - 5.5))
        depth_factor = np.exp(-depth_mm / 2.5)
        age_factor = 1.0 - np.exp(-t_seconds / (1e12))

        ratio = base_ratio * oxidation_rate * age_factor * depth_factor
        ratio += np.random.RandomState(int(ph * 100 + depth_mm * 10)).normal(0, ratio * 0.08)
        return max(0.0, min(20.0, ratio))


class BayesianPHInversion:
    def __init__(
        self,
        ph_prior_mean: float = 7.0,
        ph_prior_std: float = 1.5,
        ph_min: float = 3.0,
        ph_max: float = 11.0,
        n_particles: int = 2000,
        likelihood_sigma: float = 0.15,
        cache_dir: str = '_model_cache'
    ):
        self.ph_prior_mean = ph_prior_mean
        self.ph_prior_std = ph_prior_std
        self.ph_min = ph_min
        self.ph_max = ph_max
        self.n_particles = n_particles
        self.likelihood_sigma = likelihood_sigma
        self.cache_dir = cache_dir
        self.geochem = PHGeochemicalModel()
        self._posterior_cache: Dict[str, Dict] = {}

    def sample_prior(self, n: int = None) -> np.ndarray:
        if n is None:
            n = self.n_particles
        samples = np.random.normal(self.ph_prior_mean, self.ph_prior_std, n)
        return np.clip(samples, self.ph_min, self.ph_max)

    def log_prior(self, ph: float) -> float:
        if ph < self.ph_min or ph > self.ph_max:
            return -np.inf
        z = (ph - self.ph_prior_mean) / self.ph_prior_std
        return -0.5 * z * z - np.log(self.ph_prior_std * np.sqrt(2 * np.pi))

    def log_likelihood(self, ph: float, profile: PatinaProfile, age_years: float) -> float:
        total_ll = 0.0
        n_points = 0

        for d, observed_ratio in zip(profile.depth_mm, profile.fe3_fe2_ratio):
            if observed_ratio <= 0:
                continue
            predicted_ratio = self.geochem.theoretical_fe_ratio(ph, d, age_years)
            if predicted_ratio <= 0:
                predicted_ratio = 1e-6

            residual = np.log(observed_ratio + 1e-8) - np.log(predicted_ratio + 1e-8)
            sigma = self.likelihood_sigma * (1.0 + d * 0.1)
            ll = -0.5 * (residual / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
            total_ll += ll
            n_points += 1

        if n_points == 0:
            return -1e6

        fe3_total_obs = _np_trapz(profile.fe3_concentration, profile.depth_mm)
        fe3_total_pred_mean = self._integrated_fe3(ph, profile.depth_mm, age_years)
        if fe3_total_pred_mean > 1e-12:
            ratio_total = fe3_total_obs / fe3_total_pred_mean
            ll_total = -0.5 * ((np.log(ratio_total + 1e-8)) / 0.5) ** 2
            total_ll += ll_total * 0.5

        return total_ll

    def _integrated_fe3(self, ph: float, depths: np.ndarray, age_years: float) -> float:
        total = 0.0
        for d in depths:
            ratio = self.geochem.theoretical_fe_ratio(ph, d, age_years)
            total += ratio * 0.1
        return total

    def log_posterior(self, ph: float, profile: PatinaProfile, age_years: float) -> float:
        return self.log_prior(ph) + self.log_likelihood(ph, profile, age_years)

    def mcmc_sample(
        self,
        profile: PatinaProfile,
        age_years: float = 5000.0,
        n_burn: int = 500,
        n_samples: int = 2000,
        proposal_std: float = 0.3,
        random_seed: int = 42
    ) -> Dict:
        rng = np.random.RandomState(random_seed)

        current_ph = rng.uniform(5.0, 9.0)
        current_lp = self.log_posterior(current_ph, profile, age_years)

        chain = []
        n_accepted = 0

        total_iter = n_burn + n_samples
        for i in range(total_iter):
            proposed_ph = current_ph + rng.normal(0, proposal_std)
            proposed_ph = np.clip(proposed_ph, self.ph_min, self.ph_max)
            proposed_lp = self.log_posterior(proposed_ph, profile, age_years)

            log_alpha = proposed_lp - current_lp
            if log_alpha >= 0 or np.log(rng.rand() + 1e-12) < log_alpha:
                current_ph = proposed_ph
                current_lp = proposed_lp
                n_accepted += 1

            if i >= n_burn:
                chain.append(current_ph)

        chain = np.array(chain)

        ph_mean = float(np.mean(chain))
        ph_median = float(np.median(chain))
        ph_std = float(np.std(chain))
        ph_ci_low = float(np.percentile(chain, 2.5))
        ph_ci_high = float(np.percentile(chain, 97.5))

        acceptance_rate = n_accepted / total_iter

        hist_bins = np.linspace(self.ph_min, self.ph_max, 50)
        hist_counts, hist_edges = np.histogram(chain, bins=hist_bins, density=True)

        soil_environment = self._classify_soil_environment(ph_mean)
        redox_condition = self._classify_redox(profile)

        if ph_mean < 5.5:
            interpretation = "强酸性埋藏环境，可能为泥炭沼泽或富含腐殖质的湿地土壤，有机质分解产生大量腐殖酸。"
        elif ph_mean < 6.5:
            interpretation = "弱酸性埋藏环境，常见于森林棕壤或淋溶型土壤，气候相对湿润。"
        elif ph_mean < 7.5:
            interpretation = "中性埋藏环境，条件稳定，矿物保存状态通常较好。"
        elif ph_mean < 8.5:
            interpretation = "弱碱性埋藏环境，多见于石灰性土壤或干旱半干旱地区，钙质沉积常见。"
        else:
            interpretation = "强碱性埋藏环境，可能为盐碱土或富含碳酸盐的地质环境。"

        return {
            'ph_mean': ph_mean,
            'ph_median': ph_median,
            'ph_std': ph_std,
            'ph_95ci': [ph_ci_low, ph_ci_high],
            'ph_chain': chain.tolist(),
            'ph_histogram': {
                'counts': hist_counts.tolist(),
                'edges': hist_edges.tolist()
            },
            'acceptance_rate': acceptance_rate,
            'soil_environment': soil_environment,
            'redox_condition': redox_condition,
            'interpretation': interpretation,
            'age_years_used': age_years,
            'timestamp': datetime.now().isoformat()
        }

    def _classify_soil_environment(self, ph: float) -> Dict:
        if ph < 4.5:
            env = 'extremely_acidic'
            name = '极强酸性土'
            color = 'dark-red'
        elif ph < 5.5:
            env = 'strongly_acidic'
            name = '强酸性土'
            color = 'red'
        elif ph < 6.5:
            env = 'slightly_acidic'
            name = '弱酸性土'
            color = 'orange'
        elif ph < 7.5:
            env = 'neutral'
            name = '中性土'
            color = 'green'
        elif ph < 8.5:
            env = 'slightly_alkaline'
            name = '弱碱性土'
            color = 'teal'
        else:
            env = 'strongly_alkaline'
            name = '强碱性土'
            color = 'blue'
        return {'key': env, 'name': name, 'color': color}

    def _classify_redox(self, profile: PatinaProfile) -> Dict:
        avg_ratio = np.mean(profile.fe3_fe2_ratio[profile.fe3_fe2_ratio > 0]) if np.any(profile.fe3_fe2_ratio > 0) else 0
        if avg_ratio < 0.05:
            cond = 'reducing'
            name = '还原环境（缺氧）'
            desc = '长期处于水下或深埋缺氧环境，铁以Fe²⁺为主'
        elif avg_ratio < 0.3:
            cond = 'suboxic'
            name = '亚氧化环境'
            desc = '间歇性氧化还原波动，可能地下水位变动区域'
        elif avg_ratio < 1.0:
            cond = 'moderately_oxidizing'
            name = '中等氧化环境'
            desc = '适度通气的土壤环境'
        else:
            cond = 'oxidizing'
            name = '氧化环境'
            desc = '富氧埋藏环境，Fe³⁺占主导'
        return {'key': cond, 'name': name, 'description': desc, 'avg_fe3_fe2_ratio': float(avg_ratio)}

    def reconstruct_ph_history(
        self,
        profile: PatinaProfile,
        age_years: float = 5000.0,
        n_phases: int = 3
    ) -> Dict:
        if n_phases < 2:
            n_phases = 2
        if n_phases > 5:
            n_phases = 5

        phase_depths = np.linspace(0, profile.depth_mm[-1] if len(profile.depth_mm) > 0 else 5.0, n_phases + 1)
        phases = []

        for i in range(n_phases):
            depth_mask = (profile.depth_mm >= phase_depths[i]) & (profile.depth_mm < phase_depths[i + 1])
            if depth_mask.sum() < 2:
                continue

            sub_profile = PatinaProfile(
                depth_mm=profile.depth_mm[depth_mask] - phase_depths[i],
                fe3_concentration=profile.fe3_concentration[depth_mask],
                fe2_concentration=profile.fe2_concentration[depth_mask]
            )
            phase_age = age_years * (1.0 - (phase_depths[i] / (phase_depths[-1] + 1e-8)))
            result = self.mcmc_sample(sub_profile, age_years=max(phase_age, 100.0), n_burn=200, n_samples=800)
            phases.append({
                'phase_index': i,
                'depth_range_mm': [float(phase_depths[i]), float(phase_depths[i + 1])],
                'phase_age_years': float(phase_age),
                'ph_mean': result['ph_mean'],
                'ph_95ci': result['ph_95ci'],
                'soil_environment': result['soil_environment']
            })

        return {
            'n_phases': len(phases),
            'phases': phases,
            'overall_trend': self._compute_ph_trend(phases)
        }

    def _compute_ph_trend(self, phases: List[Dict]) -> Dict:
        if len(phases) < 2:
            return {'direction': 'insufficient_data', 'slope_per_1000_years': 0.0, 'description': '数据不足'}

        ages = np.array([p['phase_age_years'] for p in phases])
        phs = np.array([p['ph_mean'] for p in phases])

        if len(ages) >= 2 and np.std(ages) > 0:
            slope, _ = np.polyfit(ages, phs, 1)
            slope_per_ky = float(slope * 1000.0)
        else:
            slope_per_ky = 0.0

        if abs(slope_per_ky) < 0.05:
            direction = 'stable'
            desc = '埋藏环境pH长期稳定'
        elif slope_per_ky > 0:
            direction = 'alkalinizing'
            desc = '随时间推移呈碱性化趋势，可能与气候干旱化或碳酸盐沉积有关'
        else:
            direction = 'acidifying'
            desc = '随时间推移呈酸化趋势，可能与腐殖质积累或气候湿润化有关'

        return {
            'direction': direction,
            'slope_per_1000_years': slope_per_ky,
            'description': desc
        }

    def extract_profile_from_diffusion(
        self,
        diffusion_result: Dict,
        max_depth_mm: float = 5.0,
        n_points: int = 50
    ) -> PatinaProfile:
        depths = np.linspace(0, max_depth_mm, n_points)

        fe3_profile = diffusion_result.get('fe3_diffusion', {})
        mn2_profile = diffusion_result.get('mn2_diffusion', {})

        fe3_conc = np.array(fe3_profile.get('concentration_profile', np.zeros(n_points).tolist()))
        if len(fe3_conc) < n_points:
            fe3_conc = np.interp(depths, np.linspace(0, max_depth_mm, len(fe3_conc)), fe3_conc)
        elif len(fe3_conc) > n_points:
            fe3_conc = fe3_conc[:n_points]

        temperature = diffusion_result.get('temperature', 15.0)
        ph_est = 7.0 + (temperature - 15.0) * 0.03

        rng = np.random.RandomState(int(hash(diffusion_result.get('artifact_id', 'default')) % 2**32))

        base_fe2 = fe3_conc.max() * np.exp(-0.3 * (ph_est - 5.0))
        fe2_conc = base_fe2 * np.exp(-depths / 3.0)
        fe2_conc += rng.normal(0, fe2_conc.max() * 0.05, n_points)
        fe2_conc = np.maximum(fe2_conc, fe3_conc * 0.01)

        return PatinaProfile(
            depth_mm=depths,
            fe3_concentration=np.maximum(fe3_conc, 1e-8),
            fe2_concentration=np.maximum(fe2_conc, 1e-8)
        )
