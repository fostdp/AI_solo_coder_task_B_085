import os
import logging
from celery import shared_task
from datetime import datetime

from api.mongodb import get_db, get_era_key
from .models import BayesianPHInversion

logger = logging.getLogger(__name__)

_inversion_instance = None


def _get_inverter():
    global _inversion_instance
    if _inversion_instance is None:
        _inversion_instance = BayesianPHInversion()
    return _inversion_instance


CULTURE_AGES = {
    '红山文化': 5500.0,
    '良渚文化': 5000.0,
    '仰韶文化': 6000.0,
    '龙山文化': 4500.0,
    '大汶口文化': 5000.0,
    '齐家文化': 4200.0,
    '二里头文化': 3800.0,
    '商代': 3500.0,
    'default': 5000.0
}


def _estimate_age(culture: str = None) -> float:
    if culture and culture in CULTURE_AGES:
        return CULTURE_AGES[culture]
    return CULTURE_AGES['default']


@shared_task(bind=True, max_retries=3, default_retry_delay=10)
def invert_ph_history(self, artifact_id: str, diffusion_result: dict = None):
    try:
        inverter = _get_inverter()

        if diffusion_result is None:
            db = get_db()
            diffusion_result = db.diffusion_results.find_one(
                {'artifact_id': artifact_id},
                sort=[('timestamp', -1)]
            )
            if diffusion_result is None:
                raise ValueError(f"未找到玉器 {artifact_id} 的扩散计算结果")

        db = get_db()
        artifact = db.jade_artifacts.find_one({'artifact_id': artifact_id})
        culture = artifact.get('culture') if artifact else None
        age_years = _estimate_age(culture)
        era = artifact.get('era') if artifact else None
        if not era:
            era = get_era_key(culture or '')

        profile = inverter.extract_profile_from_diffusion(diffusion_result)

        inversion_result = inverter.mcmc_sample(
            profile,
            age_years=age_years,
            n_burn=500,
            n_samples=2000
        )

        history_result = inverter.reconstruct_ph_history(
            profile,
            age_years=age_years,
            n_phases=3
        )

        doc = {
            'artifact_id': artifact_id,
            'era': era,
            'age_years_used': age_years,
            'culture': culture,
            'ph_mean': inversion_result['ph_mean'],
            'ph_median': inversion_result['ph_median'],
            'ph_std': inversion_result['ph_std'],
            'ph_95ci': inversion_result['ph_95ci'],
            'soil_environment': inversion_result['soil_environment'],
            'redox_condition': inversion_result['redox_condition'],
            'interpretation': inversion_result['interpretation'],
            'ph_history': history_result,
            'acceptance_rate': inversion_result['acceptance_rate'],
            'fe_profile': {
                'depths_mm': profile.depth_mm.tolist(),
                'fe3_concentration': profile.fe3_concentration.tolist(),
                'fe2_concentration': profile.fe2_concentration.tolist(),
                'fe3_fe2_ratio': profile.fe3_fe2_ratio.tolist()
            },
            'timestamp': datetime.now()
        }

        db.ph_inversion_results.insert_one(doc)
        logger.info(
            f"[pH反演] 玉器 {artifact_id} pH历史反演完成: "
            f"pH={inversion_result['ph_mean']:.2f}±{inversion_result['ph_std']:.2f}"
            f" [{inversion_result['soil_environment']['name']}]"
        )

        if self.request.called_directly:
            return {
                'status': 'completed',
                'artifact_id': artifact_id,
                'ph_mean': inversion_result['ph_mean'],
                'ph_median': inversion_result['ph_median'],
                'ph_std': inversion_result['ph_std'],
                'ph_95ci': inversion_result['ph_95ci'],
                'soil_environment': inversion_result['soil_environment'],
                'redox_condition': inversion_result['redox_condition'],
                'interpretation': inversion_result['interpretation'],
                'ph_history': history_result,
                'acceptance_rate': inversion_result['acceptance_rate'],
                'fe_profile': {
                    'depths_mm': profile.depth_mm.tolist(),
                    'fe3_concentration': profile.fe3_concentration.tolist(),
                    'fe2_concentration': profile.fe2_concentration.tolist(),
                    'fe3_fe2_ratio': profile.fe3_fe2_ratio.tolist()
                }
            }
        return True

    except Exception as exc:
        logger.error(f"[pH反演] 任务失败: {exc}")
        if self.request.called_directly:
            raise
        raise self.retry(exc=exc)
