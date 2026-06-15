import os
import logging
from celery import shared_task
from datetime import datetime

from api.mongodb import get_db, get_era_key
from .models import (
    RandomForestProvenanceClassifier,
    TraceElementProfile
)

logger = logging.getLogger(__name__)

_classifier_instance = None


def _get_classifier():
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = RandomForestProvenanceClassifier()
    return _classifier_instance


@shared_task(bind=True, max_retries=3, default_retry_delay=10)
def trace_provenance(self, artifact_id: str, xrf_spectrum: dict = None):
    try:
        classifier = _get_classifier()

        if xrf_spectrum is None:
            db = get_db()
            latest_xrf = db.xrf_spectrum.find_one(
                {'artifact_id': artifact_id},
                sort=[('timestamp', -1)]
            )
            if latest_xrf is None:
                raise ValueError(f"未找到玉器 {artifact_id} 的XRF光谱数据")
            xrf_spectrum = latest_xrf

        profile = classifier.extract_profile_from_xrf(xrf_spectrum)
        result = classifier.predict(profile)

        db = get_db()
        artifact = db.jade_artifacts.find_one({'artifact_id': artifact_id})
        if artifact:
            era = artifact.get('era') or get_era_key(artifact.get('culture', ''))
        else:
            era = 'unknown'

        doc = {
            'artifact_id': artifact_id,
            'era': era,
            'predicted_origin': result['predicted_origin'],
            'predicted_origin_key': result['predicted_origin_key'],
            'confidence': result['confidence'],
            'top_predictions': result['top_predictions'],
            'feature_importance': result.get('feature_importance', []),
            'trace_elements': result['trace_elements'],
            'timestamp': datetime.now()
        }

        db.provenance_results.insert_one(doc)
        logger.info(f"[Provenance] 玉器 {artifact_id} 产地溯源完成: {result['predicted_origin']} (置信度 {result['confidence']:.3f})")

        if self.request.called_directly:
            return {
                'status': 'completed',
                'artifact_id': artifact_id,
                'predicted_origin': result['predicted_origin'],
                'predicted_origin_key': result['predicted_origin_key'],
                'confidence': result['confidence'],
                'top_predictions': result['top_predictions'],
                'feature_importance': result.get('feature_importance', []),
                'trace_elements': result['trace_elements']
            }
        return True

    except Exception as exc:
        logger.error(f"[Provenance] 产地溯源任务失败: {exc}")
        if self.request.called_directly:
            raise
        raise self.retry(exc=exc)
