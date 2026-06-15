import os
import logging
from celery import shared_task
from datetime import datetime

from api.mongodb import get_db, get_era_key
from .models import SVMForgeryClassifier

logger = logging.getLogger(__name__)

_classifier_instance = None


def _get_classifier():
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SVMForgeryClassifier()
    return _classifier_instance


@shared_task(bind=True, max_retries=3, default_retry_delay=10)
def classify_forgery_process(self, artifact_id: str, raman_spectrum: dict = None):
    try:
        classifier = _get_classifier()

        if raman_spectrum is None:
            db = get_db()
            latest_raman = db.raman_spectrum.find_one(
                {'artifact_id': artifact_id},
                sort=[('timestamp', -1)]
            )
            if latest_raman is None:
                raise ValueError(f"未找到玉器 {artifact_id} 的拉曼光谱数据")
            raman_spectrum = latest_raman

        features = classifier.extract_features_from_raman(raman_spectrum)
        result = classifier.predict(features)

        db = get_db()
        artifact = db.jade_artifacts.find_one({'artifact_id': artifact_id})
        if artifact:
            era = artifact.get('era') or get_era_key(artifact.get('culture', ''))
        else:
            era = 'unknown'

        doc = {
            'artifact_id': artifact_id,
            'era': era,
            'predicted_process': result['predicted_process'],
            'predicted_process_key': result['predicted_process_key'],
            'confidence': result['confidence'],
            'is_forgery': result['is_forgery'],
            'forgery_risk': result['forgery_risk'],
            'top_predictions': result['top_predictions'],
            'diagnostic_features': result['diagnostic_features'],
            'raw_features': result['raw_features'],
            'timestamp': datetime.now()
        }

        db.forgery_classification_results.insert_one(doc)
        logger.info(
            f"[ForgeryClassify] 玉器 {artifact_id} 作伪工艺分类完成: "
            f"{result['predicted_process']} (置信度 {result['confidence']:.3f}, "
            f"风险 {result['forgery_risk']:.3f})"
        )

        if result['is_forgery'] and result['forgery_risk'] > 0.7:
            try:
                from alert_ws.tasks import send_anomaly_alert
                send_anomaly_alert.delay(
                    artifact_id=artifact_id,
                    anomaly_score=result['forgery_risk'],
                    forgery_probability=result['forgery_risk'],
                    anomaly_reasons=[f"检测到作伪工艺: {result['predicted_process']} (置信度{result['confidence']:.1%})"],
                    risk_level='high' if result['forgery_risk'] > 0.85 else 'medium'
                )
            except Exception as ae:
                logger.warning(f"[ForgeryClassify] 告警推送失败: {ae}")

        if self.request.called_directly:
            return {
                'status': 'completed',
                'artifact_id': artifact_id,
                'predicted_process': result['predicted_process'],
                'predicted_process_key': result['predicted_process_key'],
                'confidence': result['confidence'],
                'is_forgery': result['is_forgery'],
                'forgery_risk': result['forgery_risk'],
                'top_predictions': result['top_predictions'],
                'diagnostic_features': result['diagnostic_features'],
                'raw_features': result['raw_features']
            }
        return True

    except Exception as exc:
        logger.error(f"[ForgeryClassify] 任务失败: {exc}")
        if self.request.called_directly:
            raise
        raise self.retry(exc=exc)
