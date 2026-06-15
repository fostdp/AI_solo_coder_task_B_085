import os
from celery import Celery
from kombu import Queue

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jade_monitor.settings')

app = Celery('jade_monitor')

app.config_from_object('django.conf:settings', namespace='CELERY')

app.autodiscover_tasks()

REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', '6379')
REDIS_DB = os.environ.get('REDIS_DB', '0')

BROKER_URL = os.environ.get(
    'CELERY_BROKER_URL',
    f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
)
RESULT_BACKEND = os.environ.get(
    'CELERY_RESULT_BACKEND',
    f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
)

app.conf.update(
    broker_url=BROKER_URL,
    result_backend=RESULT_BACKEND,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='Asia/Shanghai',
    enable_utc=True,
    task_routes={
        'diffusion_solver.tasks.*': {'queue': 'diffusion'},
        'anomaly_detector.tasks.*': {'queue': 'anomaly'},
        'alert_ws.tasks.*': {'queue': 'alerts'},
        'fiveg_receiver.tasks.*': {'queue': 'receiver'},
        'provenance_tracer.tasks.*': {'queue': 'anomaly'},
        'ph_inversion.tasks.*': {'queue': 'diffusion'},
        'forgery_classifier.tasks.*': {'queue': 'anomaly'},
    },
    task_queues=(
        Queue('default'),
        Queue('diffusion'),
        Queue('anomaly'),
        Queue('alerts'),
        Queue('receiver'),
    ),
    task_default_queue='default',
)


@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    pass
