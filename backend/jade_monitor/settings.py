import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def _load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR / '.env')
    except ImportError:
        env_path = BASE_DIR / '.env'
        if env_path.exists():
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value

_load_dotenv()

SECRET_KEY = os.environ.get(
    'DJANGO_SECRET_KEY',
    'django-insecure-jade-monitor-dev-fallback-2024-!changeme!'
)

if SECRET_KEY.startswith('django-insecure-') and os.environ.get('DJANGO_ENV') == 'production':
    import warnings
    warnings.warn(
        "生产环境使用默认 SECRET_KEY！请设置 DJANGO_SECRET_KEY 环境变量",
        RuntimeWarning
    )

DEBUG = os.environ.get('DJANGO_DEBUG', 'True').lower() in ('true', '1', 'yes')
ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', '*').split(',')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'channels',
    'django_prometheus',
    'api',
    # 'algorithms',
    # 'alerts',
    # 'simulator',
    'fiveg_receiver',
    'diffusion_solver',
    'anomaly_detector',
    'alert_ws',
    'provenance_tracer',
    'ph_inversion',
    'forgery_classifier',
]

MIDDLEWARE = [
    'django_prometheus.middleware.PrometheusBeforeMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.gzip.GZipMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django_prometheus.middleware.PrometheusAfterMiddleware',
]

CORS_ORIGIN_ALLOW_ALL = True

ROOT_URLCONF = 'jade_monitor.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'jade_monitor.wsgi.application'
ASGI_APPLICATION = 'jade_monitor.asgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

MONGODB_DATABASES = {
    'default': {
        'name': os.environ.get('MONGODB_NAME', 'jade_monitor'),
        'host': os.environ.get('MONGODB_HOST', 'localhost'),
        'port': int(os.environ.get('MONGODB_PORT', '27017')),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = 'Asia/Shanghai'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            'hosts': [(os.environ.get('REDIS_HOST', 'localhost'), int(os.environ.get('REDIS_PORT', '6379')))],
        },
    },
}

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
CELERY_TIMEZONE = 'Asia/Shanghai'
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT = ['json']

WECHAT_WEBHOOK_URL = os.environ.get(
    'WECHAT_WEBHOOK_URL',
    'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=your-key-here'
)

DIFFUSION_ALERT_THRESHOLD_MM = float(os.environ.get('DIFFUSION_ALERT_THRESHOLD_MM', '2.0'))
ANOMALY_SCORE_THRESHOLD = float(os.environ.get('ANOMALY_SCORE_THRESHOLD', '0.7'))

FIVE_G_BAND = os.environ.get('5G_BAND', 'n78')
FIVE_G_PEAK_THROUGHPUT_GBPS = float(os.environ.get('5G_PEAK_THROUGHPUT_GBPS', '1.0'))
FIVE_G_SUSTAINED_THROUGHPUT_GBPS = float(os.environ.get('5G_SUSTAINED_THROUGHPUT_GBPS', '0.8'))
FIVE_G_LATENCY_MS = int(os.environ.get('5G_LATENCY_MS', '10'))
FIVE_G_PACKET_LOSS_RATE = float(os.environ.get('5G_PACKET_LOSS_RATE', '0.001'))

ISOLATION_FOREST_N_ESTIMATORS = int(os.environ.get('ISOLATION_FOREST_N_ESTIMATORS', '50'))
ISOLATION_FOREST_MAX_SAMPLES = int(os.environ.get('ISOLATION_FOREST_MAX_SAMPLES', '256'))
ISOLATION_FOREST_CONTAMINATION = float(os.environ.get('ISOLATION_FOREST_CONTAMINATION', '0.1'))
ISOLATION_FOREST_MAX_BUFFER = int(os.environ.get('ISOLATION_FOREST_MAX_BUFFER', '10000'))

WEBSOCKET_PING_INTERVAL = int(os.environ.get('WEBSOCKET_PING_INTERVAL', '30'))
WEBSOCKET_PONG_TIMEOUT = int(os.environ.get('WEBSOCKET_PONG_TIMEOUT', '10'))
WEBSOCKET_MAX_PENDING_ALERTS = int(os.environ.get('WEBSOCKET_MAX_PENDING_ALERTS', '500'))
WEBSOCKET_PENDING_TTL = int(os.environ.get('WEBSOCKET_PENDING_TTL', '3600'))

SIMULATOR_ARTIFACT_COUNT = int(os.environ.get('SIMULATOR_ARTIFACT_COUNT', '200'))
SIMULATOR_INTERVAL = int(os.environ.get('SIMULATOR_INTERVAL', '21600'))
SIMULATOR_DEVICE_COUNT = int(os.environ.get('SIMULATOR_DEVICE_COUNT', '40'))
