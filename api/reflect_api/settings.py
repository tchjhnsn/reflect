import os
import sys
from pathlib import Path
from datetime import timedelta

import dj_database_url
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Local development reads apps/api/.env, then apps/api/.env.local.
# .env.local overrides .env so you can keep production keys in .env
# and local dev keys in .env.local (which is gitignored).
# In deployed environments, real environment variables take precedence.
load_dotenv(BASE_DIR / ".env", override=False)
load_dotenv(BASE_DIR / ".env.local", override=True)

SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-secret-key")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

raw_allowed_hosts = os.getenv("ALLOWED_HOSTS", "*")
ALLOWED_HOSTS = [host.strip() for host in raw_allowed_hosts.split(",") if host.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "rest_framework_simplejwt.token_blacklist",
    "events_api",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "reflect_api.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]

WSGI_APPLICATION = "reflect_api.wsgi.application"

DATABASES = {
    "default": dj_database_url.config(
        default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
        conn_max_age=600,
        conn_health_checks=True,
    )
}

# Default test runs to SQLite to avoid requiring CREATE DATABASE on managed Postgres roles.
if "test" in sys.argv and os.getenv("USE_POSTGRES_FOR_TESTS", "false").lower() != "true":
    DATABASES["default"] = {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": str(BASE_DIR / "test_db.sqlite3"),
    }

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

CORS_ALLOW_ALL_ORIGINS = os.getenv("CORS_ALLOW_ALL_ORIGINS", "false").lower() == "true"
raw_cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
configured_cors_origins = [origin.strip() for origin in raw_cors_origins.split(",") if origin.strip()]
default_local_cors_origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]

# Keep local frontend dev origins enabled by default unless CORS_ALLOW_ALL_ORIGINS is used.
CORS_ALLOWED_ORIGINS = list(dict.fromkeys([*configured_cors_origins, *default_local_cors_origins]))
CSRF_TRUSTED_ORIGINS = list(dict.fromkeys([*configured_cors_origins, *default_local_cors_origins]))


def _bool_env(name: str, default: bool) -> bool:
    return os.getenv(name, str(default).lower()).lower() == "true"


def _is_non_local_origin(origin: str) -> bool:
    return not (
        origin.startswith("http://127.0.0.1")
        or origin.startswith("http://localhost")
        or origin.startswith("https://127.0.0.1")
        or origin.startswith("https://localhost")
    )


cross_origin_cookie_mode = bool(configured_cors_origins) and any(
    _is_non_local_origin(origin) for origin in configured_cors_origins
)

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
    ],
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=30),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=14),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "AUTH_HEADER_TYPES": ("Bearer",),
}

CORS_ALLOW_CREDENTIALS = True
default_samesite = "None" if cross_origin_cookie_mode else "Lax"
default_secure_cookie = cross_origin_cookie_mode
SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", default_samesite)
CSRF_COOKIE_SAMESITE = os.getenv("CSRF_COOKIE_SAMESITE", default_samesite)
SESSION_COOKIE_SECURE = _bool_env("SESSION_COOKIE_SECURE", default_secure_cookie)
CSRF_COOKIE_SECURE = _bool_env("CSRF_COOKIE_SECURE", default_secure_cookie)

# Allow large request bodies for AI conversation imports
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50MB

# Neo4j AuraDB / Neomodel Configuration
NEOMODEL_NEO4J_BOLT_URL = os.environ.get(
    "NEOMODEL_NEO4J_BOLT_URL", "bolt://neo4j:neo4j@localhost:7687"
)
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

# Anthropic API Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# HuggingFace Inference API Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# LLM Pipeline Configuration
# THRIVESIGHT_LLM_PROVIDER: "anthropic" (default) or "huggingface"
LLM_PROVIDER = os.environ.get("THRIVESIGHT_LLM_PROVIDER", "anthropic")
LLM_MODEL = os.environ.get("THRIVESIGHT_LLM_MODEL", "")
LLM_MAX_RETRIES = int(os.environ.get("THRIVESIGHT_LLM_MAX_RETRIES", "1"))

# Trigger category configuration
CATEGORY_SIMILARITY_THRESHOLD = float(os.environ.get("CATEGORY_SIMILARITY_THRESHOLD", "0.7"))
CATEGORY_VALIDATION_THRESHOLD = int(os.environ.get("CATEGORY_VALIDATION_THRESHOLD", "3"))

# Trajectory computation configuration
INFLECTION_THRESHOLD = float(os.environ.get("INFLECTION_THRESHOLD", "1.5"))
