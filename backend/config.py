"""Runtime settings. Every value can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

try:  # settings can live in bikole/.env (see .env.example)
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:  # python-dotenv not installed: plain environment variables still work
    pass


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    # Folder exported by the Phase 2 notebook: phase2_outputs/multi/lora_adapter
    # (a merged model folder such as phase2_outputs/multi/merged_fp16 also works).
    model_dir: Path = field(default_factory=lambda: _env_path("BIKOLE_MODEL_DIR", ROOT_DIR / "models" / "lora_adapter"))
    # MySQL connection for feedback storage. The database and table are created automatically.
    mysql_host: str = field(default_factory=lambda: os.getenv("MYSQL_HOST", "127.0.0.1"))
    mysql_port: int = field(default_factory=lambda: int(os.getenv("MYSQL_PORT", "3306")))
    mysql_user: str = field(default_factory=lambda: os.getenv("MYSQL_USER", "root"))
    mysql_password: str = field(default_factory=lambda: os.getenv("MYSQL_PASSWORD", ""))
    mysql_database: str = field(default_factory=lambda: os.getenv("MYSQL_DATABASE", "bikole"))
    frontend_dir: Path = ROOT_DIR / "frontend"
    device: str = field(default_factory=lambda: os.getenv("BIKOLE_DEVICE", "auto"))          # auto | cpu | cuda
    num_beams: int = field(default_factory=lambda: int(os.getenv("BIKOLE_NUM_BEAMS", "4")))   # 1 = fastest (greedy)
    cpu_threads: int = field(default_factory=lambda: int(os.getenv("BIKOLE_THREADS", "0")))     # 0 = PyTorch default
    quantize_cpu: bool = field(default_factory=lambda: os.getenv("BIKOLE_QUANTIZE", "0") == "1")
    load_model: bool = field(default_factory=lambda: os.getenv("BIKOLE_LOAD_MODEL", "1") != "0")
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(o.strip() for o in os.getenv("BIKOLE_CORS_ORIGINS", "*").split(",") if o.strip())
    )


MAX_SOURCE_CHARS = 1000
MAX_TRANSLATION_CHARS = 2000
MAX_EXPLANATION_CHARS = 1000
