"""Local configuration. Secrets are read from the process or ignored .env file."""

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _env_file() -> dict[str, str]:
    path = Path.home() / ".config" / "rmg-manual-assistant" / ".env"
    if not path.exists():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def setting(name: str, default: str = "") -> str:
    return os.environ.get(name) or _env_file().get(name, default)


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    runtime: Path = ROOT / "runtime"
    api_key: str = ""
    base_url: str = "https://api.mwapi.dev/v1"
    model: str = "claude-haiku-4-5-20251001"
    embedding_model: str = "BAAI/bge-m3"
    whisper_model: str = "SayedShaun/bengali-whisper-medium-ct2"
    deepgram_api_key: str = ""

    @classmethod
    def load(cls) -> "Settings":
        runtime = Path(setting("RUNTIME_DIR", str(ROOT / "runtime"))).resolve()
        base = setting("MWAPI_BASE_URL", "https://api.mwapi.dev/v1").rstrip("/")
        if not base.startswith("https://"):
            raise ValueError("MWAPI_BASE_URL must use HTTPS")
        return cls(
            runtime=runtime,
            api_key=setting("MWAPI_API_KEY"),
            base_url=base,
            model=setting("MWAPI_MODEL", cls.model),
            embedding_model=setting("EMBEDDING_MODEL", cls.embedding_model),
            whisper_model=setting("WHISPER_MODEL", cls.whisper_model),
            deepgram_api_key=setting("DEEPGRAM_API_KEY"),
        )
