from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VaultConfig:
    root_dir: Path
    knowledge_dir: Path
    memory_dir: Path
    chroma_db: Path
    knowledge_coll_name: str
    memory_coll_name: str


@dataclass(frozen=True)
class PrompterConfig:
    root_dir: Path
    knowledge_dir: Path
    memory_dir: Path
    chroma_db: Path
