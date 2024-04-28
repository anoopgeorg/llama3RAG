from pathlib import Path

from src.utils.common import read_yaml, create_directory
from src.entities.config_entity import VaultConfig, PrompterConfig

CONFIG_FILE_PATH = Path("src/configurations/config.yaml")


class ConfigManager:
    def __init__(self, config_pth=CONFIG_FILE_PATH):
        self.config = read_yaml(config_pth)

    def getVaultConfig(self):
        config = self.config.VaultConfigurations
        create_directory(
            [config.root_dir, config.knowledge_dir, config.memory_dir, config.chroma_db]
        )
        vault_config = VaultConfig(
            root_dir=config.root_dir,
            knowledge_dir=config.knowledge_dir,
            memory_dir=config.memory_dir,
            chroma_db=config.chroma_db,
            knowledge_coll_name=config.knowledge_coll_name,
            memory_coll_name=config.memory_coll_name,
        )
        return vault_config

    def getPrompterConfig(self):
        config = self.config.PrompterConfig
        create_directory(
            [config.root_dir, config.knowledge_dir, config.memory_dir, config.chroma_db]
        )
        prompter_config = PrompterConfig(
            root_dir=Path(config.root_dir),
            knowledge_dir=Path(config.knowledge_dir),
            memory_dir=Path(config.memory_dir),
            chroma_db=Path(config.chroma_db),
        )
        return prompter_config
