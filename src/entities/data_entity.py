from dataclasses import dataclass
from werkzeug.datastructures import FileStorage
from typing import Union


@dataclass(frozen=True)
class UpdateVault:
    type: str
    file: Union[FileStorage, str]
