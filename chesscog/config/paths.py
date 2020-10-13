from fvcore.common.file_io import PathHandler, PathManager
from pathlib import Path
import typing

from chesscog import DATA_DIR, CONFIG_DIR


def make_handler(prefix: str, folder: Path) -> PathHandler:
    class HandlerImpl(PathHandler):

        PREFIX = prefix
        _folder = folder

        def _get_supported_prefixes(self):
            return [self.PREFIX]

        def _get_local_path(self, path):
            name = path[len(self.PREFIX):]
            return PathManager.get_local_path(str(self._folder / name))

        def _open(self, path, mode="r", **kwargs):
            return PathManager.open(self._get_local_path(path), mode, **kwargs)

    return HandlerImpl()


PathManager.register_handler(make_handler("data://", DATA_DIR))
PathManager.register_handler(make_handler("config://", CONFIG_DIR))
