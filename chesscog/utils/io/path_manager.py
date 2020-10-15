from pathlib import Path, PurePath, _PosixFlavour
import typing
import logging
import functools
import abc
from contextlib import contextmanager
import re
import os

logger = logging.getLogger(__name__)


class _URIFlavour(_PosixFlavour):
    has_drv = True
    is_supported = True

    def splitroot(self, part, sep=_PosixFlavour.sep):
        assert sep == self.sep

        match = re.match(rf"(.*):{re.escape(sep)}{{2}}(.*)", part)
        if match:
            drive, path = match.groups()
            drive = drive + "://"
            root = ""
            return drive, root, path
        else:
            return super().splitroot(part, sep=sep)


class URI(PurePath):
    _flavour = _URIFlavour()

    @property
    def scheme(self) -> str:
        if not self.drive:
            return ""
        return self.drive[:-len("://")]

    @property
    def path(self) -> str:
        begin = 1 if self.drive or self.root else 0
        return self.root + self._flavour.join(self.parts[begin:])

    def __str__(self) -> str:
        return str(self.resolve())

    def __repr__(self) -> str:
        s = ""
        if self.scheme:
            s += self.scheme + ":" + self._flavour.sep * 2
        s += self.path
        return "{}({!r})".format(self.__class__.__name__, s)

    def resolve(self) -> Path:
        return PathManager.resolve(self)

    @functools.wraps(Path.open)
    def open(self, *args, **kwargs):
        return self.resolve().open(*args, **kwargs)

    @functools.wraps(Path.glob)
    def glob(self, *args, **kwargs):
        return self.resolve().glob(*args, *kwargs)

    @functools.wraps(Path.exists)
    def exists(self, *args, **kwargs):
        return self.resolve().exists(*args, **kwargs)


class PathManager:

    _handlers = {}

    @classmethod
    def resolve(cls, path: os.PathLike) -> Path:
        if not isinstance(path, URI):
            path = URI(path)
        if path.scheme:
            if path.scheme not in cls._handlers:
                raise NotImplementedError
            return cls._handlers[path.scheme](path)
        else:
            return Path(path.path)

    @classmethod
    def register_handler(cls, scheme: str) -> typing.Callable:
        def decorator(func: typing.Callable[[os.PathLike], Path]):
            cls._handlers[scheme] = func
            logger.debug(f"Registered path handler for scheme {scheme}")
            return func
        return decorator


class PathTranslator(abc.ABC):

    def __init__(self, base_path: Path):
        self._base_path = base_path

    def __call__(self, url: URI) -> Path:
        return self._base_path / url.path
