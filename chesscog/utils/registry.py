import functools


class Registry:

    def __init__(self):
        self._items = dict()

    def register(self, item, name: str = None):
        if name is None:
            name = item.__name__
        self._items[name] = item
        return item

    def register_as(self, name: str):
        return functools.partial(self.register, name=name)

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def items(self):
        return self._items.items()
