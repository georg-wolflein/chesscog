"""Common interface for registries.
"""

import functools
import typing


class Registry:
    """A registry class. 

    A registry is a collection of functions or classes that are each associated with a name.
    These can be loaded dynamically based on a configuration.

    >>> my_registry = Registry()
    >>> @my_registry.register
    ... def abc():
    ...     print("abc was called")
    >>> my_registry.items()
    dict_items([('abc', <function abc at 0x103197430>)])
    >>> my_registry["abc"]()
    abc was called
    """

    def __init__(self):
        self._items = dict()

    def register(self, item: typing.Any, name: str = None) -> typing.Any:
        """Register an item.

        This function is typically used as a decorator.

        Args:
            item (typing.Any): the item to register
            name (str, optional): the name under which to register it. If not supplied, use the ``item.__name__`` attribute.

        Returns:
            typing.Any: the registered item
        """

        if name is None:
            name = item.__name__
        self._items[name] = item
        return item

    def register_as(self, name: str) -> callable:
        """Decorator for registering a function or class.

        Args:
            name (str): the name to register it under

        Returns:
            callable: the decorator
        """
        return functools.partial(self.register, name=name)

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def items(self) -> typing.Iterable:
        """Obtain a view of the registered items.

        Returns:
            typing.Iterable: the registered items
        """
        return self._items.items()
