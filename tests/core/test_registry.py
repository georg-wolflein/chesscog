import numpy as np

from chesscog.core.registry import Registry


def test_register_function_using_decorator():
    registry = Registry()

    @registry.register
    def my_func():
        pass

    assert "my_func" in registry
    assert registry["my_func"] == my_func


def test_register_class_using_decorator():
    registry = Registry()

    @registry.register
    class MyClass:
        pass

    assert "MyClass" in registry
    assert registry["MyClass"] == MyClass


def test_register_function_with_different_name():
    registry = Registry()

    @registry.register_as("abc")
    def my_func():
        pass

    assert "abc" in registry
    assert registry["abc"] == my_func


def test_registry_iter():
    registry = Registry()
    registry.register_as("abc")(None)
    registry.register_as("def")(None)
    iter = list(registry)
    assert "abc" in iter
    assert "def" in iter


def test_registry_items():
    registry = Registry()
    registry.register_as("abc")(1)
    registry.register_as("def")(2)
    items = registry.items()
    assert ("abc", 1) in items
    assert ("def", 2) in items
