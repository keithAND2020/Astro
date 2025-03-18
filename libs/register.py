import importlib
import logging
import pdb
import os

def get_modules_auto(file_dir, exclude_modules):
    MODEL_MODULES = []
    for _ in os.listdir(file_dir):
        if _ not in exclude_modules:
            if _[-3:]=='.py':
                _ = _[:-3]
            MODEL_MODULES.append(_)
    return MODEL_MODULES

def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    if not errors:
        return
    for name, err in errors:
        logging.warning("Module {} import failed: {}".format(name, err))


def import_all_modules_for_register(ALL_MODULES):
    errors = []
    for base_dir, modules in ALL_MODULES:
        for name in modules.copy():
            try:
                importlib.import_module('.'+name, base_dir)
            except ImportError as error:
                errors.append((modules, error))
    _handle_errors(errors)


class Register:

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning("Key %s already in registry %s." % (key, self._name))
        self._dict[key] = value

    def register(self, target):
        """Decorator to register a function or class."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            # @reg.register
            return add(None, target)
        # @reg.register('alias')
        return lambda x: add(target, x)

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        """key"""
        return self._dict.keys()