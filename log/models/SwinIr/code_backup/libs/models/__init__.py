from ..register import import_all_modules_for_register, Register, get_modules_auto
import os
import pdb
import importlib

EXCLUDE_MODULES = ['model_init.py','__pycache__','__init__.py','base_model']
MODEL_MODULES = get_modules_auto(os.path.dirname(os.path.realpath(__file__)),EXCLUDE_MODULES)


MODEL = Register('deep_net')
ALL_MODULES = [('libs.models', MODEL_MODULES)]
import_all_modules_for_register(ALL_MODULES)

