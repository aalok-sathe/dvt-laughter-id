#! /bin/env/ python3
'''
this file houses some miscellaneous handy utils and functions
'''


import importlib.util
import color
import sys

from pathlib import Path

def export_path():
    '''
    adds directory with vggish-related scripts to system path
    in the current session so that they can be imported with ease
    '''
    cwd = Path.cwd()
    repo = cwd.parent
    vggpath = repo.joinpath('vggish')
    sys.path.append(vggpath)


def import_by_path(name, path):
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
    except ImportError:
        color.ERR('ERR',
                  'module {} could not be imported from {}'.format(name, path))
        raise
