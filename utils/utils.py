#! /bin/env/ python3
'''
this file houses some miscellaneous handy utils and functions
'''

# local imports
import color
# stdlib and package imports
import importlib.util
import sys
from pathlib import Path


global sess
sess = None


def export_path():
    '''
    adds directory with vggish-related scripts to system path
    in the current session so that they can be imported with ease
    '''
    cwd = Path.cwd()
    repo = cwd.parent
    for child in ['vggish', 'utils']:
        childpath = repo.joinpath(child)
        if childpath.exists():
            sys.path.append(childpath)
            color.INFO('INFO', 'added "{}" to system path'.format(childpath))
        else:
            color.INFO('INFO', 'skipped "{}": did not exist'.format(childpath))


def import_by_path(name, path):
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
    except ImportError:
        color.ERR('ERR',
                  'module {} could not be imported from {}'.format(name, path))
        raise


def archive_handler(archivepath=None, filename=None, data=None):
    '''
    A middle-person function that should be used within another method to
    handle archiving of precomputed data to avoid unnecessary recomputation.
    Assumes archiving method is numpy savez_compressed. Will read data using
    a similar kind of method too.
    ---
        archivepath: the base path of the directory housing
        filename: a string filename of the archive within archives directory
                  (required)
        data: the data to archive, if archive doesn't exist (optional)
              if data is supplied, archive will be written to; else if no data
              is supplied, archive will be read and the data returned
    '''
    raise NotImplementedError
