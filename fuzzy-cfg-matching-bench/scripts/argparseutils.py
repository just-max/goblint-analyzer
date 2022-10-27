import shlex
import json
from pathlib import Path
from functools import wraps
import argparse


class SplittingArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return shlex.split(arg_line, comments=True)


def wrap_type(type_parser):
    @wraps(type_parser)
    def wrapper(s: str):
        try:
            return type_parser(s)
        except (argparse.ArgumentTypeError, TypeError, ValueError):
            raise
        except Exception as e:
            raise argparse.ArgumentTypeError(str(e)) from e
    return wrapper


@wrap_type
def absolute_path(s: str):
    return Path(s).resolve()


def existing_file(s: str):
    p = absolute_path(s)
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"{'no such file' if not p.exists() else 'not a file'}: {p}")
    return p


@wrap_type
def json_from_file(s: str):
    with open(s, "r") as json_file:
        return json.load(json_file)


# _copy_items, AppendAction, ExtendAction taken from Python 3.8 source
def _copy_items(items):
    if items is None:
        return []
    # The copy module is used only in the 'append' and 'append_const'
    # actions, and it is needed only when the default value isn't a list.
    # Delay its import for speeding up the common case.
    if type(items) is list:
        return items[:]
    import copy
    return copy.copy(items)


class AppendAction(argparse.Action):

    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        if nargs == 0:
            raise ValueError('nargs for append actions must be != 0; if arg '
                             'strings are not supplying the value to append, '
                             'the append const action may be more appropriate')
        if const is not None and nargs != OPTIONAL:
            raise ValueError('nargs must be %r to supply const' % OPTIONAL)
        super(AppendAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = _copy_items(items)
        items.append(values)
        setattr(namespace, self.dest, items)


class ExtendAction(AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        items = _copy_items(items)
        items.extend(values)
        setattr(namespace, self.dest, items)
