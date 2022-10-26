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
