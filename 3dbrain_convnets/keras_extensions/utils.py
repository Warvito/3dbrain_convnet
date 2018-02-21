from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO: Limpar essas funções de outros lugares

import re

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def sort_nicely(l):
    return sorted(l, key=alphanum_key)