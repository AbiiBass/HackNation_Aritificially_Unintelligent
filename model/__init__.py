"""
Pipeline scripts in this package use flat sibling imports (e.g.
`from model3 import ...`) instead of relative imports, so they can be
run standalone. This file adds the package dir to sys.path so
`from model import model_adapter` also works when imported as a package.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
