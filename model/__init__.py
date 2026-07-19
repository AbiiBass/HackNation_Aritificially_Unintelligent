"""
model/__init__.py

The pipeline scripts in this package (predict.py, model3.py, train_model1.py,
etc.) use flat sibling imports like `from model3 import ...` rather than
relative imports like `from .model3 import ...`. That's intentional -- it's
what lets each script also be run standalone from inside this folder
(`python3 train_model1.py`) without needing to be invoked as `python3 -m
model.train_model1`.

The one place that matters for the web app is `app.py` doing
`from model import model_adapter`, which in turn does `from predict import
predict`. For that nested import to resolve, this package's own directory
needs to be on sys.path -- so we add it here, once, the moment `model` is
first imported. Standalone script execution already gets this for free
(Python puts a script's own directory at sys.path[0]); this line is only
needed for the "imported as a package" case.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
