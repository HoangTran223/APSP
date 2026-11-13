"""Mark torchao, which does not import under this torch, unavailable to transformers (and to vLLM subprocesses via tf_torchao_site/)."""
import os

_SITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tf_torchao_site")
_pythonpath = os.environ.get("PYTHONPATH", "")
_parts = [p for p in _pythonpath.split(os.pathsep) if p]
if _SITE_DIR not in _parts:
    os.environ["PYTHONPATH"] = (
        _SITE_DIR if not _pythonpath else _SITE_DIR + os.pathsep + _pythonpath
    )

import transformers.utils.import_utils as _tf_import_utils

_tf_import_utils._torchao_available = False
