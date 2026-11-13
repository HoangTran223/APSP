"""Applies the torchao patch in subprocesses (vLLM's model registry) that have this dir on PYTHONPATH."""
try:
    import transformers.utils.import_utils as _tf_import_utils

    _tf_import_utils._torchao_available = False
except Exception:
    pass
