from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
GPTQ_DIR = REPO_ROOT / "gptq"

for path in (ROOT, GPTQ_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    import transformers  # noqa: F401
except ModuleNotFoundError:
    transformers = types.ModuleType("transformers")

    class Conv1D:  # pragma: no cover - smoke-test fallback only.
        pass

    transformers.Conv1D = Conv1D
    sys.modules["transformers"] = transformers

from quant import Quantizer, quantize  # noqa: E402
from gptq_submatrix_mixed import (  # noqa: E402
    GPTQSubmatrixMixed as LegacyGPTQSubmatrixMixed,
    _int8_fakequant_group,
    _pad_and_reshape_to_blocks,
    _vectorized_int4_fakequant_blocks,
    compute_block_sensitivity as legacy_compute_block_sensitivity,
)

__all__ = [
    "LegacyGPTQSubmatrixMixed",
    "Quantizer",
    "_int8_fakequant_group",
    "_pad_and_reshape_to_blocks",
    "_vectorized_int4_fakequant_blocks",
    "legacy_compute_block_sensitivity",
    "quantize",
]
