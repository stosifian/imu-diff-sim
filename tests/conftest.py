"""Make `src/` importable without relying on an editable install.

Belt and braces: `pip install -e .` is the documented setup, but on Colab the
repo is usually just cloned and run, and an editable install's .pth file is one
more thing that can be stale. Four lines here means `pytest` works from a bare
checkout anywhere.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Float64 for the whole test session.
#
# JAX defaults to float32. Two tests need more than that:
#   - test_gradient_matches_central_finite_differences perturbs the loss by
#     ~1e-8 on a loss of ~1.0. In float32 the representable step is ~1.2e-7, so
#     the "numeric" gradient is pure quantization noise (it comes out at
#     exactly eps/2h) and the comparison is meaningless.
#   - the Allan round-trip integrates a million samples; float32 cumsum loses
#     the long-tau end to catastrophic cancellation.
#
# This must run before the first jax array is created, which importing
# conftest at collection time guarantees.
import jax

jax.config.update("jax_enable_x64", True)
