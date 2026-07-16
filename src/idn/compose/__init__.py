"""Shared-space composition for LoRA capability adapters."""

from idn.compose.shared_basis import (
    SharedBasisError,
    build_shared_basis,
    compile_shared_scales,
    load_shared_basis_manifest,
)

__all__ = [
    "SharedBasisError",
    "build_shared_basis",
    "compile_shared_scales",
    "load_shared_basis_manifest",
]
