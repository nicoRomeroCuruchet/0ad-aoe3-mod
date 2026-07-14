"""Gather environment package.

Environment exports are resolved lazily so pure helpers such as
``rl.gather.core`` remain usable without the optional 0 A.D. client.
"""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .env import ZeroADGatherEnv
    from .factory import make_gather_env


__all__ = ["ZeroADGatherEnv", "make_gather_env"]


def __getattr__(name: str) -> Any:
    if name == "ZeroADGatherEnv":
        from .env import ZeroADGatherEnv

        return ZeroADGatherEnv
    if name == "make_gather_env":
        from .factory import make_gather_env

        return make_gather_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
