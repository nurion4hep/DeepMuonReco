import secrets

from coolname.impl import generate_slug
from omegaconf import OmegaConf


def register_resolvers() -> None:
    """Register custom OmegaConf resolvers used across scripts."""
    OmegaConf.register_new_resolver(
        "slug",
        lambda pattern=2: generate_slug(pattern=pattern),
        use_cache=True,
        replace=True,
    )

    OmegaConf.register_new_resolver(
        name="len",
        resolver=len,
        replace=True,
    )

    OmegaConf.register_new_resolver(
        name="randbits",
        resolver=secrets.randbits,
        replace=True,
    )
