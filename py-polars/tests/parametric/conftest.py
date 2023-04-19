import os

from hypothesis import settings


def load_hypothesis_profile() -> None:
    """Conditionally load different hypothesis profiles depending on env vars."""
    # Default profile (eg: running locally)
    common_settings = {"print_blob": True, "deadline": None}
    settings.register_profile(
        name="polars.default",
        max_examples=100,
        **common_settings,  # type: ignore[arg-type]
    )
    # CI 'max' profile (10x the number of iterations).
    # this is more expensive (about 4x slower), and not actually
    # enabled in our usual CI pipeline; requires explicit opt-in.
    settings.register_profile(
        name="polars.ci",
        max_examples=1000,
        **common_settings,  # type: ignore[arg-type]
    )
    if os.getenv("CI_MAX"):
        settings.load_profile("polars.ci")
    else:
        settings.load_profile("polars.default")


load_hypothesis_profile()
