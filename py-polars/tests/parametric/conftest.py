import os

from hypothesis import settings


def load_hypothesis_profile(name: str = "polars.default") -> None:
    """Conditionally load different hypothesis profiles."""
    common_settings = {"print_blob": True, "deadline": None}

    # Default profile (eg: running locally; quite fast)
    settings.register_profile(
        name="polars.default",
        max_examples=100,
        **common_settings,  # type: ignore[arg-type]
    )
    # Polars 'max' profile (10x the number of default iterations).
    # This is a bit more expensive (though not dramatically so
    # as it's only about 4-5x slower).
    settings.register_profile(
        name="polars.max",
        max_examples=1_000,
        **common_settings,  # type: ignore[arg-type]
    )
    # Polars 'ultra' profile (100x the number of default iterations).
    # This is expensive, at about 40x the default runtime.
    settings.register_profile(
        name="polars.ultra",
        max_examples=10_000,
        **common_settings,  # type: ignore[arg-type]
    )

    name = name.replace("polars.", "")
    settings.load_profile(f"polars.{name}")


load_hypothesis_profile(name=os.environ.get("POLARS_HYPOTHESIS_PROFILE", "default"))
