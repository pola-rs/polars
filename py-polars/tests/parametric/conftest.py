import os
import re

from hypothesis import settings


def load_hypothesis_profile(profile: str = "polars.default") -> None:
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
    # Polars 'custom' profile, with 'n' iterations.
    if profile.isdigit() or re.match(r"polars\.custom\.\d+$", profile):
        n_iterations = int(profile.replace("polars.custom.", ""))
        profile = f"polars.custom.{profile}"
        settings.register_profile(
            name=profile,
            max_examples=n_iterations,
            **common_settings,  # type: ignore[arg-type]
        )

    profile = profile.replace("polars.", "")
    settings.load_profile(f"polars.{profile}")


load_hypothesis_profile(
    profile=os.environ.get("POLARS_HYPOTHESIS_PROFILE", "default"),
)
