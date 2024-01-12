from collections import OrderedDict
import os
from typing import List, Optional, Set
import yaml
import logging


# Supported Languages and their metadata
LANGUAGES = OrderedDict(
    python={
        "extension": ".py",
        "display_name": "Python",
        "icon_name": "python",
        "code_name": "python",
    },
    rust={
        "extension": ".rs",
        "display_name": "Rust",
        "icon_name": "rust",
        "code_name": "rust",
    },
)

# Load all links to reference docs
with open("docs/_build/API_REFERENCE_LINKS.yml", "r") as f:
    API_REFERENCE_LINKS = yaml.load(f, Loader=yaml.CLoader)


def create_feature_flag_link(feature_name: str) -> str:
    """Create a feature flag warning telling the user to activate a certain feature before running the code

    Args:
        feature_name (str): name of the feature

    Returns:
        str: Markdown formatted string with a link and the feature flag message
    """
    return f'[:material-flag-plus:  Available on feature {feature_name}](/user-guide/installation/#feature-flags "To use this functionality enable the feature flag {feature_name}"){{.feature-flag}}'


def create_feature_flag_links(language: str, api_functions: List[str]) -> List[str]:
    """Generate markdown feature flags for the code tas based on the api_functions.
        It checks for the key feature_flag in the configuration yaml for the function and if it exists print out markdown

    Args:
        language (str): programming languages
        api_functions (List[str]): Api functions that are called

    Returns:
        List[str]: Per unique feature flag a markdown formatted string for the feature flag
    """
    api_functions_info = [
        info
        for f in api_functions
        if (info := API_REFERENCE_LINKS.get(language).get(f))
    ]
    feature_flags: Set[str] = {
        flag
        for info in api_functions_info
        if type(info) == dict and info.get("feature_flags")
        for flag in info.get("feature_flags")
    }

    return [create_feature_flag_link(flag) for flag in feature_flags]


def create_api_function_link(language: str, function_key: str) -> Optional[str]:
    """Create an API link in markdown with an icon of the YAML file

    Args:
        language (str): programming language
        function_key (str): Key to the specific function

    Returns:
        str: If the function is found than the link else None
    """
    info = API_REFERENCE_LINKS.get(language, {}).get(function_key)

    if info is None:
        logging.warning(f"Could not find {function_key} for language {language}")
        return None
    else:
        # Either be a direct link
        if type(info) == str:
            return f"[:material-api:  `{function_key}`]({info})"
        else:
            function_name = info["name"]
            link = info["link"]
            return f"[:material-api:  `{function_name}`]({link})"


def code_tab(
    base_path: str,
    section: Optional[str],
    language_info: dict,
    api_functions: List[str],
) -> str:
    """Generate a single tab for the code block corresponding to a specific language.
        It gets the code at base_path and possible section and pretty prints markdown for it

    Args:
        base_path (str): path where the code is located
        section (str, optional): section in the code that should be displayed
        language_info (dict): Language specific information (icon name, display name, ...)
        api_functions (List[str]): List of api functions which should be linked

    Returns:
        str: A markdown formatted string represented a single tab
    """
    language = language_info["code_name"]

    # Create feature flags
    feature_flags_links = create_feature_flag_links(language, api_functions)

    # Create API Links if they are defined in the YAML
    api_functions = [
        link for f in api_functions if (link := create_api_function_link(language, f))
    ]
    language_headers = " ·".join(api_functions + feature_flags_links)

    # Create path for Snippets extension
    snippets_file_name = f"{base_path}:{section}" if section else f"{base_path}"

    # See Content Tabs for details https://squidfunk.github.io/mkdocs-material/reference/content-tabs/
    return f"""=== \":fontawesome-brands-{language_info['icon_name']}: {language_info['display_name']}\"
    {language_headers}
    ```{language}
    --8<-- \"{snippets_file_name}\"
    ```
    """


def define_env(env):
    @env.macro
    def code_block(
        path: str, section: str = None, api_functions: List[str] = None
    ) -> str:
        """Dynamically generate a code block for the code located under {language}/path

        Args:
            path (str): base_path for each language
            section (str, optional): Optional segment within the code file. Defaults to None.
            api_functions (List[str], optional): API functions that should be linked. Defaults to None.
        Returns:
            str: Markdown tabbed code block with possible links to api functions and feature flags
        """
        result = []

        for language, info in LANGUAGES.items():
            base_path = f"{language}/{path}{info['extension']}"
            full_path = "docs/src/" + base_path
            # Check if file exists for the language
            if os.path.exists(full_path):
                result.append(code_tab(base_path, section, info, api_functions))

        return "\n".join(result)
