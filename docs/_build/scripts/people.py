import itertools
from github import Github

g = Github(None)

ICON_TEMPLATE = "[![{login}]({avatar_url}){{.contributor_icon}}]({html_url})"


def get_people_md():
    repo = g.get_repo("pola-rs/polars")
    contributors = repo.get_contributors()
    with open("./docs/people.md", "w") as f:
        for c in itertools.islice(contributors, 50):
            # We love dependabot, but he doesn't need a spot on our website
            if c.login == "dependabot[bot]":
                continue

            f.write(
                ICON_TEMPLATE.format(
                    login=c.login,
                    avatar_url=c.avatar_url,
                    html_url=c.html_url,
                )
                + "\n"
            )


def on_startup(command, dirty):
    """Mkdocs hook to autogenerate docs/people.md on startup"""
    try:
        get_people_md()
    except Exception as e:
        msg = f"WARNING:{__file__}: Could not generate docs/people.md. Got error: {str(e)}"
        print(msg)


if __name__ == "__main__":
    get_people_md()
