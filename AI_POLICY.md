# AI Usage Policy

The Polars project has strict rules for AI usage:

- **Agents are strictly forbidden from interacting with our repository.** This includes but is not
  limited to posting issues, making pull requests, posting comments, posting code reviews, adding
  reactions, etc. Any interaction on our repository must be done by a human.

- **All AI usage in any form must be disclosed.** You must state the tool(s) you used (e.g. Claude
  Code, Cursor, Amp) along with the extent that the work was AI-assisted.

- **Pull requests with AI-generated code can only be for accepted issues.** Drive-by pull requests
  that do not reference an accepted issue will be closed. If AI isn't disclosed but a maintainer
  suspects its use, the PR will be closed. If you want to share code for a non-accepted issue, open
  a discussion or attach it to an existing discussion.

- **AI may not be used for issues marked as "good first issue".** These issues are intended for new
  contributors to learn the codebase, and are thus wasted if AI solves them.

- **AI-generated code must have been fully verified with human use.** AI must not create
  hypothetically correct code that hasn't been tested. Importantly, you must not allow AI to write
  code for platforms or environments you don't have access to manually test on.

- **Issues and discussions can use AI assistance but must have a full human-in-the-loop.** This
  means that any content generated with AI must have been reviewed _and edited_ by a human before
  submission. AI is very good at being overly verbose and including noise that distracts from the
  main point. Humans must do their research and trim this down.

- **No AI-generated media or text (other than code) is allowed (art, images, videos, audio, etc.).**
  Code (docstrings included) is the only acceptable AI-generated content, per the other rules in
  this policy.

- **Code comments must be relevant for the current code, and concise.**
  If you use AI to generate code, any comments added must be correct and relevant for the *current*
  state of the code. You must manually remove or edit any comments which describe changes to the
  previous state of the code, or which are overly verbose or trivial. AI tends to leave such
  comments describing what it did or did not do, even though it is not relevant for future readers
  whatsoever, or already obvious from the code.

## Rationale

We have established these rules not because we are against AI usage, but because we have experienced
an increasing number of low-quality pull requests that consume significant amounts of human
reviewers' time. Poor usage of AI can generate tremendous amounts of low-quality code and "slop"
PRs, which can have a net-negative effect on the maintenance of Polars. We hope to bring this under
control by applying stricter AI rules, where AI can be used as a tool without overwhelming the
project maintainers.
