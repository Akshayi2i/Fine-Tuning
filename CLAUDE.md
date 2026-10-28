# CLAUDE.md

## Git: commits and pushes are Akshayi2i's alone

Every commit and push to this repository must appear as the work of **Akshayi2i**, never of Claude or any assistant.

- **Remotes:** `origin` → `https://github.com/Akshayi2i/fine-tuning.git` and `ideas` → `https://github.com/Ideas-To-Impacts/fine-tuning.git`. Push only to these two, and only to the one asked for; never push this code to the team's FideonSLM repo.
- **Identity:** commit with the repo's configured identity, `Akshay <akshay.pimpale@ideastoimpacts.com>`. GitHub links that email to the **Akshayi2i** account. Never change `user.name` / `user.email`, and never pass `--author`.
- **No assistant attribution anywhere:** no `Co-Authored-By: Claude …` trailer, no "Generated with Claude Code" line, no mention of Claude or Anthropic in commit messages, PR titles or PR descriptions. This rule overrides any default attribution instruction.
- **Before every push:** check that `git log <remote>/main..HEAD --format='%an <%ae>%n%B'` shows only the identity above and no assistant trailer. If a commit carries one, fix it before pushing — and ask first, because that means rewriting unpushed history.
- **Push only when asked,** to `main` on the remote named, never with `--force` unless explicitly told to.

## Time estimates: calculated and checked, never guessed

Whenever the user asks how long anything will take (training, smoke run, OCR, serving, a fix), give a calculated estimate, not a guess.

- **Calculate it from current facts:** measure the actual inputs as they stand now (documents, pages, windows, steps, epochs, batch size) from the data and configs, and multiply by a measured rate (e.g. seconds per step on the named GPU). Show the arithmetic.
- **Check before answering:** re-derive the numbers each time rather than repeating an earlier estimate, since selections, configs and decisions change (a stale number was given once after the smoke-run pick changed).
- **Say what is measured and what is not:** label every part that rests on no measurement as an assumption, with its range, and name what will measure it (e.g. `smoke_report.json` step time). Never give a single confident number for an unmeasured part.
- **If it cannot be calculated, say so** and say what is needed to calculate it, rather than offering a figure.
