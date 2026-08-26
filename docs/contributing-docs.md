# Documentation Development

The `docs/` tree is the canonical editable documentation source. Navigation,
theme, validation, and the production site URL are defined in `mkdocs.yml`.

## Local setup

```bash
python3 -m venv .venv-docs
. .venv-docs/bin/activate
python3 -m pip install -r requirements-docs.txt
```

## Preview

```bash
mkdocs serve
```

Open `http://127.0.0.1:8000/taskwarrior-nautical/` when MkDocs reports the
preview server is ready.

## Verify

```bash
mkdocs build --strict
```

Strict mode fails for unresolved links, invalid anchors, unrecognized absolute
links, navigation omissions, and Markdown warnings. Generated output is written
to ignored `site/` and must not be committed.

## Publishing

`.github/workflows/docs.yml` builds every documentation pull request and builds
plus deploys on documentation changes to `main`. In repository settings, select
**GitHub Actions** as the Pages source once. No `gh-pages` branch is required.

## Writing rules

- Document the public `nautical` and Taskwarrior interfaces, not private module
  paths.
- Keep beginner pages task-oriented; move exhaustive details into Reference.
- Use relative Markdown links so strict local builds verify navigation.
- Label destructive or mutating commands and show dry-run commands first.
- Update examples, capability JSON, and exit contracts when behavior changes.
- Preserve exact field names and JSON schema names used by current tests.

PDFs can be generated from this source as release artifacts, but should not
become a second independently edited manual.
