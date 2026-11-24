# Prompts: see `testing/prompts/`

The prompts the model is sent are generated from the code, never kept by hand:

```
python -m testing.render_prompts --write   # regenerate testing/prompts/
python -m testing.render_prompts --check   # CI: fails when a file differs from what the code renders
```

`testing/prompts/` holds every window prompt (`policy_<line>.<group>[.image_only].prompt.txt`)
of every line trained or served, in both reading modes.

The hand-made copies that were in this folder (GL, standard format, an older schema version)
were removed on 24 November 2026 (review P7): nothing regenerated them, so they no longer
matched what is sent. They are in git history before that date.
