# Change requests - pl canonical schema

- **CR-PLL-01** (open): split the coverage-code tables. `pl` and `professional_eo` carry the same PL_-prefixed codes as of 1.0.0; the owner decides each code's line and the prefix of each.
- **CR-PLL-02** (open): `fideon:lob_code` (the batch code) for pl.
- **CR-PLL-03** (open): move the pl folder's seed golds from `seeds/professional_eo` to `seeds/pl` (paths unchanged, `pl.crosswalk.yaml`), `lob_parts[].lob` -> `pl`.
