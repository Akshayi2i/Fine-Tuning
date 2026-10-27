# Changelog - pl canonical schema

## 1.0.0 - 21 October 2026 (draft for the owner's approval; DECISIONS_2026-10-19 row 48)

- **New line.** Professional Liability (L1 code `pl`), separate from Errors & Omissions (`professional_eo`). Built from `professional_eo.json` 2.0.0: the same composition (envelope plus `claims_made_terms`), the same mandatory fields and audit rules, no LOB block.
- **What a PL declarations prints and where it goes:** per-claim and aggregate limits (`coverages[].limits[]`), retention or deductible (`coverages[].deductibles[]`), defense inside or outside the limits (`claims_made_terms.defense_within_limits`), prior-acts / retroactive date (`claims_made_terms.retroactive_date`, `pending_prior_litigation_date`, `continuity_date`).
- **Coverage codes:** `pl.coverage_codes.yaml`, 22 codes, the professional_eo table as of this date (built from the pl folder first; prefix PL_).
- **Crosswalk:** `pl.crosswalk.yaml`, every professional_eo leaf to the same pl path.
- **Seeds:** none yet; the pl folder's golds are under `seeds/professional_eo` (owner / seed reviewer to move them).
