# Change requests - property (to the common-model owner, the training-side owner and the project owner)

Raised while building the property SPEC_21 schema 2.0.0 (10 October 2026). Nothing here is applied to the common
model. Labels only; no document values.

**CR-PROP-01. Products the prop folder files that other entries read (project owner).**
1. Leatherstocking's `DECLARATION, Landlords Package` and `DECLARATION, Student Housing` (4 prop documents) are the
   product two homeowners seeds are (home folder), and `policy/leatherstocking/homeowners.yaml` routes and reads them
   as homeowners by its own signals. The corpus files the same product under two lines; no property gold was drafted
   for them, and their routing is unchanged. Decide the product's line; the entry and the golds follow.
2. North Country's `Commercial Special Multi-Peril Policy Declaration` prints property (Coverage A, B) and liability
   (Coverage L, M, N) on one page; `policy/north_country/commercial_package.yaml` reads it as a package, and packages
   are out of scope here. Its prop-folder copy keeps that reading.
3. Leatherstocking's Businessowner's Policy (4 documents) stays declined as before (CR-BOPPR-06); its property
   golds are drafted so the ruling can be checked. Its Commercial Fire and Liability declarations (1) is the
   owner's Commercial Fire (C-Fire) line (ruling of 10 October 2026, resolving CR-BOPPR-06 for that product):
   the `commercial_fire` schema and entry read it, and its property draft gold is superseded by that line's.

**CR-PROP-02. Shared liability coverage codes (with CR-BOPPR-01 and CR-BOPGL-01).**
The businessowners and fire-and-liability declarations print liability beside property. Until shared codes exist
property uses `PR_LIABILITY`, `PR_PRODUCTS_COMPLETED_OPS`, `PR_PERSONAL_INJURY` and `PR_FIRE_LEGAL_LIABILITY`.

**CR-PROP-03. Training parity (fine-tuning repository owner).**
The fine-tuning repository holds the legacy `property.json`. Decide with CR-BOPPR-02 for the property_pkg family.

**CR-PROP-04. Merchants businessowners declarations (registry owner).**
Merchants' businessowners declarations (filed under prop) has no property entry and escalates. An entry was not
built: its nine declarations pages are one document of one carrier; the gold is drafted for when a second document
of the form is available to measure on.
