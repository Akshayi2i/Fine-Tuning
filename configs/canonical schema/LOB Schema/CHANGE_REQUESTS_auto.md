# Change requests - auto (commercial auto; to the common-model owner and the training-side owner)

Raised while building the auto SPEC_21 schema 2.0.0 (10 October 2026). Nothing here is applied to the common
model: the schema composes `common_model.json` 1.1.0 unchanged, and each concept below stays in
`additional_fields` or in an auto coverage code until a request is accepted. Labels only; no document values.

| Request | Change to the common model, its files or another repository | Additive? | Prompt schemas that change |
|---|---|---|---|
| CR-AU-01 | training parity: the fine-tuning repository still has the legacy auto.json | n/a | none here |
| CR-AU-02 | shared codes for hired auto liability and non-ownership liability | yes | auto (codes renamed in a major version) |
| CR-AU-03 | a vehicle's rating class, radius and size: `Vehicle` fields or `RatingExposure.applies_to` | yes (optional properties) | every LOB that composes the block |
| CR-AU-04 | `CR-GL-16` (`Deductible.basis`): no second line's evidence from commercial auto | n/a | none |

**CR-AU-01. Training parity (fine-tuning repository owner).**
The fine-tuning repository has no SPEC_21 auto schema: its `configs/canonical schema/LOB Schema/auto.json` is the
pre-SPEC_21 self-contained file (`fideon.canonical.commercial_auto` 1.5.0) this schema replaces. As for gl
(CR-GL-01), its overlay-family check refuses a layout family only partly on the common model; gl was registered
natively at c8df4bb, so casualty_fleet is now half migrated (gl yes; auto, wc and umbrella not). Decide: take this
auto overlay (2.0.0, common model 1.1.0) with `auto.coverage_codes.yaml`, or keep auto on the legacy file until
wc and umbrella follow. Until then commercial auto is served by the base model (no casualty_fleet adapter exists).
The serving repository's fine-tuning parity tests (`fideon/vlm/tests/test_*parity*.py`) cover the seven personal
lines and gl only; auto is not among them.

**CR-AU-02. Hired auto and non-ownership liability as two coverages.**
`X_HIRED_NONOWNED_AUTO` (common model) is one coverage, as a CGL or BOP prints it ("Hired and Non-Owned Auto").
A business auto policy prints them as two coverages with two premiums: Item Four (hired or borrowed autos, rated
on cost of hire) and Item Five (non-ownership liability, rated on the number of employees or volunteers); all
five carriers' declarations carry both items. auto uses `AU_HIRED_AUTO_LIABILITY` and
`AU_NON_OWNERSHIP_LIABILITY` meanwhile. Proposal: either two shared codes (`X_HIRED_AUTO_LIABILITY`,
`X_NON_OWNED_AUTO_LIABILITY`) beside the combined one, or a note on `X_HIRED_NONOWNED_AUTO` that it is the
combined coverage only. Hired auto physical damage (`AU_HIRED_AUTO_PHYSICAL_DAMAGE`) stays LOB-local (one line).

**CR-AU-03. A scheduled auto's rating class, radius and size.**
Every business auto schedule of covered autos prints, per auto, an ISO rating class code (`CLASS: 23183`,
`Class Code: 790900`, `Size Class: 21189`, `CLASS CODE 003199`), a radius of operation (`RADIUS: L`, `Radius:
Local`, `0-50`, `Radius: 300 miles`) and a size or weight (`SIZE: 17000`, `GVW/Seats: 11400`). `Vehicle` has no
field for any of them, and `RatingExposure` (which has `class_code`) cannot name the vehicle it rates: it carries a
`location_ref` and no `applies_to`. The DRAFT golds hold them as `additional_fields` with `section_hint`
`vehicles[veh_N]` (labels as printed), so one label repeats once per auto and the overflow cannot be paired by
label. Proposal (one of): (a) `Vehicle.class_code`, `Vehicle.radius`, `Vehicle.gross_vehicle_weight`; (b)
`RatingExposure.applies_to` (unit ids), so a vehicle's class code is a rating exposure of that vehicle. Personal
auto prints comparable rating symbols (`vehicles[].symbols[]` exists) but not a class code.

**CR-AU-04. `Deductible.basis` (CR-GL-16, held).**
The owner held CR-GL-16 for a second line's evidence. None of the 11 commercial auto seeds prints a deductible
basis other than per covered auto (`$ 1,000 DEDUCTIBLE`, `For Each Covered Auto`), which is the deductible's
default meaning. No evidence for it from this line.
