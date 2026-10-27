# gl - core fields (SPEC_21 4.3)

Core field list of the gl (commercial general liability) SPEC_21 schema. Seed golds are not signed off yet,
so the list is built from what the line's real CGL documents print -- 49 documents of 10 carriers under
`synth_core/fideon/synth_core/data/source data/<Carrier>/cgl/` and, from 1.1.0, 50 CGL documents of 10
carrier folders of the `policy_CGL` corpus (`policy_CGL/<Carrier>/CGL/`), read in place (text layer, first 8
pages of a seed, first 20 of a `policy_CGL` document, whose CGL declarations can follow a long renewal packet) -- every field of the pre-SPEC_21 `gl.json` 1.5.0 (`gl.crosswalk.yaml`) and the line's L1 registry YAMLs.
One file of the seed folder (Chubb's) is a personal excess liability policy and is not counted. Of `policy_CGL`,
the Businessowners (Coterie `dec_01`), professional liability (Coterie `dec_02`/`renewal_01`, Coterie Insurance
`dec_02`) and D&O (Philadelphia `policy_01`) policies, paperwork (invoices, notices, letters), text-less scans and the
unreadable XS Broker `policy_02` are not counted. Carriers marked `(policy_CGL)` are carrier or broker folders of that corpus.

| Item | Value |
|---|---|
| Schema | `gl.json` 3.1.0, composing `common_model.json` 1.1.0 by `$ref` |
| Core leaves | 172 (budget 80-300; `python -m fideon.synth_core.tools leaves`) |
| Leaves whose label the documents print | 97 of the 172 |
| Documents | 105 CGL documents: 49 seed documents of 10 carriers and 56 `policy_CGL` documents of 13 folders (FORTEGRA, Great American, Hiscox, Indium, Johnson & Johnson, Liberty Mutual, McNeil & Company, Philadelphia Insurance Companies, Russell Bond, XS Broker; from 1.2.0 The Main Street America Group, Scottsdale Insurance Company, USLI, and a Utica First renewal). Seed carriers: Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| Blocks besides the envelope | `locations`, `rating_exposures`, `claims_made_terms`, `countersignature` |
| LOB block | none (the 1.1.0-1.2.0 `gl` block was removed in 2.0.0; its concepts are common-model fields from 3.0.0, see below) |
| Coverage codes | 20 (`gl.coverage_codes.yaml`), of them 6 shared `X_` codes; 3.1.0 adds `GL_SUPPLEMENTARY_PAYMENTS`, `GL_OWNERS_CONTRACTORS_PROTECTIVE` |
| Mandatory fields | `carrier.name`, `named_insured.primary_name`, `policy.policy_number`, `policy.effective_date`, `policy.expiration_date`, `coverages[].limits[].amount` |

Field names, meanings and shapes are fixed by the common model; this line chooses the blocks, the coverage
codes and the printed labels below. **Printed labels (this line)** are captions the CGL documents print
beside or above the value that the common model does not already list; they are the schema's `fideon:aliases`
for the field. **Carriers** lists the carriers whose documents print any label of the field (this line's or
the common model's) in their first 8 pages: a label scan, not a gold count -- a carrier listed prints the
caption, which says nothing about whether every document of it fills the field. One-word labels of under six
letters are left out of the scan (they match running text), and an address part counts its address's caption.
Coverage names and limit
captions are aliases of the coverage codes, not of `coverages[].coverage_name`, and are listed in the
coverage codes file.

**Endorsement-coverages.** Additional insured, primary and noncontributory, waiver of subrogation, coverage
extension and snow plow operations are coverages granted by an endorsement. Each endorsement is one `coverages[]`
entry: `coverage_name` is the form's printed title and `form_refs` its form number as printed in
`forms_and_endorsements[].form_number` (three additional-insured forms are three entries). They usually carry no
limit, which the mandatory field and rule 5 allow: both mean that at least one coverage of the document has a limit
amount.

There is no LOB block (SPEC_21 v0.9 4.2.4). Three concepts the CGL documents print needed a field in an existing
common block, and common model 1.1.0 added them (3.0.0 uses them):
- the products-completed operations rate and premium of each classification (CR-GL-02). An ISO premium schedule
  prints `Pr/Co` and `All Other` columns per classification: each column is one `rating_exposures[]` row with the
  same `class_code`, `class_description` and `location_ref`, `rating_component` `products_completed_operations` or
  `premises_operations`, and that column's `rate` and `premium`. A classification with one rate has one row and
  no `rating_component`;
- the minimum earned premium as a percentage, `premium.minimum_earned_percent` (CR-GL-12). An amount stays in
  `premium.minimum_earned`;
- the carrier's financial rating, `carrier.financial_rating` (CR-GL-12).

gl 1.1.0-1.2.0 held them in a `gl` block and 2.0.0 in `additional_fields`. The additional insured, primary and
noncontributory and waiver of subrogation codes are the shared `X_` codes from 3.0.0 (CR-GL-03).

`rating_modifiers` (listed for gl in `lob_schema_map.yaml`) is not composed: none of the documents prints an
IRPM, schedule-rating or experience factor. `claims_made_terms` is composed although the map does not list it:
the legacy schema carries claims-made provisions and a CGL claims-made coverage part prints a retroactive date
(Gemini's declarations do).

## Core fields

| Field | Meaning | Printed labels (this line) | Carriers whose documents print a label of it |
|---|---|---|---|
| `document.transaction_type` | Kind of transaction the document records. | Declaration Type | Security Mutual Insurance Company, Utica First Insurance Company |
| `document.transaction_date` | Date the transaction was processed. | - | no label found in the documents read |
| `document.transaction_effective_date` | Date an endorsement, cancellation or reinstatement takes effect. | - | no label found in the documents read |
| `document.transaction_reason` | Reason for the transaction, as printed. | - | no label found in the documents read |
| `document.endorsement_number` | Number of the endorsement or policy change. | - | Diamond State Insurance Company, Western World Insurance Group |
| `document.title` | Document title as printed, e.g. 'Homeowners Policy Declarations'. | - | no label found in the documents read |
| `document.issue_date` | Date the document was issued. | Issued; ISSUE DATE | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Travelers, USLI, Utica First Insurance Company |
| `document.print_date` | Date the document was printed. | - | Utica First Insurance Company |
| `document.mailing_date` | Date the document was mailed; it starts notice periods on cancellation and nonrenewal notices. | - | no label found in the documents read |
| `carrier.name` | The writing company. | Insurer; Insurance Company; Insuring Company; Company | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group, FORTEGRA (policy_CGL), Great American (policy_CGL), Hiscox (policy_CGL), Indium (policy_CGL), Johnson & Johnson (policy_CGL), Liberty Mutual (policy_CGL), McNeil & Company (policy_CGL), Philadelphia Insurance Companies (policy_CGL), Russell Bond (policy_CGL), XS Broker (policy_CGL) |
| `carrier.naic_code` | NAIC company code of the writing company. | - | no label found in the documents read |
| `carrier.group_name` | Group or trade name, when printed separately from the writing company. | - | no label found in the documents read |
| `carrier.address.street` | Street address line. | - | Gemini Insurance Company, Security Mutual Insurance Company, Utica First Insurance Company, Western World Insurance Group |
| `carrier.address.street_2` | Second address line: suite, unit, attention line. | - | Gemini Insurance Company, Security Mutual Insurance Company, Utica First Insurance Company, Western World Insurance Group |
| `carrier.address.city` | City. | - | Gemini Insurance Company, Security Mutual Insurance Company, Utica First Insurance Company, Western World Insurance Group |
| `carrier.address.state` | State, two-letter USPS code in parsed. | - | Gemini Insurance Company, Security Mutual Insurance Company, Utica First Insurance Company, Western World Insurance Group |
| `carrier.address.postal_code` | ZIP code. | - | Gemini Insurance Company, Security Mutual Insurance Company, Utica First Insurance Company, Western World Insurance Group |
| `carrier.address.county` | County. | - | Gemini Insurance Company, Security Mutual Insurance Company, Utica First Insurance Company, Western World Insurance Group |
| `carrier.phone` | Main phone number of the carrier. | - | Diamond State Insurance Company, Utica First Insurance Company, Western World Insurance Group |
| `carrier.fax` | Fax number. | - | no label found in the documents read |
| `carrier.web_address` | Website address. | - | no label found in the documents read |
| `carrier.claims_phone` | Phone number for reporting a claim. | Report Claims Immediately by Calling; Report a Claim | Diamond State Insurance Company, Travelers, USLI, Western World Insurance Group |
| `carrier.claims_email` | Address claims are reported to, when printed. | - | no label found in the documents read |
| `carrier.claims_address.street` | Street address line. | - | Diamond State Insurance Company, Gemini Insurance Company |
| `carrier.claims_address.street_2` | Second address line: suite, unit, attention line. | - | Diamond State Insurance Company, Gemini Insurance Company |
| `carrier.claims_address.city` | City. | - | Diamond State Insurance Company, Gemini Insurance Company |
| `carrier.claims_address.state` | State, two-letter USPS code in parsed. | - | Diamond State Insurance Company, Gemini Insurance Company |
| `carrier.claims_address.postal_code` | ZIP code. | - | Diamond State Insurance Company, Gemini Insurance Company |
| `carrier.claims_address.county` | County. | - | Diamond State Insurance Company, Gemini Insurance Company |
| `carrier.admitted_status` | Whether the writing company is admitted or non-admitted in the state. | - | no label found in the documents read |
| `carrier.financial_rating` | Financial-strength rating of the writing company as printed, e.g. the A.M. Best rating and size class 'A+ XV'. | - | Security Mutual Insurance Company, Johnson & Johnson (policy_CGL), Philadelphia Insurance Companies (policy_CGL), XS Broker (policy_CGL) |
| `producer.agency_name` | Name of the agency or broker of record. | Agent; Agents Name and Address; Producer Name; Servicing Agent; Broker; Agent or Broker | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Travelers, USLI, Western World Insurance Group |
| `producer.producer_code` | Code the carrier assigns to the agency. | Producer Number; Agent/Broker # | Diamond State Insurance Company, Gemini Insurance Company, Western World Insurance Group |
| `producer.contact_name` | Contact person at the agency. | - | no label found in the documents read |
| `producer.address.street` | Street address line. | - | no label found in the documents read |
| `producer.address.street_2` | Second address line: suite, unit, attention line. | - | no label found in the documents read |
| `producer.address.city` | City. | - | no label found in the documents read |
| `producer.address.state` | State, two-letter USPS code in parsed. | - | no label found in the documents read |
| `producer.address.postal_code` | ZIP code. | - | no label found in the documents read |
| `producer.address.county` | County. | - | no label found in the documents read |
| `producer.phone` | Phone number of the agency. | - | Diamond State Insurance Company, Utica First Insurance Company, Western World Insurance Group |
| `producer.fax` | Fax number. | - | no label found in the documents read |
| `producer.email` | Email address. | - | no label found in the documents read |
| `producer.web_address` | Website of the agency. | - | no label found in the documents read |
| `producer.contract_number` | The agency's contract or sub-code with the carrier. | - | Diamond State Insurance Company, Gemini Insurance Company, Sutton National, Third Coast Insurance Company, USLI, Utica First Insurance Company |
| `named_insured.primary_name` | First named insured. | Named Insured and Mailing Address; Named Insured and Address | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `named_insured.additional_named_insureds[].name` | Name of the additional named insured. | - | no label found in the documents read |
| `named_insured.additional_named_insureds[].relationship` | As printed, e.g. 'Spouse', 'Subsidiary'. | - | no label found in the documents read |
| `named_insured.additional_named_insureds[].date_of_birth` | Date of birth. | - | Security Mutual Insurance Company |
| `named_insured.additional_named_insureds[].gender` | Gender as printed. | - | no label found in the documents read |
| `named_insured.additional_named_insureds[].marital_status` | Marital status as printed. | - | no label found in the documents read |
| `named_insured.additional_named_insureds[].occupation` | Occupation as printed. | - | Utica First Insurance Company |
| `named_insured.doing_business_as` | Trade name the insured does business as. | - | no label found in the documents read |
| `named_insured.entity_type` | As printed: Individual, Corporation, LLC, Partnership, Trust... | - | Diamond State Insurance Company, Travelers, USLI, Hiscox (policy_CGL), Indium (policy_CGL), Johnson & Johnson (policy_CGL), McNeil & Company (policy_CGL), Philadelphia Insurance Companies (policy_CGL), XS Broker (policy_CGL) |
| `named_insured.fein` | Federal employer identification number. | - | no label found in the documents read |
| `named_insured.business_description` | Nature of the insured's business or operations. | Business; Class Description | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `named_insured.mailing_address.street` | Street address line. | - | Diamond State Insurance Company, Gemini Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `named_insured.mailing_address.street_2` | Second address line: suite, unit, attention line. | - | Diamond State Insurance Company, Gemini Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `named_insured.mailing_address.city` | City. | - | Diamond State Insurance Company, Gemini Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `named_insured.mailing_address.state` | State, two-letter USPS code in parsed. | - | Diamond State Insurance Company, Gemini Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `named_insured.mailing_address.postal_code` | ZIP code. | - | Diamond State Insurance Company, Gemini Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `named_insured.mailing_address.county` | County. | - | Diamond State Insurance Company, Gemini Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `named_insured.phone` | Phone number of the insured. | - | no label found in the documents read |
| `named_insured.email` | Email address of the insured. | - | no label found in the documents read |
| `named_insured.date_of_birth` | Date of birth. | - | Security Mutual Insurance Company |
| `named_insured.gender` | Gender as printed. | - | no label found in the documents read |
| `named_insured.marital_status` | Marital status as printed. | - | no label found in the documents read |
| `named_insured.occupation` | Occupation as printed. | - | Utica First Insurance Company |
| `policy.policy_number` | Policy number. | Policy No.; Master Policy Number | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `policy.prior_policy_number` | Number of the policy this one renews or replaces. | Renewal of Number; Prior Policy Number | Diamond State Insurance Company, USLI, Western World Insurance Group |
| `policy.certificate_number` | Certificate number, when the policy is issued as a certificate under a master policy. | - | Great American Insurance Company |
| `policy.policy_type` | Product, program or policy form name as printed (e.g. 'Homeowners HO-3', 'Auto Special'). | - | no label found in the documents read |
| `policy.original_inception_date` | Date the insured first became a policyholder with the carrier. | - | no label found in the documents read |
| `policy.effective_date` | Date coverage starts. | Policy Period; Coverage Period; From | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `policy.expiration_date` | Date coverage ends. | To | Diamond State Insurance Company, Travelers, USLI |
| `policy.term_months` | Term length in months; 'Annual' -> 12. | - | Diamond State Insurance Company, Security Mutual Insurance Company, USLI |
| `policy.coverage_trigger` | What triggers coverage: occurrence or claims made. | Occurrence Form; Claims Made and Reported Policy | Gemini Insurance Company, Travelers |
| `policy.rating_state` | State the policy is rated in. | - | no label found in the documents read |
| `policy.audit_period` | As printed: Annual, Semi-Annual, Quarterly, Monthly, Non-Auditable. | Audit Period (if applicable) | Hiscox (policy_CGL), Johnson & Johnson (policy_CGL), McNeil & Company (policy_CGL), Philadelphia Insurance Companies (policy_CGL), XS Broker (policy_CGL) |
| `policy.subject_to_audit` | Whether the premium is subject to audit. | Subject to Audit; This Policy is Subject to Audit; Auditable; Non-Auditable | Sutton National, Third Coast Insurance Company, Travelers, Indium (policy_CGL), Johnson & Johnson (policy_CGL), Liberty Mutual (policy_CGL), XS Broker (policy_CGL) |
| `lob_parts[].title` | Title of the coverage part as printed. | - | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `lob_parts[].coverage_part_form` | Coverage-part declarations form number. | - | no label found in the documents read |
| `lob_parts[].premium` | Premium for the part. | - | Sutton National, Travelers |
| `coverages[].coverage_name` | As printed. | - | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `coverages[].included` | Whether the coverage is included, excluded or not purchased. | - | no label found in the documents read |
| `coverages[].limits[].amount` | Amount of the limit. | - | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Travelers, Utica First Insurance Company |
| `coverages[].limits[].percentage` | When the limit is stated as a percentage of another coverage. | - | no label found in the documents read |
| `coverages[].limits[].description` | Printed qualifier, e.g. 'theft of jewelry'. Required in practice for sublimits. | Any One Premises; Any One Fire; Any One Person | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Travelers, Western World Insurance Group |
| `coverages[].deductibles[].amount` | Amount of a flat deductible. | - | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Travelers, USLI, Utica First Insurance Company |
| `coverages[].deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | no label found in the documents read |
| `coverages[].deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | no label found in the documents read |
| `coverages[].deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | Diamond State Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Utica First Insurance Company |
| `coverages[].premium` | Premium for the coverage. | - | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `coverages[].valuation` | Valuation basis. | - | Security Mutual Insurance Company, Utica First Insurance Company |
| `coverages[].coinsurance_percent` | Coinsurance percentage that applies to the coverage. | - | no label found in the documents read |
| `coverages[].covered_auto_symbols[]` | ISO covered-auto symbols printed against the coverage (1, 2, 7, 8, 9...). Business auto, garage, truckers and package auto parts. | - | Travelers |
| `deductibles[].amount` | Amount of a flat deductible. | Liability Deductible; Property Damage Deductible; Self-Insured Retention (Per-Occurrence); Deductible - per occurrence or offense; Deductible Limit of Liability | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Johnson & Johnson (policy_CGL), XS Broker (policy_CGL) |
| `deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | no label found in the documents read |
| `deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | no label found in the documents read |
| `deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | Diamond State Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Utica First Insurance Company |
| `interested_parties[].rank` | 1 for first mortgagee, 2 for second... | - | no label found in the documents read |
| `interested_parties[].name` | Name of the interested party. | - | Diamond State Insurance Company, Gemini Insurance Company, Third Coast Insurance Company, Utica First Insurance Company |
| `interested_parties[].address.street` | Street address line. | - | no label found in the documents read |
| `interested_parties[].address.street_2` | Second address line: suite, unit, attention line. | - | no label found in the documents read |
| `interested_parties[].address.city` | City. | - | no label found in the documents read |
| `interested_parties[].address.state` | State, two-letter USPS code in parsed. | - | no label found in the documents read |
| `interested_parties[].address.postal_code` | ZIP code. | - | no label found in the documents read |
| `interested_parties[].address.county` | County. | - | no label found in the documents read |
| `interested_parties[].loan_number` | Loan or account number with the interested party. | - | Security Mutual Insurance Company |
| `interested_parties[].is_payor` | True when this party pays the premium (escrow billing). | - | no label found in the documents read |
| `premium.total` | Total premium for the policy. | Total Advance Premium; Total Premium (If Applicable); Total; Total Premium for Liquor Liability Coverage Part | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, USLI, Utica First Insurance Company, Western World Insurance Group |
| `premium.deposit` | Deposit premium due at inception. | Minimum & Deposit Premium; Coverage Part Minimum & Deposit Premium | Sutton National, Third Coast Insurance Company |
| `premium.minimum` | The least premium the policy (or part) is written for. | Minimum Premium for Liquor Liability Coverage Part; MP - minimum premium | Diamond State Insurance Company, Sutton National, Third Coast Insurance Company, USLI |
| `premium.minimum_earned` | The least premium kept on cancellation, as an amount; not the same as `minimum`. A percentage goes to `minimum_earned_percent`. | Minimum Retained Premium; Minimum Earned Premium at Inception | Diamond State Insurance Company, Gemini Insurance Company, Utica First Insurance Company |
| `premium.minimum_earned_percent` | The least premium kept on cancellation, printed as a percentage of the premium (e.g. 'Minimum Earned Premium: 25%'). An amount goes to `minimum_earned`. | - | Gemini Insurance Company, Johnson & Johnson (policy_CGL), XS Broker (policy_CGL) |
| `premium.change` | Premium change made by this transaction; negative for a return premium. | - | Sutton National, Third Coast Insurance Company, Utica First Insurance Company |
| `premium.items[].description` | Description of the premium line as printed. | - | no label found in the documents read |
| `premium.items[].amount` | Premium amount of the line. | Liability Premium; Advance Premium | Diamond State Insurance Company, Security Mutual Insurance Company, USLI, Utica First Insurance Company, Western World Insurance Group |
| `premium.taxes_fees[].description` | Name of the tax, fee, surcharge or discount as printed. | - | no label found in the documents read |
| `premium.taxes_fees[].amount` | Amount of the tax, fee, surcharge or discount. | Taxes/Surcharges/Fees | Diamond State Insurance Company |
| `premium.taxes_fees[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | no label found in the documents read |
| `premium.taxes_fees[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | Western World Insurance Group |
| `premium.surcharges[].description` | Name of the tax, fee, surcharge or discount as printed. | - | no label found in the documents read |
| `premium.surcharges[].amount` | Amount of the tax, fee, surcharge or discount. | - | no label found in the documents read |
| `premium.surcharges[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | no label found in the documents read |
| `premium.surcharges[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | Western World Insurance Group |
| `premium.discounts[].description` | Name of the tax, fee, surcharge or discount as printed. | - | no label found in the documents read |
| `premium.discounts[].amount` | Amount of the tax, fee, surcharge or discount. | - | no label found in the documents read |
| `premium.discounts[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | no label found in the documents read |
| `premium.discounts[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | Western World Insurance Group |
| `billing.payment_plan` | Payment plan as printed (annual, 2-pay, 12-pay EFT...). | Billing Type; Premium Payment Method; Premium shown is payable | Diamond State Insurance Company, Security Mutual Insurance Company, Utica First Insurance Company |
| `billing.bill_type` | As printed: Direct Bill, Agency Bill. | - | Utica First Insurance Company |
| `billing.payment_method` | As printed: EFT, recurring card, check... | - | Security Mutual Insurance Company |
| `billing.account_number` | Billing account number the carrier bills under. | - | Security Mutual Insurance Company |
| `billing.amount_due` | Amount currently due on the bill. | - | Sutton National, Third Coast Insurance Company |
| `billing.due_date` | Date payment is due. | - | no label found in the documents read |
| `billing.installments[].installment_number` | Number of the installment in the payment plan. | - | no label found in the documents read |
| `billing.installments[].due_date` | Date the installment is due. | - | no label found in the documents read |
| `billing.installments[].amount` | Amount of the installment. | - | no label found in the documents read |
| `forms_and_endorsements[].form_number` | Form number as printed, e.g. 'HO 00 03'. | Form #/Edition Date | Diamond State Insurance Company, Gemini Insurance Company |
| `forms_and_endorsements[].edition_date` | Edition date of the form as printed (often month and year only, e.g. '05 11'). | - | Diamond State Insurance Company, Western World Insurance Group |
| `forms_and_endorsements[].title` | Title of the form or endorsement. | - | Diamond State Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `forms_and_endorsements[].premium` | Premium charged for the form or endorsement. | - | no label found in the documents read |
| `locations[].location_number` | Location number as printed on the schedule. | Prem. Loc. No. | Diamond State Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `locations[].address.street` | Street address line. | - | Diamond State Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Western World Insurance Group |
| `locations[].address.street_2` | Second address line: suite, unit, attention line. | - | Diamond State Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Western World Insurance Group |
| `locations[].address.city` | City. | - | Diamond State Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Western World Insurance Group |
| `locations[].address.state` | State, two-letter USPS code in parsed. | - | Diamond State Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Western World Insurance Group |
| `locations[].address.postal_code` | ZIP code. | - | Diamond State Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Western World Insurance Group |
| `locations[].address.county` | County. | - | Diamond State Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Western World Insurance Group |
| `locations[].territory` | Rating territory or zone. | - | Gemini Insurance Company, USLI, Utica First Insurance Company |
| `locations[].protection_class` | Fire protection class of the premises. | - | Security Mutual Insurance Company, Travelers, USLI |
| `locations[].fire_district` | Fire district or fire protection area the premises fall in. | - | no label found in the documents read |
| `locations[].insured_interest` | The insured's interest in the premises: Owner, Deeded owner, Tenant, LLC member... | - | no label found in the documents read |
| `locations[].distance_to_fire_station` | Distance to the responding fire station. | - | no label found in the documents read |
| `locations[].distance_to_hydrant` | Distance to the nearest fire hydrant. | - | no label found in the documents read |
| `rating_exposures[].state` | State of the exposure, two-letter USPS code in parsed. | - | no label found in the documents read |
| `rating_exposures[].class_code` | Rating classification code (GL, WC or auto class). | Code No.; Classification Code No.; Classification Code; Class Code | Diamond State Insurance Company, USLI, Western World Insurance Group, Indium (policy_CGL), Johnson & Johnson (policy_CGL), McNeil & Company (policy_CGL), Philadelphia Insurance Companies (policy_CGL), XS Broker (policy_CGL) |
| `rating_exposures[].class_description` | Description of the rating classification. | Classification; Class Description; Description of Operations / Classification | Diamond State Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, USLI, Utica First Insurance Company, Western World Insurance Group |
| `rating_exposures[].exposure_basis` | Payroll, sales, units, area, receipts... | Exposure Basis | Diamond State Insurance Company, USLI, Western World Insurance Group, Johnson & Johnson (policy_CGL) |
| `rating_exposures[].exposure_amount` | Rating exposure (payroll, sales, units, area or receipts). | Premium Basis; Exposure | Diamond State Insurance Company, Great American Insurance Company, USLI, Utica First Insurance Company, Western World Insurance Group, Great American (policy_CGL), Hiscox (policy_CGL), Indium (policy_CGL), Johnson & Johnson (policy_CGL), Liberty Mutual (policy_CGL), McNeil & Company (policy_CGL), Philadelphia Insurance Companies (policy_CGL), Russell Bond (policy_CGL), XS Broker (policy_CGL) |
| `rating_exposures[].rate` | Rate applied to the exposure. | All Other; Pr/Co; Prod/Ops Rate; Prod./Comp. Ops | Diamond State Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group, Indium (policy_CGL), Johnson & Johnson (policy_CGL), Liberty Mutual (policy_CGL), McNeil & Company (policy_CGL), Philadelphia Insurance Companies (policy_CGL), XS Broker (policy_CGL) |
| `rating_exposures[].rate_basis` | Per $100 of payroll, per $1,000 of sales... | - | no label found in the documents read |
| `rating_exposures[].premium` | Premium for the exposure line. | Advance Premium; Products-Completed Operations Premium | Diamond State Insurance Company, Gemini Insurance Company, Great American Insurance Company, Security Mutual Insurance Company, Sutton National, Third Coast Insurance Company, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `rating_exposures[].rating_component` | Which printed rating column the row holds: `premises_operations` ('All Other') or `products_completed_operations` ('Pr/Co'). Omitted when the classification has one rate. Not a printed value, so not a core leaf. | - | Diamond State Insurance Company, USLI, Indium (policy_CGL), Johnson & Johnson (policy_CGL), Liberty Mutual (policy_CGL), McNeil & Company (policy_CGL), Philadelphia Insurance Companies (policy_CGL), XS Broker (policy_CGL) |
| `claims_made_terms.retroactive_date` | Retroactive date for claims-made coverage. | Retroactive Date, if any, shown here; Retroactive Date (CG 00 02 Only); Retroactive Date, if any, shown below | Gemini Insurance Company, Indium (policy_CGL), Johnson & Johnson (policy_CGL), McNeil & Company (policy_CGL) |
| `claims_made_terms.pending_prior_litigation_date` | Pending or prior litigation date for claims-made coverage. | - | no label found in the documents read |
| `claims_made_terms.continuity_date` | Continuity or prior-knowledge date for claims-made coverage. | - | no label found in the documents read |
| `claims_made_terms.extended_reporting_period` | Length, as printed. | - | no label found in the documents read |
| `claims_made_terms.extended_reporting_premium_percent` | Premium for the extended reporting period, as a percentage of the annual premium. | - | no label found in the documents read |
| `claims_made_terms.defense_within_limits` | Whether defence costs reduce the limit of liability. | - | Sutton National, Third Coast Insurance Company |
| `countersignature.representative_name` | Name of the authorised representative who countersigned. | - | Diamond State Insurance Company, Gemini Insurance Company, Security Mutual Insurance Company, Sutton National, Travelers, USLI, Utica First Insurance Company, Western World Insurance Group |
| `countersignature.date` | Date of the countersignature. | Our Authorized Representative and Countersignature Date | Gemini Insurance Company, Utica First Insurance Company |
