# wc - core fields (SPEC_21 4.3)

Core field list of the wc (workers compensation and employers liability) SPEC_21 schema, built from what the
line's real documents print -- the 39 digital documents of 3 carrier folders under
`synth_core/fideon/synth_core/data/source data/<Carrier>/wc/` (AmTrust, Hartford, NYSIF), read in place -- every
field of the pre-SPEC_21 `wc.json` 1.6.0 (`wc.crosswalk.yaml`) and the line's L1 registry YAMLs. Seed golds are
DRAFT (pending reviewer sign-off); the count column is over the 31 DRAFT golds.

| Item | Value |
|---|---|
| Schema | `wc.json` 2.1.0, composing `common_model.json` 1.1.0 by `$ref` |
| Core leaves | 180 (budget 80-300; `python -m fideon.synth_core.tools leaves`) |
| Leaves filled by at least one DRAFT gold | 76 of the 173 |
| Blocks besides the envelope | `locations`, `rating_exposures`, `rating_modifiers`, `countersignature` |
| LOB block | `wc` (`$defs/WorkersCompensationTerms`): `part_one_states[]`, `other_states_insurance`, `ncci_company_number`, `bureau_file_number`, `total_estimated_cost`, `officers[]` |
| Coverage codes | 3 (`wc.coverage_codes.yaml`): `WC_PART_ONE`, `WC_EMPLOYERS_LIABILITY`, `WC_OTHER_STATES` |
| Mandatory fields | `carrier.name`, `named_insured.primary_name`, `policy.policy_number`, `policy.effective_date`, `policy.expiration_date`, `coverages[].limits[].amount` |

Field names, meanings and shapes are fixed by the common model; this line chooses the blocks, the coverage
codes, the LOB block and the printed labels below (the schema's `fideon:aliases`; the common model's own
aliases are not repeated).

| Core leaf | Printed labels (this line) | DRAFT golds filling it |
|---|---|---|
| `document.transaction_type` |  | 0 |
| `document.transaction_date` | `Process Date` | 19 |
| `document.transaction_effective_date` | `Effective Date` | 3 |
| `document.transaction_reason` |  | 0 |
| `document.endorsement_number` | `Endorsement Number` | 3 |
| `document.title` |  | 31 |
| `document.issue_date` | `Issue Date` | 12 |
| `document.print_date` | `Printed` | 3 |
| `document.mailing_date` |  | 0 |
| `carrier.name` | `INSURER`, `NAME OF INSURER` | 31 |
| `carrier.naic_code` |  | 0 |
| `carrier.group_name` |  | 0 |
| `carrier.address.street` |  | 28 |
| `carrier.address.street_2` |  | 0 |
| `carrier.address.city` |  | 28 |
| `carrier.address.state` |  | 28 |
| `carrier.address.postal_code` |  | 28 |
| `carrier.address.county` |  | 0 |
| `carrier.phone` |  | 16 |
| `carrier.fax` |  | 0 |
| `carrier.web_address` |  | 8 |
| `carrier.claims_phone` |  | 0 |
| `carrier.claims_email` |  | 0 |
| `carrier.claims_address.street` |  | 0 |
| `carrier.claims_address.street_2` |  | 0 |
| `carrier.claims_address.city` |  | 0 |
| `carrier.claims_address.state` |  | 0 |
| `carrier.claims_address.postal_code` |  | 0 |
| `carrier.claims_address.county` |  | 0 |
| `carrier.admitted_status` |  | 0 |
| `carrier.financial_rating` |  | 0 |
| `producer.agency_name` | `Producer's Name`, `REPRESENTATIVE` | 30 |
| `producer.producer_code` | `Producer's Code` | 26 |
| `producer.contact_name` |  | 0 |
| `producer.address.street` |  | 30 |
| `producer.address.street_2` |  | 0 |
| `producer.address.city` |  | 30 |
| `producer.address.state` |  | 30 |
| `producer.address.postal_code` |  | 30 |
| `producer.address.county` |  | 0 |
| `producer.phone` |  | 0 |
| `producer.fax` |  | 0 |
| `producer.email` |  | 0 |
| `producer.web_address` |  | 0 |
| `producer.contract_number` |  | 0 |
| `named_insured.primary_name` | `Named Insured and Mailing Address`, `Named Insured and Address`, `NAMED INSURED`, `Policy Holder Details` | 31 |
| `named_insured.additional_named_insureds[].name` |  | 2 |
| `named_insured.additional_named_insureds[].relationship` |  | 0 |
| `named_insured.additional_named_insureds[].date_of_birth` |  | 0 |
| `named_insured.additional_named_insureds[].gender` |  | 0 |
| `named_insured.additional_named_insureds[].marital_status` |  | 0 |
| `named_insured.additional_named_insureds[].occupation` |  | 0 |
| `named_insured.doing_business_as` | `DBA` | 3 |
| `named_insured.entity_type` | `The Named Insured is`, `TYPE OF BUSINESS` | 27 |
| `named_insured.fein` | `FEIN Number`, `Federal Tax ID`, `Fein` | 23 |
| `named_insured.business_description` | `Business of Named Insured` | 16 |
| `named_insured.mailing_address.street` |  | 31 |
| `named_insured.mailing_address.street_2` |  | 2 |
| `named_insured.mailing_address.city` |  | 31 |
| `named_insured.mailing_address.state` |  | 31 |
| `named_insured.mailing_address.postal_code` |  | 31 |
| `named_insured.mailing_address.county` |  | 0 |
| `named_insured.phone` |  | 0 |
| `named_insured.email` |  | 0 |
| `named_insured.date_of_birth` |  | 0 |
| `named_insured.gender` |  | 0 |
| `named_insured.marital_status` |  | 0 |
| `named_insured.occupation` |  | 0 |
| `policy.policy_number` | `POLICY NUMBER`, `Policy No.` | 31 |
| `policy.prior_policy_number` | `Previous Policy Number`, `Renewal of` | 16 |
| `policy.certificate_number` |  | 0 |
| `policy.policy_type` |  | 0 |
| `policy.original_inception_date` |  | 0 |
| `policy.effective_date` |  | 28 |
| `policy.expiration_date` |  | 31 |
| `policy.term_months` |  | 0 |
| `policy.coverage_trigger` |  | 0 |
| `policy.rating_state` |  | 0 |
| `policy.audit_period` | `Audit Period` | 16 |
| `policy.subject_to_audit` |  | 0 |
| `lob_parts[].title` |  | 16 |
| `lob_parts[].coverage_part_form` |  | 0 |
| `lob_parts[].premium` |  | 0 |
| `coverages[].coverage_name` |  | 20 |
| `coverages[].included` |  | 0 |
| `coverages[].limits[].amount` |  | 20 |
| `coverages[].limits[].percentage` |  | 0 |
| `coverages[].limits[].description` |  | 20 |
| `coverages[].deductibles[].amount` |  | 0 |
| `coverages[].deductibles[].percentage` |  | 0 |
| `coverages[].deductibles[].percentage_basis` |  | 0 |
| `coverages[].deductibles[].peril` |  | 0 |
| `coverages[].premium` |  | 0 |
| `coverages[].valuation` |  | 0 |
| `coverages[].coinsurance_percent` |  | 0 |
| `coverages[].covered_auto_symbols[]` |  | 0 |
| `deductibles[].amount` |  | 0 |
| `deductibles[].percentage` |  | 0 |
| `deductibles[].percentage_basis` |  | 0 |
| `deductibles[].peril` |  | 0 |
| `interested_parties[].rank` |  | 0 |
| `interested_parties[].name` |  | 0 |
| `interested_parties[].address.street` |  | 0 |
| `interested_parties[].address.street_2` |  | 0 |
| `interested_parties[].address.city` |  | 0 |
| `interested_parties[].address.state` |  | 0 |
| `interested_parties[].address.postal_code` |  | 0 |
| `interested_parties[].address.county` |  | 0 |
| `interested_parties[].loan_number` |  | 0 |
| `interested_parties[].is_payor` |  | 0 |
| `premium.total` | `Total Estimated Annual Premium`, `TOTAL ESTIMATED ANNUAL PREMIUM` | 27 |
| `premium.deposit` | `Deposit Premium` | 3 |
| `premium.minimum` | `Policy Minimum Premium`, `Minimum Premium` | 18 |
| `premium.minimum_earned` |  | 0 |
| `premium.minimum_earned_percent` |  | 0 |
| `premium.change` |  | 2 |
| `premium.items[].description` |  | 28 |
| `premium.items[].amount` | `Expense Constant`, `Terrorism`, `Natural Disasters and Catastrophic Industrial Accidents`, `Terrorism Risk Insurance Program Reauthorization Act Disclosure Endorsement`, `Catastrophe (Other Than Certified Acts Of Terrorism)` | 28 |
| `premium.taxes_fees[].description` |  | 0 |
| `premium.taxes_fees[].amount` |  | 0 |
| `premium.taxes_fees[].percentage` |  | 0 |
| `premium.taxes_fees[].fully_earned` |  | 0 |
| `premium.surcharges[].description` |  | 28 |
| `premium.surcharges[].amount` | `Total Estimated Surcharges`, `STATE ASSESSMENT` | 28 |
| `premium.surcharges[].percentage` |  | 8 |
| `premium.surcharges[].fully_earned` |  | 0 |
| `premium.discounts[].description` |  | 7 |
| `premium.discounts[].amount` |  | 7 |
| `premium.discounts[].percentage` |  | 7 |
| `premium.discounts[].fully_earned` |  | 0 |
| `billing.payment_plan` | `Installment Term` | 16 |
| `billing.bill_type` |  | 0 |
| `billing.payment_method` |  | 0 |
| `billing.account_number` |  | 0 |
| `billing.amount_due` |  | 0 |
| `billing.due_date` |  | 0 |
| `billing.installments[].installment_number` |  | 0 |
| `billing.installments[].due_date` | `Payment Due Date`, `Due Date` | 4 |
| `billing.installments[].amount` | `Total Due`, `Amount Due` | 4 |
| `forms_and_endorsements[].form_number` |  | 30 |
| `forms_and_endorsements[].edition_date` |  | 0 |
| `forms_and_endorsements[].title` |  | 27 |
| `forms_and_endorsements[].premium` |  | 0 |
| `locations[].location_number` | `Location Number` | 4 |
| `locations[].address.street` | `Other workplaces not shown above`, `WORKPLACES`, `Named Insured and Location Address of operations covered by this schedule` | 28 |
| `locations[].address.street_2` |  | 1 |
| `locations[].address.city` | `Other workplaces not shown above`, `WORKPLACES`, `Named Insured and Location Address of operations covered by this schedule` | 27 |
| `locations[].address.state` | `Other workplaces not shown above`, `WORKPLACES`, `Named Insured and Location Address of operations covered by this schedule` | 27 |
| `locations[].address.postal_code` | `Other workplaces not shown above`, `WORKPLACES`, `Named Insured and Location Address of operations covered by this schedule` | 27 |
| `locations[].address.county` |  | 0 |
| `locations[].territory` |  | 0 |
| `locations[].protection_class` |  | 0 |
| `locations[].fire_district` |  | 0 |
| `locations[].insured_interest` |  | 0 |
| `locations[].distance_to_fire_station` |  | 0 |
| `locations[].distance_to_hydrant` |  | 0 |
| `rating_exposures[].state` |  | 2 |
| `rating_exposures[].class_code` | `Code Number`, `Code No.`, `Class code` | 30 |
| `rating_exposures[].class_description` | `Classifications`, `Class code description` | 30 |
| `rating_exposures[].exposure_basis` |  | 0 |
| `rating_exposures[].exposure_amount` | `Total Estimated Annual Remuneration`, `Total Est. Annual Remuneration` | 30 |
| `rating_exposures[].rate` | `Rates Per $100 of Remuneration`, `Rate Per $100 of Remuneration` | 30 |
| `rating_exposures[].rate_basis` |  | 0 |
| `rating_exposures[].premium` | `Estimated Annual Premium` | 30 |
| `rating_modifiers[].description` |  | 27 |
| `rating_modifiers[].factor` | `Experience modifier`, `Schedule Rating Factor`, `Merit Rating` | 16 |
| `rating_modifiers[].percent` | `Experience Modification` | 11 |
| `countersignature.representative_name` |  | 0 |
| `countersignature.date` | `Countersigned by` | 16 |
| `wc.part_one_states[]` |  | 20 |
| `wc.other_states_insurance` | `Other States Insurance` | 20 |
| `wc.ncci_company_number` | `NCCI Company Number`, `Ncci Code` | 21 |
| `wc.bureau_file_number` | `Interstate/Intrastate Identification Number`, `Risk Id`, `R.B. File No` | 20 |
| `wc.total_estimated_cost` | `Total Estimated Cost`, `Total Estimated Policy Cost` | 0 (2.1.0) |
| `wc.officers[].name` | `Name of Officer(s)` | 0 (2.1.0) |
| `wc.officers[].title` |  | 0 (2.1.0) |
| `wc.officers[].ownership_percent` |  | 0 (2.1.0) |
| `wc.officers[].included` |  | 0 (2.1.0) |
| `wc.officers[].class_code` |  | 0 (2.1.0) |
| `wc.officers[].remuneration` |  | 0 (2.1.0) |
