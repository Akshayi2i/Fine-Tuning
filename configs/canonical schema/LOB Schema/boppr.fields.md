# boppr - core fields (SPEC_21 4.3)

Core field list of the boppr (businessowners property) SPEC_21 schema, built from the 5 single-LOB seed
documents of the source-data boppr folder (3 carriers: Leatherstocking Cooperative Insurance Company, Madison
Mutual, Reinsure), every field of the pre-SPEC_21 boppr.json 1.5.0 (boppr.crosswalk.yaml) and the line's L1
registry YAMLs, read together (SPEC_21 4.3, 14.1 Phase 2). Package documents (bopgl_boppr, boppr_cgl) are not read.

| Item | Value |
|---|---|
| Schema | `boppr.json` 2.2.0, composing `common_model.json` 1.4.0 by `$ref` (2.0.0 composed 1.1.0) |
| Core leaves | 176 (budget 80-300; `python -m fideon.synth_core.tools leaves`; 170 before the six optional fields of common model 1.4.0) |
| Printed on the current seeds | 78 of the 170 |
| Seeds | 8 DRAFT seed golds, 3 carriers (4 Leatherstocking Preferred-form documents added) |
| Blocks besides the envelope | `locations`, `buildings` |
| LOB block | none |
| Coverage codes | 109 (`boppr.coverage_codes.yaml`), of them 6 shared `X_` codes; 72 have a counterpart code in bopgl (2.2.0: 13 codes added) |
| Mandatory fields | `carrier.name`, `named_insured.primary_name`, `policy.policy_number`, `policy.effective_date`, `policy.expiration_date`, `coverages[].limits[].amount` |

Field names, meanings and shapes are fixed by the common model; this line chooses the blocks, the coverage
codes and the printed labels below. **Printed labels** are the captions the seeds print beside or above the
value (read from the seed text layers, checked by hand); they are the schema's `fideon:aliases` for the field.
The common model's own aliases still apply and are not repeated. **Carriers** and **Seeds** count the DRAFT seed
golds that fill the field. Coverage names are aliases of the coverage codes, listed in the coverage codes file.

**2.2.0 (25 October 2026).** The printed labels added in 2.2.0 (`Term Length`, `Year Constructed`, `NYS INS FEE` and the others in the rows below) come from the two businessowners declarations of the `G1/bop` folder (Utica First and NYCM), not from the seeds: the Carriers and Seeds columns still count the seeds only.

## Core fields

| Field | Meaning | Printed labels (this line) | Carriers that print it | Seeds |
|---|---|---|---|---|
| `document.transaction_type` | Kind of transaction the document records. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `document.transaction_date` | Date the transaction was processed. | TRANSACTION DATE | Reinsure | 1 |
| `document.transaction_effective_date` | Date an endorsement, cancellation or reinstatement takes effect. | Amended Date; TRANSACTION EFFECTIVE DATE; Transaction Effective | Leatherstocking Cooperative Insurance Company, Madison Mutual | 2 |
| `document.transaction_reason` | Reason for the transaction, as printed. | POLICY CHANGE | Leatherstocking Cooperative Insurance Company, Madison Mutual | 2 |
| `document.endorsement_number` | Number of the endorsement or policy change. | - | not printed on the current seeds | 0 |
| `document.title` | Document title as printed, e.g. 'Homeowners Policy Declarations'. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `document.issue_date` | Date the document was issued. | - | Madison Mutual | 2 |
| `document.print_date` | Date the document was printed. | - | Reinsure | 1 |
| `document.mailing_date` | Date the document was mailed; it starts notice periods on cancellation and nonrenewal notices. | - | not printed on the current seeds | 0 |
| `carrier.name` | The writing company. | Insurance Provided By | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `carrier.naic_code` | NAIC company code of the writing company. | - | not printed on the current seeds | 0 |
| `carrier.group_name` | Group or trade name, when printed separately from the writing company. | - | not printed on the current seeds | 0 |
| `carrier.address.street` | Street address line. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `carrier.address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `carrier.address.city` | City. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `carrier.address.state` | State, two-letter USPS code in parsed. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `carrier.address.postal_code` | ZIP code. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `carrier.address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.phone` | Main phone number of the carrier. | Phone | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `carrier.fax` | Fax number. | Fax | Leatherstocking Cooperative Insurance Company | 2 |
| `carrier.web_address` | Website address. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `carrier.claims_phone` | Phone number for reporting a claim. | - | not printed on the current seeds | 0 |
| `carrier.claims_email` | Address claims are reported to, when printed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street` | Street address line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.city` | City. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.state` | State, two-letter USPS code in parsed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.postal_code` | ZIP code. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.admitted_status` | Whether the writing company is admitted or non-admitted in the state. | - | not printed on the current seeds | 0 |
| `carrier.financial_rating` | Financial-strength rating of the writing company as printed, e.g. the A.M. Best rating and size class 'A+ XV'. | - | not printed on the current seeds | 0 |
| `producer.agency_name` | Name of the agency or broker of record. | Agency; Mail To | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `producer.producer_code` | Code the carrier assigns to the agency. | CLIENTS AGENCY ID | Reinsure | 1 |
| `producer.contact_name` | Contact person at the agency. | - | not printed on the current seeds | 0 |
| `producer.address.street` | Street address line. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `producer.address.street_2` | Second address line: suite, unit, attention line. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 3 |
| `producer.address.city` | City. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `producer.address.state` | State, two-letter USPS code in parsed. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `producer.address.postal_code` | ZIP code. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `producer.address.county` | County. | - | not printed on the current seeds | 0 |
| `producer.phone` | Phone number of the agency. | Work | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `producer.fax` | Fax number. | Fax | Leatherstocking Cooperative Insurance Company | 2 |
| `producer.email` | Email address. | - | not printed on the current seeds | 0 |
| `producer.web_address` | Website of the agency. | - | not printed on the current seeds | 0 |
| `producer.contract_number` | The agency's contract or sub-code with the carrier. | - | not printed on the current seeds | 0 |
| `named_insured.primary_name` | First named insured. | Named Insured(s); NAMED INSURED(S) | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `named_insured.additional_named_insureds[].name` | Name of the additional named insured. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].relationship` | As printed, e.g. 'Spouse', 'Subsidiary'. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].date_of_birth` | Date of birth. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].gender` | Gender as printed. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].marital_status` | Marital status as printed. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].occupation` | Occupation as printed. | - | not printed on the current seeds | 0 |
| `named_insured.doing_business_as` | Trade name the insured does business as. | - | not printed on the current seeds | 0 |
| `named_insured.entity_type` | As printed: Individual, Corporation, LLC, Partnership, Trust... | Insured Is | not printed on the current seeds | 0 |
| `named_insured.fein` | Federal employer identification number. | - | not printed on the current seeds | 0 |
| `named_insured.business_description` | Nature of the insured's business or operations. | Business Operations | not printed on the current seeds | 0 |
| `named_insured.mailing_address.street` | Street address line. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `named_insured.mailing_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `named_insured.mailing_address.city` | City. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `named_insured.mailing_address.state` | State, two-letter USPS code in parsed. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `named_insured.mailing_address.postal_code` | ZIP code. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `named_insured.mailing_address.county` | County. | - | not printed on the current seeds | 0 |
| `named_insured.phone` | Phone number of the insured. | - | Reinsure | 1 |
| `named_insured.email` | Email address of the insured. | - | Reinsure | 1 |
| `named_insured.date_of_birth` | Date of birth. | - | not printed on the current seeds | 0 |
| `named_insured.gender` | Gender as printed. | - | not printed on the current seeds | 0 |
| `named_insured.marital_status` | Marital status as printed. | - | not printed on the current seeds | 0 |
| `named_insured.occupation` | Occupation as printed. | - | not printed on the current seeds | 0 |
| `policy.policy_number` | Policy number. | Policy ID; Policy Number; POLICY NUMBER | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `policy.prior_policy_number` | Number of the policy this one renews or replaces. | - | not printed on the current seeds | 0 |
| `policy.certificate_number` | Certificate number, when the policy is issued as a certificate under a master policy. | - | not printed on the current seeds | 0 |
| `policy.policy_type` | Product, program or policy form name as printed (e.g. 'Homeowners HO-3', 'Auto Special'). | Policy Type; Policy Form; POLICY TYPE | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 4 |
| `policy.original_inception_date` | Date the insured first became a policyholder with the carrier. | Policyholder Since; INCEPTION DATE | Madison Mutual, Reinsure | 3 |
| `policy.effective_date` | Date coverage starts. | Policy Term Effective Date; EFFECTIVE DATE | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `policy.expiration_date` | Date coverage ends. | Policy Term Expiration Date; EXPIRATION DATE | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `policy.term_months` | Term length in months; 'Annual' -> 12. | Term Length | not printed on the current seeds | 0 |
| `policy.coverage_trigger` | What triggers coverage: occurrence or claims made. | - | not printed on the current seeds | 0 |
| `policy.rating_state` | State the policy is rated in. | - | not printed on the current seeds | 0 |
| `policy.audit_period` | As printed: Annual, Semi-Annual, Quarterly, Monthly, Non-Auditable. | - | not printed on the current seeds | 0 |
| `policy.subject_to_audit` | Whether the premium is subject to audit. | - | not printed on the current seeds | 0 |
| `lob_parts[].title` | Title of the coverage part as printed. | - | not printed on the current seeds | 0 |
| `lob_parts[].coverage_part_form` | Coverage-part declarations form number. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `lob_parts[].premium` | Premium for the part. | - | Reinsure | 1 |
| `coverages[].coverage_name` | As printed. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `coverages[].included` | Whether the coverage is included, excluded or not purchased. | - | not printed on the current seeds | 0 |
| `coverages[].limits[].amount` | Amount of the limit. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `coverages[].limits[].percentage` | When the limit is stated as a percentage of another coverage. | - | not printed on the current seeds | 0 |
| `coverages[].limits[].description` | Printed qualifier, e.g. 'theft of jewelry'. Required in practice for sublimits. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `coverages[].deductibles[].amount` | Amount of a flat deductible. | Deductible | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `coverages[].deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | not printed on the current seeds | 0 |
| `coverages[].premium` | Premium for the coverage. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `coverages[].valuation` | Valuation basis. | Loss Settlement; Business Property Loss Settlement; Settlement - Building; Building Coverage; Business Personal Property Coverage | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `coverages[].coinsurance_percent` | Coinsurance percentage that applies to the coverage. | Coinsurance | Leatherstocking Cooperative Insurance Company | 2 |
| `coverages[].covered_auto_symbols[]` | ISO covered-auto symbols printed against the coverage (1, 2, 7, 8, 9...). Business auto, garage, truckers and package auto parts. | - | not printed on the current seeds | 0 |
| `deductibles[].amount` | Amount of a flat deductible. | Deductible | not printed on the current seeds | 0 |
| `deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | not printed on the current seeds | 0 |
| `deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | not printed on the current seeds | 0 |
| `deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | not printed on the current seeds | 0 |
| `interested_parties[].rank` | 1 for first mortgagee, 2 for second... | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `interested_parties[].name` | Name of the interested party. | 1st Mortgagee; 2nd Mortgagee; Lienholder; Lenders Loss Payable; LOSS PAYEE NAME/ADDRESS | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `interested_parties[].address.street` | Street address line. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `interested_parties[].address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `interested_parties[].address.city` | City. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `interested_parties[].address.state` | State, two-letter USPS code in parsed. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `interested_parties[].address.postal_code` | ZIP code. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `interested_parties[].address.county` | County. | - | not printed on the current seeds | 0 |
| `interested_parties[].loan_number` | Loan or account number with the interested party. | Loan Number | Leatherstocking Cooperative Insurance Company | 4 |
| `interested_parties[].is_payor` | True when this party pays the premium (escrow billing). | - | not printed on the current seeds | 0 |
| `premium.total` | Total premium for the policy. | Total; TERM AMOUNT; Total Premium; Annual Premium | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `premium.deposit` | Deposit premium due at inception. | - | not printed on the current seeds | 0 |
| `premium.minimum` | The least premium the policy (or part) is written for. | - | not printed on the current seeds | 0 |
| `premium.minimum_earned` | The least premium kept on cancellation, as an amount; not the same as `minimum`. A percentage goes to `minimum_earned_percent`. | - | not printed on the current seeds | 0 |
| `premium.minimum_earned_percent` | The least premium kept on cancellation, printed as a percentage of the premium (e.g. 'Minimum Earned Premium: 25%'). An amount goes to `minimum_earned`. | - | not printed on the current seeds | 0 |
| `premium.change` | Premium change made by this transaction; negative for a return premium. | - | not printed on the current seeds | 0 |
| `premium.items[].description` | Description of the premium line as printed. | - | Leatherstocking Cooperative Insurance Company | 2 |
| `premium.items[].amount` | Premium amount of the line. | - | Leatherstocking Cooperative Insurance Company | 2 |
| `premium.taxes_fees[].description` | Name of the tax, fee, surcharge or discount as printed. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `premium.taxes_fees[].amount` | Amount of the tax, fee, surcharge or discount. | Fees; NYS INS FEE; NY FIRE FEE | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `premium.taxes_fees[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.taxes_fees[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.surcharges[].description` | Name of the tax, fee, surcharge or discount as printed. | - | Leatherstocking Cooperative Insurance Company | 1 |
| `premium.surcharges[].amount` | Amount of the tax, fee, surcharge or discount. | - | Leatherstocking Cooperative Insurance Company | 1 |
| `premium.surcharges[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.discounts[].description` | Name of the tax, fee, surcharge or discount as printed. | - | Madison Mutual | 2 |
| `premium.discounts[].amount` | Amount of the tax, fee, surcharge or discount. | - | Madison Mutual | 2 |
| `premium.discounts[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.discounts[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `billing.payment_plan` | Payment plan as printed (annual, 2-pay, 12-pay EFT...). | Payment Plan | Reinsure | 1 |
| `billing.bill_type` | As printed: Direct Bill, Agency Bill. | Billing Type | not printed on the current seeds | 0 |
| `billing.payment_method` | As printed: EFT, recurring card, check... | - | not printed on the current seeds | 0 |
| `billing.account_number` | Billing account number the carrier bills under. | - | not printed on the current seeds | 0 |
| `billing.amount_due` | Amount currently due on the bill. | - | not printed on the current seeds | 0 |
| `billing.due_date` | Date payment is due. | - | not printed on the current seeds | 0 |
| `billing.installments[].installment_number` | Number of the installment in the payment plan. | - | not printed on the current seeds | 0 |
| `billing.installments[].due_date` | Date the installment is due. | - | not printed on the current seeds | 0 |
| `billing.installments[].amount` | Amount of the installment. | - | not printed on the current seeds | 0 |
| `forms_and_endorsements[].form_number` | Form number as printed, e.g. 'HO 00 03'. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `forms_and_endorsements[].edition_date` | Edition date of the form as printed (often month and year only, e.g. '05 11'). | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `forms_and_endorsements[].title` | Title of the form or endorsement. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `forms_and_endorsements[].premium` | Premium charged for the form or endorsement. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `locations[].location_number` | Location number as printed on the schedule. | Property; Location; LOCATION; Location # | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `locations[].address.street` | Street address line. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `locations[].address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `locations[].address.city` | City. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `locations[].address.state` | State, two-letter USPS code in parsed. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `locations[].address.postal_code` | ZIP code. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual, Reinsure | 5 |
| `locations[].address.county` | County. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `locations[].territory` | Rating territory or zone. | Zone | Leatherstocking Cooperative Insurance Company | 2 |
| `locations[].protection_class` | Fire protection class of the premises. | Protection Class; Fire Protection Class | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `locations[].fire_district` | Fire district or fire protection area the premises fall in. | - | not printed on the current seeds | 0 |
| `locations[].insured_interest` | The insured's interest in the premises: Owner, Deeded owner, Tenant, LLC member... | - | not printed on the current seeds | 0 |
| `locations[].distance_to_fire_station` | Distance to the responding fire station. | Miles to Fire Department | not printed on the current seeds | 0 |
| `locations[].distance_to_hydrant` | Distance to the nearest fire hydrant. | Feet to Hydrant | not printed on the current seeds | 0 |
| `buildings[].building_number` | Building number as printed on the schedule. | Building | Leatherstocking Cooperative Insurance Company, Madison Mutual | 3 |
| `buildings[].construction` | Construction type (frame, masonry, fire resistive...). | Construction Type | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `buildings[].occupancy` | How the building is occupied or used. | Description of Building Occupancies; Insured Occupancy Type | Leatherstocking Cooperative Insurance Company, Madison Mutual | 4 |
| `buildings[].year_built` | Year the building was built. | Year of Construction; Year Constructed | Madison Mutual | 2 |
| `buildings[].area_sq_ft` | Floor area in square feet. | Square Foot; Square Footage | not printed on the current seeds | 0 |
| `buildings[].stories` | Number of stories. | - | not printed on the current seeds | 0 |
| `buildings[].number_of_units` | Number of dwelling units or families in the building. | - | not printed on the current seeds | 0 |
| `buildings[].roof_type` | Roof type or covering material. | - | not printed on the current seeds | 0 |
| `buildings[].protective_devices[]` | Protective devices fitted (alarms, sprinklers, extinguishers...). | - | Leatherstocking Cooperative Insurance Company | 1 |
| `buildings[].basement` | Whether the building has a basement. | - | not printed on the current seeds | 0 |
| `buildings[].sprinklered` | Whether the building has an automatic sprinkler system. | - | not printed on the current seeds | 0 |
| `buildings[].heating_type` | Type of the primary heating system. | - | not printed on the current seeds | 0 |
| `buildings[].heating_fuel` | Fuel of the primary heating system. | - | not printed on the current seeds | 0 |
| `buildings[].heating_year_updated` | Year the primary heating system was last updated. | - | not printed on the current seeds | 0 |
| `policy.effective_time` | Time of day coverage starts, as printed (common model 1.4.0). | Policy Period Time; Effective Time | printed as `Policy Term Effective Date` time by Leatherstocking (an `additional_fields` entry on the current seeds) | 0 |
| `policy.expiration_time` | Time of day coverage ends, as printed (common model 1.4.0). | Expiration Time | printed by Leatherstocking (an `additional_fields` entry on the current seeds) | 0 |
| `carrier.company_type` | How the writing company describes itself: 'A Stock Company', 'A Mutual Company' (common model 1.4.0). | Company Type; Type of Company | not printed on the current seeds | 0 |
| `document.copy_designation` | Which copy of the document this is: 'Agent Copy', 'Insured Copy' (common model 1.4.0). | Copy | not printed on the current seeds | 0 |
| `named_insured.naics_code` | NAICS industry code of the named insured (common model 1.4.0). | NAICS; NAICS Code | not printed on the current seeds | 0 |
| `named_insured.sic_code` | SIC industry code of the named insured (common model 1.4.0). | SIC; SIC Code | not printed on the current seeds | 0 |
