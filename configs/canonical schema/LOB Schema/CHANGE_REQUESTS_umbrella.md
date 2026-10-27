# Change requests - umbrella (to the common-model owner, the registry owner and the project owner)

Raised while building the umbrella SPEC_21 schema 2.0.0 (11 October 2026). Nothing here is applied to the common
model. Labels only; no document values.

**CR-UM-01. Underlying coverage codes (common-model owner).**
`UnderlyingPolicy.coverage_code` is a free string. The schema lists three `UM_UNDERLYING_` codes for the underlying
policies the seeds schedule (general liability, a followed policy, an underlying excess layer). Decide whether
underlying policies should carry the underlying line's own codes (a gl code for general liability) instead.

**CR-UM-02. Sutton National's scanned declarations (registry owner).**
The Sutton National commercial excess declarations page is an image under an OCR text layer; no carrier entry exists
and no gold was drafted. It stays L3 (a layer that reads pixels).

**CR-UM-03. Training parity (fine-tuning repository owner).**
The fine-tuning repository holds the legacy `umbrella.json`. Decide with the casualty_fleet family's other lines.

**CR-UM-04. Excess over management liability (project owner).**
CNA's excess policy follows a directors and officers policy. It is read here as an excess layer (umbrella); decide
whether excess layers of management-liability towers belong to umbrella or to the line of the followed policy.
