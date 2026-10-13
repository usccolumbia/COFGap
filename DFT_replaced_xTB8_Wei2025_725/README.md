# xTB8: Wei-2025 725-COF internal validation

This is a separate labeled-data experiment, not an external test of xTB5.
It evaluates TabPFN V3 using Wei et al.'s 725 reported band-gap labels under
random 80/10/10 and family-grouped validation. The complete-case cohort will
contain only COFs that supply all three required xTB fragment records.

For the 156 CoRE identifiers shared with xTB5, existing rich-xTB calculations
are reused because the dataset identifies them as the same CoRE v5.0 positional
CIFs. The remaining 569 rows require fresh fragment/xTB extraction.
