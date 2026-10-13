# xTB9: Frozen original-model external evaluation

## Protocol

TabPFN V3 was fitted once on the original 216 complete-xTB COFs and their
original labels. It was not fine-tuned, recalibrated, or otherwise fitted to
any non-overlapping Wei COF. Of the Wei 725 rows, 476 had complete rich-xTB
features; 156 shared original-training identifiers and were excluded from the
external metric. The external evaluation therefore contains 320 COFs.

## Target alignment

The 156 shared IDs provide both Wei and original labels. An ordinary least
squares conversion was fitted only on those pairs:

`original target = 0.2039 + 1.1229 × Wei reference target`

This alignment is weak: it achieves only R2 = 0.333 and MAE = 0.551 eV on the
same-structure label pairs. Thus it is a transparent approximate conversion,
not proof that the two label conventions are equivalent.

## External result

| Evaluation target | External COFs | R2 | MAE (eV) |
|---|---:|---:|---:|
| Raw Wei labels (not directly comparable) | 320 | -5.657 | 1.136 |
| Wei labels after overlap-derived alignment | 320 | **-2.231** | **0.858** |

The frozen xTB5 model does not transfer reliably to this external cohort under
the available label alignment. Performance is poor for both external source
collections: 98 CoRE rows have aligned R2 -1.599 / MAE 0.720 eV, while 222
CURATED rows have aligned R2 -2.563 / MAE 0.919 eV.

This is not explained solely by the inclusion of CURATED-COFs: even the
non-overlapping external CoRE subset performs poorly. Likely contributors are
label-source mismatch not captured by a linear transformation, covariate and
chemical-family shift beyond the original 216 COFs, and missing external
surface-area descriptors, which are imputed using the original training data.
