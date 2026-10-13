# xTB6: Wei et al. (2025) external-data run

This folder evaluates xTB5 TabPFN V3 on the Wei-2025 CIF collection without
reusing a COF identifier from xTB5 training. The original 725 rows contain 156
CoRE COFs that overlap xTB5's 216 training identifiers; those are excluded.
The staged external cohort therefore contains 569 CIFs: 221 CoRE-COFs and 348
CURATED-COFs.

Important: Wei's reference label convention differs from the xTB5 training
target (their documentation traces the CoRE portion to PBE). Predictions can
be generated, but a comparison against those labels is a target-domain-shift
diagnostic—not a directly comparable xTB5 accuracy score—until labels are
aligned or xTB5 is fine-tuned on Wei labels.

Pipeline: stage non-overlapping CIFs → fragment node/linker/node+linker → run
rich xTB → calculate xTB4-compatible features → predict with TabPFN V3.
