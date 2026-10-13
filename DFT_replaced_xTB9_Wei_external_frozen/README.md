# xTB9: Frozen xTB5 V3 external evaluation on Wei-2025

The predictor is fitted once on the original 216 xTB5 COFs and is not
fine-tuned on Wei labels. Evaluation uses only Wei rows without a training-ID
overlap and with complete rich-xTB features. The 156 shared CoRE IDs are used
only to estimate a linear target-scale conversion from Wei's reported labels
to the original target scale.

This target conversion is imperfect and is reported transparently; it is not
part of the predictive model and does not use any external-evaluation COF.
