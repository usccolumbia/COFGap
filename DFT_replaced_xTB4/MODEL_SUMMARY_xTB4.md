# xTB4: DFT-feature-replacement band-gap model

## Executive summary

xTB4 predicts COF band gaps without using the original DFT fragment HOMO-LUMO
features as model inputs. It replaces them with inexpensive GFN2-xTB
calculations on the same three fragments (node, linker, and node+linker), plus
framework geometry and composition descriptors. The target band gap remains
the reference value in `bandgap_CoRE_COF_v5.0_DFTfree_features.csv`; a
supervised model needs such labeled examples in order to learn.

The conservative, directly validated xTB4 result is a TabPFN v2 model with
R2 = 0.624 and MAE = 0.381 eV. It was evaluated by five-fold family-grouped
cross-validation on the 216 COFs for which all required xTB fragment outputs
were available. The 75% TabPFN + 25% Random Forest blend reached R2 = 0.640
and MAE = 0.380 eV, but that blend weight was selected after inspecting these
cross-validation predictions and therefore requires a locked test set or
nested validation before it is treated as the primary score.

## Dataset and cohort selection

The canonical source table is:

`dataset_code/bandgap_CoRE_COF_v5.0_DFTfree_features.csv`

It contains 315 COFs with a finite band-gap label. The table also provides
inexpensive framework features: volume, density, surface area, atom count,
stacking/layering, periodic-bond measures, composition, and simple
fragmentation metadata.

The model cohort was *not* selected by band-gap value, photovoltaic score, or
model outcome. A COF was included only if xTB completed for all three required
fragment geometries:

1. node;
2. linker; and
3. node+linker fragment.

| Cohort | COFs |
|---|---:|
| Labeled COFs in the canonical DFT-free table | 315 |
| Complete node/linker/node+linker rich-xTB records | 216 |
| Excluded for unavailable/failed rich-xTB records | 99 |

The exclusions are technical coverage limitations. Among the 99 excluded
COFs, 54 had no generated XYZ source for any of the three fragments, 33 were
missing the node+linker XYZ only, 6 had a node+linker xTB timeout, and the
remaining 6 had an xTB timeout or SCF-convergence failure in one or more
required fragments. This means the reported score describes the 216-COF
complete-xTB cohort, not yet the full 315-COF label set.

## Feature construction

For every included COF, the workflow joins its DFT-free framework descriptors
with xTB outputs from node, linker, and node+linker fragments. The xTB4 feature
table initially requests 96 numerical descriptors. Four are entirely empty in
the current extraction and are automatically removed, leaving 92 usable source
features for the deployable model.

Feature groups are:

| Group | Examples |
|---|---|
| Framework structure | density, volume, accessible surface area, atom count, stacking length, periodic-bond density |
| Raw xTB frontier properties | HOMO, LUMO, HOMO-LUMO gap, vertical IP, vertical EA |
| Other xTB electronic properties | electronegativity, hardness, dipole magnitude, charge distribution, bond-order summaries |
| Node-linker relationships | HOMO/LUMO/IP/EA offsets, frontier cross gap, minimum/maximum gap, gap lowering on forming node+linker, charge-polarization proxy |
| Composition and normalized structure | N/B/O/heteroatom fractions, surface area per atom, surface/volume measures |
| Unit-cell geometry | lattice-length range, anisotropy, aspect ratio, cell angles and angle deviation |

No original DFT fragment HOMO, LUMO, or gap is included among the model
features. The reference band gap is used only as the prediction target during
training and scoring.

## Preprocessing and validation

All features are numeric. In each fold, preprocessing is fitted on the
training portion only, then applied to its held-out test portion. Missing
values are median-imputed from the training data. Features that contain no
observed value anywhere in the 216-COF cohort are removed.

The validation method is five-fold `GroupKFold` using family labels. A family
is constructed from the linker class and coarse B/N/O composition. In each
fold, all COFs of a family go to either train or test, never both. The code
explicitly asserts both no shared row and no shared family before a model is
fit.

| Fold | Training COFs | Held-out COFs | Shared families |
|---|---:|---:|---:|
| 0 | 172 | 44 | 0 |
| 1 | 173 | 43 | 0 |
| 2 | 173 | 43 | 0 |
| 3 | 173 | 43 | 0 |
| 4 | 173 | 43 | 0 |

Each COF receives exactly one out-of-fold prediction: a prediction from a
model that did not train on that COF or its defined family. R2 and MAE are then
computed over all 216 out-of-fold predictions. After benchmarking, the
deployable model is refit on all 216 complete-xTB COFs.

## Models and results

| Inputs | Estimator | R2 | MAE (eV) |
|---|---|---:|---:|
| xTB3 compact features (17) | Random Forest | 0.591 | 0.432 |
| Same 17 features | TabPFN v2 | 0.578 | 0.450 |
| xTB4 physics-informed features (92 usable) | Random Forest | 0.591 | 0.432 |
| xTB4 physics-informed features (92 usable) | ExtraTrees | 0.596 | 0.405 |
| xTB4 physics-informed features (92 usable) | TabPFN v2 | **0.624** | **0.381** |
| Same xTB4 features | 75% TabPFN + 25% Random Forest | 0.640* | 0.380* |

`*` Exploratory blend: confirm using nested cross-validation or an untouched
external test set.

R2 measures how much of the variation in band gaps is explained relative to a
mean-only prediction. A value of 1 is perfect; 0 means no improvement over
predicting the average. MAE is the mean absolute prediction error in eV; lower
is better. Thus the standalone TabPFN result means that, for the complete-xTB
cohort and this family-grouped validation, the average absolute error is about
0.381 eV and the model captures roughly 62% of the observed band-gap
variation.

## Prediction workflow

For a new COF, the deployment workflow is:

1. fragment the COF in the same way as the training data;
2. calculate xTB outputs for node, linker, and node+linker fragments;
3. calculate the same DFT-free framework descriptors from the COF;
4. assemble the 92 usable xTB4 feature columns;
5. apply the training-fitted median imputer; and
6. predict using TabPFN, or the saved 75/25 TabPFN-Random-Forest blend.

The saved ensemble consists of `03_package_model/DFT_replaced_xTB4_bundle.joblib`
and `03_package_model/xTB4_tabpfn_v2.tabpfn_fit`; `03_package_model/predict.py`
loads both correctly.

## Interpretation and next step

The principal result is that richer xTB electronic information becomes useful
when modeled with TabPFN: simply changing Random Forest to TabPFN with the old
17 features did not help. The current limitation is coverage: xTB4 cannot
score all 315 labeled COFs until the 99 missing/failed fragment records are
recovered or a well-defined fallback feature pathway is added. The most useful
next benchmark is therefore a 315-COF model with a documented fallback for
missing xTB features, evaluated with the same family-grouped split protocol.
