# Machine Learning Project: Report Reader

A self-contained pipeline for predicting domain-specific numerical targets from unstructured financial and operational reports (train.csv, test.csv). Full problem specifications are in [ML Project Statement.pdf](ML%20Project%20Statement.pdf).

## Repository Structure

- Report_Reader.ipynb — End-to-end notebook covering normalization, extraction, modeling, and submission generation.
- train.csv / test.csv — Labeled training reports and unlabeled test reports.
- ML Project Statement.pdf — Project guidelines and domain specifications.
- LICENSE — MIT License.

## Approach

1. *Text & Table Normalization*: Standardizes spelled-out numbers, Indian/international currency units (₹, Rs, INR, lakh, crore, million, billion), percentages, fiscal-year labels, and Markdown tables into uniform statements.
2. *Domain Routing*: Assigns each report to one of four domains (Hospital, Farmer, Real Estate, Corporate) using deterministic domain cues, cross-checked with a TF–IDF + Logistic Regression classifier trained on the training split.
3. *Target-Free Structured Extraction*: Dedicated parsers isolate main-entity actuals while filtering out rivals, prior-period comparisons, forecasts, benchmarks, and superseded figures:
   - *Corporate*: Builds a year-indexed ledger to compute operating profit across the three most recent consecutive fiscal years.
   - *Hospital*: Extracts inpatient revenue (direct or derived from total/outpatient figures) and annualized inpatient admissions, with a label-distance recovery fallback.
   - *Farmer*: Extracts yield per acre (or production ÷ area), realised price per quintal, and cultivation cost per acre.
   - *Real Estate*: Extracts total apartments (checked against tower/floor dimensions), booked units, and current price per apartment (direct or rate × size).
4. *Robust Formula Fitting*: Fits each domain's parametric formula using iteratively reweighted bounded least squares (scipy.optimize.lsq_linear) to resist label noise:
   - *Hospital*: Log-linear power law with 3-MAD trimmed weights.
   - *Real Estate*: Linear inventory–price features with 3-MAD trimmed weights.
   - *Farmer*: Linear revenue–cost features with Tukey biweight IRLS.
   - *Corporate*: 3-year operating profit combination with Cauchy weights.

## How to Run

### Requirements
- Python 3.10+
- numpy, pandas, scipy, scikit-learn

### Execution
1. Keep train.csv and test.csv alongside Report_Reader.ipynb (or set DATA_DIR in the first cell).
2. Run all cells in Report_Reader.ipynb in order.
3. The final cell validates and writes the submission to report_reader_outputs/submission.csv.

## License

MIT — see [LICENSE](LICENSE).

## Team Members

- [Nishant Jangid](https://github.com/nishantj27)
- [Ritanshu Sharma](https://github.com/RS7143)
- [Nikhil Singhal](https://github.com/nikhilsinghal66)
- [Satyajeet Singh](https://github.com/satyajeet-byte)
