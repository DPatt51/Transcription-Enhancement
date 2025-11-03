# Harnessing AI for Smart Medical Transcription Enhancement

A minimal, PyTorch-first scaffold for **Project 2: Transcriptive** (specialty classification, entity extraction, QA/error checks, summary).

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (Classification)
```bash
# Put/inspect example CSVs in data/
python src/classify/train_distilbert.py  # trains DistilBERT on data/train.csv & data/val.csv
python src/classify/predict.py models/cls_distilbert "Patient presents with chest pain radiating to left arm."
python src/classify/baselines/predict_sklearn.py models/cls_tfidf_logreg "26-year-old with Crohn’s disease (ileocolonic) presenting with 3–4 loose stools/day, mild RLQ cramping, no bleeding. Last colonoscopy 10 months ago showed shallow ulcers in terminal ileum; fecal calprotectin previously elevated. Current meds: adalimumab q2w, azathioprine 100 mg daily; reports partial response and occasional missed doses. Labs today: CRP 9 mg/L, Hgb 12.0. Assessment: probable mild–moderate active Crohn’s with suboptimal biologic trough. Plan: check adalimumab level/anti-drug antibodies, reinforce adherence, stool calprotectin, consider dose escalation or switch to ustekinumab if low trough/high ADA; nutrition counseling; follow-up in 6 weeks."
```

## Data Format
Each CSV has columns: `id,text,specialty`
```
id,text,specialty
1,"Chief complaint: fever and cough for 3 days...", "Pediatrics"
2,"Chest pain on exertion, ECG shows ST changes...", "Cardiology"
```

## Components
- `src/classify/` — PyTorch (Transformers) document classifier
- `src/tokencls/` — optional PyTorch token classification (NER) head
- `src/weak_ner/` — dictionary/regex-based extraction (silver labels optional)
- `src/qa/` — rule-based error checks (spelling/structure/range)
- `src/summarize/` — SOAP/brief summary builder (template-based)

## Reports
- Validation and test metrics JSON will be saved under `reports/metrics/`.
- Put illustrative before→after examples under `reports/examples/`.

## Notes
- This is an **educational prototype**, not a medical device.
- Pin library versions for reproducibility.
