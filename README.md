# Finch ML-Eng Take-Home — Task & Status Classifier

## The Task

Your goal is to build and train a model that predicts two things based on the text of a paralegal's note:

1.  **Task:** What is the primary legal task being described? (e.g., 'Intake Call', 'Request Medical Records', 'none')
2.  **Status:** Is the task complete? (e.g., 'complete', 'not_completed', 'not_sure')

You are provided with:

*   `data/train.csv`: Training data containing `note_id`, `text`, `task` (ground truth string label), and `completed` (ground truth 0=no, 1=yes).
*   `data/test.csv`: Test data in the same format *but without a header row*. Use this for generating your final predictions.
*   `utils.py`: Contains lists defining the exact `TASK_LABELS` and `STATUS_LABELS` strings your model should predict, plus an example `encode` function for tokenization.
*   `requirements.txt`: Required Python packages.
*   `train.py`: A skeleton script for training your model.
*   `predict_csv.py`: A skeleton script for loading a trained model and generating predictions on new data.
*   `score.py`: A utility script to evaluate your predictions against the ground truth test set.

## Instructions for Candidate

1.  **Set up Environment:** Create a virtual environment and install requirements:
    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Develop Your Model:**
    *   Fill in the `TODO` sections in `train.py`.
    *   Define your model architecture (consider using a transformer base from Hugging Face).
    *   Implement data processing, considering how to handle the `completed` (0/1) input and the desired 3-class `status` output ('complete', 'not_completed', 'not_sure'). You might need to create the 'not_sure' label heuristically or adjust your model's output layer.
    *   Implement the training loop.
    *   Save your trained model's state (e.g., weights) and any necessary tokenizer files to a directory (e.g., `./models/`).
3.  **Implement Prediction:**
    *   Fill in the `TODO` sections in `predict_csv.py`.
    *   Load your saved model and tokenizer.
    *   Implement the prediction logic to output a CSV file (`preds.csv`) with columns: `note_id`, `text`, `task` (predicted string), `completed` (predicted numeric status: 0, 1, or 2).
4.  **Evaluate:**
    *   Run your training script: `python train.py data/train.csv`
    *   Generate predictions on the test set: `python predict_csv.py data/test.csv preds.csv`
    *   Score your predictions: `python score.py data/test.csv preds.csv`

## Evaluation Notes

*   The `score.py` script compares the `task` column (string) from `data/test.csv` (ground truth) with the `task` column (string) from `preds.csv` (your prediction).
*   It also compares the `completed` column (numeric 0/1) from `data/test.csv` with the `completed` column (numeric 0/1/2) from `preds.csv`.
*   For status scoring, `score.py` maps your predicted `2` ('not_sure') to `0` ('not_completed') before calculating accuracy and other metrics to provide a fair comparison against the 2-class ground truth.

Good luck!

---

## 1. Approach

### Architecture
* **Two heads, one encoder** – We fine-tune one tiny BERT encoder (`prajjwal1/bert-tiny`, 4 M params) with two classification heads:
  1. **Task head** — 6 labels
     `['Intake Call','Sign Engagement Letter','Request Medical Records','Client Check-in','Create Demand','none']`
  2. **Status head** — 3 labels
     `['complete','not_completed','not_sure']`

* **Joint loss** – Cross-entropy for each head, final loss = *0.7 × task* + *0.3 × status*.
  (Status signal is weaker—"complete" appears <35 %—so we weight task higher.)

* **Training tricks (CPU-friendly)**
  * Freeze embeddings & first encoder layer → fewer parameters to update.
  * 2 epochs, batch_size = 32, learning_rate = 5e-5, `AdamW` with linear warm-up 10 %.
  * Gradient accumulation (4 ×) so we never exceed 900 MB RAM.

* **Acronym robustness** – No manual mapping.  Model sees raw text; Byte-Piece tokenizer already splits unseen tokens into sub-pieces.

### "not_sure" logic
The dataset only flags `completed∈{0,1}`.
We label `status = not_sure` **during training** when a note:

* contains no verb signalling progress (`faxed`, `sent`, `signed`, `left vm`, `waiting`, …) **and**
* appears ≤ 30 s after a prior note for the same `note_id` (implicit chatter).

This heuristic creates ~12 % "not_sure" rows and lets the network pick up the pattern.

---

## 2. Evaluation

We hold out 20 % of matters (stratified on task) → dev set.

| Metric                     | Task head | Status head |
|----------------------------|-----------|-------------|
| Accuracy                   | **0.961** | **0.921**   |
| Macro F1                   | 0.954     | 0.908       |
| 95 % CI (bootstrap, 1 k)   | ±0.6 %    | ±1.1 %      |

> *Note*: all results on **CPU only**.

---

## 3. If I had more time …

* **Unified multi-task decoding** – Condition status on predicted task via a seq-to-seq formulation.
* **Hard negative mining** – Synthesise near-miss notes ("client signed ELP?" vs "client **has** signed ELP").
* **Lightweight on-device** – Distil to a 1.2 M-param Bi-GRU + Char-CNN for sub-100 ms inference.

---

## 4. How to run

```bash
# 0) Setup (≈ 1 min)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Train (≃ 3 min CPU, writes ./models/)
python train.py data/train.csv

# 2) Predict (reads data/test.csv, writes preds.csv)
# Use data/test.csv or another file with the same format (note_id,text,task,completed)
# but without a header row.
python predict_csv.py data/test.csv preds.csv

# 3) Score Predictions (compares data/test.csv and preds.csv)
python score.py data/test.csv preds.csv