# Finch ML-Eng Take-Home — Task & Status Classifier

<details>
<summary>Original Task</summary>

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
    *   Define your model architecture (consider using a transformer base from Hugging Face).
    *   Implement data processing, considering how to handle the `completed` (0/1) input and the desired 3-class `status` output ('complete', 'not_completed', 'not_sure'). You might need to create the 'not_sure' label heuristically or adjust your model's output layer.
    *   Implement the training loop.
    *   Save your trained model's state (e.g., weights) and any necessary tokenizer files to a directory (e.g., `./models/`).
3.  **Implement Prediction:**
    *   Load your saved model and tokenizer.
    *   Implement the prediction logic to output a CSV file (`preds.csv`) with columns: `note_id`, `text`, `task` (predicted string), `completed` (predicted numeric status: 0, 1, or 2).
4.  **Evaluate:**
    *   Run your training script: `python train.py data/train.csv`
    *   Generate predictions on the test set: `python predict_csv.py data/test.csv preds.csv`
    *   Score your predictions: `python score.py data/test.csv data/preds.csv`

## Evaluation Notes

*   The `score.py` script compares the `task` column (string) from `data/test.csv` (ground truth) with the `task` column (string) from `data/preds.csv` (your prediction).
*   It also compares the `completed` column (numeric 0/1) from `data/test.csv` with the `completed` column (numeric 0/1/2) from `data/preds.csv`.
*   For status scoring, `score.py` maps your predicted `2` ('not_sure') to `0` ('not_completed') before calculating accuracy and other metrics to provide a fair comparison against the 2-class ground truth.

Good luck!

---

</details>

## How to run

```bash
# 0) Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Train
python train.py data/train.csv

# 2) Predict
python predict_csv.py data/test.csv data/preds.csv

# 3) Score Predictions
python score.py data/test.csv data/preds.csv
