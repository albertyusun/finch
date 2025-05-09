# Finch ML-Eng Take-Home — Task & Status Classifier

## Approach

As suggested, I fine-tuned `distilbert-base-uncased` model from HF with a **multitask setup** on the 400 datapoints:
- **Shared encoder**: DistilBERT processes the input text.
- **Two classifier heads**:
  - 6-way task classification (`task`)
  - 2-way status classification (`status`)
- Loss = task loss + status loss (`CrossEntropy`)

Training took 82.58 seconds to train the model on 400 datapoints. I used a basic VM from Google Cloud with machine type e2-medium (2 vCPUs, 4 GB Memory).

---

## Evaluation

I used `score.py` for:
- **Task Accuracy** (string match)
- **Status Accuracy** (with `2 → 0` mapping for fair comparison)

Key metrics:
- **Task Accuracy**: 1.0000
- **Status Accuracy (2→0)**: 1.0000

Classification reports (shown below) show perfect precision/recall across all classes.

Normally, for a classification task like this, I typically care most about F1-score to understand how strong a model is.

I use precision, recall, and FPR / FNR to debug performance of a model if a model is shown to have a weak F1-score.

However, for our task, I didn't have to do much analysis for our experiments, because we achieved 100% F1 score for each category and overall.

---

## With More Time

With more time, I'd do the following:
- Train with real `status = 2` examples
- Tune hyperparameters & use stratified K-fold CV
- Experiment with larger models (`bert-base`, `roberta`)
- Add uncertainty-aware predictions

## Overall Metrics

```
--- Scoring Results ---

Task Prediction Accuracy: 1.0000
Task Classification Report:
                         precision    recall  f1-score   support

        Client Check-in       1.00      1.00      1.00        15
          Create Demand       1.00      1.00      1.00        15
            Intake Call       1.00      1.00      1.00        15
Request Medical Records       1.00      1.00      1.00        15
 Sign Engagement Letter       1.00      1.00      1.00        15
                   none       1.00      1.00      1.00        25

               accuracy                           1.00       100
              macro avg       1.00      1.00      1.00       100
           weighted avg       1.00      1.00      1.00       100


Status Prediction Accuracy (mapping predicted 'not_sure' [2] to 'not_completed' [0]): 1.0000
Status Classification Report (after mapping 2->0):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        64
           1       1.00      1.00      1.00        36

    accuracy                           1.00       100
   macro avg       1.00      1.00      1.00       100
weighted avg       1.00      1.00      1.00       100
```

---

<sub><i>All scores are rounded to two decimal places where applicable.</i></sub>

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
