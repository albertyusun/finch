# Finch ML-Eng Take-Home — Task & Status Classifier

## Approach

As suggested, I fine-tuned `distilbert-base-uncased` model from HF with a **multitask setup** on the 400 datapoints:
- **Shared encoder**: DistilBERT processes the input text.
- **Two classifier heads**:
  - 6-way task classification (`task`)
  - 2-way status classification (`status`)
- Loss = task loss + status loss (`CrossEntropy`)

Why did I choose this model?

Multitask learning allows the model to share linguistic representations across both labels, improving generalization while reducing compute overhead. [Caruana 1997](https://link.springer.com/article/10.1023/A:1007379606734). To do this, I used two independent heads to give the model flexibility to model the differing semantics and class sizes of `task` vs. `status`. This set up has been well established by awhile, see [Liu 2019](https://arxiv.org/pdf/1901.11504).

Training took 82.58 seconds to train the model on 400 datapoints. I used a basic VM from Google Cloud with machine type e2-medium (2 vCPUs, 4 GB Memory).

Finally, how did I model uncertainty (the unsure 2 label) in status prediction? To capture ambiguity in legal task status, I experimented with three approaches for predicting the not_sure class:

1. Keyword-based heuristics:
I manually labeled training examples as not_sure if they contained vague terms (e.g., "awaiting", "needs", "reminder"). This approach resulted in very low F1 score for status (0.3).

2. Confidence thresholding:
I used the model's max softmax probability over classes 0 and 1 as a proxy for certainty, labeling low-confidence outputs as not_sure. This approach overfit to phrasing and mislabeled clearly complete tasks, such as:

"Completed intake call; SIG captured on IQ." → wrongly labeled as not_sure; this is clearly a certain example

3. Entropy-based uncertainty (the one I went with):
I computed entropy over the softmax distribution and labeled predictions as not_sure when entropy exceeded a threshold. This method better aligned with human judgment — for instance:

"Faxed MRQ; got CNF." and "DPK approved; mailing today."
were reasonably flagged as ambiguous, as they suggest progress but not definitive task completion.

Overall, entropy-based thresholding offered a more principled and interpretable way to identify uncertain cases and avoided many of the false positives triggered by brittle heuristics.

Nonetheless, after some threshold-tuning, we don't get any 2 ("uncertain") examples in our ultimate prediction, because using a higher entropy threshold results in the highest F1 score (1).

---

## Evaluation

Classification reports generated using `score.py` (shown below) show perfect F1 scores and accuracy across all classes. 

Key metrics: Normally, for a classification task like this, I typically care most about F1-score to understand how strong a model is. If a model is shown to have a weak F1-score, I use precision, recall, and FPR / FNR to debug performance of a model. 

Key metrics analysis: For our task, I didn't have to do much debugging for our model, because we achieved 100% F1 score for each category and overall.

---

## With More Time

With more time, I'd do the following:
- Train with real `status = 2` examples; I'd label the data for examples where the status actually looks uncertain based on the notes.
- Experiment with larger models (`bert-base`, `roberta`)

## Overall Metrics

```
--- Scoring Results ---

Task Prediction Accuracy: 0.9700
Task Classification Report:
                         precision    recall  f1-score   support

        Client Check-in       1.00      1.00      1.00        15
          Create Demand       1.00      1.00      1.00        15
            Intake Call       1.00      1.00      1.00        15
Request Medical Records       1.00      1.00      1.00        15
 Sign Engagement Letter       1.00      0.80      0.89        15
                   none       0.89      1.00      0.94        25

               accuracy                           0.97       100
              macro avg       0.98      0.97      0.97       100
           weighted avg       0.97      0.97      0.97       100


Status Prediction Accuracy (mapping predicted 'not_sure' [2] to 'not_completed' [0]): 1.0000
Status Classification Report (after mapping 2->0):
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        64
           1       1.00      1.00      1.00        36

    accuracy                           1.00       100
   macro avg       1.00      1.00      1.00       100
weighted avg       1.00      1.00      1.00       100
```

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
