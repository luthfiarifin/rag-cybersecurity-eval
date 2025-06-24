import pandas as pd
import re
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from nltk.translate.bleu_score import sentence_bleu
from src.config import Config


def parse_generated_answer(answer_text: str) -> int:
    """
    Robustly parses the generated answer text to find the chosen index.
    Searches for numbers in parentheses or the first number it can find.
    """
    # Regex to find a number inside parentheses, e.g., (3)
    match = re.search(r"\((\d+)\)", answer_text)
    if match:
        return int(match.group(1))

    # If no number in parentheses, find the first digit in the string
    match = re.search(r"\d", answer_text)
    if match:
        return int(match.group(0))

    # If no number is found, return a value indicating failure
    return -1


def evaluate_performance(
    questions: list, generated_answers: list, ground_truths: list, choices_list: list
):
    """
    Evaluates the RAG system's performance with improved logging and confusion matrix.
    """
    print("--- Starting Evaluation ---")

    predicted_indices = []
    correct_answers_text = []
    predicted_answers_text = []

    for i, gen_answer in enumerate(generated_answers):
        true_index = ground_truths[i]
        choices = ast.literal_eval(
            choices_list[i]
        )  # Safely evaluate string representation of list

        # Get the text of the correct answer
        correct_text = (
            choices[true_index - 1] if 0 < true_index <= len(choices) else "N/A"
        )
        correct_answers_text.append(correct_text)

        # Parse the generated answer to get the predicted index
        predicted_index = parse_generated_answer(gen_answer)
        predicted_indices.append(predicted_index)

        # Get the text of the predicted answer
        predicted_text = (
            choices[predicted_index - 1]
            if 0 < predicted_index <= len(choices)
            else "N/A"
        )
        predicted_answers_text.append(predicted_text)

    # --- Quantitative Metrics ---
    # Accuracy
    correct_predictions = sum(
        1 for i, p_idx in enumerate(predicted_indices) if p_idx == ground_truths[i]
    )
    accuracy = correct_predictions / len(ground_truths) if len(ground_truths) > 0 else 0

    # F1 Score (Macro)
    # Use indices for classification report. Ensure labels are consistent.
    labels = sorted(list(set(ground_truths)))
    f1 = f1_score(
        ground_truths,
        predicted_indices,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    # BLEU Score (Average)
    bleu_scores = [
        sentence_bleu([str(gt).split()], str(p).split())
        for gt, p in zip(correct_answers_text, predicted_answers_text)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    print("\n--- Quantitative Metrics Summary ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Hallucination Rate: Not calculated (context unavailable)")

    # --- Qualitative Analysis ---
    print("\n--- Detailed Comparison Log ---")
    comparison_df = pd.DataFrame(
        {
            "Question": questions,
            "Correct Answer": correct_answers_text,
            "Predicted Answer": predicted_answers_text,
            "Is Correct?": [p == t for p, t in zip(predicted_indices, ground_truths)],
        }
    )
    print(comparison_df.to_markdown(index=False))

    # --- Confusion Matrix ---
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(ground_truths, predicted_indices, labels=labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[f"Choice {l}" for l in labels]
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix of Predicted vs. True Answers")
    plt.show()
