import json
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from nltk.translate.chrf_score import sentence_chrf
from Levenshtein import distance as levenshtein_distance

# Load the annotated results from the JSON file
with open("code_completion_results_annotated.json", "r") as f:
    results = json.load(f)

# Ensure every object has the "correct" key (just to be sure)
for entry in results:
    if "correct" not in entry:
        raise ValueError("Missing 'correct' key in entry")

    # Calculate each metric and add to the entry if not already present
    entry["exact_match"] = int(entry["fill"].strip() == entry["middle"].strip())
    entry["chrf_score"] = sentence_chrf([entry["middle"]], entry["fill"])
    entry["bleu_score"] = sentence_bleu([entry["middle"].split()], entry["fill"].split())
    entry["levenshtein"] = levenshtein_distance(entry["middle"], entry["fill"])

with open("code_completion_results_annotated_with_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

df = pd.DataFrame(results)
correlations = {
    "exact_match": df["exact_match"].corr(df["correct"]),
    "chrf_score": df["chrf_score"].corr(df["correct"]),
    "bleu_score": df["bleu_score"].corr(df["correct"]),
    "levenshtein": (df["levenshtein"].corr(df["correct"]))
}

# Display the correlation results and choose the "best" metric
best_metric = max(correlations, key=correlations.get)
print("Metric Correlations with 'correct' Label:")
for metric, corr in correlations.items():
    print(f"{metric}: {corr:.4f}")

print(f"\nBest metric based on correlation: {best_metric} (Correlation: {correlations[best_metric]:.4f})")