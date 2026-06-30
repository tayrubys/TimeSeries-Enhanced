from src.experiments.evaluator import evaluate_binary_classification, print_metrics


y_true = [0, 0, 1, 1, 0, 1]
y_pred = [0, 1, 1, 1, 0, 0]

metrics = evaluate_binary_classification(
    y_true=y_true,
    y_pred=y_pred
)

print_metrics(metrics)