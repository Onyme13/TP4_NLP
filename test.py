import json

def parse_baseline(file_path):
    baseline_data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) == 5:  
                label, precision, recall, f1 = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                baseline_data[label] = {'precision': precision, 'recall': recall, 'f1': f1}
    return baseline_data

def compare_metrics(baseline_data, fine_tuned_data):
    comparison_results = {}
    for label, baseline_metrics in baseline_data.items():
        fine_tuned_label = f'test_{label}'
        if fine_tuned_label in fine_tuned_data:
            fine_tuned_metrics = fine_tuned_data[fine_tuned_label]
            comparison_results[label] = {
                'Baseline Precision': baseline_metrics['precision'],
                'Fine-tuned Precision': fine_tuned_metrics['precision'],
                'Precision Difference': fine_tuned_metrics['precision'] - baseline_metrics['precision'],
                'Baseline Recall': baseline_metrics['recall'],
                'Fine-tuned Recall': fine_tuned_metrics['recall'],
                'Recall Difference': fine_tuned_metrics['recall'] - baseline_metrics['recall'],
                'Baseline F1': baseline_metrics['f1'],
                'Fine-tuned F1': fine_tuned_metrics['f1'],
                'F1 Difference': fine_tuned_metrics['f1'] - baseline_metrics['f1']
            }
    return comparison_results

# Replace 'baseline.txt' with your baseline file path
baseline_data = parse_baseline('Baseline.txt')

# Load your fine-tuned model results (assuming this is your JSON data)
fine_tuned_data = {
    #... (your fine-tuned model results here)
}

# Compare the metrics
comparison_results = compare_metrics(baseline_data, fine_tuned_data)

# Print or process the comparison_results as needed
print(comparison_results)
