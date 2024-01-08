label2id = {label: i for i, label in enumerate(['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'])}
id2label = {i: label for label, i in label2id.items()}

print(label2id)
print()
print(id2label)