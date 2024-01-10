import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from load_models import load_models
import os
from compare import compare_to_baseline

def load_data(lang):
    data = [
        ('Chinese-GSDSIMP', 'bert-base-chinese', 'zh_gsdsimp-ud-train.iob2', 'zh_gsdsimp-ud-test.iob2', 'zh_gsdsimp-ud-dev.iob2'),
        ('Croatian-SET', 'bert-base-multilingual-uncased', 'hr_set-ud-train.iob2', 'hr_set-ud-test.iob2', 'hr_set-ud-dev.iob2'),
        ('Danish-DDT', 'Maltehb/danish-bert-botxo', 'da_ddt-ud-train.iob2', 'da_ddt-ud-test.iob2', 'da_ddt-ud-dev.iob2'),
        ('English-EWT', 'bert-base-uncased', 'en_ewt-ud-train.iob2', 'en_ewt-ud-test.iob2', 'en_ewt-ud-dev.iob2'),
        ('Portuguese-Bosque', 'neuralmind/bert-base-portuguese-cased', 'pt_bosque-ud-train.iob2', 'pt_bosque-ud-test.iob2', 'pt_bosque-ud-dev.iob2'),
        ('Serbian-SET', 'bert-base-multilingual-uncased', 'sr_set-ud-train.iob2', 'sr_set-ud-test.iob2', 'sr_set-ud-dev.iob2'),
        ('Slovak-SNK', 'bert-base-multilingual-uncased', 'sk_snk-ud-train.iob2', 'sk_snk-ud-test.iob2','sk_snk-ud-test.iob2'),
        ('Swedish-Talbanken', 'KB/bert-base-swedish-cased', 'sv_talbanken-ud-train.iob2', 'sv_talbanken-ud-test.iob2','sv_talbanken-ud-dev.iob2'),
    ]
    # Load dataset
    for language in data:
        if lang in language[0]:
            language_name = language[0]
            model_name = language[1]
            train_file = f"data/UNER_{language[0]}/{language[2]}"
            test_file = f"data/UNER_{language[0]}/{language[3]}"
            dev_file = f"data/UNER_{language[0]}/{language[4]}"
            return language_name, model_name, train_file, test_file

    print("Language not found")
    pass

def iob2_to_sentences(file_path):
    sentences = []
    tags = []
    current_sentence_words = []
    current_sentence_tags = []
    full_sentences = [] # For a list of sentences 

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line.startswith('#'):
                continue

            parts = line.split('\t')

            if len(parts) > 2:
                current_sentence_words.append(parts[1])
                current_sentence_tags.append(parts[2])
            else:
                if current_sentence_words:
                    sentences.append(current_sentence_words)
                    tags.append(current_sentence_tags)
                    #full_sentences.append(current_sentence_words)
                current_sentence_words = []
                current_sentence_tags = []

    return sentences, tags        


def load_model_and_tokenizer(model_name, num_labels):
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def convert_labels_to_ids(labels, label_to_id):
    return [[label_to_id[label] for label in sentence_labels] for sentence_labels in labels]


def tokenize_and_align_labels(sentences, labels, tokenizer):
    tokenized_inputs = tokenizer(sentences, truncation=True, padding=True, is_split_into_words=True)
    aligned_labels = []

    for i, label_list in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Maps each token to word in the original sentence
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Special tokens have a word id of None. We set the label to -100 so they are automatically ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_list[word_idx])
            # For the other tokens in a word, if the word is a part of an entity, we continue the entity label
            else:
                if label_list[word_idx].startswith("B-"):
                    label_ids.append("I" + label_list[word_idx][1:])
                else:
                    label_ids.append(label_list[word_idx])

            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def convert_labels_to_ids(labels, label2id):
    converted_labels = []
    for label_list in labels:
        converted_label_list = []
        for label in label_list:
            if label == -100:
                # Keep -100 as it is
                converted_label_list.append(-100)
            elif label not in label2id:
                # Convert the label to the label for unknown tokens - 'O'
                converted_label_list.append(label2id['O'])
            else:
                # Convert string labels to their corresponding integer IDs
                converted_label_list.append(label2id[label])
        converted_labels.append(converted_label_list)
    return converted_labels



class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # Encodings are your tokenized input ids
        self.labels = labels  # Labels are the integer labels for each token

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def train_model(model, train_dataset, test_dataset):

    # For AMD GPU 6800xt:

    training_args = TrainingArguments(
        output_dir="./model_output",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128, 
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    evaluation_results = trainer.evaluate()

    # Generate predictions
    predictions, label_ids, _ = trainer.predict(test_dataset)

    return evaluation_results, predictions, label_ids




def evaluate_model(predictions, labels, test_dataset, id2label):
    # Convert logits to label IDs
    predictions = np.argmax(predictions, axis=2)

    # Flatten the predictions and true labels
    flat_true_labels = []
    flat_predictions = []

    for i in range(len(labels)):
        # Filter out '-100' used for special tokens and padding
        label_seq = [l for l in labels[i] if l != -100]
        pred_seq = [predictions[i][j] for j, l in enumerate(labels[i]) if l != -100]

        # Extend the flat list
        flat_true_labels.extend(label_seq)
        flat_predictions.extend(pred_seq)

    # Convert label IDs to label names
    flat_true_labels = [id2label[label_id] for label_id in flat_true_labels]
    flat_pred_labels = [id2label[pred_id] for pred_id in flat_predictions]

    tag_names =  ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    # Generate classification report
    report = classification_report(flat_true_labels, flat_pred_labels, target_names=tag_names, zero_division=1)
    return report





def main():

    label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}
    id2label = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}

    # Preload the models if needed or wanted
    #load_models()

    # First parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('language', help='Language for which to run the NER task')
    args = parser.parse_args()
    language = args.language
    print(f'Running NER for language: {language}')

    # Load the data
    language_name, model_name, train_file, test_file = load_data(language)

    # Process the data
    train_sentences, train_tags = iob2_to_sentences(train_file) 
    test_sentences, test_tags = iob2_to_sentences(test_file) 

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label2id))



    # Save the model if needed
    save_directory = f"models/{language}"
    model.save_pretrained(save_directory)
    #tokenizer.save_pretrained(save_directory)


    # Tokenize the data and align the labels
    train_tokenized = tokenize_and_align_labels(train_sentences, train_tags, tokenizer)
    test_tokenized = tokenize_and_align_labels(test_sentences, test_tags, tokenizer)



    # Convert the labels to ids
    train_tokenized['labels'] = convert_labels_to_ids(train_tokenized['labels'], label2id)
    test_tokenized['labels'] = convert_labels_to_ids(test_tokenized['labels'], label2id)



    # Creating Dataset
    train_dataset = NERDataset(train_tokenized, train_tokenized['labels'])
    test_dataset = NERDataset(test_tokenized, test_tokenized['labels'])

    # Train and evaluate the model
    evaluation_results, predictions, labels = train_model(model, train_dataset, test_dataset)

    # Generate report
    report = evaluate_model(predictions, labels, test_dataset, id2label)



    # Save the evaluation results
    save_file = f"results/{language_name}_results.txt"
    with open(save_file, 'w') as f:
        f.write(f"UNER_{language_name}\n")
        f.write("---------------------------------------------------------\n")
        f.write(report)


    # Compare to baseline
    compare_to_baseline(language)

    print("Evaluation results:")
    print(evaluation_results)


if __name__ == '__main__':
    main()