import argparse
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import os
import torch


# Load DistilBERT tokenizer
model_name = 'distilbert-base-uncased'  # change as needed
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)




# Function to process the IOB2 files and extract sentences and tags
def process_iob2_file(file_path):
    sentences = []
    sentence_tags = []
    current_sentence = []
    current_tags = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Skip lines with comments
            if line.startswith('#'):
                continue
            
            # Handle end of a sentence
            if line.strip() == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    sentence_tags.append(current_tags)
                    current_sentence = []
                    current_tags = []
                continue

            # Extract word and tag
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                word, tag = parts[1], parts[2]
                current_sentence.append(word)
                current_tags.append(tag)

    return sentences, sentence_tags

# Process the train and test files

def load_data(lang):
    data = [
        ('Chinese-GSDSIMP', 'bert-base-chinese', 'zh_gsdsimp-ud-train.iob2', 'zh_gsdsimp-ud-test.iob2'),
        ('Croatian-SET', 'classla/bcms-bertic', 'hr_set-ud-train.iob2', 'hr_set-ud-test.iob2'),
        ('Danish-DDT', 'Maltehb/danish-bert-botxo', 'da_ddt-ud-train.iob2', 'da_ddt-ud-test.iob2'),
        ('English-EWT', 'bert-base-uncased', 'en_ewt-ud-train.iob2', 'en_ewt-ud-test.iob2'),
        ('Portuguese-Bosque', 'neuralmind/bert-base-portuguese-cased', 'pt_bosque-ud-train.iob2', 'pt_bosque-ud-test.iob2'),
        ('Serbian-SET', 'classla/bcms-bertic', 'sr_set-ud-train.iob2', 'sr_set-ud-test.iob2'),
        ('Slovak-SNK', 'bert-base-multilingual-uncased', 'sk_snk-ud-train.iob2', 'sk_snk-ud-test.iob2'),
        ('Swedish-Talbanken', 'KB/bert-base-swedish-cased', 'sv_talbanken-ud-train.iob2', 'sv_talbanken-ud-test.iob2'),
    ]
    # Load dataset
    for language in data:
        if lang in language[0]:
            model = language[1]
            train_sentences, train_tags = process_iob2_file(f"data/UNER_{language[0]}/{language[2]}")
            test_sentences, test_tags = process_iob2_file(f"data/UNER_{language[0]}/{language[3]}")
            
            return model, train_sentences, train_tags, test_sentences, test_tags
    print("Language not found")
    pass





def transform_to_dict(train_token, train_label, test_token, test_label,label_to_id):
    raw_dict_train = {}
    raw_dict_test = {}
    for i in range(len(train_label)):
        raw_dict_train[i] = {}
        raw_dict_train[i]['words'] = train_token[i]
        raw_dict_train[i]['labels'] = train_label[i]

        ner_tags = []
        for j in range(len(train_label[i])):
            ner_tags.append(label_to_id[train_label[i][j]])
        
        raw_dict_train[i]['ner_tags'] = ner_tags

    for i in range(len(test_label)):
        raw_dict_test[i] = {}
        raw_dict_test[i]['words'] = test_token[i]
        raw_dict_test[i]['labels'] = test_label[i]

        ner_tags = []
        for j in range(len(test_label[i])):
            ner_tags.append(label_to_id[test_label[i][j]])
        
        raw_dict_test[i]['ner_tags'] = ner_tags
    
    return raw_dict_train, raw_dict_test



def raw_data_to_list_of_dict(raw_dict_train,raw_dict_test):
    train_list_prep = []
    test_list_prep = []

    for i, data in raw_dict_train.items():
        train_list_prep.append({
            'id': i,
            'words': data['words'],
            'ner_tags': data['ner_tags'],
            'pos_tags': [],
            'chunk_tags': []
        })
    for i, data in raw_dict_test.items():
        test_list_prep.append({
            'id': i,
            'words': data['words'],
            'ner_tags': data['ner_tags'],
            'pos_tags': [],
            'chunk_tags': []
        })

    return train_list_prep, test_list_prep


def align_labels_with_tokens(examples):
    label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", ...]  # Add all labels
    label_to_id = {label: i for i, label in enumerate(label_list)}
    
    tokenized_inputs = tokenizer(examples["words"], truncation=True, padding="max_length", is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None, we set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label_list[label[word_idx]]])
            # For the other tokens in a word, we set the label to -100
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs



"""
def align_tags_with_tokens(tokenized_data, tags, label_to_id):
    aligned_labels = []

    for i, sentence_tags in enumerate(tags):
        word_ids = tokenized_data.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_str = sentence_tags[word_idx]  # Get the string label
                label_id = label_to_id[label_str]    # Convert to integer ID
                label_ids.append(label_id)
            # For the other tokens in a word, we set the label to -100
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        aligned_labels.append(torch.tensor(label_ids))

    return aligned_labels
"""

def train_and_evaluate(model, train_dataset, test_dataset, model_name):


    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # Output directory
        num_train_epochs=3,              # Total number of training epochs
        per_device_train_batch_size=16,  # Batch size per device during training
        per_device_eval_batch_size=64,   # Batch size for evaluation
        warmup_steps=500,                # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # Strength of weight decay
        logging_dir='./logs',            # Directory for storing logs
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'],
        eval_dataset=test_dataset['test']
    )

    # Train and Evaluate
    trainer.train()
    evaluation_result = trainer.evaluate()

    # Save the model
    model.save_pretrained(f'./{model_name}')

    return evaluation_result



def main():

    # For reference, the labels are:
    REF_label_to_id ={
    "O": 0,
    "B-LOC": 1,
    "B-ORG": 2,
    "B-PER": 3,
    "I-LOC": 4,
    "I-ORG": 5,
    "I-PER": 6
    }

    def tokenize_function(data):
        return tokenizer(data["words"], truncation=True, padding="max_length", is_split_into_words=True)


    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('language', help='Language for which to run the NER task')
    args = parser.parse_args()


    mod, train_sentences, train_tags, test_sentences, test_tags = load_data(args.language)

    # Transform the data into a dictionary
    train_dict, test_dict = transform_to_dict(train_sentences, train_tags, test_sentences, test_tags, REF_label_to_id)

    # Transform the dictionary into a list of dictionariesssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMqXrquTlkttdgsbkhVWDrAcI/TbQxwDgFKcc8xLrDAC jocorl@hotmail.com

    train_list, test_list = raw_data_to_list_of_dict(train_dict, test_dict)

    # Convert it to Hugging Face Dataset format
    train_dataset = Dataset.from_dict({k: [d[k] for d in train_list] for k in train_list[0]})
    test_dataset = Dataset.from_dict({k: [d[k] for d in test_list] for k in test_list[0]})

    # Create a DatasetDict
    raw_data_train = DatasetDict({"train": train_dataset})
    raw_data_test = DatasetDict({"test": test_dataset})

    # Tokenizing the data
    tokenized_train_data = raw_data_train.map(tokenize_function, batched=True)
    tokenized_test_data = raw_data_test.map(tokenize_function, batched=True)

    tokenized_train_data = raw_data_train.map(align_labels_with_tokens, batched=True)
    tokenized_test_data = raw_data_test.map(align_labels_with_tokens, batched=True)





    # Tokenizing the data 
    #tokenized_train_data = tokenizer(train_sentences, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
    #tokenized_test_data = tokenizer(test_sentences, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)


    #train_labels = align_tags_with_tokens(tokenized_train_data, train_tags, label_to_id)
    #tokenized_train_data['labels'] = train_labels
    #test_labels = align_tags_with_tokens(tokenized_train_data, test_tags, label_to_id)
    #tokenized_test_data['labels'] = test_labels



    # Load DistilBERT model
    model = DistilBertForTokenClassification.from_pretrained(model_name, num_labels=7)
    #model = DistilBertForTokenClassification.from_pretrained(mod, num_labels=9)


    evaluation_result = train_and_evaluate(model, tokenized_train_data, tokenized_test_data, model_name)

    # Print evaluation result
    print(evaluation_result)




if __name__ == "__main__":
    main()

