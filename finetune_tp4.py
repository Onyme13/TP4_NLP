import argparse
import sys
import torch
import re, string
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
#from datasets import load_dataset


punctuation = string.punctuation

other_punctuation = "。，、；：？！“”‘’（）《》【】「」『』——…－"

all_punctuation = punctuation + other_punctuation
special_chars_pattern = re.compile(f"[{re.escape(all_punctuation)}]")


def read_iob2(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    sentences, sentence = [], []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split()
            token, tag = parts[1], parts[2]  
            token = special_chars_pattern.sub('', token)
            if not token:
                continue
            sentence.append((token, tag))
        elif line.startswith("#"):
            continue  
        else:
            if sentence:
                sentences.append(sentence)
                sentence = []
    
    if sentence:
        sentences.append(sentence)

    return sentences
# https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb

def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]


def load_data(lang):
    data = [
        ('Chinese-GSDSIMP', 'zh_core_web_sm', 'zh_gsdsimp-ud-train.iob2', 'zh_gsdsimp-ud-test.iob2'),
        ('Croatian-SET', 'hr_core_news_sm', 'hr_set-ud-train.iob2', 'hr_set-ud-test.iob2'),
        ('Danish-DDT', 'da_core_news_sm', 'da_ddt-ud-train.iob2', 'da_ddt-ud-test.iob2'),
        ('English-EWT', 'en_core_web_sm', 'en_ewt-ud-train.iob2', 'en_ewt-ud-test.iob2'),
        ('Portuguese-Bosque', 'pt_core_news_sm', 'pt_bosque-ud-train.iob2', 'pt_bosque-ud-test.iob2'),
        ('Serbian-SET', 'xx_ent_wiki_sm', 'sr_set-ud-train.iob2', 'sr_set-ud-test.iob2'),
        ('Slovak-SNK', 'xx_ent_wiki_sm', 'sk_snk-ud-train.iob2', 'sk_snk-ud-test.iob2'),
        ('Swedish-Talbanken', 'sv_core_news_sm', 'sv_talbanken-ud-train.iob2', 'sv_talbanken-ud-test.iob2'),
    ]
    # Load dataset
    for language in data:
        if lang in language[0]:

            train_dataset = read_iob2(f"data/UNER_{language[0]}/{language[2]}")
            test_dataset = read_iob2(f"data/UNER_{language[0]}/{language[3]}")
            train_data = [(sent2features(s), sent2labels(s)) for s in train_dataset]
            test_data = [(sent2features(s), sent2labels(s)) for s in test_dataset]


            
            return train_dataset, test_dataset, train_data, test_data
    print("Language not found")
    pass

def preprocess_data(data, tokenizer):
    # Implement function to preprocess data
    # Tokenize, encode, and format the data for the model
    pass

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
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # Train and Evaluate
    trainer.train()
    evaluation_result = trainer.evaluate()

    # Save the model
    model.save_pretrained(f'./{model_name}')

    return evaluation_result

def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('language', help='Language for which to run the NER task')
    args = parser.parse_args()



    # Load dataset
    train_dataset, test_dataset, train_data, test_data = load_data(args.language)

    # Load a pretrained model and tokenizer
    model_name = 'bert-base-multilingual-cased'  # Example model, change as needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # Preprocess data
    #train_dataset = preprocess_data(train_dataset, tokenizer)
    #test_dataset = preprocess_data(test_dataset, tokenizer)

    # Train and evaluate the model
    evaluation_result = train_and_evaluate(model, train_dataset, test_dataset, model_name)

    # Print evaluation result
    print(evaluation_result)

if __name__ == "__main__":
    main()
