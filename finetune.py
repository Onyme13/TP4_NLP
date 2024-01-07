import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification



def load_data(lang):
    data = [
        ('Chinese-GSDSIMP', 'bert-base-chinese', 'zh_gsdsimp-ud-train.iob2', 'zh_gsdsimp-ud-test.iob2', 'zh_gsdsimp-ud-dev.iob2'),
        ('Croatian-SET', 'classla/bcms-bertic', 'hr_set-ud-train.iob2', 'hr_set-ud-test.iob2', 'hr_set-ud-dev.iob2'),
        ('Danish-DDT', 'Maltehb/danish-bert-botxo', 'da_ddt-ud-train.iob2', 'da_ddt-ud-test.iob2', 'da_ddt-ud-dev.iob2'),
        ('English-EWT', 'bert-base-uncased', 'en_ewt-ud-train.iob2', 'en_ewt-ud-test.iob2', 'en_ewt-ud-dev.iob2'),
        ('Portuguese-Bosque', 'neuralmind/bert-base-portuguese-cased', 'pt_bosque-ud-train.iob2', 'pt_bosque-ud-test.iob2', 'pt_bosque-ud-dev.iob2'),
        ('Serbian-SET', 'classla/bcms-bertic', 'sr_set-ud-train.iob2', 'sr_set-ud-test.iob2', 'sr_set-ud-dev.iob2'),
        ('Slovak-SNK', 'bert-base-multilingual-uncased', 'sk_snk-ud-train.iob2', 'sk_snk-ud-test.iob2','sk_snk-ud-test.iob2'),
        ('Swedish-Talbanken', 'KB/bert-base-swedish-cased', 'sv_talbanken-ud-train.iob2', 'sv_talbanken-ud-test.iob2','sv_talbanken-ud-dev.iob2'),
    ]
    # Load dataset
    for language in data:
        if lang in language[0]:
            train_file = f"data/UNER_{language[0]}/{language[2]}"
            test_file = f"data/UNER_{language[0]}/{language[3]}"
            dev_file = f"data/UNER_{language[0]}/{language[4]}"
            return  train_file, test_file

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


#FOR LATER USE
def load_model_and_tokenizer(model_name, num_labels):
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def tokenize_and_align_labels(sentences, labels, tokenizer):
    tokenized_inputs = tokenizer(sentences, truncation=True, padding=True, is_split_into_words=True)
    aligned_labels = []

    for i, label_list in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Maps each token to word in the original sentence
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # Special tokens have a word id of None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_list[word_idx])
            # For the other tokens in a word, we set the label to -100.
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx
        
        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def main():

    # First parse the arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument('language', help='Language for which to run the NER task')
    args = parser.parse_args()
    language = args.language
    print(f'Running NER for language: {language}')

    # Load the data
    train_file, test_file = load_data(language)

    # Process the data
    train_sentences, train_tags = iob2_to_sentences(train_file) 
    test_sentences, test_tags = iob2_to_sentences(test_file) 

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
    model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=7)

    # Save the model
    save_directory = f"models/{language}"
    #model.save_pretrained(save_directory)
    #tokenizer.save_pretrained(save_directory)

    # Tokenize the data and align the labels
    train_tokenized = tokenize_and_align_labels(train_sentences, train_tags, tokenizer)
    test_tokenized = tokenize_and_align_labels(test_sentences, test_tags, tokenizer)
    print("Train tokenized:")
    print(train_tokenized[0])
    




if __name__ == '__main__':
    main()