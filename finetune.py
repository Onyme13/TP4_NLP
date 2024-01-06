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

def iob2_to_senteces(file_path):
    sentences = []
    current_sentence_words = []
    current_sentence_tags = []
    full_sentences = [] # For a list of sentences 

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            # Capture full sentence from metadata
            if line.startswith("# text ="):
                full_sentence = line.split("= ")[1]
                full_sentences.append(full_sentence)
                continue

            # Check for the end of a sentence
            if not line:
                if current_sentence_words:
                    sentences.append((current_sentence_words, current_sentence_tags))
                    current_sentence_words = []
                    current_sentence_tags = []
            else:
                parts = line.split('\t')
                if len(parts) > 2:
                    word, tag = parts[1], parts[3]  
                    current_sentence_words.append(word)
                    current_sentence_tags.append(tag)

        # Add the last sentence if the file doesn't end with a newline
        if current_sentence_words:
            sentences.append((current_sentence_words, current_sentence_tags))

    return full_sentences


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
    train_sentences = iob2_to_senteces(train_file)
    test_sentences = iob2_to_senteces(test_file)

    # Load the model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
    model = AutoModelForTokenClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=7)

    # Tokenize the data
    train_batch = tokenizer(train_sentences, truncation=True, padding=True, return_tensors="pt")




if __name__ == '__main__':
    main()