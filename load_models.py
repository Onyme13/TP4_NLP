""" Script to load models from the models folder. """

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer


def load_models():
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


    for i in range(len(data)):
        model_name = data[i][1]
        AutoModelForTokenClassification.from_pretrained(model_name, num_labels=7)
        AutoTokenizer.from_pretrained(model_name)


def main():
    load_models()


if __name__ == "__main__":
    main()