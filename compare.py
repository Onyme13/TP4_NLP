"""Script that compare the results of the models to the Baseline.txt """

import os

def compare_to_baseline(lang):

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

    for language in data:
        if lang in language[0]:
            language_name = language[0]
    # Get the baseline of the language
    baseline = {}
    language_found = False
    with open("Baseline.txt", "r", encoding="utf-8") as f:
        index = 0
        for line in f:
            if lang in line:
                language_found = True
            if language_found :
                index += 1
            if index > 4 and index < 15 and not index == 11 and not index == 12:
                parts = line.split()
                if "avg" in parts:
                    baseline[parts[0]] = [parts[2], parts[3], parts[4]]
                else:
                    baseline[parts[0]] = [parts[1], parts[2], parts[3]]

    # Get the results of the model
    language = "English-EWT"
    path = f"results/{language}_results.txt"

    if not os.path.exists(path):
        print("File not found")
    
    model_metrics = {}
    with open(path, "r", encoding="utf-8") as f:
        index = 0
        for line in f:
            index += 1
            if index > 4 and index < 16 and not index == 12 and not index == 13:
                parts = line.split()
                if "avg" in parts:
                    model_metrics[parts[0]] = [parts[2], parts[3], parts[4]]
                else:
                    model_metrics[parts[0]] = [parts[1], parts[2], parts[3]]


    difference = {}
    # Compare the results
    for key in baseline:
        difference[key] = [round(float(model_metrics[key][0]) - float(baseline[key][0]),2), round(float(model_metrics[key][1]) - float(baseline[key][1]),2), round(float(model_metrics[key][2]) - float(baseline[key][2]),2)]
    print(difference)

    if not os.path.exists("compare_eval.txt"):
        with open("compare_eval.txt", "w", encoding="utf-8") as f:
            f.write("")        

    # Write the results to a file
    with open(f"compare_eval.txt", "a", encoding="utf-8") as f:
        f.write(f"UNER_{language_name}\n")
        f.write("---------------------------------------------------------\n")
        f.write("\t\tprecision\t\trecall\t\tf1-score\n")
        f.write("\n")
        for key in difference:
            f.write(f"{key}\t\t{difference[key][0]}\t\t{difference[key][1]}\t\t{difference[key][2]}\n")
        f.write("\n")
    print("\nComparason done.\n")

