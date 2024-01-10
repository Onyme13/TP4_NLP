 ## To run the script 

1. Install required dependencies:
    ``
    pip install -r requirements.txt
    ``
2. To train the model, run:
``
    python finetune_tp4.py [Language]
``
Replace [Language] with the desired language code (e.g., Portuguese, Chinese, etc.).

## Pretrained Models

For the NER tasks, different pre-trained models were used for each language. Below is the list of these models:

- **English (EWT)**
  - **Model**: bert-base-uncased
  - **Type**: BERT
  - **Parameters**: ~110 million

- **Chinese (GSDSIMP)**
  - **Model**: bert-base-chinese
  - **Type**: BERT
  - **Parameters**: ~110 million (Typical for BERT base models)

- **Portuguese (Bosque)**
  - **Model**: neuralmind/bert-base-portuguese-cased
  - **Type**: BERT
  - **Parameters**: Similar to BERT base models

- **Swedish (Talbanken)**
  - **Model**: KB/bert-base-swedish-cased
  - **Type**: BERT
  - **Parameters**: Similar to BERT base models

- **Slovak (SNK)**
  - **Model**: bert-base-multilingual-uncased
  - **Type**: BERT
  - **Parameters**: ~110 million (Similar to BERT base models)

- **Danish (DDT)**
  - **Model**: Maltehb/danish-bert-botxo
  - **Type**: BERT
  - **Parameters**: Specific details might vary

- **Croatian (SET)**
  - **Model**: classla/bcms-bertic
  - **Type**: BERT
  - **Parameters**: Specific details might vary

- **Serbian (SET)**
  - **Model**: classla/bcms-bertic
  - **Type**: BERT
  - **Parameters**: Specific details might vary

Note: The parameters count for some models are approximate.
