# Concept Extraction for Course Description

This project provides a concept extractor for college course descriptions. The current models mainly focus on courses in computer science and information science domains. The extraction models were trained with weak labels from public resources such as [IIR dataset](https://github.com/PAWSLabUniversityOfPittsburgh/Concept-Extraction/tree/master/IIR-dataset), [KP20 dataset](https://github.com/memray/seq2seq-keyphrase) or Wikipedia pages. The extractor is the combination of different pretrained BERT-based and BiLSTM-based cased/uncased models on different datasets.

## Quick Links

- [Model](#model)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Prediction](#prediction)

## Model

BERT architecture adaption for concept extraction 

<p align="center"><img width="50%" src="docs/bert_CE.png"/></p>

- Pretrained BERT-based models stored on [Hugging Face account](https://huggingface.co/HungChau): specify the models you want to use in the config file (files/config.json). When running a prediction script for the first time, it will automatically download to your local machine.
    - Uncased model trained with IIR dataset: ``HungChau/distilbert-base-uncased-concept-extraction-iir-v1.2``
    - Cased model trained with IIR dataset: ``HungChau/distilbert-base-cased-concept-extraction-iir-v1.2``
    - Uncased model trained with KP20K dataset: ``HungChau/distilbert-base-uncased-concept-extraction-kp20k-v1.2``
    - Cased model trained with KP20K dataset: ``HungChau/distilbert-base-cased-concept-extraction-kp20k-v1.2``
    - Uncased model trained with Wikipedia dataset: ``HungChau/distilbert-base-uncased-concept-extraction-kp20k-v1.2-concept-extraction-wikipedia-v1.2``
    - Cased model trained with Wikipedia dataset: ``HungChau/distilbert-base-cased-concept-extraction-kp20k-v1.2-concept-extraction-wikipedia-v1.2``


Bi-LSTM-CRF architecture adaption for concept extraction 

<p align="center"><img width="70%" src="docs/bilstm_CE.png"/></p>

- Pretrained BiLSTM-based models: Models (including ``*.model`` and ``*.json`` files) must be downloaded to a local directory to run a prediction script (e.g., /checkpoints/). Specify the directories to the downloaded models in the config files (files/config.json).
    - Uncased model trained with IIR dataset: 
    - Cased model trained with IIR dataset: 
    - Uncased model trained with KP20K dataset: 
    - Cased model trained with KP20K dataset:
    - Uncased model trained with Wikipedia dataset:
    - Cased model trained with Wikipedia dataset:

## Installation
Install the following required libraries:
- spaCy and NLTK
- scikit-learn
- PyTorch
- transformers

## Data

## Usage
Config file (files/config.json): you can provide paths to the pretrained models you want to use.

We provide several scripts. The usages of these scripts can be accessed by the parameter -h, i.e.,
```
python predict.py -h
python predict_txt.py -h
```

## Prediction
```predict.py``` is provided to extract concepts from a direct text (a sequence of words). A running command example is provided below:
```
python predict.py --input_text "Machine learning is an important subject in Computer Science." --output_file outputs/output.txt
```

```predict_txt.py``` is provided to extract concepts from a text file (for multiple documents). A running command example is provided below:
```
python predict_txt.py --input_file data/sample/input_sample.txt --output_file outputs/output.txt
```
