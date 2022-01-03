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

- Pretrained BERT-based models stored on [Hugging Face account](https://huggingface.co/HungChau): specify the models you want to use in the config file (files/config.json). When running a prediction script for the first time, it will automatically download to your local machine (it will take some time to download the models).
    - Uncased model trained with IIR dataset: ``HungChau/distilbert-base-uncased-concept-extraction-iir-v1.2``
    - Cased model trained with IIR dataset: ``HungChau/distilbert-base-cased-concept-extraction-iir-v1.2``
    - Uncased model trained with KP20K dataset: ``HungChau/distilbert-base-uncased-concept-extraction-kp20k-v1.2``
    - Cased model trained with KP20K dataset: ``HungChau/distilbert-base-cased-concept-extraction-kp20k-v1.2``
    - Uncased model trained with Wikipedia dataset: ``HungChau/distilbert-base-uncased-concept-extraction-kp20k-v1.2-concept-extraction-wikipedia-v1.2``
    - Cased model trained with Wikipedia dataset: ``HungChau/distilbert-base-cased-concept-extraction-kp20k-v1.2-concept-extraction-wikipedia-v1.2``


Bi-LSTM-CRF architecture adaption for concept extraction 

<p align="center"><img width="70%" src="docs/bilstm_CE.png"/></p>

- Pretrained BiLSTM-based models: Models (including ``*.model`` and ``*.json`` files) must be downloaded to a local directory to run a prediction script (e.g., /bilstm-checkpoints/). Specify the directories to the downloaded models in the config files (files/config.json). (download all the models [[here]](https://sites.pitt.edu/~hkc6/bilstm-checkpoints.zip))
    - Uncased model trained with IIR dataset: [model checkpoint](https://drive.google.com/file/d/1LqWLRNEN0lSalOEMBtcpWQvPyWvJocwJ/view?usp=sharing) and [model argument](https://drive.google.com/file/d/1TSXMp4StGIR2zfKoh39Qu_dGyzOV-ecj/view?usp=sharing)
    - Cased model trained with IIR dataset: [model checkpoint](https://drive.google.com/file/d/1fqoROKiwNF-Oty0yNOv64qIRE4QC9XQR/view?usp=sharing) and [model argument](https://drive.google.com/file/d/1NwQzQEKDtibmM6Pa-V22tu7jaj-GEY6k/view?usp=sharing)
    - Uncased model trained with KP20K dataset: [model checkpoint](https://drive.google.com/file/d/1wkEeDzCxvKlEdvkOZR3TtTL8j2ZZJtS6/view?usp=sharing) and [model argument](https://drive.google.com/file/d/1bsWofpw9z27q57IefJricwefQ6ct6Kf_/view?usp=sharing)
    - Cased model trained with KP20K dataset: [model checkpoint](https://drive.google.com/file/d/1YtTdbEc4TpKv6eHF8Ic4x_jp2mMdsDYn/view?usp=sharing) and [model argument](https://drive.google.com/file/d/1JsLt-4VJLJsxYd1gMKKG-PWF3vp320_s/view?usp=sharing)
    - Uncased model trained with Wikipedia dataset: [model checkpoint](https://drive.google.com/file/d/1_bZyqd_Dvihrhn-s0KiwbCgOyDOMaczZ/view?usp=sharing) and [model argument](https://drive.google.com/file/d/1xUO-G2fzsbfUIK1qD1g8-S0q8rD4LiJb/view?usp=sharing)
    - Cased model trained with Wikipedia dataset: [model checkpoint](https://drive.google.com/file/d/1wQ9DBrZoaS6TXFLlIVfJBxzVs8IKulCf/view?usp=sharing) and [model argument](https://drive.google.com/file/d/1Iv0zdtWOS963_bvpuh08VSPrizWBht6G/view?usp=sharing)

BERT-based NER model: the extractor also uses a pretrained NER model to extract named entities. Specify the model name  (e.g., ``dslim/bert-base-NER``) on Hugging Face in the config files (files/config.json). 

## Installation
Install the following required libraries:
- spaCy and NLTK
- scikit-learn
- PyTorch ([guide](https://pytorch.org/get-started/locally/))
- transformers ([guide](https://huggingface.co/docs/transformers/installation))
- datasets  ([guide](https://huggingface.co/docs/datasets/installation.html))

## Data
The input text file follows the format: <doc_id>\<tab>\<text>
```
000001	This course emphasizes the study of the basic data structures of computer science (stacks, queues, trees, lists) and their implementations using the java language included in this study are programming techniques which use recursion, reference variables, and dynamic memory allocation.  Students in this course are also introduced to various searching and sorting methods and also expected to develop an intuitive understanding of the complexity of these algorithms.
000002	This course aims to expose students to different data management, data manipulation, and data analysis techniques. The class will cover all the major data management paradigms (relational/SQL, XML/Xquery, RDF/SPARQL) including NOSQL and data stream processing approaches. Going beyond traditional data management techniques, the class will expose students to information retrieval, data mining, data warehousing, network analysis, and other data analysis topics. Time permitting, the class will include big data processing techniques, such as the map/reduce framework.
```

## Usage
Config file (files/config.json): you can provide paths to the pretrained models you want to use.

We provide several scripts. The usages of these scripts can be accessed by the parameter -h, i.e.,
```
python predict.py -h
python predict_txt.py -h
```

## Prediction
It will take some time to load the models. To extract concepts from multiple documents, the best practice is to input a text file and run the second script ``predict_txt.py``.

- Input as a string

```predict.py``` is provided to extract concepts from a direct text (a sequence of words). A running command example is provided below:
```
python predict.py --input_text "Machine learning is an important subject in Computer Science." --output_file outputs/output.txt
```
The corresponding output follows the format: "doc_id"\<tab>\<list_of_concepts>
```
doc_id	['Computer Science', 'Machine learning']
```

- Input as a text file (described in [Data](#data) section)

```predict_txt.py``` is provided to extract concepts from a text file (for multiple documents). A running command example is provided below:
```
python predict_txt.py --input_file data/sample/input_sample.txt --output_file outputs/output.txt
```

The corresponding output follows the format: <doc_id>\<tab>\<list_of_concepts>
```
000001	['data structures', 'queues', 'stacks', 'reference variables', 'trees', 'programming techniques', 'dynamic memory allocation', 'complexity', 'searching', 'recursion', 'java language', 'algorithms', 'computer science', 'sorting methods', 'lists']
000002	['relational/SQL', 'data management', 'RDF/SPARQL', 'data stream processing approaches', 'data mining', 'data analysis topics', 'data management paradigms', 'data analysis techniques', 'XML/Xquery', 'information retrieval', 'big data processing techniques', 'map/reduce framework', 'NOSQL', 'data management techniques', 'data warehousing', 'class', 'data manipulation', 'network analysis']
```
