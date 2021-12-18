## NOTE: NEED TO LOAD UTILS FILES FIRST, ELSE, IT WILL THROW ERRORS. INVESTIGAGTE THIS WHEN HAVING TIME.
import files.utils_bert as utils_bert
import files.utils_bilstm as utils_bilstm

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
from transformers import pipeline, logging
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset, load_metric, Dataset

import torch
import spacy
import json
import numpy as np
from nltk import sent_tokenize
import sys
import time
import argparse

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Using the trained models to extract concepts from an input text as a string')
    parser.add_argument('--input_file', default='./data/sample/input_sample.txt', help='path to input txt file')
    parser.add_argument('--output_file', default='./outputs/output.txt', help='path to output txt file')
    args = parser.parse_args()

    print('Loading the model...')
    
    with open('files/config.json') as json_file:
        data = json.load(json_file)
        
    ### Load BERT-based models
    MODEL_NAME_UNCASED = [v for k, v in data['BertModel']['Uncased'].items()]

    MODEL_NAME_CASED = [v for k, v in data['BertModel']['Cased'].items()]

    label_list = ['B-KEY', 'I-KEY', 'O'] ## Keep the same item orders as when training models

    logging.set_verbosity_error()

    tokenizer_list_uncased = []
    model_list_uncased = []
    for m_name in MODEL_NAME_UNCASED:
        tokenizer_list_uncased.append(AutoTokenizer.from_pretrained(m_name))
        model_list_uncased.append(AutoModelForTokenClassification.from_pretrained(m_name))

    tokenizer_list_cased = []
    model_list_cased = []
    for m_name in MODEL_NAME_CASED:
        tokenizer_list_cased.append(AutoTokenizer.from_pretrained(m_name))
        model_list_cased.append(AutoModelForTokenClassification.from_pretrained(m_name))

    # #NER FROM BERT
    tokenizer_ner = AutoTokenizer.from_pretrained(data['NERModel']['Path'])
    model_ner = AutoModelForTokenClassification.from_pretrained(data['NERModel']['Path'])
    label_list_ner = list(model_ner.config.id2label.values())

    nlp = spacy.load("en_core_web_sm")


    ### Load BiLSTM-based Models
    MODEL_NAME = [v for k, v in data['BiLSTMModel']['ModelName'].items()]

    MODEL_ARG = [v for k, v in data['BiLSTMModel']['ModelArgument'].items()]

    model_list = []
    predictor_list = []

    for m_idx in range(len(MODEL_NAME)):
        _model, _predictor = utils_bilstm.initialize_models(MODEL_NAME[m_idx], MODEL_ARG[m_idx])
        model_list.append(_model)
        predictor_list.append(_predictor)


    ### Extract Concepts
    print('Extracting concepts...')
    output_id_concept_combined = {}

    with open(args.input_file, 'r') as fr:
        for line in fr:
            _id, _text = line.split('\t')

            concepts_uncased = []
            concepts_cased = []
            concepts_bilstm = []
            concepts_ner = []

            for sub_str in sent_tokenize(_text):
                idx_str = _text.find(sub_str) # the index of the sentence in the entire text

                if sub_str[-1] not in ['.', '?', '!']:
                    sub_str += '.'

                # Since all words in title are upper case, converting them to title format
            #     if col_name=='COURSE_TITLE':
            #         sub_str = utils_bert.generateTitleCase(sub_str)

                ## Predicting with BERT UNCASED models
                ## ===================================
                true_predictions_list = []

                for m_idx in range(len(MODEL_NAME_UNCASED)):
                    tokens = tokenizer_list_uncased[m_idx].tokenize(tokenizer_list_uncased[m_idx].decode(tokenizer_list_uncased[m_idx].encode(sub_str)), return_offsets_mapping=True)
                    offset_mapping = tokenizer_list_uncased[m_idx](sub_str, return_offsets_mapping=True)['offset_mapping']
                    inputs = tokenizer_list_uncased[m_idx].encode(sub_str, return_tensors="pt")

                    outputs = model_list_uncased[m_idx](inputs)[0]
                    predictions = torch.argmax(outputs, dim=2)

                    true_predictions = [label_list[p] for p in predictions[0][1:-1]]
                    true_predictions_list.append(true_predictions)

                true_predictions_combined = true_predictions_list[0]
                for i in range(1, len(true_predictions_list)):
                    true_predictions_combined = utils_bert.combine_predicted_lists(true_predictions_combined, true_predictions_list[i])

                true_predictions_combined = utils_bert.refine_prediction(true_predictions_combined)
                index_predictions = [2] + [label_list.index(p) for p in true_predictions_combined] + [2]

                token_label = [(token, offset, label_list[prediction]) for token, offset, prediction in zip(tokens, offset_mapping, index_predictions)]
                concepts_uncased = concepts_uncased + utils_bert.extraction_concepts_from_token_label(sub_str, idx_str, token_label) 

                ## Predicting with BERT CASED models
                ## ===================================
                true_predictions_list = []

                for m_idx in range(len(MODEL_NAME_CASED)):
                    tokens = tokenizer_list_cased[m_idx].tokenize(tokenizer_list_cased[m_idx].decode(tokenizer_list_cased[m_idx].encode(sub_str)), return_offsets_mapping=True)
                    offset_mapping = tokenizer_list_cased[m_idx](sub_str, return_offsets_mapping=True)['offset_mapping']
                    inputs = tokenizer_list_cased[m_idx].encode(sub_str, return_tensors="pt")

                    outputs = model_list_cased[m_idx](inputs)[0]
                    predictions = torch.argmax(outputs, dim=2)

                    true_predictions = [label_list[p] for p in predictions[0][1:-1]]
                    true_predictions_list.append(true_predictions)

                true_predictions_combined = true_predictions_list[0]
                for i in range(1, len(true_predictions_list)):
                    true_predictions_combined = utils_bert.combine_predicted_lists(true_predictions_combined, true_predictions_list[i])

                true_predictions_combined = utils_bert.refine_prediction(true_predictions_combined)
                index_predictions = [2] + [label_list.index(p) for p in true_predictions_combined] + [2]

                token_label = [(token, offset, label_list[prediction]) for token, offset, prediction in zip(tokens, offset_mapping, index_predictions)]
                concepts_cased = concepts_cased + utils_bert.extraction_concepts_from_token_label(sub_str, idx_str, token_label) 

                ## Predicting with BILSTM models
                ## ===================================
                docs = []
                nlp_s = nlp(sub_str)
                tokens = list(map(str, list(nlp_s)))
                docs.append(tokens)
                offset_mapping = [(token.idx, token.idx + len(token)) for token in nlp_s]

                true_predictions_list = []
                for m_idx in range(len(model_list)):
                    true_predictions_list.append(predictor_list[m_idx].output_batch(model_list[m_idx], [docs])[0])

                true_predictions_combined = true_predictions_list[0]
                for i in range(1, len(true_predictions_list)):
                    true_predictions_combined = utils_bilstm.combine_predicted_lists(true_predictions_combined, true_predictions_list[i])

                true_predictions_combined = utils_bilstm.refine_prediction(true_predictions_combined)

                token_label = [(token, offset, prediction) for token, offset, prediction in zip(tokens, offset_mapping, true_predictions_combined)]
                concepts_bilstm = concepts_bilstm + utils_bilstm.extraction_concepts_from_token_label(sub_str, idx_str, token_label)

                ## NER
                ## ===================================
                tokens_ner = tokenizer_ner.tokenize(tokenizer_ner.decode(tokenizer_ner.encode(sub_str)), return_offsets_mapping=True)
                offset_mapping_ner = tokenizer_ner(sub_str, return_offsets_mapping=True)['offset_mapping']
                inputs_ner = tokenizer_ner.encode(sub_str, return_tensors="pt")
                outputs_ner = model_ner(inputs_ner)[0]
                predictions_ner = torch.argmax(outputs_ner, dim=2)  

                # process ner and convert to 'KEY' labels
                true_predictions_ner = [label_list_ner[p] for p in predictions_ner[0][1:-1]]
                index_predictions_ner = [0] + [label_list_ner.index(p) for p in true_predictions_ner] + [0]
                token_label_ner = [(token, offset, label_list_ner[prediction]) for token, offset, prediction in zip(tokens_ner, offset_mapping_ner, index_predictions_ner)]
                concepts_ner = concepts_ner + utils_bert.extraction_concepts_from_token_label(sub_str, idx_str, token_label_ner)    

            combined_concepts = concepts_uncased + concepts_cased + concepts_bilstm + concepts_ner
            combined_concepts.sort(key=lambda x:x[1])
            combined_concepts = utils_bert.resolve_overlapping(combined_concepts)
            output_id_concept_combined[_id] = combined_concepts

    # Write extracted concepts to DataFrame
    output_id_concept_only = {}
    for _id, _concept in output_id_concept_combined.items():
        output_id_concept_only[_id] = list(set([c[0] for c in _concept]))

    # Filtering the results (only for concepts, not concepts with positions for now)
    utils_bert.filtering_title_concepts(output_id_concept_only)
    
    # Save the result to file
    with open(args.output_file, 'w') as fr:
        for k, v in output_id_concept_only.items():
            fr.write(k + '\t' + str(v))
            fr.write('\n')

    print("\nRunning time:", round(time.time()-start_time), 'seconds')
