import sys, os

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

import copy
import glob
import json
import os.path
from argparse import ArgumentParser, Namespace
from typing import Optional, Set
import stanza


from nltk.tokenize import WordPunctTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer

nlp = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'}) # initialize English neural pipeline
stanza.download('en', package='mimic', processors={'ner': 'i2b2'}) # download English model

from re_utils.common import (
    NerAnnotation,
    ReAnnotation,
    lower_bound,
    upper_bound,
    binary_search,
    save_jsonl,
    save_json,
)

NOT_A_NAMED_ENTITY = "O"
FIRST_TOKEN_TAG_PREFIX = "B"
SUBSEQUENT_TOKEN_TAG_PREFIX = "I"

global_sentence_count = 0
extracted_texts = []

def configure_arg_parser():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--dir",
        type=str,
        default="/home/chudeo/bert-crf-project/2", #CHUDE replace this folder with my own datapath
        help="Directory where the source data is located",
    )
    arg_parser.add_argument(
        "--hf-tokenizer",
        type=str,
        default="admis-lab/biobert-large-cased-v1.1", #CHUDE replace this model with a biomedical vert model
        help="The name of the tokenizer with which to tokenize the text. "
        "This can be a tokenizer from the hf pub or a local path.",
    )
    arg_parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length in tokens.",
    )
    arg_parser.add_argument(
        "--label2id",
        type=str,
        default=None,
        help="json file with mapping from label name to id",
    )
    arg_parser.add_argument(
        "--retag2id",
        type=str,
        default=None,
        help="json file with mapping from relation tag to id",
    )
    return arg_parser


def get_mapping_to_id(argument: Optional[str], set: Set[str]):
    if argument is not None:
        with open(argument, "r") as label2id_file:
            label2id = json.load(label2id_file)
    else:
        label2id = {label: id for id, label in enumerate(set)}
    return label2id

# Inside the function process_text_with_stanza, update the global_sentence_count
def process_text_with_stanza(text):
    global global_sentence_count
    doc = nlp(text)
    global_sentence_count += len(doc.sentences)  # Increment global_sentence_count by the number of sentences in the current document
    return doc

def find_word_indices(words, begin, end):
    # Create an empty list to store the indices of words found within the given range
    word_indices = []

    # Iterate through each token in the list of tokens
    for idx, word in enumerate(words):
        # Check if the start character of the token matches the beginning of the range
        # or if the token spans the beginning of the range
        if word.start_char == begin or (word.start_char < begin and word.end_char > begin):
            # If it matches, add the index of the token to the list of word indices
            word_indices.append(idx)
        # Check if the end character of the token matches the end of the range
        # or if the token spans the end of the range
        if word.end_char == end or (word.start_char < end and word.end_char > end):
            # If it matches, add the index of the token to the list of word indices
            word_indices.append(idx)
            break

    # Return the list of word indices found within the given range
    return word_indices


def generate_labels(words, annotations):
    labels = ['O'] * len(words)

    for annotation in annotations:
        code = annotation['code']
        # start_token_idx = annotation['start_token_idx']
        # end_token_idx = annotation['end_token_idx']

        # labels[start_token_idx] = f'B-{code}'
        # for idx in range(start_token_idx + 1, end_token_idx + 1):
        #     labels[idx] = f'I-{code}'

    return labels

def search_files(folder_path):
    hadm_id_set = set()

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.json'):
                json_file_path = os.path.join(root, filename)
                print("Processing:", json_file_path)
                extract_info_from_json(json_file_path, hadm_id_set)
                print("=" * 50)

    print("Total unique hadm_id count:", len(hadm_id_set))

def main(args):
    global extracted_texts
    global global_sentence_count
    last_sentence_id = global_sentence_count
    labels_set = set()


    search_files(args.dir)

        # Save extracted data to files
    labeled_texts_path = os.path.join(args.dir, "labeled_texts.jsonl")
    label2id_path = os.path.join(args.dir, "label2id.json")

    # Save labeled_texts
    with open(labeled_texts_path, 'w') as f:
        for idx, text_info in enumerate(extracted_texts):
            f.write(json.dumps(text_info))
            f.write('\n')

    # Save label2id
    label2id = {label: idx for idx, label in enumerate(labels_set)}
    with open(label2id_path, 'w') as f:
        json.dump(label2id, f)

def extract_info_from_json(json_file_path, hadm_id_set):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    hadm_id = data.get('hadm_id', None)
    comment = data.get('comment', None)

    if hadm_id is None:
        print(f"Warning: 'hadm_id' not found in {json_file_path}")
        return None

    hadm_id_set.add(hadm_id)

    notes = data.get('notes', [])

    for note in notes:
        note_info = {}

        note_info['hadm_id'] = hadm_id
        note_info['note_id'] = note.get('note_id', None)
        note_info['category'] = note.get('category', None)
        note_info['description'] = note.get('description', None)

        annotations = note.get('annotations', [])
        annotations_info = []

        for annotation in annotations:
            annotation_info = {}

            annotation_info['begin'] = annotation.get('begin', None)
            annotation_info['end'] = annotation.get('end', None)
            annotation_info['code'] = annotation.get('code', None)
            annotation_info['code_system'] = annotation.get('code_system', None)
            annotation_info['description'] = annotation.get('description', None)
            annotation_info['type'] = annotation.get('type', None)
            annotation_info['covered_text'] = annotation.get('covered_text', None)

            annotations_info.append(annotation_info)

        note_info['annotations'] = annotations_info
        note_info['text'] = note.get('text', None)

        if note_info['text']:
            processed_text = process_text_with_stanza(note_info['text'])

            # Extract lemma from processed_text
            tokens = []
            lemmas = []
            for sent in processed_text.sentences:
                for word in sent.words:
                    tokens.append(word.text)
                    lemmas.append(word.lemma)

            note_info['tokens'] = tokens
            note_info['lemmas'] = lemmas

        extracted_texts.append(note_info)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)


