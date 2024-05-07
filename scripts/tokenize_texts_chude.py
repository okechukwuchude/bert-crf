import sys, os

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

import copy
import glob
import json
import os.path
from argparse import ArgumentParser, Namespace
from typing import Optional, Set

from nltk.tokenize import WordPunctTokenizer
from tqdm import tqdm
from transformers import AutoTokenizer

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


def configure_arg_parser():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--dir",
        type=str,
        default="/home/chudeo/bert-crf-project/work_sentence.csv", #CHUDE replace this folder with my own datapath
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

def generate_labels(words, annotations, start_token_idx, end_token_idx):
    labels = ['O'] * len(words)

    for annotation in annotations:
        code = annotation['code']

        # # Update labels for the matched words
        # if start_token_idx is not None and end_token_idx is not None:
        #     labels[start_token_idx] = f'B-{code}'
        #     for idx in range(start_token_idx + 1, end_token_idx + 1):
        #         labels[idx] = f'I-{code}'

    return labels

# Function to search for JSON files in a given folder and its subfolders
def search_files(folder_path):
    # Set to store unique hadm_id values
    hadm_id_set = set()

    # Recursively search for JSON files in the folder and its subfolders
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.json'):
                # Construct the full path to the JSON file
                json_file_path = os.path.join(root, filename)
                print("Processing:", json_file_path)
                # Call extract_info_from_json function to extract information from the JSON file
                extract_info_from_json(json_file_path, hadm_id_set)
                # Print a separator after processing each file
                print("=" * 50)

    # Print the count of unique hadm_id values
    print("Total unique hadm_id count:", len(hadm_id_set))


def main(args: Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer)

    tokenized_texts = []
    #relations = []

    #skipped_relations = 0

    labels_set = set()
    retags_set = set()

    text_id = 0
    for text_path in tqdm(
        glob.glob(
            f"{args.dir}/**/*.txt",
            recursive=True,
        )
    ):
        with open(text_path, "r") as text_file:
            text = text_file.read()

        annotation_path = os.path.join(
            os.path.dirname(text_path),
            os.path.basename(text_path).split(".")[0] + ".ann",
        )
        ner_annotations = []

        with open(annotation_path, "r") as annotation_file:
            for annotation_line in annotation_file.readlines():
                annotation_data = annotation_line.split()
                annotation_id = annotation_data[0]
                if annotation_id.startswith("T"):
                    #CHUDE modify the for loop between line 89 and 109 to read annotations from mdace data
                    ner_annotation = NerAnnotation(
                        id=annotation_id,
                        tag=annotation_data[1],
                        start_ch_pos=int(annotation_data[2]),
                        end_ch_pos=int(annotation_data[3]),
                        phrase=" ".join(annotation_data[4:]),
                    )
                    ner_annotations.append(ner_annotation)

                    labels_set.add(f"{FIRST_TOKEN_TAG_PREFIX}-{ner_annotation.tag}")
                    labels_set.add(f"{SUBSEQUENT_TOKEN_TAG_PREFIX}-{ner_annotation.tag}")


        id2annotation = {ann.id: ann for ann in ner_annotations}
        tokenized_text_spans = list(WordPunctTokenizer().span_tokenize(text))

        for id in id2annotation.keys():
            start_ch_pos = id2annotation[id].start_ch_pos
            end_ch_pos = id2annotation[id].end_ch_pos

            start_word_pos_ind = upper_bound(tokenized_text_spans, start_ch_pos, key=lambda x: x[0])
            start_word_pos_ind -= 1
            start = tokenized_text_spans[start_word_pos_ind][0]
            if start != start_ch_pos:
                end = tokenized_text_spans[start_word_pos_ind][1]
                tokenized_text_spans[start_word_pos_ind] = (start, start_ch_pos)
                tokenized_text_spans.insert(start_word_pos_ind + 1, (start_ch_pos, end))

            end_word_pos_ind = lower_bound(tokenized_text_spans, end_ch_pos, lambda x: x[1])
            end = tokenized_text_spans[end_word_pos_ind][1]
            if tokenized_text_spans[end_word_pos_ind][1] != end_ch_pos:
                start = tokenized_text_spans[end_word_pos_ind][0]
                tokenized_text_spans[end_word_pos_ind] = (end_ch_pos, end)
                tokenized_text_spans.insert(end_word_pos_ind, (start, end_ch_pos))

        for id in id2annotation.keys():
            id2annotation[id].start_word_pos = binary_search(
                tokenized_text_spans, id2annotation[id].start_ch_pos, lambda x: x[0]
            )
            id2annotation[id].end_word_pos = binary_search(
                tokenized_text_spans, id2annotation[id].end_ch_pos, lambda x: x[1]
            )

        for annotation in id2annotation.values():
            assert annotation.start_word_pos != -1 and annotation.end_word_pos != -1

        words = [text[span[0] : span[1]] for span in tokenized_text_spans]
        encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        words_ids_for_tokens = encoded.word_ids()

        for id in id2annotation.keys():
            id2annotation[id].start_token_pos = lower_bound(words_ids_for_tokens, id2annotation[id].start_word_pos)
            id2annotation[id].end_token_pos = upper_bound(words_ids_for_tokens, id2annotation[id].end_word_pos)

        text_labels = ["O"] * len(input_ids)
        for annotation in id2annotation.values():
            text_labels[annotation.start_token_pos] = f"{FIRST_TOKEN_TAG_PREFIX}-{annotation.tag}"
            for i in range(annotation.start_token_pos + 1, annotation.end_token_pos):
                text_labels[i] = f"{SUBSEQUENT_TOKEN_TAG_PREFIX}-{annotation.tag}"

        labels_set.add(NOT_A_NAMED_ENTITY)

        current_seq_ids = []
        current_seq_labels = []
        dump = {"input_ids": [], "text_labels": [], "labels": []}
        total_token_dumped = 0

        
    label2id = get_mapping_to_id(args.label2id, labels_set)

    for i in range(len(tokenized_texts)):
        tokenized_texts[i]["labels"] = [label2id[label] for label in tokenized_texts[i]["text_labels"]]

    save_jsonl(tokenized_texts, os.path.join(args.dir, "labeled_texts.jsonl"))
    #save_jsonl(relations, os.path.join(args.dir, "relations.jsonl"))
    save_json(label2id, os.path.join(args.dir, "label2id.json"))
    #save_json(retag2id, os.path.join(args.dir, "retag2id.json"))
    #print(skipped_relations)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
