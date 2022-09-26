import os
import re
import json
import numpy as np
from typing import Dict, List, Union
from dadc_tools.utils import get_unique_order_preserving
from dadc_tools.settings import RANDOM_SEED


def load_squad(data_path: str, return_flat: bool=False) -> Union[List, Dict]:
    """Load a SQuAD format dataset"""
    with open(os.path.expanduser(data_path), "r") as f:
        squad_data = json.load(f)

    squad_data = squad_data['data']

    if return_flat:
        return flatten_squad(squad_data)

    return squad_data


def save_squad(squad_dict: Dict, data_path: str) -> None:
    """Save a SQuAD format dataset"""
    with open(os.path.expanduser(data_path), "w") as f:
        json.dump(squad_dict, f)


def flatten_squad(squad_dict: Dict, ref: str='', verbose:bool=False) -> List:
    """Convert a loaded SQuAD file into a flat list of dictionaries - one for each QA pair"""
    squad_flat = [
        {
            'id': qa['id'],
            'title': dp['title'],
            'ref': ref,
            'context': para['context'],
            'question': qa['question'],
            'answers': qa['answers']
        }
        for dp in squad_dict
            for para in dp['paragraphs']
                for qa in para['qas']
    ]

    if verbose:
        print("Flattened to {} QA pairs".format(len(squad_flat)))

    return squad_flat


def unflatten_squad(squad_flat: List, version: str='', has_metadata: bool=False, verbose: bool=False) -> Dict:
    """Convert a flat list of SQuAD examples to the standard SQuAD format"""
    squad_dict = {
        'data': [],
        'version': version
    }

    count = 0
    unique_titles = get_unique_order_preserving([dp['title'] for dp in squad_flat])

    if verbose:
        print(f"Flat version has {len(squad_flat)} QAs")
    
    dps_by_title = {}
    dps_by_title_and_context = {}
    for dp in squad_flat:
        dps_by_title.setdefault(dp['title'], []).append(dp)
        dps_by_title_and_context.setdefault((dp['title'], dp['context']), []).append(dp)

    for title in unique_titles:
        title_dps = dps_by_title[title]
        unique_contexts = get_unique_order_preserving([dp['context'] for dp in title_dps])

        data_item = {
            'title': title,
            'paragraphs': []
        }

        for context in unique_contexts:
            context_dps = dps_by_title_and_context[(title, context)]
            if has_metadata:
                context_qas = [
                    {
                        'id': dp['id'],
                        'question': dp['question'],
                        'answers': dp['answers'],
                        'metadata': json.dumps(dp['metadata']) if 'metadata' in dp else {}

                    } for dp in context_dps
                ]
            else:
                context_qas = [
                    {
                        'id': dp['id'],
                        'question': dp['question'],
                        'answers': dp['answers']
                    } for dp in context_dps
                ]
            count += len(context_qas)

            dict_para = {
                'context': context,
                'qas': context_qas
            }

            data_item['paragraphs'].append(dict_para)

        squad_dict['data'].append(data_item)

    if verbose:
        print("Total QAs added: {}".format(count))

    return squad_dict


def finalise_squad_data_list(squad_data_list: List, version: str='') -> Dict:
    """Convert a SQuAD data list to SQuAD final format"""
    return {
        'data': squad_data_list,
        'version': version
    }


def shuffle_flat_squad(squad_flat: List, random_seed: int=RANDOM_SEED) -> List:
    """Shuffle the order of a flat list of SQuAD examples"""
    np.random.seed(random_seed)  # set random seed

    before_shuffle_first_five_ids = (x['id'] for x in squad_flat)
    np.random.shuffle(squad_flat)  # shuffle the list in-place
    after_shuffle_first_five_ids = (x['id'] for x in squad_flat)
    assert before_shuffle_first_five_ids != after_shuffle_first_five_ids, "Failed to shuffle"

    return squad_flat


def load_and_shuffle_squad(data_path: str, num_to_return: int=-1, return_flat: bool=False, verbose: bool=False) -> Union[List, Dict]:
    """Load and shuffle a SQuAD dataset and return num requested"""
    flat_squad = flatten_squad(load_squad(data_path=data_path), verbose=verbose)
    shuffled_squad = shuffle_flat_squad(flat_squad)
    if num_to_return > 0:
        shuffled_squad = shuffled_squad[:num_to_return]
    if return_flat is False:
        shuffled_squad = unflatten_squad(shuffled_squad)
    return shuffled_squad


def get_squad_len(data_path: str, verbose: bool=False) -> int:
    """Get the number of QA pairs"""
    flat_squad = flatten_squad(load_squad(data_path=data_path), verbose=verbose)
    return len(flat_squad)


def get_squad_titles(data_path: str, verbose: bool=False) -> List:
    """Get all unique passage titles"""
    flat_squad = flatten_squad(load_squad(data_path=data_path), verbose=verbose)
    return get_unique_order_preserving(flat_squad, apply_custom_fn=lambda x: x['title'])


def get_squad_contexts(data_path: str, verbose: bool=False) -> List:
    """Get all unique passages/contexts"""
    flat_squad = flatten_squad(load_squad(data_path=data_path), verbose=verbose)
    return get_unique_order_preserving(flat_squad, apply_custom_fn=lambda x: x['context'])


def build_context_to_title_map(data_path: str, verbose: bool=False) -> Dict:
    """Create a dict of contexts to titles"""
    flat_squad = flatten_squad(load_squad(data_path=data_path), verbose=verbose)
    return {x['context'].strip(): x['title'] for x in flat_squad}


def get_majority(answer_list: List)  -> List:
    """Extract the majority vote answer or first answer in case of no majority for the dev data for consistency"""
    dict_answer_counts = {}
    for i, ans in enumerate(answer_list):
        if ans['text'] in dict_answer_counts:
            dict_answer_counts[ans['text']]['count'] += 1
        else:
            dict_answer_counts[ans['text']] = {
                'id': i,  # return the first occurring answer or first answer seen
                'count': 1
            }

    # Extract counts and indices
    count_indices = [(ans['count'], ans['id']) for ans in dict_answer_counts.values()]
    # Sort, first by index ascending, then by count descending
    count_indices = sorted(sorted(count_indices, key=lambda x: x[1]), key=lambda x: x[0], reverse=True)

    # Check that we have as many counts as expected
    assert len(answer_list) == sum(x[0] for x in count_indices)

    # Return the most common answer by index in the list
    return [answer_list[count_indices[0][1]]]


def convert_squad_multiple_answers_to_majority(squad_dict: Dict) -> Dict:
    """Converts a SQuAD dev or test file with multiple answers to majority vote"""
    for dp in squad_dict:
        for para in dp['paragraphs']:
            for qa in para['qas']:
                qa['answers'] = get_majority(qa['answers'])
    return squad_dict


def remove_answers(squad_dict: Dict) -> Dict:
    """Remove answers from a SQuAD dev or test file"""
    for dp in squad_dict:
        for para in dp['paragraphs']:
            for qa in para['qas']:
                qa['answers'] = []
    return squad_dict


def clean_context(text: str, replace_linebreaks: bool=True) -> str:
    """Cleans a SQuAD context"""
    if replace_linebreaks:
        text = re.sub('[\r\n]+', ' ', text)  # remove linebreaks
    text = re.sub('\s+', ' ', text)  # fix multiple spaces
    return text.strip()


if __name__ == '__main__':
    print(get_squad_len('~/_data/mrqa/train/SQuAD.json'))
