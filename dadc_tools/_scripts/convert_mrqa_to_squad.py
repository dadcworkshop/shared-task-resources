import argparse
import gzip
import json
import os
from datetime import datetime

"""
Example Usage:
python convert_mrqa_to_squad.py ~/_data/mrqa/train/SQuAD.jsonl.gz,~/_data/mrqa/train/HotpotQA.jsonl.gz

MRQA data format:
https://github.com/mrqa/MRQA-Shared-Task-2019#mrqa-format
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_paths", help="Comma-separated list of the MRQA files to convert", type=str)
    parser.add_argument("--output_dir", help="Where to store the output files. If not set, will store in the same directory as source", type=str, default='')
    args = parser.parse_args()

    file_paths = [os.path.expanduser(f.strip()) for f in args.file_paths.split(',')]
    if not file_paths:
        raise BaseException(f"No file paths detected: {args.file_paths}")

    for file_path in file_paths:
        print(f'Processing {file_path}')

        file_dirname = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        if args.output_dir:
            file_dirname = args.output_dir

        if not filename.endswith('.jsonl.gz'):
            raise BaseException(f"{filename} is not a recognised MRQA format")
        filename = filename.replace('.jsonl.gz', '')

        # Convert to SQuAD format
        count = 0
        version = ''
        squad_data = []
        with gzip.open(file_path, 'r') as f:
            for line in f:
                line = json.loads(line.strip())
                if 'header' in line:
                    version = line['header']
                    version['mrqa_conversion'] = 'Converted from MRQA format on {}'.format(datetime.now().strftime('%Y%m%d-%H%M'))
                else:
                    context_dict = {
                        'context': line['context'],
                        'qas': [
                            {
                                'id': qa['qid'],
                                'question': qa['question'],
                                'answers': [{
                                    'text': a['text'],
                                    'answer_start': a['char_spans'][0][0]
                                } for a in qa['detected_answers']],
                            } for qa in line['qas']]
                    }
                    count += len(context_dict['qas'])

                    data_item = {
                        'title': '',
                        'paragraphs': [context_dict]
                    }
                    squad_data.append(data_item)

        # Convert to final squad_dict format
        squad_dict = {
            'data': squad_data,
            'version': version
        }

        # Save
        output_data_path = os.path.join(file_dirname, filename+'.json')
        with open(output_data_path, "w") as f:
            json.dump(squad_dict, f)

        print("Successfully converted with {} QAs added. Saved to: {}".format(count, output_data_path))
        print("-"*3)
