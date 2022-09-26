import argparse
import sys
import os
import jsonlines

MODULE_DIR = os.path.join(os.path.dirname(__file__), '..')
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)

from squad import load_squad, save_squad, flatten_squad, unflatten_squad, shuffle_flat_squad, get_squad_len


"""
Usage:
python join_squad.py '~/_data/squad_v1.1/train.json,~/_data/adversarialQA/1_dbidaf/train.json,~/_data/adversarialQA/2_dbert/train.json,~/_data/adversarialQA/3_droberta/train.json' --train_weights 1,5,5,5 --output_dir ~/_data/adversarialQA/combined/ --output_filename train_squad_plus_5xdcombined
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_paths", help="Comma-separated list of the paths to the files to join", type=str)
    parser.add_argument("--train_weights", help="How much to upsample each training set", type=str, default='')
    parser.add_argument("--output_dir", help="Directory in which to save the output file", type=str, default='')
    parser.add_argument("--output_filename", help="Name of the output file", type=str, default='')
    parser.add_argument("--output_flat_jsonl", help="Whether to output as flattened jsonlines file", action='store_true')
    args = parser.parse_args()

    # Process file paths as list
    file_paths = [os.path.expanduser(x.strip()) for x in args.file_paths.split(',')]
    if args.train_weights:
        train_weights = [int(x.strip()) for x in args.train_weights.split(',')]
    else:
        train_weights = [1] * len(file_paths)
    assert len(train_weights) == len(file_paths)

    # Extract source directory and filenames
    output_dir = os.path.dirname(file_paths[0])
    if args.output_dir:
        output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    joined_filename = 'joined'
    joined_data = []
    for f, train_weight in zip(file_paths, train_weights):
        f_filename, f_ext = os.path.splitext(os.path.basename(f))
        joined_filename += '_{}x{}'.format(train_weight,f_filename)
        # Load and flatten
        this_data = flatten_squad(load_squad(data_path=f))
        for i in range(train_weight):
            joined_data.extend(this_data)

    if args.output_filename:
        joined_filename = args.output_filename

    # Shuffle
    joined_data = shuffle_flat_squad(joined_data)

    if args.output_flat_jsonl:
        with jsonlines.open(os.path.join(output_dir, f'{joined_filename}.jsonl'), 'w') as writer:
            writer.write_all(joined_data)

    else:
        # Convert back to squad format
        joined_data = unflatten_squad(joined_data)

        # Save
        joined_path = os.path.join(output_dir, '{}{}'.format(joined_filename, '.json'))
        print(f"Saving file to {joined_path}")
        save_squad(joined_data, data_path=joined_path)

        print(f"File saved to {joined_path}")

        # Check sample sizes
        assert sum([train_weight*get_squad_len(f) for f, train_weight in zip(file_paths, train_weights)]) == get_squad_len(joined_path), \
            "Joined file has different number of QAs than sum of individual files."

        print("Datasets joined and sizes validated. Total dataset size is {}".format(get_squad_len(joined_path)))
