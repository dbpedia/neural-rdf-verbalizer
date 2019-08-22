import argparse
import os
import sys


def prepare_files_ter(inputdir, predsFile, partition):
    """
    Generate files for METEOR and TER input.
    :param inputdir: directory with bleu files
    :return:
    """
    references = []  # each element is a list of references
    pure_references = []
    initialref = inputdir + partition + '-all-notdelex-reference0.lex'
    # complete refs with references for all sents
    with open(initialref, 'r') as f:
        for i, line in enumerate(f):
            references.append([line.strip() + ' (id' + str(i) + ')\n'])
            pure_references.append([line])

    # create a file with only one reference for TER
    with open(partition + '-all-notdelex-oneref-ter.txt', 'w+') as f:
        for ref in references:
            f.write(''.join(ref))

    files = [(inputdir, filename) for filename in os.listdir(inputdir)]
    for filepath in files:
        if partition + '-all-notdelex-reference' in filepath[1] and 'reference0' not in filepath[1]:
            with open(filepath[0] + filepath[1], 'r') as f:
                for i, line in enumerate(f):
                    if line != '\n':
                        references[i].append(line.strip() + ' (id' + str(i) + ')\n')
                        pure_references[i].append(line)

    with open(partition + '-all-notdelex-refs-ter.txt', 'w+') as f:
        for ref in references:
            f.write(''.join(ref))

    # prepare generated hypotheses
    # with open('relexicalised_predictions.txt', 'r') as f:
    with open(predsFile, 'r') as f:
        geners = [line.strip() + ' (id' + str(i) + ')\n' for i, line in enumerate(f)]
    # with open('relexicalised_predictions-ter.txt', 'w+') as f:
    with open(predsFile.replace('.txt', '-ter.txt'), 'w+') as f:
        f.write(''.join(geners))

    # data for meteor
    # For N references, it is assumed that the reference file will be N times the length of the test file,
    # containing sets of N references in order.
    # For example, if N=4, reference lines 1-4 will correspond to test line 1, 5-8 to line 2, etc.
    with open(partition + '-all-notdelex-refs-meteor.txt', 'w+') as f:
        for ref in pure_references:
            empty_lines = 8 - len(ref)  # calculate how many empty lines to add (8 max references)
            f.write(''.join(ref))
            if empty_lines > 0:
                f.write('\n' * empty_lines)
    print('Input files for METEOR and TER generated successfully.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pred', help="Predictions file.", required=True)
    parser.add_argument('--td', help="Top dir.", required=True)
    parser.add_argument('--p', help="Partition.", required=True)

    args = parser.parse_args(sys.argv[1:])

    # topdir = './'
    # prepare_files_ter(topdir, args)
    prepare_files_ter(args.td, args.pred, args.p)
