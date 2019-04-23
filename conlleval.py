#!/usr/bin/env python

# Python version of the evaluation script from CoNLL'00-

# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

import sys
import re

from collections import defaultdict, namedtuple

ANY_SPACE = '<SPACE>'

class FormatError(Exception):
    pass

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')

class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = []    # number of correctly identified chunks
        self.correct_tags = 0     # number of correct chunk tags
        self.found_correct = []    # number of chunks in corpus
        self.found_guessed = []    # number of identified chunks
        self.token_counter = 0    # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(list)
        self.t_found_correct = defaultdict(list)
        self.t_found_guessed = defaultdict(list)

    def __repr__(self):
        s = "correct_chunk " + str(self.correct_chunk) + "\n"
        s += "correct_tags " + str(self.correct_tags) + "\n"
        s += "found_correct " + str(self.found_correct) + "\n"
        s += "found_guessed " + str(self.found_guessed) + "\n"

        # counts by type
        s += "t_correct_chunk " + str(self.t_correct_chunk)
        #print(self.t_found_correct)
        #print(self.t_found_guessed)

        return s

def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description='Evaluate tagging results using CoNLL criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-',
        help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
        help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O',
        help='alternative outside tag')
    arg('-t', '--trainfile', metavar='STR', default=None)
    arg('file', nargs='?', default=None)
    return parser.parse_args(argv)

def parse_tag(t):
    """
    This takes a tag t, and returns the chunk label and type as a
    tuple. For example, given B-MISC, it returns (B, MISC).

    Given O, it returns (O,)
    """
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')

def evaluate(iterable, options=None):
    if options is None:
        options = parse_args([])    # use defaults

    trainnames = defaultdict(int)
    if options.trainfile:
        last_pref = 'O'
        last_label = ''
        with open(options.trainfile) as f:
            # get all phrases from trainfile
            # assume word ... tag
            curr = []
            for line in f:
                line = line.rstrip('\r\n')
                if len(line) > 0:
                    sline = line.split()
                    word = sline[0]
                    tag = sline[-1]
                    pref, label = parse_tag(sline[-1])

                    start = start_of_chunk(last_pref, pref,
                                                   last_label, label)
                    end = end_of_chunk(last_pref, pref,
                                               last_label, label)
                    last_pref = pref
                    last_label = label

                    if end:
                        trainnames[" ".join(curr)] += 1
                        curr = []
                    if start or len(curr) > 0:
                        curr.append(word)
        print("Loaded trainfile: {}, with {} total names, {} unique names".format(options.trainfile, sum(trainnames.values()), len(trainnames)))

    counts = EvalCounts()
    num_features = None       # number of features per line
    in_correct = False        # currently processed chunks is correct until now
    last_correct = 'O'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    # running tally
    curr_guessed = []
    curr_correct = []

    for line in iterable:

        if type(line) is str:
            line = line.rstrip('\r\n')
            if options.delimiter == ANY_SPACE:
                features = line.split()
            else:
                features = line.split(options.delimiter)
        elif type(line) is list or type(line) is tuple:
            features = list(line)
        else:
            raise FormatError("unexpected type of input line! Type is: " + str(type(line)))

        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), num_features))

        if len(features) == 0 or features[0] == options.boundary:
            features = [options.boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)

        # guessed is chunk signifier (B, I, etc.),
        # guessed_type is the label (PER, ORG, etc.)
        guessed, guessed_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())
        first_item = features.pop(0)

        if first_item == options.boundary:
            guessed = 'O'

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        if in_correct:
            if (end_correct and end_guessed and
                last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk.append(" ".join(curr_guessed))
                counts.t_correct_chunk[last_correct_type].append(" ".join(curr_guessed))
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct = False

        if end_correct:
            s = " ".join(curr_correct)
            counts.found_correct.append(s)
            counts.t_found_correct[curr_correct_type].append(s)
            curr_correct = []
        if end_guessed:
            s = " ".join(curr_guessed)
            counts.found_guessed.append(s)
            counts.t_found_guessed[curr_guessed_type].append(s)
            curr_guessed = []

        if start_correct or len(curr_correct) > 0:
            curr_correct.append(first_item)
            if start_correct:
                curr_correct_type = correct_type

        if start_guessed or len(curr_guessed) > 0:
            curr_guessed.append(first_item)
            if start_guessed:
                curr_guessed_type = guessed_type

        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True

        if first_item != options.boundary:
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:
        counts.correct_chunk.append(" ".join(curr_guessed))
        counts.t_correct_chunk[last_correct_type].append(" ".join(curr_guessed))

    if len(curr_guessed) > 0:
        s = " ".join(curr_guessed)
        counts.found_guessed.append(s)
        counts.t_found_guessed[curr_guessed_type].append(s)

    if len(curr_correct) > 0:
        s = " ".join(curr_correct)
        counts.found_correct.append(s)
        counts.t_found_correct[curr_correct_type].append(s)

    return counts

def uniq(iterable):
  seen = set()
  return [i for i in iterable if not (i in seen or seen.add(i))]

def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)

def metrics(counts):
    c = counts
    overall = calculate_metrics(
        len(c.correct_chunk), len(c.found_guessed), len(c.found_correct)
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct) + list(c.t_found_guessed)):
        by_type[t] = calculate_metrics(
            len(c.t_correct_chunk[t]), len(c.t_found_guessed[t]), len(c.t_found_correct[t])
        )
    return overall, by_type

def report(counts, out=None):
    if out is None:
        out = sys.stdout

    overall, by_type = metrics(counts)

    c = counts
    out.write('processed %d tokens with %d phrases; ' %
              (c.token_counter, len(c.found_correct)))
    out.write('found: %d phrases; correct: %d.\n' %
              (len(c.found_guessed), len(c.correct_chunk)))

    if c.token_counter > 0:
        out.write('accuracy: %6.2f%%; ' %
                  (100.*c.correct_tags/c.token_counter))
        out.write('precision: %6.2f%%; ' % (100.*overall.prec))
        out.write('recall: %6.2f%%; ' % (100.*overall.rec))
        out.write('FB1: %6.2f\n' % (100.*overall.fscore))

    for i, m in sorted(by_type.items()):
        out.write('%17s: ' % i)
        out.write('precision: %6.2f%%; ' % (100.*m.prec))
        out.write('recall: %6.2f%%; ' % (100.*m.rec))
        out.write('FB1: %6.2f  %d\n' % (100.*m.fscore, len(c.t_found_guessed[i])))

def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end

def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start

def main(argv):
    args = parse_args(argv[1:])

    if args.file is None:
        counts = evaluate(sys.stdin, args)
    else:
        with open(args.file) as f:
            counts = evaluate(f, args)
    report(counts)

if __name__ == '__main__':
    sys.exit(main(sys.argv))
