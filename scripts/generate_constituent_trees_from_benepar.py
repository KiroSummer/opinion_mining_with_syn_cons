import nltk
import sys
import benepar


if __name__ == "__main__":
    filepath = sys.argv[1]
    sentences = []
    with open(filepath, 'r') as input_file:
        for line in input_file.readlines():
            sentence = line.strip()
            words = sentence.split(' ')
            sentences.append(words)

    parser = benepar.Parser("benepar_en2")
    constituent_trees = []
    for sentence in sentences:
        tree = parser.parse(sentence)
        constituent_trees.append(tree)

    with open(filepath + '.constituent.txt', 'w') as output_file:
        for t in constituent_trees:
            output_file.write(str(t) + '\n' + '\n')

