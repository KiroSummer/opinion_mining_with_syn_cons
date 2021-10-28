from os.path import join
import os
import random

ROOT_DIR = join(os.path.dirname(os.path.abspath(__file__)), '../../../')

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)

SRL_CONLL_EVAL_SCRIPT = join(ROOT_DIR, '../run_eval.sh')

START_MARKER = '<S>'
END_MARKER = '</S>'
PADDING_TOKEN = '*PAD*'
UNKNOWN_TOKEN = '*UNKNOWN*'
NULL_LABEL = 'O'

TEMP_DIR = join(ROOT_DIR, '../temp')

# assert os.path.exists(SRL_CONLL_EVAL_SCRIPT)
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
