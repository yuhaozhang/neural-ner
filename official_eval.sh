#!/bin/bash

# Run CoNLL13 official evaluation scripts
# First run `eval.py --out EVAL_OUTPUT` to generate an output file,
# then use this script to evaluate.
# 
# Usage: ./official_eval.sh EVAL_OUTPUT

if [ $# -eq 0 ]; then
    echo "Usage: ./official_eval.sh EVAL_OUTPUT"
    exit
fi

./utils/conlleval -d '\t' < $1

