#!/usr/bin/env bash

set -eu
# python ./la_evaluate.py ./exp_result/LA_AASIST_ep100_bs24/eval_scores_using_best_dev_model.txt  /data3/DB/FakeAudio/LA/keys/LA eval
python ./la_evaluate.py ./exp_result/<FILE_NAME>.txt DB/FakeAudio/LA/keys/LA eval
