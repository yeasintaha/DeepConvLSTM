#!/usr/bin/env bash
# Student ablation study for DeepConvLSTM on OPPORTUNITY (gestures task).
# Usage:  bash ablations.sh A        # run one block (A..H)
#         SEEDS="0 1 2" EPOCHS=60 bash ablations.sh C
# Every block changes ONE thing relative to the paper configuration ("ref").
# Report the mean +- std of test weighted F1 AND macro F1 over seeds.
set -e
BLOCK=${1:?give a block letter A..H}
SEEDS=${SEEDS:-"0 1 2"}
EPOCHS=${EPOCHS:-100}
T="--task gestures --epochs $EPOCHS"

run() {  # run <name> <extra args...>
  local name=$1; shift
  for s in $SEEDS; do
    python train.py $T --seed $s --name "${BLOCK}_${name}_s${s}" "$@"
  done
}

case $BLOCK in
A)  # Is recurrence what helps?  (paper's main claim)
    run ref
    run cnn        --model cnn
    run gru        --rnn gru
    run bilstm     --bidirectional
    run lstm1      --rnn-layers 1
    run lstm3      --rnn-layers 3 ;;
B)  # Do the convolutions matter?  (Fig. 8 and beyond)
    for c in 0 1 2 3; do run conv$c --n-conv $c; done
    run k3   --kernel 3
    run k6   --kernel 6   # window 24 needs n_conv*(k-1) < 24, so k <= 6
    run f32  --filters 32
    run f128 --filters 128 ;;
C)  # How much context is needed?  (Fig. 7)
    run w24
    run w36 --window 36 --step 12
    run w48 --window 48 --step 12
    run w72 --window 72 --step 12 ;;
D)  # Which sensors matter?  (Table 10, with Null and without)
    for s in acc gyro acc_gyro acc_gyro_mag all; do
      run sens_$s --sensors $s
      run sens_${s}_nonull --sensors $s --no-null
    done ;;
E)  # Regularization
    for d in 0.0 0.25 0.5 0.7; do run drop$d --dropout $d; done
    run wd1e-4 --weight-decay 1e-4
    run aug    --augment ;;
F)  # Optimization
    run adam        --optimizer adam
    run adam_cos    --optimizer adam --cosine
    run lr3e-4      --lr 3e-4
    run lr3e-3      --lr 3e-3
    run bs32        --batch-size 32
    run bs256       --batch-size 256 ;;
G)  # Class imbalance (watch macro F1, not only weighted F1)
    run ref
    run cw_inv     --class-weights inverse
    run cw_sqrt    --class-weights sqrt_inverse
    run ls0.1      --label-smoothing 0.1
    run sel_macro  --select-on f1_macro ;;
H)  # Model size vs. accuracy
    for u in 32 64 128 256; do run units$u --rnn-units $u; done ;;
*)  echo "unknown block $BLOCK"; exit 1 ;;
esac
python summarize.py
