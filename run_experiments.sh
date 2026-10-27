set -e
EPOCHS=${EPOCHS:-100}

echo "=== Table 4: both models x both tasks x with/without Null ==="
for model in deepconvlstm cnn; do
  for task in locomotion gestures; do
    python train.py --model $model --task $task --epochs $EPOCHS
    python train.py --model $model --task $task --no-null --epochs $EPOCHS
  done
done

echo "=== Table 10: sensor modalities (gestures, no Null) ==="
for s in acc gyro acc_gyro acc_gyro_mag all; do
  python train.py --model deepconvlstm --task gestures --no-null --sensors $s --epochs $EPOCHS
done

echo "=== Figure 8: number of conv layers (gestures, with Null) ==="
for c in 1 2 3; do   # 4 is already covered by Table 4
  python train.py --model deepconvlstm --task gestures --n-conv $c --epochs $EPOCHS
done

echo "=== Figure 7: window length (gestures, no Null). 24 = default ==="
for w in 42 82; do   # 1400 ms and 2750 ms at 30 Hz
  python train.py --model deepconvlstm --task gestures --no-null --window $w --step $((w/2)) --epochs $EPOCHS
done

python summarize.py
