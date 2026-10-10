# Archived ADL14 Experiment

This folder contains the unfinished ADL14 custom-object YOLO experiment.

It was moved out of the active `yolo_features_extraction/` workflow because the
project did not continue with ADL14 as the final lower-feature pipeline.

## Contents

```text
adl_14_object_detection_csv_output.py
merge_coco_and_adl14_yolo_csv.py
prepare_real_adl_hands_yolo_dataset.py
run_adl14_yolo_output.sh
run_yolo11s_coco_plus_adl14_output.sh
train_yolo11s_real_adl_14cls_gpu.sh
```

The experiment tried to train and run a 14-class specialist model for:

```text
pen, paper, bowl, plate, spoon, cup, knife, fork, container, bottle, bar, glass, key, wallet
```

The intended flow was:

1. Prepare a YOLO-format dataset from labeled ADL frames.
2. Train a YOLO11s ADL14 model.
3. Run both COCO YOLO and ADL14 YOLO on world-camera videos.
4. Merge the two CSV outputs and remove overlapping semantic duplicates.

Treat these files as historical reference unless the ADL14 approach is revived.
