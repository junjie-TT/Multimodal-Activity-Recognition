# Upper Feature Extraction

This package starts from cleaned lower-level outputs and builds interaction
features.

## Current Pipeline

### Head IMU Feature Extraction

Input:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/aligned_sensor_with_YOLO/
  {participant_id}/{full_id}/eye_tracker__imu.csv
```

Output:

```text
/home/julius/datasets/seminar_object_detection/upper_feature_outcome/head_imu_features/
  head_imu_feature_index.csv
  head_imu_feature_summary.json
  frame_features/{participant_id}/{full_id}_head_imu_frame_features.csv
```

Run:

```bash
conda run --no-capture-output -n object_detection python -m upper_features_extraction.head_features.eye_imu --overwrite
```

Useful filters:

```bash
--participant IA_4_H_14
--full-id IA_4_H_14_Brief_schreiben1
--limit 10
```

This extractor builds upper-level head-motion features from the eye-tracker IMU
on the video-aligned time axis. Rolling statistics use a causal window over the
current sample and the previous `0.35s` by default.

Feature groups:

```text
head motion intensity:
  acc_mag, gyro_mag, acc_dyn_mag, motion_intensity,
  gyro_energy, acc_energy

head orientation:
  roll, pitch, yaw,
  roll_range, pitch_range, yaw_range,
  orientation_change, orientation_velocity

head stability:
  gyro_std, roll_std, pitch_std, yaw_std,
  low_motion_ratio, head_stability_score

motion derivatives:
  jerk_x/y/z, jerk_mag,
  angular_acc_x/y/z, angular_acc_mag
```

`acc_dyn_mag` uses quaternion-based gravity compensation when possible. The
default `--gravity-magnitude auto` infers the acceleration unit per recording,
which supports eye-tracker IMU files whose gravity magnitude is around `1`
instead of `9.80665`.

### Skeleton Upper-Limb Kinematics Feature Extraction

Input:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/aligned_sensor_with_YOLO/
  {participant_id}/{full_id}/zed__skeleton__Body_*.csv
```

Output:

```text
/home/julius/datasets/seminar_object_detection/upper_feature_outcome/skeleton_upper_limb_kinematics_features/
  skeleton_upper_limb_kinematics_index.csv
  skeleton_upper_limb_kinematics_summary.json
  frame_features/{participant_id}/{full_id}_Body_{body_id}_skeleton_upper_limb_kinematics_frame_features.csv
```

Run:

```bash
conda run --no-capture-output -n object_detection python -m upper_features_extraction.body_features.skeleton_upper_limb_kinematics --overwrite
```

This extractor builds one row per original ZED skeleton sensor sample using
`time_s` and `frame_idx`. It does not use `video_time_s`, `video_frame_idx`, or
video FPS. Rolling statistics use a causal `0.35s` window.

Feature groups:

```text
per-side hand/arm motion:
  {left/right}_wrist_speed, wrist_acceleration_mag, wrist_jerk_mag,
  hand_center_speed, hand_center_acceleration_mag, hand_center_jerk_mag,
  elbow_speed, shoulder_speed

joint angles:
  {left/right}_elbow_angle, elbow_angle_velocity, elbow_angle_acceleration,
  wrist_angle, wrist_angle_velocity, wrist_angle_acceleration,
  shoulder_angle, shoulder_angle_velocity, shoulder_angle_acceleration

hand-body spatial relations:
  {left/right}_wrist_to_shoulder_dist, wrist_to_neck_dist,
  wrist_to_hip_dist, wrist_to_trunk_dist, hand_to_trunk_dist,
  hand_height_relative_to_shoulder, hand_forward_relative_to_trunk,
  hand_lateral_relative_to_trunk

hand shape:
  {left/right}_thumb_index_dist, finger_spread, hand_shape_area

trunk/neck:
  neck_speed, neck_acceleration_mag, neck_jerk_mag,
  trunk_angle, trunk_angle_velocity, trunk_angle_acceleration,
  trunk_stability_0p35s, neck_stability_0p35s,
  neck_to_hip_dist, neck_spine_angle

bilateral coordination:
  left_right_wrist_dist, left_right_hand_center_dist,
  relative_hand_speed_diff, relative_hand_speed_abs_diff,
  both_hands_active, dominant_hand_speed_ratio,
  hands_moving_together_corr

causal 0.35s window statistics:
  *_mean_0p35s, *_std_0p35s, *_range_0p35s,
  angle_velocity_abs_mean_0p35s,
  {left/right}_hand_motion_energy_0p35s
```

### Depth Scene Feature Extraction

Input:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/aligned_sensor_with_YOLO/
  {participant_id}/{full_id}/zed__depth.csv
```

Output:

```text
/home/julius/datasets/seminar_object_detection/upper_feature_outcome/depth_scene_features/
  depth_scene_feature_index.csv
  depth_scene_feature_summary.json
  frame_features/{participant_id}/{full_id}_depth_scene_frame_features.csv
```

Run:

```bash
conda run --no-capture-output -n object_detection python -m upper_features_extraction.depth_features.scene_depth --overwrite
```

This extractor builds frame-level depth proximity and depth-change features on
the video-aligned time axis. Rolling statistics use a causal window over the
current sample and the previous `0.35s` by default.

Feature groups:

```text
distance/proximity:
  min_depth, max_depth, mean_depth, median_depth, valid_ratio,
  depth_range, near_scene_score,
  foreground_contrast, relative_foreground_contrast

dynamic change:
  delta_min_depth, delta_median_depth,
  min_depth_velocity, median_depth_velocity, depth_velocity,
  min_depth_acceleration, median_depth_acceleration, depth_acceleration

window stability/change:
  min_depth_std, median_depth_std, mean_depth_std, valid_ratio_std,
  depth_change_peak, min_depth_change_peak, median_depth_change_peak,
  foreground_contrast_std, foreground_contrast_mean,
  relative_foreground_contrast_mean
```

`foreground_contrast = median_depth - min_depth` is kept as the main foreground
approach signal. `depth_velocity` and `depth_acceleration` use median depth by
default, while min-depth variants are retained separately for nearby hand/object
motion.

### IMU Joint Kinematics Feature Extraction

Input:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/aligned_sensor_with_YOLO/
  {participant_id}/{full_id}/movella__DOT_*.csv
```

Output:

```text
/home/julius/datasets/seminar_object_detection/upper_feature_outcome/imu_joint_kinematics_features/
  imu_joint_kinematics_index.csv
  imu_joint_kinematics_summary.json
  frame_features/{participant_id}/{full_id}_imu_joint_kinematics_frame_features.csv
```

Run:

```bash
conda run --no-capture-output -n object_detection python -m upper_features_extraction.imu_features.joint_kinematics --overwrite
```

This extractor is independent from the older `feature_extraction/modalities/imu.py`
implementation. It outputs one row per original IMU sensor sample using
`time_s` and `frame_idx` at the Movella sampling rate, not `video_time_s` or
`video_frame_idx`. It uses the previously inferred Movella DOT body mapping:

```text
D422CD008603 -> left_forearm
D422CD008CC7 -> right_hand
D422CD009F54 -> left_hand
D422CD009F5B -> right_forearm
D422CD009F60 -> right_upper_arm
D422CD00A220 -> left_upper_arm
D422CD00A795 -> trunk
```

Feature groups:

```text
absolute segment posture:
  {body}_abs_roll, {body}_abs_pitch, {body}_abs_yaw,
  {body}_abs_gyro_mag

trunk posture:
  trunk_posture_roll, trunk_posture_pitch, trunk_posture_yaw,
  trunk_posture_gyro_mag

relative joint kinematics:
  {left/right}_{shoulder/elbow/wrist}_angle_deg,
  angle_velocity_deg_s,
  relative_roll, relative_pitch, relative_yaw,
  relative_gyro_mag

causal 0.35s window statistics:
  angle_mean_0p35s, angle_std_0p35s, angle_range_0p35s,
  trunk_abs_roll/pitch/yaw_std_0p35s,
  trunk_abs_roll/pitch/yaw_range_0p35s
```

Relative joint orientation is computed from segment quaternions:

```text
q_relative = conjugate(q_parent_segment) * q_child_segment
angle_deg = 2 * acos(abs(q_relative.w))
```

### Hand-Object Interaction Feature Layer

Input:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo_object_outcome_csv/
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo_left_right_hand_outcome_csv/
/home/julius/datasets/seminar_object_detection/upper_feature_outcome/skeleton_upper_limb_kinematics_features/frame_features/
/home/julius/datasets/seminar_object_detection/upper_feature_outcome/imu_joint_kinematics_features/frame_features/
```

Output:

```text
/home/julius/datasets/seminar_object_detection/upper_feature_outcome/hand_object_interaction_features/
  hand_object_interaction_index.csv
  hand_object_interaction_summary.json
  missing_left_right_hand_yolo.csv
  pair_features/{participant_id}/{full_id}_hand_object_pair_features.csv
  frame_features/{participant_id}/{full_id}_hand_object_frame_features.csv
```

Run:

```bash
conda run --no-capture-output -n object_detection python -m upper_features_extraction.hand_object_interaction.pair_features --overwrite
```

Useful filters:

```bash
--participant IA_4_H_02
--full-id IA_4_H_02_Chaos_several_Tasks_combine1
--limit 10
```

This layer builds one row per `{frame, hand_side, object}` pair from ordinary
YOLO object detections and left/right hand YOLO detections. Since the ordinary
YOLO object CSVs do not contain stable object identities, the extractor assigns
a lightweight per-class `object_id` by matching nearby boxes across adjacent
frames. Skeleton and IMU upper-limb features are merged with nearest-time
matching when their feature CSVs are available.

`object_class` is mapped from COCO class id to a readable name such as
`bottle`, `cup`, or `cell phone`. The numeric detector class is retained as
`object_class_id`.

Pair-level feature groups:

```text
2D spatial relation:
  center_distance_norm, distance_to_object_size,
  bbox_iou, bbox_gap_norm, expanded_iou,
  hand_center_in_object_bbox, object_center_in_hand_bbox

temporal relation:
  approach_speed, separation_speed,
  hand_velocity_2d, object_velocity_2d,
  velocity_similarity, relative_velocity,
  contact_candidate, contact_duration_s

skeleton / IMU side-specific motion:
  wrist_velocity_3d, hand_velocity_3d, elbow_angle,
  wrist_angle_velocity, hand_local_motion_energy,
  hand_gyro_mag, imu_wrist_angle_velocity, imu_wrist_gyro_mag

rule baseline:
  visual_score, temporal_score, skeleton_motion_score, imu_motion_score,
  large_object_penalty, cell_phone_penalty, interaction_score,
  interaction_phase_name

1s temporal hand proxy:
  temporal_proxy_score, temporal_proxy_support_frames,
  temporal_proxy_support_ratio, temporal_proxy_phase

0.5s temporal hand proxy:
  temporal_proxy_0p5_score, temporal_proxy_0p5_support_frames,
  temporal_proxy_0p5_support_ratio, temporal_proxy_0p5_phase
```

The frame-level summary keeps the best active object per hand:

```text
L_active_object, L_interaction_score, L_interaction_phase,
R_active_object, R_interaction_score, R_interaction_phase,
two_hand_object, two_hand_interaction_score
```

It also keeps a smoothed hand proxy over a causal 1s window:

```text
L_temporal_object, L_temporal_score, L_temporal_phase,
R_temporal_object, R_temporal_score, R_temporal_phase,
two_hand_temporal_object, two_hand_temporal_score
```

For comparison, a more responsive causal 0.5s proxy is also written:

```text
L_temporal_0p5_object, L_temporal_0p5_score, L_temporal_0p5_phase,
R_temporal_0p5_object, R_temporal_0p5_score, R_temporal_0p5_phase,
two_hand_temporal_0p5_object, two_hand_temporal_0p5_score
```

### Gaze Fixation Extraction

Input:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/gaze_xy_filled_sensor_csv/gaze_xy_filled_index.csv
```

Output:

```text
/home/julius/datasets/seminar_object_detection/upper_feature_outcome/gaze_fixation_features/
```

Run:

```bash
conda run --no-capture-output -n object_detection python -m upper_features_extraction.gaze_features.fixation
```

The fixation extractor uses the filled gaze CSVs and applies three layers:

```text
1. worn + blink-like filtering
2. x/y velocity + head-fused gaze angular velocity candidate detection
3. duration + dispersion segment confirmation
```

The optical-axis vectors are expressed in the eye-tracker/head coordinate frame,
so the extractor also reads the neighboring `eye_tracker__imu.csv` file and
interpolates `gyro_x/y/z` onto gaze timestamps. Candidate detection uses:

```text
world_gaze_angular_velocity ~= norm(d(optical_axis_head)/dt + head_gyro x optical_axis_head)
```

The original optical-axis angular velocity, head gyro magnitude, and fused
angular velocity are all kept in the sample/segment outputs for comparison.

Default parameters are in `config.py`:

```text
FIXATION_MIN_DURATION_MS = 100
FIXATION_SLIDING_WINDOW_MS = 200
FIXATION_MAX_GAP_DURATION_MS = 50
FIXATION_MAX_DISPERSION_PX = 60
FIXATION_MAX_GAZE_VELOCITY_PX_S = 2500
FIXATION_MAX_OPTICAL_AXIS_ANGULAR_VELOCITY_DEG_S = 80  # applied to fused angular velocity
BLINK_MIN_EYELID_APERTURE = 1
```

For each recording, two files are written:

```text
sample_labels/{participant_id}/{full_id}_gaze_fixation_samples.csv
segments/{participant_id}/{full_id}_gaze_fixation_segments.csv
```

The segment file contains fixation parameters such as start/end time, duration,
center point, dispersion, mean gaze velocity, mean optical-axis angular
velocity, mean head gyro magnitude, and mean head-fused gaze angular velocity.

### Gaze-Object Point-in-BBox Baseline

Input:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/gaze_xy_filled_sensor_csv/gaze_xy_filled_index.csv
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo_object_outcome_csv/
```

Output:

```text
/home/julius/datasets/seminar_object_detection/upper_feature_outcome/gaze_object_features/point_in_bbox_frame_aggregation/
```

Run:

```bash
conda run --no-capture-output -n object_detection python -m upper_features_extraction.gaze_object.point_in_bbox
```

The baseline does not downsample gaze before matching. Every gaze sample is
tested against YOLO boxes from its `video_frame_idx`, then the result is
aggregated to one row per video frame.

Frame-level output columns:

```text
video_frame_idx
gaze_sample_count
valid_gaze_sample_count
yolo_detection_count
hit_box_count
hit_sample_count
hit_detection_count
hit_ratio
hit_class
```

`hit_class` values:

```text
NO_VALID_GAZE
NO_DETECTION
NONE
[class_id, ...]
```

For each recording, a second long-format bbox detail file is also written:

```text
{full_id}_gaze_yolo_point_in_bbox_frame_boxes.csv
```

Each row is one YOLO box that contains at least one gaze sample in that video
frame:

```text
video_frame_idx
detection_idx_in_frame
class_id
confidence
x1
y1
x2
y2
hit_gaze_sample_count
valid_gaze_sample_count
hit_ratio_in_box
```

### Gaze-Object Scale-Aware Gaussian

Input:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/gaze_xy_filled_sensor_csv/gaze_xy_filled_index.csv
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo_object_outcome_csv/
```

Output:

```text
/home/julius/datasets/seminar_object_detection/upper_feature_outcome/gaze_object_features/scale_aware_gaussian_frame_aggregation/
```

Run:

```bash
conda run --no-capture-output -n object_detection python -m upper_features_extraction.gaze_object.scale_aware_gaussian
```

This keeps gaze at its original sampling rate. For every gaze sample and YOLO
box in the same `video_frame_idx`, it computes:

```text
sigma = GAUSSIAN_ALPHA * bbox_diagonal
score = exp(-(distance_to_bbox_center^2) / (2 * sigma^2)) * yolo_confidence
```

Classes listed in `GAZE_OBJECT_PENALIZED_CLASS_IDS` use a smaller sigma and a
lower score prior. This is useful when person/hand-like boxes dominate
egocentric hand-object scenes.

Frame-level output columns:

```text
video_frame_idx
gaze_sample_count
valid_gaze_sample_count
yolo_detection_count
best_detection_idx_in_frame
best_class
best_score
score_threshold
```

For each recording, a second long-format bbox score file is also written:

```text
{full_id}_gaze_yolo_scale_aware_gaussian_frame_boxes.csv
```

Each row is one YOLO box in one video frame:

```text
video_frame_idx
detection_idx_in_frame
class_id
confidence
x1
y1
x2
y2
sigma
valid_gaze_sample_count
mean_score
max_score
```

### Gaze Heatmap And Object Uncertainty

Input:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/gaze_xy_filled_sensor_csv/gaze_xy_filled_index.csv
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo_object_outcome_csv/
```

Output:

```text
/home/julius/datasets/seminar_object_detection/upper_feature_outcome/gaze_object_features/probabilistic_heatmap_frame_aggregation/
```

Run:

```bash
conda run --no-capture-output -n object_detection python -m upper_features_extraction.gaze_object.probabilistic_heatmap
```

This model treats both gaze and objects as probability distributions:

```text
gaze heatmap = Gaussian mixture over all gaze samples in the video frame
object distribution = Gaussian centered at bbox center
object sigma = OBJECT_DISTRIBUTION_ALPHA * bbox_diagonal
object prior = YOLO confidence
```

Classes listed in `GAZE_OBJECT_PENALIZED_CLASS_IDS` use:

```text
object sigma *= PENALIZED_CLASS_SIGMA_FACTOR
object prior *= PENALIZED_CLASS_SCORE_FACTOR
```

For every frame-object pair, it computes distribution overlap between the gaze
mixture and object distribution. The frame-object scores are normalized within
each frame into `object_posterior_probability`.

Frame-level output columns:

```text
video_frame_idx
gaze_sample_count
valid_gaze_sample_count
yolo_detection_count
best_detection_idx_in_frame
best_class
best_object_probability
best_confidence_weighted_score
object_probability_entropy
object_probability_normalized_entropy
gaze_mean_x
gaze_mean_y
gaze_std_x
gaze_std_y
```

Per-object posterior output:

```text
{full_id}_gaze_yolo_probabilistic_heatmap_frame_objects.csv
```

Each row is one probabilistic object candidate in one frame:

```text
video_frame_idx
detection_idx_in_frame
class_id
confidence
x1
y1
x2
y2
object_center_x
object_center_y
gaze_sigma
object_sigma
valid_gaze_sample_count
raw_overlap_score
confidence_weighted_score
object_posterior_probability
```
