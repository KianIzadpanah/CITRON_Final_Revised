#!/usr/bin/env bash
# run_all.sh — execute all phases in order
# On Kaggle, copy the entire citron_revision/ folder to /kaggle/working/citron_revision/
# then run:  bash /kaggle/working/citron_revision/run_all.sh

set -e
cd "$(dirname "$0")"

# Create required output directories (needed on fresh Kaggle sessions)
mkdir -p /kaggle/working/citron_revision/data/processed/{scenes,crops,stitched,metadata}
mkdir -p /kaggle/working/citron_revision/outputs/{dataset_summary/qc_panels,overlap,stitched,detection,ablation,simulation}
mkdir -p /kaggle/working/citron_revision/outputs/figures/{scene_examples,ablation_examples,network_plots}

RESNET_CKPT="/kaggle/working/citron_revision/outputs/overlap/overlap_resnet50_best.pt"
MOBILENET_CKPT="/kaggle/working/citron_revision/outputs/overlap/overlap_mobilenet_best.pt"
DETECTOR_WEIGHTS="/kaggle/working/citron_revision/outputs/detection/crop_mode/train/weights/best.pt"
SCENE_CSV="/kaggle/working/citron_revision/outputs/data/processed/metadata/scene_test.csv"
CROP_META="/kaggle/working/citron_revision/outputs/data/processed/metadata/crop_metadata.csv"

# echo "=== Phase A: Build dataset ==="
# python src/dataset/build_citron_dataset.py --config configs/dataset.yaml --mode overlap
# python src/dataset/build_citron_dataset.py --config configs/dataset.yaml --mode scene --force

echo "=== Phase B: Train ResNet-50 overlap model ==="
python src/overlap/train_overlap_resnet50.py --config configs/overlap_resnet50.yaml

echo "=== Phase B: Predict overlap masks on test set ==="
python src/overlap/predict_overlap_masks.py \
    --config configs/overlap_resnet50.yaml \
    --checkpoint "${RESNET_CKPT}" \
    --scene_csv "${SCENE_CSV}" \
    --out_dir /kaggle/working/citron_revision/outputs/overlap/predicted_masks

echo "=== Phase D: Train detector (crop_mode) ==="
python src/detection/train_detector.py --config configs/detector.yaml --mode crop_mode

echo "=== Phase C: Run fusion ablation ==="
python src/fusion/run_fusion_ablation.py \
    --detector_weights "${DETECTOR_WEIGHTS}" \
    --resnet_ckpt "${RESNET_CKPT}" \
    --scene_csv "${SCENE_CSV}" \
    --crop_meta "${CROP_META}" \
    --out_dir /kaggle/working/citron_revision/outputs/ablation

echo "=== Phase D: Scene-level ODO vs CITRON evaluation ==="
python src/detection/evaluate_scene_level.py \
    --detector_weights "${DETECTOR_WEIGHTS}" \
    --overlap_ckpt "${RESNET_CKPT}" \
    --scene_csv "${SCENE_CSV}" \
    --crop_meta "${CROP_META}" \
    --out_dir /kaggle/working/citron_revision/outputs/detection \
    --fig_dir /kaggle/working/citron_revision/outputs/figures/scene_examples

echo "=== Phase E: Network/energy simulator ==="
python src/simulation/network_simulator.py --config configs/network.yaml

echo "=== Phase F: Train lightweight overlap model ==="
python src/overlap/train_overlap_lightweight.py --config configs/overlap_mobilenet.yaml

echo "=== Phase F: Update ablation with MobileNet ==="
python src/fusion/run_fusion_ablation.py \
    --detector_weights "${DETECTOR_WEIGHTS}" \
    --resnet_ckpt "${RESNET_CKPT}" \
    --mobilenet_ckpt "${MOBILENET_CKPT}" \
    --scene_csv "${SCENE_CSV}" \
    --crop_meta "${CROP_META}" \
    --out_dir /kaggle/working/citron_revision/outputs/ablation \
    --fig_dir /kaggle/working/citron_revision/outputs/figures/ablation_examples

echo "=== Phase G: Leader delay estimator ==="
python src/simulation/leader_delay_estimator.py --config configs/network.yaml

echo "=== Phase G: Control overhead ==="
python src/simulation/control_overhead.py --config configs/network.yaml

echo "=== All phases complete ==="
