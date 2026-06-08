#!/bin/bash
# Eldarin Dataset Download Helper
# ================================
# Downloads and organizes datasets for Eldarin training.
#
# Datasets supported:
#   - VisDrone: https://github.com/VisDrone/VisDrone-Dataset
#   - UAVDT: https://datasetninja.com/uavdt
#   - FRED: https://github.com/francesco-p/FRED
#   - UAV3D: https://uav3d.github.io/

set -e

DATA_DIR="${1:-data}"

echo "=== Eldarin Dataset Download ==="
echo "Data directory: ${DATA_DIR}"
mkdir -p "$DATA_DIR"

download_visdrone() {
    echo ""
    echo "[VisDrone] Downloading VisDrone2019-DET dataset..."
    echo "  VisDrone is the primary UAV benchmark (288 video clips, 10k+ images)."
    echo "  Manual download required from: https://github.com/VisDrone/VisDrone-Dataset"
    echo "  After downloading, extract to: ${DATA_DIR}/VisDrone2019-DET/"
    echo ""
    echo "  Expected structure:"
    echo "    ${DATA_DIR}/VisDrone2019-DET/"
    echo "      VisDrone2019-DET-train/images/"
    echo "      VisDrone2019-DET-train/annotations/"
    echo "      VisDrone2019-DET-val/images/"
    echo "      VisDrone2019-DET-val/annotations/"
    echo ""
    echo "  For tracking (MOT format):"
    echo "    ${DATA_DIR}/VisDrone2019-MOT/"
    echo "      VisDrone2019-MOT-train/images/"
    echo "      VisDrone2019-MOT-train/annotations/"
}

download_fred() {
    echo ""
    echo "[FRED] RGB-Event Drone Dataset..."
    echo "  Repository: https://github.com/francesco-p/FRED"
    echo "  Manual download required."
    echo ""
    echo "  After downloading, extract to: ${DATA_DIR}/FRED/"
    echo "    ${DATA_DIR}/FRED/train/rgb/"
    echo "    ${DATA_DIR}/FRED/train/events/"
    echo "    ${DATA_DIR}/FRED/train/annotations/"
}

download_uavdt() {
    echo ""
    echo "[UAVDT] Vehicle detection/tracking dataset..."
    echo "  Available at: https://datasetninja.com/uavdt"
    echo "  Manual download required."
}

download_synthetic() {
    echo ""
    echo "[Synthetic] Generating synthetic dataset..."
    echo "  Run: python scripts/generate_synthetic_data.py --output ${DATA_DIR}/synthetic"
}

echo ""
echo "Datasets are typically large (10s of GB)."
echo "Manual download from the official sources is recommended."
echo ""

# Show options
echo "Available download options:"
echo "  1) VisDrone (recommended starting point)"
echo "  2) FRED (RGB + Event)"
echo "  3) UAVDT"
echo "  4) All"
echo "  5) Show expected directory structure"
echo ""

read -p "Select option [1-5]: " choice

case $choice in
    1) download_visdrone ;;
    2) download_fred ;;
    3) download_uavdt ;;
    4)
        download_visdrone
        download_fred
        download_uavdt
        ;;
    5)
        echo ""
        echo "Expected directory structure:"
        echo "  ${DATA_DIR}/"
        echo "    VisDrone2019-DET/"
        echo "      VisDrone2019-DET-train/"
        echo "        images/"
        echo "        annotations/"
        echo "      VisDrone2019-DET-val/"
        echo "        images/"
        echo "        annotations/"
        echo "    FRED/"
        echo "      train/"
        echo "        rgb/"
        echo "        events/"
        echo "        annotations/"
        echo "    UAVDT/"
        echo "      train/"
        echo "        images/"
        echo "        annotations/"
        echo "    UAV3D/"
        echo "      train/"
        echo "        images/"
        echo "        annotations.json"
        ;;
    *) echo "Invalid option" ;;
esac

echo ""
echo "After downloading, run preprocessing:"
echo "  python scripts/prepare_visdrone.py --data_root ${DATA_DIR}/VisDrone2019-DET"
echo ""
echo "Then start training:"
echo "  python main.py --config config/train_visdrone.yaml --data_root ${DATA_DIR}/VisDrone2019-DET"