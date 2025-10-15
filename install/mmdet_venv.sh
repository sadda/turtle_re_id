#!/bin/bash
if [ ! -d "venv_detection" ]; then
    python3 -m venv venv_detection
fi
source venv_detection/bin/activate
pip install -r mmdet_requirements.txt
mim install mmengine
mim install 'mmcv==2.1.0'
if [ ! -d "mmdetection" ]; then
    git clone https://github.com/open-mmlab/mmdetection.git
fi
pip install -v -e mmdetection
python3 -c 'import mmdet; print(mmdet.__version__)'