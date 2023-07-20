# common_rose
This is an image classification project designed for assessing junior data scientist candidates for hire.
The name common_rose is one of the butterfly species found in Singapore. The company im working for also have its name inspired by a endangered butterfly species. Naming in this way makes it harder for potential candidate to find this repo.

# Setup Environment
``` shell
python -m venv venv
source venv/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -U scikit-learn
```

# Dataset
Download dataset from
https://www.kaggle.com/datasets/gpreda/chinese-mnist/download?datasetVersionNumber=7
Unzip downloaded
``` shell
unzip archive.zip
```

# Run Script
``` shell
python image_classification.py
```
best_model.pth will be updated when accuracy of test dataset is better than previous during training loop