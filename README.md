# Weakly-Supervised-Street-Text-Detection
Weakly supervised street text detection , localisation and segmentation in Pytorch  

## Instructions

1. Install the required python packages by running

   ```pip install -r requirements.txt```

2. Download Chars74k dataset () and place it in the root directory
3.
4. Train a charecter recognition network by running

   ```python3 train_charmodel.py```

5. Label the images using the following command

   ```python3 label_images.py```

6. Train a localisation , detection and segmentation network by running the command

   ```python3 train_localizationmodel.py```
