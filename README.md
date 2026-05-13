## Overview

Repo for evaluating robustness of Catboost-based LR generation approach for digital activity traces ([paper link](https://www.sciencedirect.com/science/article/pii/S2666281726000041).)

Two datasets: [NFI-FARED](https://huggingface.co/datasets/NetherlandsForensicInstitute/NFI_FARED_Digital_Traces), and one from Amsterdam University of Applied Sciences. 

## Example Commands:

```bash
# Cross-dataset: train on NFI-FARED, test on AUAS
python main.py eval.test_data_path=data/AUAS/clean/auas_processed.csv

# Same dataset, test filtered to iPhone 7 only
python main.py eval.test_phone_types=['Iphone7_IOS_14.7.1']

# Train on iPhone 7 (NFI), test on iPhone 7 (NFI)
python main.py eval.phone_types=['Iphone7_IOS_14.7.1'] eval.test_phone_types=['Iphone7_IOS_14.7.1']

# Cross-dataset + same iOS filter (iOS 14.7.1 in AUAS)
python main.py eval.test_data_path=data/AUAS/clean/auas_processed.csv eval.test_ios_versions=['14.7.1']
```

## AUAS details

activities present: running, walking, standing, train, bus, cycling, stair_up, escalator_up, tram, elevator_up