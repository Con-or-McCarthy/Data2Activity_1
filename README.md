## Overview

Repo for evaluating robustness of Catboost-based LR generation approach for digital activity traces ([paper link](https://www.sciencedirect.com/science/article/pii/S2666281726000041).)

Two datasets: [NFI-FARED](https://huggingface.co/datasets/NetherlandsForensicInstitute/NFI_FARED_Digital_Traces), and one from Amsterdam University of Applied Sciences. 

## Key Configuration Parameters

### `eval.setup`
Controls whether train and test come from the same or different datasets.
- `same` (default): train and test split from the same dataset by subject
- `cross`: train on `eval.train_data`, test on the other dataset

### `eval.train_data`
Which dataset to train on (`NFI` or `AUAS`, default: `NFI`).  
When `eval.setup=cross`, the other dataset is used for testing.

### `eval.phone_types`
Restricts both train and test to specific phone models. Uses **normalised matching** — the same value works across both datasets regardless of capitalisation or device-number suffixes (e.g. `Iphone7_IOS_14.7.1` matches both NFI's `Iphone7_IOS_14.7.1` and AUAS's `iPhone_7_iOS_14.7.1_nr20`).

**Available phone types (use NFI-style names):**
| Name | Dataset |
|------|---------|
| `Iphone6+_IOS_11.4.1` | NFI only |
| `Iphone7_IOS_14.7.1` | NFI + AUAS |
| `Iphone7_IOS_13.5.1` | AUAS only |
| `Iphone11_IOS_13.1.1` | NFI only |
| `IphoneXR_IOS_15.4.1` | NFI only |

## Example Commands

```bash
# Same-dataset baseline (NFI train + test)
python main.py eval.activity_pair=[walking,standing]

# Cross-dataset: train on NFI, test on AUAS
python main.py eval.activity_pair=[walking,standing] eval.setup=cross

# Same-dataset, restrict to iPhone 7 iOS 14.7.1 only
python main.py eval.activity_pair=[walking,standing] \
  eval.phone_types=['Iphone7_IOS_14.7.1']

# Cross-dataset, restrict both to the iPhone 7 iOS 14.7.1 (NFI name works for both)
python main.py eval.activity_pair=[walking,standing] eval.setup=cross \
  eval.phone_types=['Iphone7_IOS_14.7.1']

# Cross-dataset, train on non-iPhone-7 NFI, test on all AUAS (all iPhone 7)
python main.py eval.activity_pair=[walking,standing] eval.setup=cross \
  eval.phone_types=['Iphone6+_IOS_11.4.1','Iphone11_IOS_13.1.1','IphoneXR_IOS_15.4.1']
```

## AUAS details

activities present: running, walking, standing, train, bus, cycling, stair_up, escalator_up, tram, elevator_up