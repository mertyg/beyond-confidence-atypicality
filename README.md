# Beyond Confidence: Reliable Models Should Also Consider Atypicality (NeurIPS 2023)
This is the repository for the paper [**Beyond Confidence: Reliable Models Should Also Consider Atypicality (NeurIPS 2023)**](https://arxiv.org/abs/2305.18262). An earlier version of the paper appeared as a Contributed Talk at ICLR 2023 Workshop on Trustworthy Machine Learning.

Overall, we demonstrate that machine learning models should also consider atypicality (i.e. how 'rare' a sample is) when making predictions. We show simple- and easy-to-implement atypicality estimators can provide significant value.
# Atypicality Estimation

## LLM CLassification
For large language models, the log-likelihood provided by the model already estimates how `typical` a prompt is. Therefore, we simply use
```python
from model_zoo import get_alpaca7b
# Assume we have the path to Alpaca7B under `model_path`
model = get_alpaca7b(model_path=model_path)
test_atypicality = -model.get_batch_loglikelihood(test_prompts)
```
We will later use the atypicality value in the recalibration algorithm to boost accuracy and calibration. See `model_zoo/llm.py` for further details.

**Obtaining Alpaca7b:** Please refer to `https://github.com/tatsu-lab/stanford_alpaca` for instructions on how to obtain Alpaca7b.

## Image Classification
For image classifiers, there is no out-of-the-box atypicality estimator. Therefore, we use the following simple estimator:
```python
from atypicality import GMMAtypicalityEstimator
from model_zoo import get_image_classifier
from data_zoo import get_image_dataset

# First, load the dataset and model of interest.
model, preprocess = get_image_classifier("resnext50_imagenet_lt", device="cuda")
train_dataset, val_dataset, test_dataset = get_image_dataset("imagenet_lt", preprocess=preprocess)
# We first extract the features for the training and test sets:
train_features, train_logits, train_labels = model.run_and_cache_outputs(train_dataset, batch_size=16, output_dir="./outputs/")
calib_features, calib_logits, calib_labels = model.run_and_cache_outputs(val_dataset, batch_size=16, output_dir="./outputs/")
test_features, test_logits, test_labels = model.run_and_cache_outputs(test_dataset, batch_size=16, output_dir="./outputs/")
# Then, we fit the estimator to the embeddings of the training set:
atypicality_estimator = GMMAtypicalityEstimator()
atypicality_estimator.fit(train_features, train_labels)
# We then use the estimator to estimate atypicality of the calibration and test sets:
calib_atypicality = atypicality_estimator.predict(calib_features)
test_atypicality = atypicality_estimator.predict(test_features)
```

# Improving Calibration (and Performance) with Atypicality-Aware Recalibration (AAR)
Once the atypicality of different inputs are computed, we can perform atypicality-aware recalibration with a few simple lines:
```python
from calibration import AtypicalityAwareCalibrator
aar_calib = AtypicalityAwareCalibrator()
# Train the AAR calibrator: 
aar_calib.fit(calib_logits, calib_atypicality, calib_labels, max_iters=1500)
# Use the AAR calibrator:
test_pred_probs = aar_calib.predict_proba(test_logits, test_atypicality)
```

# Reproducing Results
To reproduce our LLM classification experiments, please refer `LLM Classification.ipynb`. 

Similarly, to reproduce our image classification experiments, please refer to `Imbalanced Image Classification.ipynb` and `Balanced Image Classification.ipynb`. 

In each of the above notebooks, we demonstrate how atypicality-awareness improves the accuracy and the calibration of the models.

# Citation
If you find this repository or the ideas therein useful, please consider citing our paper:
```
@article{yuksekgonul2023beyond,
  title={Beyond Confidence: Reliable Models Should Also Consider Atypicality},
  author={Yuksekgonul, Mert and Zhang, Linjun and Zou, James and Guestrin, Carlos},
  journal={arXiv preprint arXiv:2305.18262},
  year={2023}
}
```
