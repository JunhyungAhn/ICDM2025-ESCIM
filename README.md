# On Predicting Post-Click Conversion Rate via Counterfactual Inference

## Table of Contents

1. [Introduction](#introduction)
2. [Framework](#framework)
3. [Experimental Results](#results)
4. [Setup](#setup)

## Introduction

Accurately estimating a conversion rate (CVR) is essential in various recommendation domains, such as online advertising systems and e-commerce. Such systems utilize user interaction logs, which consist of impressions, clicks, and conversions. Since both clicked and converted instances are very sparse, it is crucial to collect a substantial amount of logs. Recent works address this issue by devising frameworks that can leverage non-clicked items. However, the discrepancy between clicked and non-clicked items often causes biases in the naively trained CVR prediction model.

Against this background, we attempt to answer "Would the user have converted if he/she had clicked the recommended item?" by proposing a novel conversion label generation method based on counterfactual inference, named the Entire Space Counterfactual Inference Multi-task Model (ESCIM). We initially train a structure causal model (SCM) of user sequential behaviors and conduct a hypothetical intervention (i.e., click) on non-clicked items to infer counterfactual CVR for them. We then introduce several approaches to transform predicted counterfactual CVRs into binary counterfactual conversion labels for the non-clicked samples. Finally, the generated samples are incorporated into the training process. 

Extensive experiments on public datasets illustrate the superiority of the proposed algorithm. Online A/B testing further empirically validates the effectiveness of our proposed algorithm in real-world scenarios. In addition, we demonstrate improved performances of the proposed method on latent conversion data, showcasing its robustness and superior generalization capabilities.


## Framework
![Entire Space Counterfactual Inference Multi-task Model](./ESCIM.pdf)


## Experimental Results

### Ali-CCP across Backbones

| Method       | MLP           | DeepFM        | AutoInt       | DCN-V2        |
|--------------|---------------|---------------|---------------|---------------|
| ESMM         | 0.6474 / 0.6379 | 0.6398 / 0.6213 | 0.6543 / 0.6302 | 0.6495 / 0.6373 |
| Multi-IPS    | 0.6523 / 0.6390 | 0.6431 / 0.6244 | 0.6587 / 0.6357 | 0.6535 / 0.6379 |
| Multi-DR     | 0.6437 / 0.6305 | 0.6388 / 0.6199 | 0.6542 / 0.6340 | 0.6488 / 0.6353 |
| ESCM<sup>2</sup>-IPS | 0.6691 / 0.6424 | 0.6487 / 0.6281 | 0.6681 / 0.6407 | 0.6609 / 0.6374 |
| ESCM<sup>2</sup>-DR  | 0.6609 / 0.6353 | 0.6502 / 0.6280 | 0.6485 / 0.6329 | 0.6489 / 0.6313 |
| DCMT         | 0.6743 / 0.6451 | 0.6528 / 0.6338 | 0.6702 / 0.6389 | 0.6603 / 0.6350 |
| **ESCIM-max**    | **0.6792** / 0.6487 | **0.6585** / **0.6401** | **0.6737** / 0.6413 | **0.6698** / 0.6467 |
| **ESCIM-ratio**  | 0.6756 / **0.6566** | 0.6523 / 0.6382 | 0.6774 / **0.6475** | 0.6683 / **0.6534** |

### Ali-Express with MLP across Countries

| Method       | AE-ES         | AE-FR         | AE-NL         | AE-US         |
|--------------|---------------|---------------|---------------|---------------|
| ESMM         | 0.8099 / 0.8717 | 0.7949 / 0.8500 | 0.7795 / 0.8479 | 0.7952 / 0.8425 |
| Multi-IPS    | 0.8205 / 0.8592 | 0.7947 / 0.8504 | 0.7829 / 0.8377 | 0.8068 / 0.8433 |
| Multi-DR     | 0.8135 / 0.8600 | 0.7943 / 0.8538 | 0.7803 / 0.8292 | 0.7956 / 0.8396 |
| ESCM<sup>2</sup>-IPS | 0.8199 / 0.8777 | 0.8078 / 0.8511 | 0.7867 / 0.8511 | 0.8018 / 0.8573 |
| ESCM<sup>2</sup>-DR  | 0.8112 / 0.8739 | 0.8073 / 0.8535 | 0.7772 / 0.8473 | 0.7917 / 0.8404 |
| DCMT         | 0.8251 / 0.8838 | 0.8089 / 0.8628 | 0.7897 / 0.8525 | 0.8140 / 0.8620 |
| **ESCIM-max**    | **0.8314** / 0.8935 | **0.8210** / 0.8776 | **0.7951** / **0.8613** | **0.8332** / **0.8782** |
| **ESCIM-ratio**  | 0.8305 / **0.8951** | 0.8192 / **0.8821** | 0.7907 / 0.8583 | 0.8201 / 0.8643 |

### Online A/B Test

Metrics with an upward-pointing arrow (↑) indicate that higher values are better, while metrics with a downward-pointing arrow (↓) indicate that lower values are preferable.

| Metric   | Day 1    | Day 2    | Day 3    | Day 4    | Day 5    |
|----------|----------|----------|----------|----------|----------|
| CVR (↑)  | +21.55%  | +16.94%  | +12.59%  | +17.42%  | +18.24%  |
| CTCVR (↑)| +7.41%   | +4.76%   | +3.98%   | +5.56%   | +6.31%   |
| CPA (↓)  | -27.16%  | -26.67%  | -16.05%  | -22.53%  | -24.38%  |

## Setup

### Prerequisites

Required software, libraries, or tools to run the code or reproduce the results.

- Python (3.10.13)
- PyTorch (2.0.1+cu118)
- NumPy (1.23.5)
- Pandas (1.5.3)

### Installation

Step-by-step instructions on how to set up the environment and install dependencies.

```sh
# Example installation commands
git clone {git url}
pip install -r requirements.txt
```

### Download Ali-CCP and Ali-Express Dataset

- Ali-CCP: https://tianchi.aliyun.com/dataset/408
- Ali-Express: https://tianchi.aliyun.com/dataset/74690

### Preprocessing
```sh
# 1. Preprocess Ali-CCP dataset
python Ali_CCP/preprocess.py

# 2. Preprocess Ali-Express dataset
python Ali_Express/preprocess.py --country=ES
```

### Command
```sh
# 1. Run ESCIM for the Ali-CCP dataset
python run.py --dataset=Ali-CCP

# 2. Run ESCIM for the Ali-Express ES dataset
python run.py --dataset=Ali-Express --country=ES
```

### Hyperparameters
Change the hyperparameters related to training in the 'config.json' file under each dataset's repository.
