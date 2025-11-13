# ECE537 Final Project: Network Traffic Classification and Adversarial Generation

## Project Overview

This project explores network traffic classification using machine learning techniques and adversarial attacks on trained classifiers. The project consists of three main tasks:

1. **Task 1**: Centralized learning-based traffic classification
2. **Task 2**: Federated learning-based traffic classification
3. **Task 3**: Adversarial attack using Wasserstein GAN with Gradient Penalty (WGAN-GP)

## Dataset

The project uses encrypted network traffic data from five different applications:
- **bilibili** (Class 0): Chinese video sharing platform
- **flashscore** (Class 1): Sports live score website
- **tiktok** (Class 2): Short video social media platform
- **youtube** (Class 3): Video sharing platform
- **chatgpt** (Class 4): AI chatbot service

Each packet is represented as a sequence of 1456 bytes, normalized to the range [-1, 1]. The dataset is split with 90% for training and 10% for testing.

### Download Data Files
- [data.json](https://fireflies3072.blob.core.windows.net/fireflies/2025/25-11-12-ece537-final-project/data/data.json) - Main dataset with all 5 classes
- [data_youtube.json](https://fireflies3072.blob.core.windows.net/fireflies/2025/25-11-12-ece537-final-project/data/data_youtube.json) - YouTube-only dataset for Task 3

## Model Architectures

### Classifier Network
A fully connected neural network with the following architecture:
- **Input layer**: 1456 dimensions (packet length)
- **Hidden layers**: Two hidden layers with 128 neurons each
- **Activation function**: SiLU (Sigmoid Linear Unit)
- **Output layer**: 5 neurons (one per class)

### Generator Network (Task 3)
A deep neural network for generating synthetic network packets:
- **Input**: 91-dimensional latent vector (random noise)
- **Architecture**: Four fully connected layers with batch normalization
- **Layer sizes**: 91 → 182 → 364 → 728 → 1456
- **Activation**: SiLU
- **Output**: 1456-dimensional packet representation

### Discriminator Network (Task 3)
A critic network for distinguishing real from generated packets:
- **Input**: 1456-dimensional packet
- **Architecture**: Five fully connected layers with instance normalization
- **Layer sizes**: 1456 → 728 → 364 → 182 → 91 → 1
- **Activation**: SiLU
- **Output**: Wasserstein distance estimate

## Task 1: Centralized Learning

### Methodology
- **Approach**: Standard supervised learning with centralized training
- **Optimizer**: Adam optimizer with learning rate 0.0001
- **Loss function**: Cross-entropy loss
- **Batch size**: 32
- **Early stopping**: Training stops after 5 epochs without improvement
- **Best epoch**: 9

### Hyperparameters
- `packet_length`: 1456
- `num_class`: 5
- `batch_size`: 32
- `hidden_dim`: 128
- `learning_rate`: 0.0001
- `tolerate_epoch`: 5

### Results

| Class | Application | Precision (%) | Recall (%) | Accuracy (%) | F1 Score (%) |
|-------|-------------|---------------|------------|--------------|--------------|
| 0 | bilibili | 100.00 | 53.33 | 91.00 | 69.57 |
| 1 | flashscore | 68.81 | 81.52 | 83.60 | 74.63 |
| 2 | tiktok | 73.02 | 83.64 | 91.64 | 77.97 |
| 3 | youtube | 98.11 | 98.11 | 99.36 | 98.11 |
| 4 | chatgpt | 85.19 | 90.20 | 95.82 | 87.62 |

**Average Performance**: 
- Mean Precision: 85.03%
- Mean Recall: 81.36%
- Mean Accuracy: 92.28%
- Mean F1 Score: 81.58%

### Analysis
The centralized model performs exceptionally well on YouTube traffic (98.11% F1 score), likely due to distinctive traffic patterns. The model shows high precision for bilibili (100%) but lower recall (53.33%), indicating conservative classification. ChatGPT and TikTok traffic are classified with good balanced performance (F1 scores of 87.62% and 77.97% respectively).

### Training Statistics

![Task 1 Training Statistics](https://fireflies3072.blob.core.windows.net/fireflies/2025/25-11-12-ece537-final-project/model/task1_stat.png)

The plot above shows the progression of precision, recall, accuracy, and F1 score across training epochs for the centralized learning model.

### Download Model
- [task1_best.pt](https://fireflies3072.blob.core.windows.net/fireflies/2025/25-11-12-ece537-final-project/model/task1_best.pt) - Best centralized model checkpoint

## Task 2: Federated Learning

### Methodology
- **Approach**: Federated averaging with 3 clients
- **Data distribution**: Training data evenly split among 3 clients (each gets 30% of data)
- **Aggregation method**: FedAvg (model parameter averaging)
- **Optimizer**: Adam optimizer (re-initialized for each client per round)
- **Learning rate**: 0.0001
- **Batch size**: 32
- **Best epoch**: 28

### Hyperparameters
- `num_client`: 3
- `learning_rate`: 0.0001
- `batch_size`: 32
- `hidden_dim`: 128

### Federated Learning Process
1. Initialize global model
2. For each communication round:
   - Distribute global model to all clients
   - Each client trains on local data
   - Aggregate client models by averaging parameters
   - Update global model
3. Evaluate global model on test set

### Results

| Class | Application | Precision (%) | Recall (%) | Accuracy (%) | F1 Score (%) |
|-------|-------------|---------------|------------|--------------|--------------|
| 0 | bilibili | 93.94 | 51.67 | 90.03 | 66.67 |
| 1 | flashscore | 75.56 | 73.91 | 85.21 | 74.73 |
| 2 | tiktok | 67.50 | 98.18 | 91.32 | 80.00 |
| 3 | youtube | 91.23 | 98.11 | 98.07 | 94.55 |
| 4 | chatgpt | 88.24 | 88.24 | 96.14 | 88.24 |

**Average Performance**:
- Mean Precision: 83.29%
- Mean Recall: 82.02%
- Mean Accuracy: 92.15%
- Mean F1 Score: 80.84%

### Comparison with Task 1

| Metric | Task 1 (Centralized) | Task 2 (Federated) | Difference |
|--------|---------------------|-------------------|------------|
| Mean Precision | 85.03% | 83.29% | -1.74% |
| Mean Recall | 81.36% | 82.02% | +0.66% |
| Mean Accuracy | 92.28% | 92.15% | -0.13% |
| Mean F1 Score | 81.58% | 80.84% | -0.74% |

### Analysis
The federated learning model achieves comparable performance to the centralized model despite data being distributed across 3 clients. The slight performance drop (-0.74% in F1 score) is expected due to the challenges of federated learning, including:
- Non-IID data distribution across clients
- Communication overhead
- Reduced effective training data per client

Notable observations:
- TikTok recall improved significantly (83.64% → 98.18%), suggesting federated learning captured diverse traffic patterns
- YouTube classification remained strong (94.55% F1 score)
- Bilibili precision decreased slightly but remained high (93.94%)

### Training Statistics

![Task 2 Training Statistics](https://fireflies3072.blob.core.windows.net/fireflies/2025/25-11-12-ece537-final-project/model/task2_stat.png)

The plot above shows the progression of precision, recall, accuracy, and F1 score across training epochs for the federated learning model.

### Download Model
- [task2_best.pt](https://fireflies3072.blob.core.windows.net/fireflies/2025/25-11-12-ece537-final-project/model/task2_best.pt) - Best federated model checkpoint

## Task 3: Adversarial Attack with WGAN-GP

### Methodology
- **Target**: Generate synthetic YouTube (Class 3) network traffic
- **Attack model**: Wasserstein GAN with Gradient Penalty (WGAN-GP)
- **Target classifiers**: Both Task 1 (centralized) and Task 2 (federated) models
- **Objective**: Generate packets that are classified as YouTube by both classifiers
- **Training approach**: Train generator and discriminator adversarially on real YouTube traffic

### Hyperparameters
- `latent_size`: 91
- `batch_size`: 64
- `learning_rate_G`: 0.0001 (Generator)
- `learning_rate_D`: 0.0004 (Discriminator)
- `lambda_gp`: 10 (gradient penalty coefficient)
- `betas`: (0.0, 0.9) for Adam optimizer
- **Best epoch**: 1

### WGAN-GP Training
- **Generator loss**: Negative Wasserstein distance (maximize discriminator score for fake samples)
- **Discriminator loss**: Wasserstein distance + gradient penalty
- **Gradient penalty**: Enforces 1-Lipschitz constraint for stable training

### Results

#### Classifier 1 (Task 1 - Centralized Model)

| Class | Application | Precision (%) | Recall (%) | Accuracy (%) | F1 Score (%) |
|-------|-------------|---------------|------------|--------------|--------------|
| 3 | youtube | 100.00 | 100.00 | 100.00 | 100.00 |

#### Classifier 2 (Task 2 - Federated Model)

| Class | Application | Precision (%) | Recall (%) | Accuracy (%) | F1 Score (%) |
|-------|-------------|---------------|------------|--------------|--------------|
| 3 | youtube | 100.00 | 100.00 | 100.00 | 100.00 |

### Attack Success Rate

| Target Classifier | Success Rate |
|------------------|--------------|
| Centralized Model (Task 1) | 100% |
| Federated Model (Task 2) | 100% |
| **Overall Attack Success** | **100%** |

### Analysis
The WGAN-GP successfully generated synthetic network packets that achieved **perfect classification** as YouTube traffic by both the centralized and federated classifiers. This demonstrates:

1. **Vulnerability of Traffic Classifiers**: Both centralized and federated models are susceptible to adversarial attacks
2. **Effectiveness of WGAN-GP**: The Wasserstein GAN with gradient penalty effectively learned the distribution of YouTube network traffic
3. **Transferability**: Generated samples fool both independently trained classifiers, indicating that the attack captures fundamental traffic patterns
4. **Early Convergence**: Achieving 100% success rate at epoch 1 suggests the YouTube traffic has distinctive patterns that are relatively easy to replicate

### Security Implications
The perfect attack success rate highlights critical security concerns:
- **Privacy risks**: Adversaries could generate fake traffic to evade detection
- **Traffic analysis limitations**: ML-based traffic classification may be unreliable against sophisticated adversaries
- **Need for robust defenses**: Adversarial training and defense mechanisms are essential for production systems

### Training Statistics

![Task 3 Training Statistics](https://fireflies3072.blob.core.windows.net/fireflies/2025/25-11-12-ece537-final-project/model/task3_stat.png)

The plot above shows the progression of precision, recall, accuracy, and F1 score for both Classifier 1 (left) and Classifier 2 (right) when evaluating the generated samples across training epochs.

### Download Model
- [task3_best.pt](https://fireflies3072.blob.core.windows.net/fireflies/2025/25-11-12-ece537-final-project/model/task3_best.pt) - Best WGAN-GP model checkpoint (Generator and Discriminator)

## Implementation Details

### Project Structure
```
ECE537FinalProject/
├── data/
│   ├── data.json              # Main dataset (5 classes)
│   └── data_youtube.json      # YouTube-only dataset for Task 3
├── model/
│   ├── task1_best.pt          # Best centralized model
│   ├── task2_best.pt          # Best federated model
│   ├── task3_best.pt          # Best GAN model
│   └── *.json, *.png          # Statistics and plots
├── src/
│   ├── dataset.py             # Dataset loader
│   ├── model.py               # Neural network architectures
│   ├── utils.py               # Utility functions
│   ├── task1.py               # Centralized learning
│   ├── task2.py               # Federated learning
│   └── task3.py               # WGAN-GP attack
└── README.md                  # This file
```

### Dependencies
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- tqdm
- json
- base64

### Running the Code

```bash
# Task 1: Centralized Learning
python src/task1.py

# Task 2: Federated Learning
python src/task2.py

# Task 3: WGAN-GP Attack
python src/task3.py
```

## Key Findings

1. **Centralized vs Federated Learning**: Federated learning achieves comparable performance (0.74% F1 score difference) to centralized learning while preserving data privacy across distributed clients.

2. **Model Robustness**: Both centralized and federated models show vulnerability to adversarial attacks, with the WGAN-GP achieving 100% attack success rate.

3. **Traffic Patterns**: YouTube traffic shows the most distinctive patterns (highest F1 scores in both Task 1 and 2), making it both easy to classify and to generate synthetically.

4. **Federated Learning Benefits**: Despite slight performance trade-offs, federated learning provides privacy guarantees by keeping data distributed while maintaining competitive accuracy.

5. **Adversarial Vulnerability**: The perfect attack success rate demonstrates that current ML-based traffic classifiers require additional defensive mechanisms to be deployment-ready.

## Conclusions

This project successfully implements and evaluates three critical aspects of network traffic analysis:

### Task 1 - Centralized Learning
Established a strong baseline classifier with 81.58% mean F1 score, demonstrating that neural networks can effectively classify encrypted network traffic based on packet-level patterns.

### Task 2 - Federated Learning
Proved the feasibility of privacy-preserving federated learning for traffic classification, achieving 80.84% mean F1 score with data distributed across 3 clients. This approach enables collaborative model training without centralizing sensitive network data.

### Task 3 - Adversarial Attack
Exposed critical vulnerabilities in ML-based traffic classifiers through a successful WGAN-GP attack achieving 100% fooling rate. This highlights the need for:
- Adversarial training techniques
- Robust model architectures
- Multi-modal verification systems
- Continuous model monitoring and updating

### Future Work

1. **Defense Mechanisms**: Implement adversarial training and certified defenses
2. **Non-IID Federated Learning**: Explore performance under more realistic heterogeneous data distributions
3. **Differential Privacy**: Add privacy guarantees to federated learning
4. **Multi-class Attacks**: Extend adversarial attacks to target multiple traffic classes
5. **Real-time Deployment**: Optimize models for low-latency online classification
6. **Explainability**: Analyze what features the models learn to improve interpretability

## Authors

ECE537 Course Project

## License

This project is part of academic coursework for ECE537.
