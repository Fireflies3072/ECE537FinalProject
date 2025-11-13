# ECE537 Final Project: Network Traffic Classification and Adversarial Generation

## Project Overview

This project explores network traffic classification using machine learning techniques and adversarial attacks on trained classifiers. The project consists of three main tasks:

1. **Task 1**: Centralized learning-based traffic classification
2. **Task 2**: Federated learning-based traffic classification
3. **Task 3**: Adversarial attack using Wasserstein GAN with Gradient Penalty (WGAN-GP)

### Project Description
For detailed project requirements and specifications, refer to the course project description:
- [ECE537 Course Project Description (PDF)](https://fireflies3072.blob.core.windows.net/fireflies/2025/25-11-12-ece537-final-project/ECE537-course-project.pdf)

## Dataset

The project uses encrypted network traffic data from five different applications:
- **bilibili** (Class 0): Chinese video sharing platform
- **flashscore** (Class 1): Sports live score website
- **tiktok** (Class 2): Short video social media platform
- **youtube** (Class 3): Video sharing platform
- **chatgpt** (Class 4): AI chatbot service

### Data Collection

Network traffic data was captured using **Wireshark** by monitoring download packets from each of the five different websites during normal usage sessions. The collection process involved:

1. **Packet Capture**: Using Wireshark to capture network traffic while accessing each application/website
   - Monitor network interface during active sessions (e.g., watching videos, browsing)
   - Focus on download packets containing encrypted payload data

2. **Export to JSON**: Export captured packets from Wireshark to JSON format
   - In Wireshark: `File` → `Export Packet Dissections` → `As JSON`
   - This creates a tshark-compatible JSON file with packet details and hex-encoded payloads

3. **Payload Extraction**: Use `extract_payload.py` script to process the exported JSON
   - Extracts encrypted on-wire payloads from network layers (prioritizes `udp.payload` > `tcp.payload` > `data.data`)
   - Filters and validates hex-encoded payload data
   - Converts valid payloads to Base64 string format for portability
   
   ```bash
   # Example usage
   python src/extract_payload.py
   # Modify in_json_path, out_json_path, and label in the script as needed
   ```

4. **Dataset Organization**: The script outputs data in the required format
   - Creates labeled JSON files (e.g., `data_youtube.json`, `data_chatgpt.json`)
   - Each file contains Base64-encoded payloads organized by application label
   - Multiple class datasets can be merged into the main `data.json` file

### Data Format

The dataset is stored in JSON format with the following structure:

```json
{
  "bilibili": [
    "base64_encoded_packet_1",
    "base64_encoded_packet_2",
    ...
  ],
  "flashscore": [
    "base64_encoded_packet_1",
    "base64_encoded_packet_2",
    ...
  ],
  "tiktok": [...],
  "youtube": [...],
  "chatgpt": [...]
}
```

Each entry represents a network packet's payload encoded as a Base64 string. The dictionary keys serve as class labels for the classification tasks.

### Special Dataset for Task 3

For Task 3 (adversarial attack), we collected a **separate set of YouTube traffic data** (`data_youtube.json`) that is completely independent from the YouTube data in the main training dataset. This ensures:
- **Unbiased Evaluation**: The generated adversarial samples are not evaluated against the same data used to train the victim classifiers
- **Fair Assessment**: Prevents data leakage and provides a more realistic measure of the attack's effectiveness
- **Generalization Testing**: Validates that the generated traffic can fool classifiers trained on different YouTube traffic patterns

### Data Preprocessing

The raw Base64-encoded packets undergo several preprocessing steps before being fed to the neural networks:

1. **Base64 Decoding**: Convert Base64 strings back to raw binary data
   ```python
   raw_packet = base64.b64decode(str_packet)
   ```

2. **Truncation**: Packets longer than 1456 bytes are truncated to maintain consistent input dimensions
   ```python
   raw_packet = raw_packet[:packet_length]  # packet_length = 1456
   ```

3. **Zero Padding**: Packets shorter than 1456 bytes are padded with zeros at the beginning to reach the required length
   ```python
   raw_packet = raw_packet.rjust(packet_length, b'\x00')
   ```

4. **Normalization**: Each byte value (0-255) is normalized to the range [-1, 1] for stable neural network training
   
   ```python
   normalized_value = (byte_value / 127.5) - 1.0
   ```
   
5. **Tensor Conversion**: The normalized packet is converted to a PyTorch tensor for model input
   ```python
   packet_tensor = torch.tensor(packet, dtype=torch.float32)
   ```

### Dataset Split

- **Training Set**: 90% of the data (0.0 - 0.9 split ratio)
- **Test Set**: 10% of the data (0.9 - 1.0 split ratio)

For Task 2 (Federated Learning), the training set is further divided equally among 3 clients:
- **Client 1**: 0.0 - 0.3 (30% of total data)
- **Client 2**: 0.3 - 0.6 (30% of total data)
- **Client 3**: 0.6 - 0.9 (30% of total data)

Each packet is represented as a sequence of 1456 bytes, normalized to the range [-1, 1].

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

### Running on Custom Dataset

To generate adversarial samples for a different traffic class, edit the following lines in `src/task3.py`:

```python
# Line 25-26: Specify the target class
data_label = 3              # Change to target class ID (0-4)
data_label_str = 'youtube'  # Change to corresponding class name

# Line 30: Specify the dataset path
data_path = os.path.join(base_dir, 'data', 'data_youtube.json')  # Change to your dataset file
```

**Available Classes:**
- `0` - `'bilibili'`
- `1` - `'flashscore'`
- `2` - `'tiktok'`
- `3` - `'youtube'`
- `4` - `'chatgpt'`

**Example:** To generate adversarial ChatGPT traffic, change:
```python
data_label = 4
data_label_str = 'chatgpt'
data_path = os.path.join(base_dir, 'data', 'data_chatgpt.json')
```

**Note:** Make sure you have a separate dataset file for the target class to ensure fair evaluation, independent from the training data used in Task 1 and Task 2.

### Download Model
- [task3_best.pt](https://fireflies3072.blob.core.windows.net/fireflies/2025/25-11-12-ece537-final-project/model/task3_best.pt) - Best WGAN-GP model checkpoint (Generator only)

## Implementation Details

### Project Structure
```
ECE537FinalProject/
├── data/
│   ├── data.json              # Main dataset (5 classes)
│   └── data_youtube.json      # YouTube-only dataset for Task 3
├── model/
│   ├── task1_best.pt          # Best centralized model
│   ├── task1_latest.pt        # Latest centralized model checkpoint
│   ├── task1_stat.json        # Task 1 training statistics
│   ├── task1_stat.png         # Task 1 performance plots
│   ├── task2_best.pt          # Best federated model
│   ├── task2_latest.pt        # Latest federated model checkpoint
│   ├── task2_stat.json        # Task 2 training statistics
│   ├── task2_stat.png         # Task 2 performance plots
│   ├── task3_best.pt          # Best GAN model (Generator + Discriminator)
│   ├── task3_latest.pt        # Latest GAN model checkpoint
│   ├── task3_stat1.json       # Task 3 Classifier 1 statistics
│   ├── task3_stat2.json       # Task 3 Classifier 2 statistics
│   └── task3_stat.png         # Task 3 performance plots
├── src/
│   ├── dataset.py             # Dataset loader and preprocessing
│   ├── model.py               # Neural network architectures (Classifier, Generator, Discriminator)
│   ├── utils.py               # Utility functions (training, saving, statistics)
│   ├── extract_payload.py     # Wireshark JSON to base64 payload converter
│   ├── task1.py               # Centralized learning implementation
│   ├── task2.py               # Federated learning implementation
│   └── task3.py               # WGAN-GP adversarial attack implementation
├── ECE537-course project.pdf  # Project description and requirements
├── LICENSE                    # License file
├── README.md                  # This file (project documentation)
├── requirements.txt           # Python dependencies
└── pyproject.toml             # Project configuration and metadata
```

### Installation

#### Step 1: Install PyTorch

First, install PyTorch based on your system configuration (CPU/CUDA version). Visit the official [PyTorch](https://pytorch.org/) website to get the appropriate installation command for your system.

Select your system configuration and install PyTorch.

#### Step 2: Clone Repository

```bash
git clone https://github.com/Fireflies3072/ECE537FinalProject.git
cd ECE537FinalProject
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The `requirements.txt` includes PyTorch, but if you've already installed it in Step 1, pip will skip it or verify the installation.

### Dependencies

**Core Requirements:**
- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- tqdm >= 4.62.0

**Standard Library (included with Python):**
- json
- base64
- os
- platform
- copy

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

## Authors

- Chengling Xu
- Junwen Gu
- Jinyao Sun

## License

This project is part of academic coursework for ECE537.
