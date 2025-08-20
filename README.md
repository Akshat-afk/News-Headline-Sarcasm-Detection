# Sarcasm Detection with Data Augmentation & Transformer Models

## 📌 Overview
This project implements a **sarcasm detection pipeline** using advanced **data augmentation** strategies and **transformer-based classification** models.  
The primary goal is to improve model robustness and generalization by **expanding the dataset** with synthetically generated variations while training on a **RoBERTa classifier**.

---

## 🔑 Key Features
1. **Multi-Stage Data Augmentation**  
   - **PPDB-like Synonym Replacement (WordNet-based)**  
   - **Word2Vec Similarity Swaps (GloVe embeddings)**  
   - **BERT-based Word Insertion (Masked Language Modeling)**  
   - **Fast Augmentation with Batch BERT Predictions**

2. **Dataset Preparation**  
   - Source: [Sarcasm Headlines Dataset v2](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)  
   - Filtered headlines between **45–180 characters** for meaningful context.  
   - Augmented dataset size: **2x original**.  
   - Final shuffled dataset exported as `fast_augmented_dataset.csv`.

3. **Model Training**  
   - Fine-tuned **RoBERTa-base** for **binary classification (sarcastic vs non-sarcastic)**.  
   - Implemented with **PyTorch + HuggingFace Transformers**.  
   - Optimized with **AdamW + Linear Scheduler**.  
   - Hardware: CUDA GPU-enabled training.

4. **Evaluation**  
   - Train/validation split: **80/20**.  
   - Achieved **97% accuracy** on validation set.  
   - Balanced performance across sarcastic (1) and non-sarcastic (0) classes.

---

## 📂 Project Structure
```
.
├── Sarcasm_Headlines_Dataset_v2.json   # Raw dataset
├── augmented_sarcasm_dataset.csv       # Full augmentation (3-stage)
├── fast_augmented_dataset.csv          # Fast BERT-only augmentation
├── sarcasm_training.py                 # Training & evaluation script
└── README.md                           # Project documentation
```

---

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sarcasm-detection.git
   cd sarcasm-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages include:
   - `torch`
   - `transformers`
   - `scikit-learn`
   - `pandas`
   - `nltk`
   - `gensim`
   - `tqdm`

3. Download NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

---

## 🚀 Usage

### 1. Data Augmentation
```python
python augment.py
```
- Generates `fast_augmented_dataset.csv` with doubled samples.

### 2. Model Training
```python
python train.py
```
- Trains RoBERTa on augmented dataset for sarcasm detection.

### 3. Evaluation
- Produces classification report:
```
Accuracy: 0.9699
Precision (macro): 0.9714
Recall (macro): 0.9688
F1 Score (macro): 0.9697
```

---

## 📊 Results
| Metric        | Score  |
|---------------|--------|
| Accuracy      | 96.99% |
| Precision     | 97.14% |
| Recall        | 96.88% |
| F1 (macro)    | 96.97% |
| F1 (weighted) | 96.98% |

---

## 🔮 Future Work
- Explore **back-translation augmentation** for richer paraphrasing.  
- Experiment with **larger transformer models** (e.g., DeBERTa, XLNet).  
- Deploy as an **API endpoint** using FastAPI or Flask.  
- Extend to **multi-class sarcasm detection** with fine-grained humor categories.  

