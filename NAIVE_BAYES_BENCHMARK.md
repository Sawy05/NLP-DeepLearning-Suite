# Text Classification with Naive Bayes

**Author:** Ahmed ElSawy  
**Date:** October 2026  

---

## 1. Abstract & Problem Overview
Text classification is a foundational task in Natural Language Processing (NLP), enabling automated document tagging, sentiment analysis, topic detection, and spam filtering. This report presents the methodology and results for building a multi-class topical text classifier using a **Bag-of-Words (BoW)** feature model and the **Multinomial Naive Bayes (MNB)** algorithm.

The objective was twofold:
1. Implement a complete, from-scratch Multinomial Naive Bayes model using pure Python/NumPy (without external machine learning libraries).
2. Empirically validate the custom model against Scikit-Learn's reference implementation under strict experimental constraints (zero test data leakage, reproducible stratified splitting, Laplace smoothing, and comprehensive visualization).

---

## 2. Dataset Description & Preprocessing Pipeline
The dataset (`synthetic_text_data.csv`) contains 85 short sentences categorized into four distinct topical domains:
- **Technology** (30 samples, 35.3%)
- **Politics** (21 samples, 24.7%)
- **Entertainment** (19 samples, 22.4%)
- **Sports** (15 samples, 17.6%)

### Text Preprocessing Protocol
Raw text documents underwent a systematic multi-step normalization pipeline:
1. **Case Normalization:** Converted all characters to lowercase to eliminate casing discrepancies (e.g., `"Football"` vs. `"football"`).
2. **Noise and Punctuation Removal:** Filtered punctuation marks, digits, and special characters using the regular expression `[^a-z\s]`.
3. **Tokenization:** Segmented normalized text into distinct lexical tokens based on whitespace boundaries.
4. **Stop Word Filtering:** Discarded standard English stop words (function words such as *"the"*, *"is"*, *"at"*, *"which"*) that lack category-discriminating power.
5. **Length Filtering:** Pruned residual single-letter artifacts (retaining tokens with length $\ge 2$).

---

## 3. Train/Test Partitioning & Feature Extraction

### Avoiding Data Leakage
To strictly obey the rule *"Never use test data for anything except final evaluation"*, the dataset was partitioned before constructing the feature space:
- **Split Ratio:** 80% training (68 samples) and 20% test (17 samples).
- **Stratification:** Applied stratified sampling (`stratify=y`) to maintain identical class proportions across both splits.
- **Reproducibility:** Seeded with `random_state=42`.

### Vocabulary Construction
The vocabulary $\mathcal{V}$ was extracted **strictly from the training partition**, producing $|\mathcal{V}| = 345$ unique tokens. Test-set tokens unobserved during training were treated as Out-of-Vocabulary (OOV) terms and ignored during vectorization.

### Bag-of-Words Representation
Each document $d$ was transformed into an integer frequency vector $\mathbf{x} = [x_1, x_2, \dots, x_{|\mathcal{V}|}]^T \in \mathbb{R}^{|\mathcal{V}|}$, where $x_j$ denotes the term frequency of word $w_j$ in document $d$. The resulting training matrix had a sparsity of $97.7\%$, with an average of $7.85$ active tokens per sentence.

---

## 4. Manual Multinomial Naive Bayes Formulation

### 4.1 Probabilistic Model
By Bayes' Theorem, the posterior probability of class $c \in \mathcal{C}$ given document vector $\mathbf{x}$ is:
$$P(c \mid \mathbf{x}) = \frac{P(c) P(\mathbf{x} \mid c)}{P(\mathbf{x})} \propto P(c) \prod_{j=1}^{|\mathcal{V}|} P(w_j \mid c)^{x_j}$$

### 4.2 Class Prior Estimation
Class priors were calculated via Maximum Likelihood Estimation (MLE):
$$P(c) = \frac{N_c}{N_{total}} \implies \log P(c) = \log(N_c) - \log(N_{total})$$

### 4.3 Likelihood with Laplace (Add-1) Smoothing
To avoid zero-probability estimates for words unobserved within a particular training class, additive Laplace smoothing ($\alpha = 1.0$) was incorporated:
$$P(w_j \mid c) = \frac{\text{count}(w_j, c) + \alpha}{\sum_{k=1}^{|\mathcal{V}|} \text{count}(w_k, c) + \alpha \cdot |\mathcal{V}|}$$

### 4.4 Log-Space Inference
To prevent floating-point underflow caused by multiplying numerous fractional probabilities, predictions were computed in log-space:
$$\log P(c, \mathbf{x}) = \log P(c) + \sum_{j=1}^{|\mathcal{V}|} x_j \cdot \log P(w_j \mid c)$$
$$\hat{c} = \arg\max_{c \in \mathcal{C}} \log P(c, \mathbf{x})$$

Normalized posterior log-probabilities were computed using the numerically stable **Log-Sum-Exp** formulation:
$$\log P(c \mid \mathbf{x}) = \log P(c, \mathbf{x}) - \left( m + \log \sum_{c'} \exp(\log P(c', \mathbf{x}) - m) \right), \quad m = \max_{c'} \log P(c', \mathbf{x})$$

---

## 5. Experimental Results & Validation

### 5.1 Validation Against Scikit-Learn
Both the custom `ManualMultinomialNB` and Scikit-Learn's `MultinomialNB(alpha=1.0)` were evaluated on the exact same feature matrices:
- **Prediction Alignment:** $100\%$ identical predictions across all training ($68/68$) and test ($17/17$) instances.
- **Numerical Difference:** Maximum absolute difference in posterior log-probabilities was $< 7.11 \times 10^{-15}$, confirming complete mathematical equivalence to machine precision.

### 5.2 Quantitative Performance Metrics
The classifier demonstrated strong generalization on the unseen test set:
- **Training Accuracy:** $100.0\%$ (68/68)
- **Test Accuracy:** $94.12\%$ (16/17)

#### Per-Class Performance Breakdown (Test Set)
| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Entertainment** | 1.0000 | 1.0000 | 1.0000 | 4 |
| **Politics** | 1.0000 | 0.7500 | 0.8571 | 4 |
| **Sports** | 1.0000 | 1.0000 | 1.0000 | 3 |
| **Technology** | 0.8571 | 1.0000 | 0.9231 | 6 |
| **Macro Average** | **0.9643** | **0.9375** | **0.9451** | **17** |
| **Weighted Average** | **0.9496** | **0.9412** | **0.9419** | **17** |

### 5.3 Test Confusion Matrix
$$\begin{pmatrix} & \text{Pred: Ent} & \text{Pred: Pol} & \text{Pred: Spo} & \text{Pred: Tech} \\ \text{True: Ent} & 4 & 0 & 0 & 0 \\ \text{True: Pol} & 0 & 3 & 0 & 1 \\ \text{True: Spo} & 0 & 0 & 3 & 0 \\ \text{True: Tech} & 0 & 0 & 0 & 6 \end{pmatrix}$$

---

## 6. Qualitative Analysis & Diagnostic Keywords
An analysis of learned word likelihoods $\log P(w \mid c)$ revealed distinctive topical anchor keywords:
- **Technology:** *ai, computing, quantum, tech, data, intelligence, models*
- **Politics:** *government, policy, debate, election, leaders, arena, climate*
- **Entertainment:** *movie, music, festival, box, streaming, blockbuster, artist*
- **Sports:** *cup, league, football, olympics, match, tournament, fans*

### Error Analysis
Across the test set, only a single sentence was misclassified:
- **Sentence:** *"New digital privacy regulations were passed by the government committee."*
- **Ground Truth:** *Politics* | **Predicted:** *Technology*
- **Explanation:** The co-occurrence of terms like *"digital"* and *"privacy"* alongside regulatory terms shifted the posterior distribution slightly in favor of Technology ($P=0.62$ vs. $P=0.38$), exposing an inherent semantic boundary overlap in interdisciplinary topics.

---

## 7. Conclusion
1. **Bag-of-Words Viability:** Despite discarding syntactic word order and context, BoW features provide sufficient semantic discrimination for domain-level categorization.
2. **Laplace Smoothing:** Crucial for robust generalization, successfully mitigating zero-frequency probability penalties.
3. **Leakage Prevention:** Building the vocabulary strictly on training data ensures authentic out-of-sample evaluation.
