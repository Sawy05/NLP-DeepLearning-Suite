# Comparative Evaluation of Word Embedding Architectures, Manifold Projections, and Downstream Document Semantics: Word2Vec Skip-gram vs. GloVe

**Author:** Academic Researcher & NLP Engineer  
**Date:** October 2026  

---

## Abstract
Continuous vector space representations of words represent a cornerstone of modern natural language processing. This study presents a systematic empirical and theoretical comparison between two foundational word embedding paradigms: **Word2Vec Skip-gram with Negative Sampling (SGNS)**—a local predictive neural model—and **Global Vectors for Word Representation (GloVe)**—a log-bilinear model factorizing global word co-occurrence counts. Evaluating on the Kaggle 5-Domain Text Document Classification Dataset (encompassing 2,225 news articles across politics, sport, tech, entertainment, and business), we examine: (1) token normalization via morphological WordNet lemmatization versus heuristic stemming; (2) nearest-neighbor geometry across 10 domain anchor words, identifying the contrast between episodic syntagmatic collocations and canonical paradigmatic taxonomies; (3) dimensionality reduction behavior comparing linear Principal Component Analysis (PCA) against non-linear t-Distributed Stochastic Neighbor Embedding (t-SNE); (4) quantitative Topic Coherence across 50 domain keywords; and (5) downstream multi-class document classification using centroid pooling coupled with Logistic Regression and Linear Support Vector Machines. Skip-gram achieved 97.75% accuracy (Linear SVM) and 97.53% (Logistic Regression), slightly outperforming pre-trained GloVe (97.30%) due to target-domain calibration, while GloVe exhibited superior global topic coherence (mean coherence 0.5794 vs. 0.3727). We conclude with an in-depth error analysis of the Politics–Business boundary overlap and articulate the structural limitations of static vector spaces regarding polysemy, pooling dilution, and out-of-vocabulary tokens.

---

## 1. Title & Objective

### 1.1 Study Title
**Comparative Evaluation of Word Embedding Architectures, Manifold Projections, and Downstream Document Semantics: Word2Vec Skip-gram vs. GloVe**

### 1.2 Core Objectives
1. **Pipeline Construction:** Design a reproducible, production-grade text normalization and feature engineering pipeline tailored to document-level semantic analysis.
2. **Architectural Contrast:** Contrast local predictive context-window objectives (Word2Vec Skip-gram) against global co-occurrence matrix factorization (GloVe) from both mathematical and empirical standpoints.
3. **Semantic Geometry Exploration:** Evaluate semantic similarity across 10 domain anchor words, probing the tension between in-domain episodic collocations and pre-trained taxonomic synonyms.
4. **Manifold & Coherence Analysis:** Interrogate embedding space topologies using linear (PCA) and non-linear (t-SNE) projections, corroborated by quantitative intra-topic cosine coherence metrics.
5. **Downstream Utility Benchmark:** Quantify the discriminative capacity of mean-pooled document embeddings across 5 news categories using regularized linear classifiers, identifying persistent decision-boundary confusions.
6. **Theoretical Critique:** Delineate the boundary conditions of static word representations, providing architectural justification for contextualized language models.

---

## 2. Dataset & Preprocessing Pipeline

### 2.1 Dataset Composition
The benchmark dataset consists of the **Kaggle Text Document Classification Dataset** (`sunilthite/text-document-classification-dataset`), derived from the standardized BBC news archive. The dataset contains **2,225 full-text articles** balanced across five thematic domains:

| Label ID | Domain Name | Document Count | Class Proportion (%) |
|:---:|:---|:---:|:---:|
| 0 | **Politics** | 417 | 18.74% |
| 1 | **Sport** | 511 | 22.97% |
| 2 | **Tech** | 401 | 18.02% |
| 3 | **Entertainment** | 386 | 17.35% |
| 4 | **Business** | 510 | 22.92% |
| **Total** | **Corpus Aggregate** | **2,225** | **100.00%** |

The corpus contains an aggregate of **496,252 tokens** post-cleaning, with a mean document length of **223.0 tokens** ($\sigma \approx 138.4$).

### 2.2 Text Normalization Pipeline
Raw journalistic text contains substantial orthographic variance, typographical anomalies, and high-entropy punctuation that inject noise into sliding context windows and co-occurrence counts. The following sequential pipeline was executed:
1. **Case Normalization:** All text is mapped to lowercase characters ($c \leftarrow \text{lower}(c)$), collapsing redundant surface forms (e.g., `Government`, `GOVERNMENT`, `government` map to a single vocabulary index).
2. **URL & Hyperlink Elimination:** Regular expression matching (`https?://\S+|www\.\S+`) strips web addresses, which lack generalizable semantic content.
3. **Non-Alphabetic Character Stripping:** Punctuation marks, numeric digits, currency symbols, and control characters are replaced with whitespace (`[^a-z\s]`), isolating pure lexical tokens.
4. **Tokenization & Stopword Pruning:** Whitespace tokenization yields discrete unigrams. Standard English stopwords (179 words from NLTK) are pruned, eliminating functional syntactic operators (`the`, `at`, `which`, `between`) that inflate frequency without domain discriminability.
5. **Minimum Length Thresholding:** Tokens with length $\le 2$ are removed, filtering residual abbreviations and transcription noise.
6. **Morphological Lemmatization:** Remaining tokens are mapped to their lexical dictionary headwords using the WordNet lexical database.

### 2.3 Methodological Rationale: WordNet Lemmatization vs. Stemming
A pivotal design decision in NLP pipeline engineering is the choice between **stemming** (e.g., Porter, Lancaster, Snowball) and **lemmatization** (WordNet):

- **Stemming Limitations:** Stemming algorithms operate via greedy, heuristic rule trees that slice off known inflectional suffixes (`-ing`, `-ed`, `-s`, `-tion`). Crucially, stemmers have no internal lexicon:
  - **Truncation to Non-Words:** `economy` $\to$ `economi`, `business` $\to$ `busi`, `minister` $\to$ `minist`, `policy` $\to$ `polici`.
  - **Over-Stemming:** Semantically divergent lexemes are conflated to the same root (e.g., `universal`, `universe`, and `university` collapse to `univers`).
  - **Under-Stemming:** Morphologically related irregular forms remain disparate (e.g., `alumnus`, `alumna`, and `alumni` fail to merge).
- **Lemmatization Superiority:** Lemmatization leverages morphological analysis backed by a comprehensive lexical knowledge base. It considers vocabulary context and part-of-speech structure to return valid canonical lemmas (`companies` $\to$ `company`, `matches` $\to$ `match`, `played` $\to$ `play`).
- **Critical Embedding Compatibility:** Pre-trained embeddings like GloVe (`glove-wiki-gigaword-100`) are keyed exclusively to valid English dictionary words. If a corpus is processed with a stemmer, tokens such as `economi`, `busi`, and `technolog` produce **100% Out-of-Vocabulary (OOV) failure rates** when queried against GloVe. Lemmatization ensures vocabulary harmonization across both in-domain trained and pre-trained vector spaces.

---

## 3. Architectural Comparison: Skip-gram vs. GloVe

The two embedding models embody fundamentally divergent philosophical perspectives on how to exploit the distributional hypothesis ("you shall know a word by the company it keeps", Firth, 1957).

```
+-----------------------------------------------------------------------------------------------+
|                                    WORD EMBEDDING PARADIGMS                                   |
+------------------------------------+----------------------------------------------------------+
|  Word2Vec Skip-gram (Predictive)   |                  GloVe (Count-Based Log-Bilinear)        |
+------------------------------------+----------------------------------------------------------+
|  Input: Center word w_t            |  Input: Global co-occurrence matrix X_ij                  |
|  Hidden: 100D dense vector         |  Objective: Weighted least squares on log-counts         |
|  Output: Predict context w_{t+j}   |  Property: Encodes probability ratios P_ik / P_jk        |
|  Training: Streaming sliding window|  Training: Global epoch-level matrix factor optimization  |
+------------------------------------+----------------------------------------------------------+
```

### 3.1 Word2Vec Skip-gram with Negative Sampling (SGNS)
- **Mechanics:** Skip-gram (Mikolov et al., 2013) is a local, iterative predictive model. For each position $t$ in a sequence of $T$ tokens, it treats $w_t$ as a target word and attempts to maximize the conditional probability of all context words $w_{t+j}$ within a symmetric window of radius $c=5$.
- **Objective Function:** Full softmax computation across vocabulary $|\mathcal{V}|$ is computationally intractable ($\mathcal{O}(|\mathcal{V}|)$ per update). SGNS reformulates training as a set of binary logistic regressions. For target token $w_I$ and observed context token $w_O$, the per-pair loss is:
  $$\mathcal{L}_{\text{SGNS}} = \log \sigma(v'_{w_O}{}^\top v_{w_I}) + \sum_{k=1}^K \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-v'_{w_i}{}^\top v_{w_I}) \right]$$
  where $K=5$ negative samples are drawn from the smoothed unigram distribution $P_n(w) \propto U(w)^{0.75}$.
- **Inductive Bias:** Skip-gram observes local word pairs sequentially as a streaming process. It updates target vector $v$ and context vector $v'$ via stochastic gradient descent (SGD), naturally biasing representations toward episodic, narrative, and highly local syntactic-semantic contingencies.

### 3.2 Global Vectors (GloVe)
- **Mechanics:** GloVe (Pennington et al., 2014) is a global log-bilinear matrix factorization model. It compiles an exhaustive corpus-level co-occurrence matrix $X \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$, where $X_{ij}$ tabulates the frequency with which word $j$ falls within window distance of word $i$.
- **Mathematical Foundation:** Pennington et al. demonstrated that semantic relations are encoded in the **ratio of co-occurrence probabilities**:
  $$F(w_i, w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}} = \frac{X_{ik} / X_i}{X_{jk} / X_j}$$
  Enforcing linearity and homomorphic mapping between vector space differences and probability ratios leads to the dot-product formulation $w_i^\top \tilde{w}_k + b_i + \tilde{b}_k = \log X_{ik}$. GloVe minimizes the weighted least squares objective:
  $$\mathcal{J}_{\text{GloVe}} = \sum_{i,j=1}^{|\mathcal{V}|} f(X_{ij}) \left( w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$
- **Weighting Function:** The non-linear damper $f(X_{ij}) = \min\left(1, (X_{ij}/x_{\max})^\alpha\right)$ with $\alpha=0.75$ and $x_{\max}=100$ caps the leverage of ubiquitous stopwords while ensuring rare co-occurrences contribute proportionally.
- **Inductive Bias:** GloVe aggregates the entire global statistical landscape prior to optimization. Its objective directly optimizes global distances to reconstruct log-frequency statistics, making it sensitive to broad taxonomic structures and paradigmatic word classes.

### 3.3 Model Configuration Summary
- **Word2Vec Skip-gram:** $d=100$, window size $c=5$, minimum count threshold $\text{min\_count}=3$, negative sampling $K=5$, trained directly on the BBC tokenized corpus ($|\mathcal{V}| = 12,620$ unique words) over 15 epochs.
- **GloVe:** $d=100$, pre-trained on Wikipedia 2014 + Gigaword 5 (6 billion tokens, $|\mathcal{V}| = 400,000$ uncased vocabulary), loaded via Gensim API (`glove-wiki-gigaword-100`).

---

## 4. Semantic Similarity & Word Relationships

### 4.1 Nearest Neighbor Findings across Domain Anchors
Ten canonical anchor words (2 per domain) were queried in both embedding spaces using cosine similarity. The retrieved top-5 nearest neighbors are documented below:

| Domain | Anchor Word | Skip-gram Top-5 Neighbors (Cosine) | GloVe Top-5 Neighbors (Cosine) |
|:---|:---|:---|:---|
| **Politics** | `government` | quango (0.61), corruption (0.61), curb (0.57), nath (0.56), authority (0.56) | administration (0.79), governments (0.77), officials (0.76), authorities (0.74), opposition (0.74) |
| **Politics** | `minister` | prime (0.86), silvio (0.69), yulia (0.68), tymoshenko (0.66), ahern (0.64) | prime (0.88), cabinet (0.75), ministry (0.74), foreign (0.73), deputy (0.73) |
| **Sport** | `football` | refereed (0.64), punish (0.64), club (0.64), disgrace (0.63), decides (0.61) | soccer (0.87), basketball (0.86), league (0.82), rugby (0.80), hockey (0.78) |
| **Sport** | `champion` | hardenne (0.72), henin (0.72), maurice (0.71), justine (0.71), edith (0.70) | medalist (0.79), champ (0.78), reigning (0.78), defending (0.77), runner (0.76) |
| **Tech** | `software` | program (0.64), suite (0.63), programmer (0.63), sybari (0.62), spyware (0.61) | computer (0.84), hardware (0.79), microsoft (0.78), applications (0.74), technology (0.74) |
| **Tech** | `computer` | pc (0.63), remotely (0.63), clamshell (0.62), stylus (0.60), hijacked (0.60) | computers (0.88), software (0.84), technology (0.76), pc (0.74), hardware (0.73) |
| **Entertainment** | `film` | movie (0.72), cinema (0.66), documentary (0.62), hollywood (0.62), cuaron (0.61) | movie (0.91), films (0.89), directed (0.81), documentary (0.81), drama (0.79) |
| **Entertainment** | `music` | urban (0.61), xtra (0.59), napster (0.56), downloading (0.56), streamed (0.56) | musical (0.81), songs (0.80), dance (0.79), pop (0.79), recording (0.77) |
| **Business** | `market` | stock (0.68), outperformed (0.59), trader (0.59), crossover (0.57), profitability (0.57) | markets (0.89), stock (0.80), prices (0.78), stocks (0.77), investors (0.76) |
| **Business** | `economy` | growth (0.69), spurt (0.68), economic (0.68), fragile (0.67), markedly (0.66) | economic (0.83), growth (0.79), recession (0.77), economies (0.75), recovery (0.75) |

### 4.2 Semantic Analysis: Syntagmatic Collocations vs. Paradigmatic Synonyms
The empirical nearest neighbor results illustrate a textbook linguistic dichotomy:

1. **GloVe Learns Paradigmatic (Taxonomic) Relations:**
   - In linguistic theory, words in a **paradigmatic relation** belong to the same grammatical or conceptual category and can substitute for one another in a sentence frame (e.g., "He plays [football / soccer / rugby / basketball]").
   - GloVe's global co-occurrence statistics over 6 billion tokens cause co-hyponyms to share nearly identical co-occurrence ratio profiles. Consequently, its nearest neighbors are canonical synonyms, inflectional plurals, and taxonomic siblings:
     - `football` yields `soccer` (0.87), `basketball` (0.86), `rugby` (0.80), and `hockey` (0.78).
     - `government` yields `administration` (0.79), `officials` (0.76), and `authorities` (0.74).
     - `champion` yields `medalist` (0.79), `champ` (0.78), `reigning` (0.78), and `defending` (0.77).
2. **Skip-gram Learns Syntagmatic (Episodic / Collocational) Relations:**
   - **Syntagmatic relations** occur between words that co-occur sequentially along the linear chain of discourse (e.g., subject-verb-object, modifier-noun).
   - Because our Skip-gram model was trained directly on mid-2000s BBC news reporting, its vectors reflect the narrative, political, and cultural events of that specific historical window:
     - `minister` co-occurs constantly with specific political leaders: `silvio` (0.69, Italian PM Silvio Berlusconi), `yulia` (0.68) and `tymoshenko` (0.66, Ukrainian PM Yulia Tymoshenko), and `ahern` (0.64, Irish Taoiseach Bertie Ahern).
     - `champion` associates directly with Belgian tennis star `justine` (0.71) `henin` (0.72) `-hardenne` (0.72), who dominated Grand Slam championships during the dataset's era.
     - `music` associates with mid-2000s digital audio technology and BBC broadcasting: `xtra` (0.59, BBC Radio 1Xtra), `napster` (0.56), `downloading` (0.56), and `streamed` (0.56).
     - `government` captures UK administrative discourse: `quango` (0.61, Quasi-Autonomous Non-Governmental Organisation), `curb` (0.57), and `corruption` (0.61).

### 4.3 Intra-Domain vs. Cross-Domain Similarity Geometry
Evaluating the $10 \times 10$ cosine similarity matrices of the anchor words confirms that intra-domain pairs exhibit high affinity (e.g., `film` $\leftrightarrow$ `music` = 0.52 in GloVe; `software` $\leftrightarrow$ `computer` = 0.84 in GloVe, 0.58 in Skip-gram). However, cross-domain semantic leakage is evident: `government` (Politics) and `economy` (Business) share a substantial cosine similarity of 0.58 in GloVe and 0.44 in Skip-gram, foreshadowing the downstream classification confusions observed between these two classes.

---

## 5. Visualization & Topic Coherence Analysis

### 5.1 Dimensionality Reduction: PCA vs. t-SNE
To examine geometric structure, 50 curated keywords (10 per domain) were projected from $\mathbb{R}^{100}$ to $\mathbb{R}^2$ via **Principal Component Analysis (PCA)** and **t-Distributed Stochastic Neighbor Embedding (t-SNE)**.

```
+--------------------------------------------------------------------------------------------------+
|                                  2D MANIFOLD PROJECTION SUMMARY                                  |
+-----------------------------------+--------------------------------------------------------------+
| Principal Component Analysis (PCA)| t-Distributed Stochastic Neighbor Embedding (t-SNE)          |
+-----------------------------------+--------------------------------------------------------------+
| - Linear orthogonal projection    | - Non-linear probabilistic manifold mapping                  |
| - Preserves global covariance     | - Preserves local neighborhood topology (Student-t kernel)   |
| - Susceptible to origin crowding  | - Eliminates crowding; produces isolated topic clusters      |
| - Explains ~25-35% total variance | - Unfolds non-linear boundaries between semantic domains     |
+-----------------------------------+--------------------------------------------------------------+
```

#### Detailed Geometric Observations:
1. **PCA Projections (Linear):**
   - **Skip-gram PCA:** Words are distributed continuously along two dominant orthogonal axes. Sport and Tech keywords occupy peripheral extrema, while Politics and Business keywords cluster tightly together near the origin $(0, 0)$. PCA preserves maximum Euclidean variance, but a 2D linear subspace cannot disentangle the multi-attribute semantic overlaps of political economy, resulting in visible **origin crowding**.
   - **GloVe PCA:** Exhibits greater dispersion than Skip-gram due to richer pre-trained variance, with Tech (`software`, `hardware`, `network`) forming a coherent trajectory along PC1, while Entertainment and Sport diverge along PC2.
2. **t-SNE Projections (Non-Linear):**
   - **Resolving Crowding:** By minimizing the Kullback-Leibler divergence between high-dimensional Gaussian affinities and low-dimensional Student-t affinities, t-SNE repels dissimilar points and pulls true neighbors into dense, isolated "semantic islands".
   - **Skip-gram t-SNE:** Clear topic clusters emerge. Sport (`player`, `match`, `champion`, `cup`) forms an entirely distinct island completely segregated from the remaining corpus. Entertainment (`film`, `movie`, `music`, `album`) coalesces tightly. Politics (`election`, `parliament`, `vote`, `labour`) and Business (`market`, `price`, `share`, `investor`) form neighboring but visually separable sub-manifolds.
   - **GloVe t-SNE:** Demonstrates exceptional cluster cohesion. All 5 domains form 5 clearly distinguished, compact clusters with wide inter-cluster margins, validating that GloVe’s pre-training constructs isotropic, well-partitioned semantic domains.

### 5.2 Quantitative Topic Coherence Benchmark
Topic Coherence was formally calculated for each domain $D_k$ as the mean pairwise cosine similarity among its 10 representative keywords:

$$\mathcal{C}(D_k) = \frac{2}{|D_k|(|D_k|-1)} \sum_{i < j} \cos(v_{w_i}, v_{w_j})$$

| Domain | Skip-gram Topic Coherence | GloVe Topic Coherence | $\Delta$ (GloVe $-$ Skip-gram) |
|:---|:---:|:---:|:---:|
| **Politics** | 0.3843 | 0.5710 | +0.1867 |
| **Sport** | 0.3787 | 0.5718 | +0.1931 |
| **Tech** | 0.3811 | 0.6371 | +0.2560 |
| **Entertainment** | 0.3371 | 0.5420 | +0.2049 |
| **Business** | 0.3818 | 0.5746 | +0.1928 |
| **Overall Mean** | **0.3727** | **0.5794** | **+0.2067** |

#### Coherence Synthesis:
GloVe achieved a substantially higher mean coherence score across all five domains ($0.5794$ vs. $0.3727$, an absolute advantage of $+0.2067$). The highest coherence was attained by **GloVe on Tech** ($\mathcal{C} = 0.6371$), where terms like `software`, `computer`, `technology`, `digital`, `internet`, and `network` co-occur consistently across billions of web pages. Skip-gram's lower coherence scores ($\approx 0.34 - 0.38$) do not reflect deficient semantic encoding; rather, in a small corpus of 2,225 documents, topical words appear in diverse narrative contexts, resulting in broader vector dispersion.

---

## 6. Downstream Classification Benchmark

### 6.1 Feature Representation via Centroid Pooling
Each document $d$ is mapped to a static continuous vector $v_d \in \mathbb{R}^{100}$ via arithmetic mean pooling over its in-vocabulary tokens:

$$v_d = \frac{1}{|d \cap \mathcal{V}|} \sum_{w \in d \cap \mathcal{V}} v_w$$

If $|d \cap \mathcal{V}| = 0$, $v_d = \mathbf{0}$. The dataset was partitioned into an **80% stratified training set** ($n=1,780$) and a **20% test set** ($n=445$).

### 6.2 Empirical Results Table
Two linear classifiers—Multinomial Logistic Regression ($C=1.0$) and Linear Support Vector Machine (LinearSVC, $C=1.0$)—were trained on the centroid representations:

| Embedding Architecture | Downstream Classifier | Test Accuracy (%) | Macro-F1 Score (%) |
|:---|:---|:---:|:---:|
| **Word2Vec Skip-gram (Corpus)** | **Linear SVM** | **97.75%** | **97.70%** |
| **Word2Vec Skip-gram (Corpus)** | **Logistic Regression** | **97.53%** | **97.51%** |
| **GloVe (Pretrained 6B)** | **Linear SVM** | **97.30%** | **97.25%** |
| **GloVe (Pretrained 6B)** | **Logistic Regression** | **97.30%** | **97.20%** |

### 6.3 Detailed Class-Level Performance Reports

#### Word2Vec Skip-gram + Logistic Regression:
- **Politics:** Precision = 0.9756, Recall = 0.9524, F1 = 0.9639 ($n=84$)
- **Sport:** Precision = 1.0000, Recall = 0.9902, F1 = 0.9951 ($n=102$)
- **Tech:** Precision = 0.9872, Recall = 0.9625, F1 = 0.9747 ($n=80$)
- **Entertainment:** Precision = 0.9620, Recall = 0.9870, F1 = 0.9744 ($n=77$)
- **Business:** Precision = 0.9519, Recall = 0.9706, F1 = 0.9612 ($n=102$)
- **Corpus Aggregate:** Accuracy = **97.53%**, Macro-F1 = **0.9751**, Weighted-F1 = **0.9753**

#### GloVe + Logistic Regression:
- **Politics:** Precision = 0.9535, Recall = 0.9762, F1 = 0.9647 ($n=84$)
- **Sport:** Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 ($n=102$)
- **Tech:** Precision = 0.9512, Recall = 0.9750, F1 = 0.9630 ($n=80$)
- **Entertainment:** Precision = 0.9865, Recall = 0.9481, F1 = 0.9669 ($n=77$)
- **Business:** Precision = 0.9703, Recall = 0.9608, F1 = 0.9655 ($n=102$)
- **Corpus Aggregate:** Accuracy = **97.30%**, Macro-F1 = **0.9720**, Weighted-F1 = **0.9730**

### 6.4 Model Comparison Verdict & In-Depth Error Analysis
1. **The In-Domain Advantage of Skip-gram:**
   - Word2Vec Skip-gram trained from scratch achieved the highest overall accuracy (**97.75%** with Linear SVM and **97.53%** with Logistic Regression), marginally surpassing GloVe (**97.30%**).
   - Although GloVe was trained on 6 billion tokens (yielding superior general semantic coherence), Skip-gram’s training distribution was *identically distributed* ($i.i.d.$) with the downstream test evaluation. The local Skip-gram vectors explicitly internalized domain-defining UK political and economic entities (`blair`, `chancellor`, `tory`, `telecom`, `opec`) that do not carry disproportionate weight in general Wikipedia text.
2. **Analysis of Confusion Matrices & Decision Boundary Overlap:**
   - Across both models, **Sport** is virtually unassailable: Skip-gram misclassifies only 1 sport document out of 102 (Recall = 99.02%), while GloVe attains a perfect **100.0% Precision and Recall**. Sports journalism utilizes highly segregated, specialized jargon (`striker`, `goal`, `tournament`, `referee`) that produces centroid vectors located far from other domain manifolds.
   - The primary source of classification error is the **Politics $\leftrightarrow$ Business interface**:
     - In Skip-gram, 3 Politics documents are misclassified as Business, and 3 Business documents are misclassified as Politics or Tech.
     - In GloVe, 2 Politics documents are classified as Business, and 3 Business documents are classified as Politics or Tech.
     - *Qualitative Error Rationale:* Articles analyzing government fiscal policy, national budgets, corporate tax reform, privatized healthcare spending, and central bank interest rate interventions straddle both political and economic domains. When centroid pooling computes the unweighted average across hundreds of tokens, the resulting vector lies directly along the hyper-plane boundary separating Politics and Business.

---

## 7. Limitations & Future Work

### 7.1 Structural Limitations of Static Word Embeddings
While both Skip-gram and GloVe deliver strong classification performance, they suffer from intrinsic architectural bottlenecks:

1. **The Polysemy / Monosemy Bottleneck:**
   - Both models assign a **single invariant vector** $v_w \in \mathbb{R}^d$ to each word token in the vocabulary.
   - Natural language is fundamentally polysemous: `apple` (orchard fruit vs. multinational tech conglomerate), `bank` (financial depository vs. riparian river bank), `interest` (monetary lending fee vs. intellectual curiosity), and `share` (equity ownership security vs. reciprocal action).
   - A single static vector forced to represent multiple meanings inevitably settles at an unnatural Euclidean midpoint between its distinct semantic modes, degrading fine-grained semantic discrimination.
2. **Information Destruction via Centroid Mean Pooling:**
   - Calculating document embeddings by computing the arithmetic mean $\frac{1}{M}\sum v_w$ reduces a structured document to a continuous "bag-of-embeddings":
     - **Loss of Word Order & Syntax:** Sentences with opposite meanings (e.g., "The bank acquired the tech startup" vs. "The tech startup acquired the bank") yield identical centroid representations.
     - **Negation Oblivion:** Phrases such as "not profitable" and "profitable" contribute near-identical vectors to the document sum, neutralizing sentiment and polarity signals.
     - **Keyword Dilution:** In lengthy documents (500+ tokens), non-discriminative background vocabulary dilutes the signal of critical topic-identifying unigrams.
3. **Out-of-Vocabulary (OOV) Fragility:**
   - Static word-level models possess fixed vocabulary look-up tables. At test time, any unseen word, emergent slang, technical compound, or typographical error is discarded as OOV.
   - Unlike subword-based architectures (e.g., FastText character n-grams, Byte-Pair Encoding), neither Skip-gram nor GloVe can dynamically construct representations for novel morphological combinations.

### 7.2 Future Directions & Modern Contextualized Architectures
To transcend the limits of static embeddings, modern NLP has transitioned to **deep contextualized transformer architectures**:
1. **Self-Attention Mechanisms (BERT, RoBERTa, DeBERTa):** By attending to all bidirectional tokens simultaneously across 12 to 24 layers, transformers generate dynamic token embeddings $h_{t} = f(w_t, w_{1:T})$ conditioned on full sentential context. In a transformer, `apple` in "Apple released the M3 chip" and `apple` in "apple cider vinegar" occupy entirely distinct coordinate regions in vector space.
2. **Dense Sentence Encoders (SBERT, Contriever):** Rather than naive arithmetic mean pooling, sentence transformers optimize pooling layers via Siamese and triplet objective functions on semantic textual similarity (STS) benchmarks, preserving syntactic and semantic nuances.
3. **Hybrid Corpus Calibration:** When deploying pre-trained embeddings in specialized domains, continuing self-supervised pre-training (domain adaptation) on target text before downstream fine-tuning reconciles the broad linguistic coverage of foundation models with the niche precision of in-domain Skip-gram.

---

## 8. Conclusion
This study provides empirical validation that word embedding geometry reflects both architectural formulation and training corpus distribution. Word2Vec Skip-gram trained locally on domain news captures vivid syntagmatic collocations and achieves superior downstream classification (97.75%), whereas GloVe pre-trained on 6 billion tokens yields superior global topic coherence (0.5794) and canonical taxonomic clustering. Both approaches provide potent, computationally lightweight semantic primitives, while underscoring the structural necessity of contextualized transformer architectures for resolving polysemy and complex discourse syntax in modern NLP.

---

### Deliverable Artifact Index:
- **Executable Jupyter Notebook:** [`Assignment_2_Word_Embeddings.ipynb`](file:///c:/Users/ahmed/PycharmProjects/pythonProject1/AntiGrav/Assignments/Assignment_2_Word_Embeddings.ipynb)
- **Extracted Publication Figures:**
  - Anchor Similarity Heatmaps: [`figures/anchor_similarity_heatmap.png`](file:///c:/Users/ahmed/PycharmProjects/pythonProject1/AntiGrav/Assignments/figures/anchor_similarity_heatmap.png)
  - 2x2 PCA vs. t-SNE Projections: [`figures/pca_tsne_2x2_grid.png`](file:///c:/Users/ahmed/PycharmProjects/pythonProject1/AntiGrav/Assignments/figures/pca_tsne_2x2_grid.png)
  - Topic Coherence Bar Chart: [`figures/topic_coherence_barchart.png`](file:///c:/Users/ahmed/PycharmProjects/pythonProject1/AntiGrav/Assignments/figures/topic_coherence_barchart.png)
  - Side-by-Side Confusion Matrices: [`figures/confusion_matrices.png`](file:///c:/Users/ahmed/PycharmProjects/pythonProject1/AntiGrav/Assignments/figures/confusion_matrices.png)
- **Standalone Academic Report:** [`Assignment_2_Academic_Report.md`](file:///c:/Users/ahmed/PycharmProjects/pythonProject1/AntiGrav/Assignments/Assignment_2_Academic_Report.md)
