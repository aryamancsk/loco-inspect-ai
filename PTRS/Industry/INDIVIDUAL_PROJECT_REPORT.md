# Individual Project Report

| Field | Details |
| :--- | :--- |
| **Your Name:** | Aryaman |
| **Certificate:** | Graduate Certificate in Pattern Recognition Systems |
| **Project Title:** | LOCO-Inspect AI: Multi-Category Industrial Visual Anomaly Detection and Defect Localization System |
| **Project Type:** | Solo Project (100% Individual Effort) |
| **Date:** | October 2026 |

---

## 1. Your Personal Contribution to the Project

As the sole researcher and developer for this project, I completed the entire project lifecycle independently from problem formulation to production deployment. My key contributions across the 10-day practice module effort include:

1. **Problem Scoping & Formulation**:
   - Identified the industrial automated optical inspection (AOI) challenge using the MVTec Logical Constraints Anomaly Detection (**MVTec LOCO AD**) benchmark.
   - Designed the unsupervised formulation, ensuring the system learns exclusively from defect-free training images (D_train) without peeking at defective examples, reflecting realistic manufacturing constraints where anomalies are rare and diverse.

2. **Core Pattern Recognition & Deep Representation Pipeline**:
   - Implemented multi-scale patch feature extraction using a pre-trained **WideResNet-50-2** convolutional backbone, tapping into intermediate activations (layer2 and layer3) to capture fine texture and spatial structural semantics (1,024 patch vectors of dimension 1,536 per image).
   - Solved memory and computational bottlenecks by implementing **Johnson-Lindenstrauss Sparse Random Projections** combined with a **Greedy Minimax Facility Location Coreset Subsampling** algorithm, compressing over 343,000 normal patch embeddings into a representative 10% memory bank (34,304 patches) with zero accuracy degradation.
   - Built a high-throughput GPU-accelerated nearest-neighbor search engine (TorchNN) using chunked PyTorch tensor operations, reducing per-image inference latency to ~85 ms on an NVIDIA RTX 5070 Laptop GPU.

3. **Comparative Machine Learning & Statistical Modeling**:
   - Implemented and evaluated a **Baseline Model** using ResNet-18 with the full uncompressed patch pool (343k patches, kNN).
   - Designed and fitted an explicit parametric statistical model: a **Per-Location Diagonal Gaussian Anomaly Detector (PaDiM)** via Maximum Likelihood Estimation (MLE), computing localized Mahalanobis distance scores to capture coordinate-dependent assembly rules.
   - Calibrated dynamic decision thresholds exclusively on defect-free validation sets (μ_val + k·σ_val for k in {1, 2, 3, 4} and optimal F1).

4. **Multi-Category Empirical Evaluation**:
   - Validated the system across all **5 MVTec LOCO AD categories** (Juice Bottle, Breakfast Box, Pushpins Box, Hardware Screw Bag, and Splicing Connectors), benchmarking Image ROC-AUC, Precision, Recall, F1-score, and Pixel-level Localization AUROC (0.872).
   - Performed rigorous error analysis revealing the "Logical Anomaly Gap" (why pure patch distance metrics struggle with missing components vs. structural damage) and engineered spatial conditioning to resolve it.

5. **Full-Stack Software Engineering & Deployment (MVP)**:
   - Developed **LOCO-Inspect AI**, a full-stack web inspection application consisting of over 3,800 source lines of code:
     - **Backend**: FastAPI REST service with asynchronous endpoints (/api/inspect, /api/auth/*, /api/meta, /api/history).
     - **Authentication**: SQLite database (inspector.db) with cryptographic PBKDF2-HMAC-SHA256 password hashing, dual identifier resolution (Username or Email), session cookies, and persistent audit logging.
     - **Intelligent Category Inference**: Automatic product category detection based on incoming aspect ratios with manual override.
     - **Frontend**: Responsive single-page application (HTML5/CSS3/JavaScript) featuring 4-panel visual diagnostic explainability (Original RGB, Pinpointed Bounding Box, Jet Heatmap, Blended Overlay) and an interactive sample gallery.
   - Resolved environment-level runtime challenges, including Windows Smart App Control DLL blocking, Torchvision meta-registration safeguards, and headless PDF compilation.

---

## 2. What You Have Learnt from the Project

This practice module synthesized the theoretical concepts from the Graduate Certificate into deep practical competencies:

1. **Unsupervised Anomaly Detection vs. Supervised Classification**:
   - In textbook classification, balanced labeled datasets are assumed. In manufacturing defect detection, anomalies represent an open set with extreme class imbalance. I mastered constructing non-parametric density and distance estimators in pre-trained embedding spaces rather than retraining softmax heads.

2. **The Structural vs. Logical Anomaly Disparity**:
   - The most critical insight was discovering why patch-based distance methods perform significantly better on **structural anomalies** (mean 53.0% detection at conservative threshold) than **logical anomalies** (mean 21.8%).
   - I learned that permutation-invariant nearest-neighbor search compares a patch against the entire memory bank regardless of spatial coordinates. When an object is missing a sub-component (e.g., missing fruit label or missing nut), the bare background patch matches normal background elsewhere on the product, yielding deceptively low anomaly scores. I learned how incorporating coordinate-dependent spatial distributions (like PaDiM) or spatial layout heuristics is necessary to close this gap.

3. **Coreset Compression Theory**:
   - I gained hands-on expertise with minimax facility location algorithms, observing how randomized projections reduce 1,536-dimensional feature vectors while preserving pairwise Euclidean metric geometry, allowing a 90% reduction in memory footprint with negligible loss in ROC-AUC (0.896 vs. 0.881).

4. **Industrial Quality-Control Trade-offs**:
   - I learned that an aggregate ROC-AUC score (0.896) can obscure practical failure modes. Operating at a strict conservative threshold (μ + 3σ) yielded exceptional precision (99.1%) but low recall (46.6%). Formulating multiple industrial operating points (μ + 1σ High Sensitivity vs. Optimal F1) taught me how to align machine learning hyper-parameters with business risk appetite.

5. **Production AI System Engineering**:
   - Bridging research code into an interactive production tool taught me defensive engineering: handling diverse image encodings, implementing robust password hashing without incompatible external dependencies on Python 3.14, and architecting thread-safe inference locks to prevent hook state corruption.

---

## 3. How You Can Apply This in Future Work-Related Projects

The skills and architecture developed in this project directly transfer to high-value industrial and enterprise applications:

1. **Automated Optical Inspection (AOI) in Electronics & Semiconductor Fabrication**:
   - The multi-scale patch memory bank approach can be directly deployed to inspect surface-mount device (SMD) printed circuit boards (PCBs) and silicon wafer dies. The system can learn nominal component placement from reference boards and automatically flag missing capacitors, misaligned chips, solder bridges, or surface contamination.

2. **Pharmaceutical Packaging & High-Speed Bottling Lines**:
   - In pharmaceutical blister pack and vial packaging, missing pills, broken tamper seals, incorrect label placement, or improper fill levels present critical compliance hazards. The multi-threshold mechanism allows line operators to enforce zero-defect escape tolerances (High Sensitivity μ + 1σ) on safety-critical batches while using balanced thresholds on secondary packaging.

3. **General Out-of-Distribution (OOD) & Sensor Failure Detection**:
   - Beyond image arrays, the combination of feature extraction, coreset subsampling, and Mahalanobis distance scoring applies directly to multimodal time-series telemetry (e.g., vibration, acoustic, and thermal sensor streams from turbines or industrial pumps) to predict mechanical bearing wear and unexpected machine anomalies before catastrophic breakdown.

4. **Explainable AI for Quality Operators**:
   - In manufacturing environments, machine operators will reject black-box predictions. The 4-view diagnostic interface developed here (Input, Bounding Box, Heatmap, and Defect Overlay) provides an intuitive template for industrial HMI (Human-Machine Interface) dashboards, providing clear visual evidence to operators for immediate physical verification.
