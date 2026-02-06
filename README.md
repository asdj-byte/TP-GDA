# TP-GDA

This repository contains the official implementation of the paper **"A Topology-Driven Domain Adaptation Framework with Physical Consistency for Cross-Scale Microarchitecture Power Modeling".

## Abstract

Accurate architecture-level power modeling is a critical enabler for early Design Space Exploration (DSE). However, existing learning-based approaches often exhibit fragile behavior when extrapolating across architectural scales. This paper proposes **TP-GDA**, a physics-consistent, topology-driven domain adaptation framework designed to resolve complex dynamic coupling and cross-scale transfer issues. TP-GDA reconstructs flat performance counters into a **Component Dependency Graph (CDG)** to capture hardware signal flows and dynamic cascading effects. By integrating a **Scale-Denoising Adversarial mechanism** with **Microarchitectural Scaling Laws**, the framework distills scale-invariant features and ensures predictions strictly adhere to physical plausibility. Extensive evaluations on the open-source **ArchPower** dataset demonstrate that TP-GDA reduces prediction errors by **17.2% to 30.7%** compared to state-of-the-art methods.

## Requirements

* Python >= 3.8
* PyTorch >= 2.0 


* XGBoost >= 1.7.1 


* scikit-learn >= 1.2.2 


* NumPy, Matplotlib

## Framework Overview

TP-GDA consists of three coupled physics-aware stages:

1. **Physics-Aware Graph Construction**: Reconstructs unstructured signals into a directed attributed graph  and injects **Physical Identity Embeddings** to resolve feature ambiguity among heterogeneous components.


2. **Topology-Driven Domain-Invariant Learning**: Employs a dual-layer **Graph Attention Network (GAT)** and a **Domain-Adversarial Neural Network (DANN)** to align distributions between source and target domains.


3. **Hybrid Enhanced Inference**: Combines a graph-tree hybrid architecture with a **Microarchitectural Scaling Law Decoder** to balance non-linear fitting with physical consistency.



## Dataset Selection

The framework is validated using the **ArchPower** dataset, the community's only open-source microarchitectural power dataset generated via commercial-grade EDA flows.

* **Architectures**: Covers **BOOM** (out-of-order) and **XiangShan** (deep out-of-order) RISC-V processors.


* **Scale**: Includes 25 diverse architectural configurations across 8 real-world workloads.


* **Heterogeneity**: Features 14-dimensional hardware parameters and 87-dimensional performance event counters.



## Data Split Scenarios

To evaluate generalization, we employ three challenging scenarios:

* **Balanced Scenario**: Training and testing sets cover all configurations (interpolation).


* **Small-to-Large Extrapolation**: Trained exclusively on small-scale cores and tested on unseen large-scale configurations to simulate early-phase DSE.


* **Large-to-Small Extrapolation**: Trained on large-scale configurations to predict small-scale performance.



## Model Training

The training process is divided into two synergistic stages:

1. **Pre-training**: Joint optimization of the topology encoder and the domain adversarial network for 1,200 epochs using the Adam optimizer.


2. **Ensemble Inference**: Freezing the encoder to generate node embeddings for the **XGBoost** regressor (200 estimators, max depth 5).



### Usage

Run the following commands to execute the ensemble training and evaluation:

**BOOM Architecture:**

```bash
# Balanced scenario
python TP-GDA.py --uarch BOOM --scenario evenly

# Small-to-Large extrapolation
python TP-GDA.py --uarch BOOM --scenario small

```

**XiangShan Architecture:**

```bash
# Small-to-Large extrapolation (Critical Benchmark)
python TP-GDA.py --uarch XS --scenario small

```

## Model Inference

The inference engine fuses deep topological semantics with numerical robustness:

1. Extracts contextual node embeddings via the trained GAT.
2. Reduces dimensionality using PCA and concatenates with raw features.
3. Executes prediction in logarithmic space to enhance sensitivity to low-power fluctuations.
4. Applies the scaling function  to ensure physical self-consistency.
