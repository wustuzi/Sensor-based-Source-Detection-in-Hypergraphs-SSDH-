# SSDH: Sensor-based Source Detection in Hypergraphs

[cite_start]This repository contains the reference implementation and datasets for the paper **"Entropy-Driven Sensor Deployment and Source Detection in Hypergraphs"**, submitted to *Information Processing & Management (IPM)*[cite: 1].

The project implements the **SSDH framework**, which includes:
1.  [cite_start]An **Entropy-based Sensor Deployment Strategy** to maximize information gain[cite: 9].
2.  [cite_start]A **Source Detection Algorithm** that combines topological distance with a novel path uncertainty-based score[cite: 10].

## 📂 Repository Structure

The repository is organized as follows:

```text
.
├── Algorithm/
│   ├── SSDH.py            # Main source code for sensor deployment and source detection
│   └── requirements.txt   # Python dependencies
├── Data/                  # Contains 12 hypergraph datasets (6 synthetic, 6 empirical)
│   ├── Algebra.txt
│   ├── Bars-Rev.txt
│   ├── Geometry.txt
│   └── ... (other datasets)
└── README.md
