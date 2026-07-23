# GEO Satellite Maneuver Detection and Prediction

MSc thesis project developed at Cranfield University in industrial collaboration with INDRA UK.

This repository presents a framework for detecting and predicting maneuvers of geosynchronous satellites using Two-Line Element (TLE) data. The work combines orbital dynamics, statistical anomaly detection, uncertainty modelling and machine learning for Space Situational Awareness applications.

## Main Contributions

* TLE parsing and orbital-element preprocessing
* SGP4-based orbit propagation
* RTN residual and orbital-feature generation
* Covariance propagation and uncertainty calibration
* Statistical and unsupervised maneuver detection
* Supervised temporal classification using BiLSTM networks
* Maneuver-event grouping and validation
* Analysis of GEO station-keeping behaviour
* Initial investigation of maneuver prediction methods

## Methods

The project investigates and compares several approaches:

* Adaptive statistical thresholds
* Local Outlier Factor
* Isolation Forest
* Hidden Markov Models
* Boosted classification models
* Bidirectional Long Short-Term Memory networks

## Technologies

* MATLAB
* Python
* SGP4
* Machine learning
* Time-series analysis
* Orbital dynamics
* Space Situational Awareness

## Repository Structure

```text
data/             Example or synthetic input data
preprocessing/    TLE parsing and data preparation
features/         Orbital and temporal feature generation
detection/        Statistical and machine-learning detectors
prediction/       Maneuver prediction models
validation/       Event matching and performance evaluation
figures/          Selected results and visualisations
```

## Data and Confidentiality

Operational datasets, proprietary information and industrially sensitive material are not included. Public examples use open, anonymised or synthetic data.

## Status

This repository accompanies an MSc thesis currently under development. Code and documentation will be added progressively as results are reviewed and prepared for public release.

## Author

Pau Costa Aura
MSc Astronautics and Space Engineering, Cranfield University
