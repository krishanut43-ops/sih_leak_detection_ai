# AI-Enabled Leak Detection for Skilled Trades (SIH)

Acoustic leak detection model for a low-cost trainee wearable.
Classifies 1-second audio windows as: no_leak, leak, background_noise.

## Models
- Random Forest on MFCC features (baseline)
- Small CNN on log-mel spectrograms

## Run
pip install numpy scipy scikit-learn librosa matplotlib torch
python leak_detection_ai.py

Uses synthetic data by default. For real recordings:
python leak_detection_ai.py --data_dir dataset

## Status
Proof of concept on simulated acoustic data. Field validation with a
real sensor prototype is planned.
