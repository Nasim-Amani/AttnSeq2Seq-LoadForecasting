# AttnSeq2Seq-LoadForecasting

Sequence-to-sequence (encoder–decoder LSTM) models with three attention mechanisms for **electrical load forecasting**.

## Overview
This project compares how different attention mechanisms affect load forecasting:

| File | Description |
|------|-------------|
| `additive_attention.py` | Seq2Seq + additive (Bahdanau) attention |
| `global_attention.py` | Seq2Seq + global (Luong) attention |
| `multihead_attention.py` | Seq2Seq + multi-head attention |
| `hyperparameter_tuning.py` | Hyperparameter search for the models |
