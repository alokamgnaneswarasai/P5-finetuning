# Recommendation Systems using Large Language Models (LLM)

## Table of Contents
- [Project Overview](#project-overview)
- [Models and Architectures](#models-and-architectures)
  - [P5 Recommendation Model](#p5-recommendation-model)
  - [Transformer-Decoder Architecture](#transformer-decoder-architecture)
  - [Recurrent Neural Network (RNN) Models](#recurrent-neural-network-rnn-models)
- [Datasets](#datasets)


## Project Overview

This project focuses on implementing recommendation systems using large language models (LLMs). Specifically, we employ the P5 recommendation model for rating prediction and sequential recommendation tasks. Additionally, we explore the transformer-decoder architecture, including fine-tuning with GPT-2 and traditional RNN models.

## Models and Architectures

### P5 Recommendation Model
The P5 model is utilized for two primary tasks:
1. **Rating Prediction**: Predicting user ratings for items.
2. **Sequential Recommendation**: Recommending the next item in a user's sequence of interactions.

### Transformer-Decoder Architecture
We experiment with the transformer-decoder architecture, fine-tuning it using GPT-2 for enhanced sequential recommendation performance. This architecture helps in capturing long-term dependencies in user interaction sequences.

### Recurrent Neural Network (RNN) Models
Traditional RNN models are also implemented to compare their performance with the transformer-based approaches. RNNs are useful for modeling sequential data, though they may struggle with long-term dependencies compared to transformer models.

## Datasets

The models are evaluated on various datasets, including:
- **MovieLens**
- **Amazon Beauty**
- **Delicious**

These datasets provide a diverse set of user-item interactions, enabling comprehensive evaluation of the models' performance.
