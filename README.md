# Sentiment Classifier From Scratch

A simple binary sentiment classifier built entirely from scratch in Python without using machine-learning libraries.

## What This Project Does

The model classifies text as either:

* `1` → Positive
* `0` → Negative

It uses a Bag-of-Words representation and logistic regression trained with gradient descent.

## Project Pipeline

```text
Text
↓
Tokenization
↓
Vocabulary
↓
Word-to-ID Mapping
↓
Bag-of-Words Vector
↓
Weighted Sum + Bias
↓
Sigmoid
↓
Probability
↓
Positive / Negative
```

## Machine Learning Concepts

This project implements:

* Vocabulary construction
* Text preprocessing
* Bag-of-Words
* Feature vectors
* Weights and bias
* Forward propagation
* Sigmoid activation
* Binary cross-entropy loss
* Gradients
* Gradient descent
* Training
* Train/test split
* Accuracy evaluation

## Example

The training data contains simple positive and negative sentences such as:

```text
"I love this" → 1
"This is amazing" → 1
"I hate this" → 0
"This is terrible" → 0
```

## Requirements

Python 3.x

The project uses only Python's built-in `math` module.

## Run

```bash
python sentiment_classifier.py
```

## Important Limitation

This is an educational project with a very small dataset.

The model is designed to demonstrate how a text classifier works internally rather than provide reliable real-world sentiment analysis.

The test set also contains words that do not appear in the training set, which makes the evaluation intentionally limited.

## Why I Built It

This project was built to understand machine learning fundamentals by implementing the core training process manually instead of relying on libraries such as scikit-learn, PyTorch, or TensorFlow.

## Next Step

The next stage of my AI learning is to build a neural network from scratch and implement neurons, layers, forward propagation, backpropagation, and training.
