# Sentiment Classifier From Scratch

A simple binary sentiment classifier built from scratch in Python without using machine-learning libraries.

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

## Important Limitations

This is an educational project with a very small dataset.

The model demonstrates the internal mechanics of a text classifier rather than providing reliable real-world sentiment analysis.

The test set is also very small, and some test words do not appear in the training vocabulary, so the reported accuracy should not be treated as a meaningful benchmark.

## Why I Built It

I built this project to understand machine-learning fundamentals by implementing the core training process manually instead of relying on libraries such as scikit-learn, PyTorch, or TensorFlow.

## What I Learned

Through this project I implemented and explored the complete basic workflow of a machine-learning classifier:

```text
Data
↓
Representation
↓
Model
↓
Prediction
↓
Loss
↓
Gradients
↓
Parameter Updates
↓
Training
↓
Testing
```

## Next Step

The next stage of my AI learning is to build a neural network from scratch and implement neurons, layers, forward propagation, backpropagation, and training.

## README Note

This README was drafted with assistance from OpenAI's ChatGPT.

The project code and implementation were written and developed by me as part of my AI learning process.
