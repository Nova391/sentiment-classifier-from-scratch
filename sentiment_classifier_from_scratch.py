import math

# 1. DATA
train_texts = [
    "I love this",
    "This is amazing",
    "I really enjoyed it",
    "I hate this",
    "This is terrible",
    "This is awful"
]

train_labels = [
    1,
    1,
    1,
    0,
    0,
    0
]

test_texts = [
    "This is fantastic",
    "I really disliked it"
]

test_labels = [
    1,
    0
]


# 2. VOCABULARY
vocabulary = []

for text in train_texts:
    words = text.lower().split()

    for word in words:
        if word not in vocabulary:
            vocabulary.append(word)

word_to_id = {}

for index in range(len(vocabulary)):
    word_to_id[vocabulary[index]] = index

# 3. MODEL PARAMETERS
weights = []

for _ in range(len(vocabulary)):
    weights.append(0)
bias = 0
learning_rate = 0.1

# 4. TRAINING
for epoch in range(1000):
    for sentence_index in range(len(train_texts)):
        words = train_texts[sentence_index].lower().split()
        text_vector = []
        for _ in range(len(vocabulary)):
            text_vector.append(0)
        for word in words:
            if word in word_to_id:
                word_id = word_to_id[word]
                text_vector[word_id] = 1
        weighted_sum = 0
        for index in range(len(weights)):
            weighted_sum += text_vector[index] * weights[index]
        prediction = weighted_sum + bias
        probability = 1 / (1 + math.exp(-prediction))
        label = train_labels[sentence_index]
        loss = -(label * math.log(probability) + (1 - label) * math.log(1 - probability))
        for index in range(len(weights)):
            weight_gradient = (probability - label) * text_vector[index]
            weights[index] -= learning_rate * weight_gradient
        bias_gradient = probability - label
        bias -= learning_rate * bias_gradient

# 5. TESTING
correct = 0

for test_index in range(len(test_texts)):
    words = test_texts[test_index].lower().split()
    text_vector = []
    for _ in range(len(vocabulary)):
        text_vector.append(0)
    for word in words:
        if word in word_to_id:
            word_id = word_to_id[word]
            text_vector[word_id] = 1
    weighted_sum = 0
    for index in range(len(weights)):
        weighted_sum += text_vector[index] * weights[index]
    prediction = weighted_sum + bias
    probability = 1 / (1 + math.exp(-prediction))
    if probability >= 0.5:
        predicted_label = 1
    else:
        predicted_label = 0
    actual_label = test_labels[test_index]
    if predicted_label == actual_label:
        correct += 1
    print("Test:", test_texts[test_index])
    print("Predicted:", predicted_label)
    print("Actual:", actual_label)
    print("Probability:", round(probability * 100, 2), "%")
    print()

# 6. EVALUATION
accuracy = correct / len(test_texts)
print("Correct:", correct, "/", len(test_texts))
print("Accuracy:", round(accuracy * 100, 2), "%")
