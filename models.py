# models.py

import numpy as np
import torch
from torch import nn
import collections

#####################
# MODELS FOR PART 1 #
#####################

class ConsonantVowelClassifier(object):

    def predict(self, context):
        """
        :param context:
        :return: 1 if vowel, 0 if consonant
        """
        raise Exception("Only implemented in subclasses")

class FrequencyBasedClassifier(ConsonantVowelClassifier):
    """
    Classifier based on the last letter before the space. If it has occurred with more consonants than vowels,
    classify as consonant, otherwise as vowel.
    """
    def __init__(self, consonant_counts, vowel_counts):
        self.consonant_counts = consonant_counts
        self.vowel_counts = vowel_counts

    def predict(self, context):
        # Look two back to find the letter before the space
        if self.consonant_counts[context[-1]] > self.vowel_counts[context[-1]]:
            return 0
        else:
            return 1


class RNNClassifier(ConsonantVowelClassifier,nn.Module):

    def __init__(self, vcidx, embedding_size, hidden_size, output_size):
        super().__init__()
        self.vocab_index_size = vcidx
        self.no_of_embeddings = embedding_size
        self.hidden_dim = hidden_size
        self.output_dim = output_size
        self.create_model()

    def create_model(self):
        # Basic Architecture
        """
        1. Embedding Layer (with vocabulary size and embedding size)
        2. GRU RNN Layer (with input size of embedding size and output hidden dimension of hidden_size)
        3. Fully Connected Layer
        4. Softmax Output Layer
        """
        self.embeddings = nn.Embedding(self.vocab_index_size, self.no_of_embeddings)
        self.rnn = nn.GRU(self.no_of_embeddings, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_sequence):
        # embedding
        seq_embedding = self.embeddings(torch.tensor(input_sequence).unsqueeze(0))  # Add batch dimension

        # RNN
        _, hidden = self.rnn(seq_embedding)

        # Fully Connected and Output
        logits = self.fc(hidden.squeeze(0))
        output = self.softmax(logits)
        return output

    def predict(self, input_sequence):
        with torch.no_grad():
            output = self.forward(input_sequence)
            return torch.argmax(output, dim=1).item()

def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_cons_exs: list of strings followed by consonants
    :param train_vowel_exs: list of strings followed by vowels
    :param dev_cons_exs: list of strings followed by consonants
    :param dev_vowel_exs: list of strings followed by vowels
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNClassifier instance trained on the given data
    """
    # parameters
    embedding_size = 20
    hidden_size = 50
    output_size = 2

    # Model compiling
    model = RNNClassifier(len(vocab_index), embedding_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        correct_train_predictions = 0
        total_train_examples = len(train_cons_exs) + len(train_vowel_exs)

        # Training on train set
        for example, label in [(ex, 0) for ex in train_cons_exs] + [(ex, 1) for ex in train_vowel_exs]:
            # Convert example to indices
            input_sequence = [vocab_index.index_of(char) for char in example]
            target = torch.tensor([label])
            optimizer.zero_grad()
            output = model(input_sequence)
            loss = criterion(output, target)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy on training data
            _, predicted = torch.max(output, 1)
            correct_train_predictions += (predicted == target).sum().item()

        avg_train_loss = total_loss / total_train_examples
        train_accuracy = correct_train_predictions / total_train_examples * 100

        # Evaluate on dev set
        model.eval()
        correct_dev_predictions = 0
        total_dev_examples = len(dev_cons_exs) + len(dev_vowel_exs)
        
        with torch.no_grad():
            for example, label in [(ex, 0) for ex in dev_cons_exs] + [(ex, 1) for ex in dev_vowel_exs]:
                input_sequence = [vocab_index.index_of(char) for char in example]
                target = torch.tensor([label])
                output = model(input_sequence)
                
                # Calculate accuracy on dev data
                _, predicted = torch.max(output, 1)
                correct_dev_predictions += (predicted == target).sum().item()

        dev_accuracy = correct_dev_predictions / total_dev_examples * 100

        print(f"Epoch {epoch + 1}/10, Loss: {avg_train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Dev Accuracy: {dev_accuracy:.2f}%")
        
    return model

def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)


#####################
# MODELS FOR PART 2 #
#####################


class LanguageModel(object):

    def get_log_prob_single(self, next_char, context):
        """
        Scores one character following the given context. That is, returns
        log P(next_char | context)
        The log should be base e
        :param next_char:
        :param context: a single character to score
        :return:
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context):
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return:
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_log_prob_single(self, next_char, context):
        return np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class RNNLanguageModel(LanguageModel):
    def __init__(self, model_emb, model_dec, vocab_index):
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index

    def get_log_prob_single(self, next_char, context):
        raise Exception("Implement me")

    def get_log_prob_sequence(self, next_chars, context):
        raise Exception("Implement me")


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
    raise Exception("Implement me")

if __name__ == "__main__":
    embeddings = nn.Embedding(26,10)
    print(embeddings.weight.shape)