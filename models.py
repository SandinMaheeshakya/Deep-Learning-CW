# models.py

import numpy as np
import torch
from torch import nn
import collections
import random
from torch.utils.data import DataLoader, TensorDataset
import math

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

    def __init__(self, vcidx, embedding_size, hidden_size, num_layers, output_dim):
        super(RNNClassifier,self).__init__()
        self.vocab_index_size = vcidx
        self.no_of_embeddings = embedding_size
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
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
        self.gru = nn.GRU(self.no_of_embeddings, self.hidden_dim,batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_sequence):

        # embedding
        seq_embedding = self.embeddings(torch.tensor(input_sequence).unsqueeze(0)) # Add batch dimension

        # first hidden state
        h0 = torch.zeros(self.num_layers,seq_embedding.size(0),self.hidden_dim)

        # RNN
        output, _ = self.gru(seq_embedding,h0)

        # Fully Connected and Output
        out = output[:, -1, :]  
        out = self.fc(out)
        out = self.softmax(out)
        return out

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
    embedding_size = 10
    hidden_size = 10
    layers = 1

    # Model compiling
    model = RNNClassifier(len(vocab_index), embedding_size, hidden_size, layers, output_dim=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)    

    # Shuffling data each epoch
    train_data = [(ex, 0) for ex in train_cons_exs] + [(ex, 1) for ex in train_vowel_exs]
    dev_data = [(ex, 0) for ex in dev_cons_exs] + [(ex, 1) for ex in dev_vowel_exs]

    # Random shuflling the data
    random.shuffle(train_data)
    random.shuffle(dev_data)
    
    # Training Loop
    for epoch in range(20):
        model.train()
        total_loss = 0
        correct_train_predictions = 0
        total_train_examples = len(train_cons_exs) + len(train_vowel_exs)

        for example, label in train_data:
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

        print(f"Epoch {epoch + 1}/10, Loss: {avg_train_loss:.4f}, " f"Train Accuracy: {train_accuracy:.2f}%")
        
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

class RNNLanguageModel(LanguageModel,nn.Module):
    def __init__(self, model_emb, model_dec, vocab_index):
        super(RNNLanguageModel,self).__init__()
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index

        # Network Design
        self.lstm = nn.LSTM( input_size = 10 , hidden_size = 50, num_layers = 4, batch_first=True)

    def forward(self, input_sequence,inital_states=None):

        text_embeddings = self.model_emb(input_sequence)

        # cell state and hidden state
        if inital_states is None:
            batch_size = input_sequence.size(0)
            h0 = torch.zeros(4, batch_size, 50)
            c0 = torch.zeros(4, batch_size, 50)

        # Go through LSTM layer
        lstm_output, h_state = self.lstm(text_embeddings, (h0,c0))

        # feed through a fully connected and output layer
        fc_out = self.model_dec(lstm_output[:-1:])
        
        return fc_out

    def get_log_prob_single(self, next_char, context):
        """
        Computes log probability for a single character given context.
        """
        self.eval()
        with torch.no_grad():
            context_indices = torch.tensor([self.vocab_index.index_of(c) for c in context]).unsqueeze(0)
            context_emb = self.model_emb(context_indices)
            _, hidden = self.lstm(context_emb)
            next_char_idx = torch.tensor([self.vocab_index.index_of(next_char)]).unsqueeze(0)
            logits, _ = self.forward(next_char_idx, hidden)
            log_probs = torch.log_softmax(logits, dim=-1)
            return log_probs[0, -1, self.vocab_index.index_of(next_char)].item()

    def get_log_prob_sequence(self, next_chars, context):
        """
        Computes log probabilities for a sequence of characters given context.
        """
        log_probs = []
        for char in next_chars:
            log_prob = self.get_log_prob_single(char, context)
            log_probs.append(log_prob)
            context += char  # Update context by adding the current character
        return log_probs

# Helper function for text chunking with <SOS> and target label creation
def prepare_texts(text,vocab_index):
    context_window = 7
    overlap_seq = 2
    chunked_data = []

    """ Text Chunking Process """
    words = text.split() # word list
    for i in range(0, len(words), context_window - overlap_seq):
        current_chunk = words[i:i + context_window]
        chunked_data.append(" ".join(current_chunk))
        
        if i + (context_window - overlap_seq) >= len(words):
            break
  
    """Inputs and target labels creation"""
    inputs =  []
    targets = []
    for text_chunk in chunked_data:
        input_data  = "<SOS>" + " ".join(text_chunk.split()[:-1]) # removing last token in order to predict it
        target_label = text_chunk

        # add data
        inputs.append(input_data)
        targets.append(target_label)

    # Adding sos token to the indexer object
    sos_index = vocab_index.add_and_get_index("<SOS>")

    # Returning the tensor wrapped data
    inputs =   [[sos_index] + [vocab_index.index_of(char) for char in sequence[5:]] for sequence in inputs]
    targets =  [[vocab_index.index_of(char) for char in sequence] for sequence in targets]

    return inputs,targets


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev texts as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: an RNNLanguageModel instance trained on the given data
    """
if __name__ == "__main__":
    embeddings = nn.Embedding(28,10)
    X = embeddings(torch.tensor([12,11,5,4,3]))
    print(X)
    print(embeddings.weight.shape)