# models.py

import numpy as np
import torch
from torch import nn
import collections
import random
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
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

# Define the RNN Language Model
class RNNLanguageModel(LanguageModel, nn.Module):
    def __init__(self, model_emb, model_dec, vocab_index):
        super(RNNLanguageModel, self).__init__()
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index

        # Define the LSTM layers
        self.lstm = nn.LSTM(input_size=256, hidden_size=512, dropout=0.2, num_layers=2,batch_first=True)
        self.fc = nn.Linear(512, len(vocab_index))  # Output layer

    def forward(self, input_sequence, initial_states=None):
        text_embeddings = self.model_emb(input_sequence)  # Get embeddings

        if initial_states is None:
            batch_size = input_sequence.size(0)
            h0 = torch.zeros(2, batch_size, 512).to(input_sequence.device) 
            c0 = torch.zeros(2, batch_size, 512).to(input_sequence.device)
        else:
            h0, c0 = initial_states

        # Run through LSTM
        lstm_output, (hn, cn) = self.lstm(text_embeddings, (h0, c0))
        
        fc_out = self.fc(lstm_output)  # Shape: (batch_size, seq_length, vocab_size)
        
        return fc_out, (hn, cn)  # Return output and the final hidden state
    
    def get_log_prob_single(self, next_char, context):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Convert context to indices
        context_indices = [self.vocab_index.index_of(char) for char in context]
        
        # Add a batch dimension and convert to tensor
        context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0).to(device)
        
        # Pass the context through the model
        hidden = None  # Start with no hidden state
        output, _ = self.forward(context_tensor, hidden)  # Shape: [1, seq_length, vocab_size]
        
        # Get the log-probabilities for the last character in the context
        logits = output[0, -1]  # Shape: [vocab_size]
        log_probs = torch.log_softmax(logits, dim=0)  # Log probabilities over the vocabulary
        
        # Get the log-probability of the next_char
        next_char_index = self.vocab_index.index_of(next_char)

        return log_probs[next_char_index].item()

    def get_log_prob_sequence(self, next_chars, context):
        log_prob_seq = 0.0
        for next_char in next_chars:
            # Calculate log-probability for the current character
            log_prob_seq += self.get_log_prob_single(next_char, context)
            
            # Update the context to include the current character
            context += next_char
    
        return log_prob_seq
# Training function
def train_lm(args, train_text, dev_text, vocab_index):

    # Hyperparameters
    embed_size = 256
    batch_size = 64
    seq_length = 10
    epochs = 10
    lr = 0.0005

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare input-output pairs for the model
    input_seqs = []
    target_seqs = []

    # Adding SOS Token
    sos_token = vocab_index.add_and_get_index("<SOS>")
    for i in range(len(train_text) - seq_length):
        input_seqs.append([vocab_index.index_of(char) for char in train_text[i:i+seq_length]])
        target_seqs.append([vocab_index.index_of(char) for char in train_text[i+1:i+seq_length+1]])


    # Convert to tensors and create DataLoader
    input_tensor = torch.tensor(input_seqs, dtype=torch.long)
    target_tensor = torch.tensor(target_seqs, dtype=torch.long)
    dataset = TensorDataset(input_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)

    # Initialize the embedding and decoder models
    embedding_model = nn.Embedding(len(vocab_index), embed_size).to(device)
    decoder_model = nn.Embedding(len(vocab_index), embed_size).to(device)  # Assuming decoder also uses embedding

    # Initialize the combined RNNLanguageModel
    model = RNNLanguageModel(embedding_model, decoder_model, vocab_index).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_tokens = 0  # For perplexity calculation
        total_correct = 0  # For accuracy calculation

        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            # Initialize hidden states
            batch_size = input_seq.size(0)
            hidden = None  # No initial hidden state passed

            optimizer.zero_grad()  # Clear gradients

            # Forward pass
            output, hidden = model(input_seq, hidden)

            # Compute loss
            loss = criterion(output.view(-1, len(vocab_index)), target_seq.view(-1))
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Update total loss and tokens count for perplexity
            total_loss += loss.item()
            total_tokens += target_seq.size(0) * target_seq.size(1)  # Total number of tokens

            # Calculate accuracy
            _, predicted = torch.max(output, dim=2)
            correct = (predicted == target_seq).float()
            total_correct += correct.sum().item()

        # Compute accuracy
        accuracy = total_correct / total_tokens * 100  # Accuracy as percentage

        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")

    # Return trained model
    return model
        
if __name__ == "__main__":
    embeddings = nn.Embedding(28,10)
    X = embeddings(torch.tensor([12,11,5,4,3]))
    print(X)
    print(embeddings.weight.shape)