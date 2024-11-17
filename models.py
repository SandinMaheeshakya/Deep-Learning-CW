# models.py
import numpy as np
import torch
from torch import nn
import collections
from torch.utils.data import DataLoader, Dataset,TensorDataset

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

# Helper Function - Creating a tensor type dataset to store the text indexes
class TensoredDataset(Dataset):
    def __init__(self, consonant_data, vowel_data, vocab_index):

        self.data = []
        for word in consonant_data:
            self.data.append((word, 0))  # Consonant = 0
        for word in vowel_data:
            self.data.append((word, 1))  # Vowel = 1
        self.vocab_index = vocab_index

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, label = self.data[idx]
        input_sequence = [self.vocab_index.index_of(char) for char in data]
        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        input_label = torch.tensor(label, dtype=torch.long)
        return input_tensor, input_label
    

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
        self.gru = nn.GRU(self.no_of_embeddings, self.hidden_dim,bidirectional=True,batch_first=True,num_layers=self.num_layers)
        self.dropout = nn.Dropout(p=0.7)
        self.fc = nn.Linear(2 * self.hidden_dim, self.output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_sequence):

        # Value checks
        if not isinstance(input_sequence, torch.Tensor):
            raise TypeError("Input has to be a tensor")
        
        # embedding
        seq_embedding = self.embeddings(input_sequence)

        # first hidden state
        h0 = torch.zeros(self.num_layers * 2,seq_embedding.size(0),self.hidden_dim)

        # RNN
        output, _ = self.gru(seq_embedding,h0)

        # Fully Connected and Output
        out = output[:, -1, :]  
        out = self.fc(self.dropout(out)) # droput

        # Returns through a softmax layer
        return self.softmax(out)

    def predict(self, input_sequence):

        try:
            input_sequence = torch.tensor([self.vocab_index.index_of(char) for char in input_sequence], 
                                        dtype=torch.long
                ) 
            
            input_sequence = input_sequence.unsqueeze(0) # add another dimension to match the required dimension

            with torch.no_grad():
                output = self.forward(input_sequence)
                prediction = torch.argmax(output, dim=1)
                return prediction.item()
            
        except Exception as e:
            print(f"Error -> {e}")

def train_rnn_classifier(args, train_cons_exs, train_vowel_exs, dev_cons_exs, dev_vowel_exs, vocab_index):

    try:
        # parameters
        embedding_size = 50
        hidden_size = 144
        layers = 2

        # Model compiling
        model = RNNClassifier(len(vocab_index), embedding_size, hidden_size, layers, output_dim=2)
        model.vocab_index = vocab_index # storing for feature reference
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create datasets and dataloaders
        train_dataset = TensoredDataset(train_cons_exs, train_vowel_exs, vocab_index)
        dev_dataset = TensoredDataset(dev_cons_exs, dev_vowel_exs, vocab_index)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=False)
        
    except Exception as e:
        print(f'Error occured during model comiplation stage -> {e}')

    try:
        # Training loop
        for epoch in range(12):
            model.train() # training state
            
            total_loss = 0
            correct_train_predictions = 0
            total_train_examples = len(train_loader.dataset)

            for input_sequence, target in train_loader:
                
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

            # Print epoch results
            print(f"Epoch {epoch + 1}/{12}, Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

            # Evaluate on the development data
            model.eval()
            correct_development_predictions = 0
            total_development_examples = len(dev_loader.dataset)

            with torch.no_grad():
                for input_sequence, target in dev_loader:
                    output = model(input_sequence)

                    _, predicted = torch.max(output, 1)
                    correct_development_predictions += (predicted == target).sum().item()

            dev_accuracy = correct_development_predictions / total_development_examples * 100
            print(f"Dev Accuracy: {dev_accuracy:.2f}%")
        
    except RuntimeError as e:
        print(f"Runtime Error -> {e}")

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


class RNNLanguageModel(LanguageModel, nn.Module):
    def __init__(self, model_emb, model_dec, vocab_index):
        super(RNNLanguageModel, self).__init__()
        self.model_emb = model_emb
        self.model_dec = model_dec
        self.vocab_index = vocab_index

        # change the device to GPU if available 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LSTM layer
        self.lstm = nn.LSTM(input_size=256, hidden_size=512, dropout=0.2, num_layers=2,batch_first=True)

    def forward(self, input_sequence, initial_states=None):

        if input_sequence is None:
            raise ValueError("No value for the input sequence!")
        
        # embeded text
        text_embeddings = self.model_emb(input_sequence)

        if initial_states is None:
            batch_size = input_sequence.size(0)
            h0 = torch.zeros(2, batch_size, 512).to(input_sequence.device) 
            c0 = torch.zeros(2, batch_size, 512).to(input_sequence.device)
        else:
            h0, c0 = initial_states

        # LSTM output
        lstm_output, (hn, cn) = self.lstm(text_embeddings, (h0, c0))
        
        fc_out = self.model_dec(lstm_output) # Fully connected Layer
        
        return fc_out, (hn, cn)
    
    def get_log_prob_single(self, next_char, context):

        try:
            # Sequence Conversion
            context_indices = [self.vocab_index.index_of(char) for char in context]
            context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # model output
            hs_cs = None 
            output, _ = self.forward(context_tensor, hs_cs) 
            
            # log-probabilities of the last character
            log_char = output[0, -1]
            log_prob = torch.log_softmax(log_char, dim=0) 
            
            # log probability of the next character
            next_char_index = self.vocab_index.index_of(next_char)
            return log_prob[next_char_index].item()
        
        except Exception as e:
            print(f'Error occured during single log probability calculation -> {e}')
            
    def get_log_prob_sequence(self, next_chars, context):
        log_prob_seq = 0.0
        try:
            for char in next_chars:
                # Calculate log-probability for each character and add the context in order to use the log_prob_single method
                log_prob_seq += self.get_log_prob_single(char, context)
                context += char
        
            return log_prob_seq
        
        except Exception as e:
            raise RuntimeError(f'Error Occured in sequence log calculation -> {e}')
            
# Training function
def train_lm(args, train_text, dev_text, vocab_index):

    # Hyperparameters
    embed_size = 256
    batch_size = 64
    seq_length = 12
    epochs = 10
    lr = 0.0005

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare input-output pairs for the model
    input_sequence = []
    target_sequence = []

    try:
        # Adding SOS Token
        sos_token = vocab_index.add_and_get_index("<SOS>")
        for i in range(len(train_text) - seq_length):
            input_sequence.append([sos_token] + [vocab_index.index_of(char) for char in train_text[i:i + (seq_length - 1)]])
            target_sequence.append([vocab_index.index_of(char) for char in train_text[i : i + seq_length]])

        # Convert to tensors and create DataLoader
        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        target_tensor = torch.tensor(target_sequence, dtype=torch.long)
        dataset = TensorDataset(input_tensor, target_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)

        # Initialize the embedding and decoder models
        embedding_model = nn.Embedding(len(vocab_index), embed_size).to(device)
        decoder_model = nn.Linear(512, len(vocab_index))
        model = RNNLanguageModel(embedding_model, decoder_model, vocab_index).to(device)

        # Optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    except Exception as e:
        print(f'Error -> {e}')

    try:
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0

            for batch_idx, (input_seq, target_seq) in enumerate(dataloader):

                # Cuda support
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)

                batch_size = input_seq.size(0)

                optimizer.zero_grad()  # Clear gradients

                # Forward pass
                output , _ = model(input_seq)

                # loss calculation
                loss = criterion(output.view(-1, len(vocab_index)), target_seq.view(-1))
                loss.backward()  

                # Gradiant clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Total loss
                total_loss += loss.item()

            # Calculate perplexity
            avg_loss = total_loss / len(dataloader)
            perplexity = torch.exp(torch.tensor(avg_loss))
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Perplexity: {perplexity.item():.4f}")

        scheduler.step(avg_loss)
        
        # Return trained model
        return model
    
    except RuntimeError as re:
        print(f"Runtime Error Occured, Error ->{re}")
        