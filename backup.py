class RNNClassifier(nn.Module):

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
    hidden_size = 10
    output_size = 2

    # Model compiling
    model = RNNClassifier(len(vocab_index), embedding_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    for epoch in range(5):
        model.train()
        total_loss = 0

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

        print(f"Epoch {epoch + 1}/{5}, Loss: {total_loss / len(train_cons_exs + train_vowel_exs):.4f}")

        # Evaluation (for simplicity, compute and print accuracy on dev set here)
        model.eval()  # Set to evaluation mode
        correct = 0
        with torch.no_grad():
            for example, label in [(ex, 0) for ex in dev_cons_exs] + [(ex, 1) for ex in dev_vowel_exs]:
                input_sequence = [vocab_index.index_of(char) for char in example]
                prediction = model.predict(input_sequence)
                if prediction == label:
                    correct += 1
        accuracy = correct / len(dev_cons_exs + dev_vowel_exs)
        print(f"Development Set Accuracy: {accuracy * 100:.2f}%")

    return model
    
def train_frequency_based_classifier(cons_exs, vowel_exs):
    consonant_counts = collections.Counter()
    vowel_counts = collections.Counter()
    for ex in cons_exs:
        consonant_counts[ex[-1]] += 1
    for ex in vowel_exs:
        vowel_counts[ex[-1]] += 1
    return FrequencyBasedClassifier(consonant_counts, vowel_counts)