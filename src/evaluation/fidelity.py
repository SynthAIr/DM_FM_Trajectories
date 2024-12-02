"""
Code taken from https://github.com/SynthAIr/TimeGAN_Trajectories/blob/main/eval_fidelity.py - Paper: https://www.sesarju.eu/sites/default/files/documents/sid/2024/papers/SIDs_2024_paper_054%20final.pdf

Converted from Keras (original) to PyTorch 
"""

"""
Generation of Synthetic Aircraft Landing Trajectories Using Generative Adversarial Networks [Codebase]

File name:
    eval_fidelity.py

Description:
    Data fidelity assessment by measuring discriminative score of a LSTM binary (original vs. synthetic data) classifier.

Author:
    Sebastiaan Wijnands
    S.C.P.Wijnands@student.tudelft.nl
    August 10, 2024    
"""
# Import required packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, (hn, _) = self.lstm3(x)
        x = self.fc(hn[-1])
        return x

def split_data(original_data, synthetic_data, test_size=0.25):
    """
    Split original and synthetic data into training and testing sets.

    Inputs:
    - original_data (array): original data
    - synthetic_data (array): synthetic data
    - test_size (float, optional): fraction of the data to reserve as test data

    Outputs:
    - ori_train (array): original data for training
    - ori_test (array): original data for testing
    - syn_train (array): synthetic data for training
    - syn_test (array): synthetic data for testing
    """
    #print(original_data.shape, synthetic_data.shape)
    ori_train, ori_test = train_test_split(original_data, test_size=test_size)
    syn_train, syn_test = train_test_split(synthetic_data, test_size=test_size)
    return ori_train, ori_test, syn_train, syn_test

def train_discriminator(ori_train_data, syn_train_data, input_dim):
    """
    Train the discriminator model on original and synthetic training data.

    Inputs:
    - ori_train_data (array): Original data for training
    - syn_train_data (array): Synthetic data for training
    - input_dim (int): The dimension of the input features

    Outputs:
    - discriminator (nn.Module): Trained discriminator model
    """
    # Convert data to PyTorch tensors
    train_data = np.concatenate([ori_train_data, syn_train_data])
    train_labels = np.concatenate([np.ones(len(ori_train_data)), np.zeros(len(syn_train_data))])
    idx = np.random.permutation(len(train_data))
    train_data = train_data[idx]
    train_labels = train_labels[idx]

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)

    # Create model
    discriminator = Discriminator(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    # Training loop
    discriminator.train()
    num_epochs = 10
    batch_size = 128
    for epoch in range(num_epochs):
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]

            optimizer.zero_grad()
            outputs = discriminator(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return discriminator

def evaluate_discriminator(discriminator, ori_test_data, syn_test_data):
    """
    Evaluate the discriminator model on original and synthetic test data.

    Inputs:
    - discriminator (nn.Module): Trained discriminator model
    - ori_test_data (array): Original data for testing
    - syn_test_data (array): Synthetic data for testing

    Outputs:
    - accuracy (float): Accuracy of the discriminator on the test data
    - discriminative_score (float): Discriminative score of the discriminator
    - conf_matrix (array): Confusion matrix of the discriminator's predictions
    - tpr (float): True Positive Rate (Sensitivity)
    - tnr (float): True Negative Rate (Specificity)
    """
    # Concatenate and prepare test data
    test_data = np.concatenate([ori_test_data, syn_test_data])
    test_labels = np.concatenate([np.ones(len(ori_test_data)), np.zeros(len(syn_test_data))])
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

    # Evaluate
    discriminator.eval()
    with torch.no_grad():
        outputs = discriminator(test_data)
        predictions = (torch.sigmoid(outputs) > 0.5).int().flatten()
        acc = (predictions == test_labels.flatten().int()).float().mean().item()
        conf_matrix = confusion_matrix(test_labels.numpy(), predictions.numpy())
        tn, fp, fn, tp = conf_matrix.ravel()
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
        discriminative_score = abs(0.5 - acc)

    return acc, discriminative_score, conf_matrix, tpr, tnr

def discriminative_score(original_data, synthetic_data, test_size=0.2):
    """
    Compute the accuracy, discriminative score, confusion matrix, TPR, and TNR of the discriminator on original and synthetic data.

    Inputs:
    - original_data (array): Array containing original data
    - synthetic_data (array): Array containing synthetic data
    - test_size (float, optional): Fraction of the data to reserve as test data

    Outputs:
    - accuracy (float): Accuracy of the discriminator on the test data
    - discriminative_score (float): Discriminative score of the discriminator
    - confusion_matrix (array): Confusion matrix of the discriminator's predictions
    - tpr (float): True Positive Rate (Sensitivity)
    - tnr (float): True Negative Rate (Specificity)
    """
    print(original_data.shape, synthetic_data.shape)
    ori_train, ori_test, syn_train, syn_test = split_data(original_data, synthetic_data, test_size)
    print(ori_train.shape, ori_test.shape, syn_train.shape, syn_test.shape)
    input_dim = ori_train.shape[2]
    discriminator = train_discriminator(ori_train, syn_train, input_dim)
    accuracy, score, conf_matrix, tpr, tnr = evaluate_discriminator(discriminator, ori_test, syn_test)
    return accuracy, score, conf_matrix, tpr, tnr

if __name__ == '__main__':
    # Data loading and processing part remains similar to the original
    ori_data = np.random.rand(100, 200, 2)  # 100 samples, each with a sequence of length 10 and 5 features
    generated_data = np.random.rand(100, 200, 2)  # Synthetic data with the same shape as ori_data
    accuracy, score, conf_matrix, tpr, tnr = discriminative_score(ori_data, generated_data)
    print("Accuracy on test data:", accuracy)
    print("Discriminative Score:", score)
    print("Confusion Matrix:\n", conf_matrix)
    print("True Positive Rate (TPR):", tpr)
    print("True Negative Rate (TNR):", tnr)

    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Synthetic', 'Original'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("fidelity")

