"""
Code adapted from https://github.com/SynthAIr/TimeGAN_Trajectories/blob/main/eval_diversity.py - Paper: https://www.sesarju.eu/sites/default/files/documents/sid/2024/papers/SIDs_2024_paper_054%20final.pdf
"""


"""
Generation of Synthetic Aircraft Landing Trajectories Using Generative Adversarial Networks [Codebase]

File name:
    eval_diversity.py

Description:
    Data diversity assessment using PCA or t-SNE for original & synthetic distribution visualization.

Author:
    Sebastiaan Wijnands
    S.C.P.Wijnands@student.tudelft.nl
    August 10, 2024    
"""

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  
import matplotlib.pyplot as plt 
import numpy as np  

def data_diversity(ori_data, generated_data, analysis, average_dimension='sequence', max_sample_size=1000, model_name='model'):
    """
    Data diversity assessment using PCA or t-SNE for original & synthetic distribution visualization.

    Inputs:
        - ori_data (array): original data
        - generated_data (array): synthetic data
        - analysis (str): 'PCA' or 't-SNE'
        - average_dimension (string): flatten along 'sequence' or 'samples' dimension
        - max_sample_size (int): maximum sample size for computational efficiency
    """
    fig = plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 14})
    
    # Determine the analysis sample size
    anal_sample_no = min(max_sample_size, len(ori_data), len(generated_data))
    
    # Randomly select indices
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    idx_gen = np.random.permutation(len(generated_data))[:anal_sample_no]
    
    # Subset the data
    ori_data, generated_data = np.asarray(ori_data)[idx], np.asarray(generated_data)[idx_gen]
    
    # Preprocess the data
    if average_dimension == 'sequence':
        prep_data = np.mean(ori_data, axis=1)
        prep_data_hat = np.mean(generated_data, axis=1)
    elif average_dimension == 'samples':
        prep_data = np.mean(ori_data, axis=2)
        prep_data_hat = np.mean(generated_data, axis=2)
    else:
        prep_data = ori_data.reshape(ori_data.shape[0], -1)
        prep_data_hat = generated_data.reshape(generated_data.shape[0], -1)
    
    # Combine original and generated data for a single transformation
    combined_data = np.concatenate((prep_data, prep_data_hat), axis=0)
    
    # Define colors (red for original, blue for synthetic)
    colors = ["red"] * anal_sample_no + ["blue"] * anal_sample_no
    
    if analysis == 'PCA':
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(combined_data)
        
        # Plot PCA results
        plt.scatter(pca_results[:anal_sample_no, 0], pca_results[:anal_sample_no, 1], c=colors[:anal_sample_no], alpha=0.35, label="Original")
        plt.scatter(pca_results[anal_sample_no:, 0], pca_results[anal_sample_no:, 1], c=colors[anal_sample_no:], alpha=0.35, label="Synthetic")
    
    elif analysis == 't-SNE':
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(combined_data)
        
        # Plot t-SNE results
        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1], c=colors[:anal_sample_no], alpha=0.35, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1], c=colors[anal_sample_no:], alpha=0.35, label="Synthetic")
    
    # Plot settings
    plt.title(f'{analysis} Plot')
    plt.xlabel(f'x-{analysis.lower()}')
    plt.ylabel(f'y-{analysis.lower()}')
    plt.legend()
    #plt.savefig(f'./figures/{model_name}_diversity_{analysis}.png', dpi=400, bbox_inches='tight')
    
    return fig


