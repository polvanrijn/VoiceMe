import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import os

models = {
    'speakernet-cv': 'pca/cv-corpus-8.0-2022-01-19-en.csv',
    'vctk-vits': 'pca/vctk-vits-speaker_embeddings.csv',
}

for key, csv_path in models.items():
    df = pd.read_csv(csv_path)
    N_PCA = 10
    X = df.values[:, :256]

    pca = PCA(n_components=N_PCA)
    pca_result = pca.fit_transform(X)


    os.makedirs('pca', exist_ok=True)
    mu = np.mean(X, axis=0)
    np.save(f'pca/{key}-mu.npy', mu)

    components = pca.components_[:N_PCA, :]
    np.save(f'pca/{key}-components.npy', components)

    # Print variance
    explained_var = pca.explained_variance_ratio_
    print(f'Explained variance {sum(explained_var)} per principal component in {key}: {explained_var}')