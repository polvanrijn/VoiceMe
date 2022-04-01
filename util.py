import io
import numpy as np

def convert_pca_weights_to_speaker_embedding(pca_weights, model, normalize_embedding=True):
    print(normalize_embedding)
    def normalize(x):
        return x / np.linalg.norm(x)

    N_PCA = 10
    assert model in ['speakernet-cv', 'vctk-vits']
    components = np.load(f'pca/{model}-components.npy', allow_pickle=True)
    mu = np.load(f'pca/{model}-mu.npy', allow_pickle=True)

    # Make assertions
    assert components.shape[0] == N_PCA

    pca_weights = np.array(pca_weights, dtype=np.float32)
    spk_embeddings = []
    for i in range(pca_weights.shape[0]):
        pca_vector = pca_weights[i]
        assert pca_vector.shape[0] == N_PCA
        speaker_embedding = np.dot(pca_vector, components) + mu

        if normalize_embedding:
            speaker_embedding = normalize(speaker_embedding)

        spk_embeddings.append(speaker_embedding.astype(np.float32))
    spk_embeddings = np.array(spk_embeddings)

    return spk_embeddings

def to_bytes(obj):
    b = io.BytesIO()
    np.save(b, obj)
    b.seek(0)
    return b