import numpy as np

class ANNFaiss:
    def __init__(self, dimension, n_neighbors) -> None:
        import faiss
        self.index = faiss.IndexFlatL2(dimension)
        self.n_neighbors = n_neighbors
        self.samples = None
    def fit(self, samples):
        self.index.add(samples.astype(np.float32))
    def query(self, coor):
        distances, indices = self.index.search(coor.astype(np.float32), self.n_neighbors)
        return indices, distances

class ANNAnnoy:
    def __init__(self, dimension, n_neighbors) -> None:
        from annoy import AnnoyIndex
        self.index = AnnoyIndex(dimension, 'euclidean')
        self.index.set_seed(42)
        self.index.build(10)
        self.n_neighbors = n_neighbors
        self.samples = None
    def fit(self, samples):
        for i, s in enumerate(samples):
            self.index.add_item(i, s)
        self.samples = np.concatenate((self.samples, samples)) if self.samples is not None else samples
    def query(self, coor):
        indices,  distances = self.index.get_nns_by_vector(coor[0], self.n_neighbors, include_distances=True)
        return [indices], [distances]

class ANNHnswlib:
    def __init__(self, dimension, n_neighbors, seed=None) -> None:
        import hnswlib
        self.index = hnswlib.Index(space='l2', dim=dimension)
        if seed is not None:
            self.index.init_index(max_elements=200, ef_construction=200, M=48, random_seed = seed)
        else:
            self.index.init_index(max_elements=200, ef_construction=200, M=48)
        self.n_neighbors = n_neighbors
        self.samples = None
    def fit(self, samples):
        self.index.add_items(samples, np.arange(samples.shape[0]))
        self.index.set_ef(50)
    def query(self, coor):
        indices,  distances = self.index.knn_query(coor, self.n_neighbors)
        return indices, distances
    
class SKNN:
    def __init__(self, dimension, n_neighbors) -> None:
        from sklearn.neighbors import NearestNeighbors
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.samples = None
    def fit(self, samples):
        self.nn.fit(samples)
    def query(self, coor):
        distances, indices = self.nn.kneighbors(coor)
        return indices, distances

class NN:
    def __init__(self, ann, dimension, n_neighbors, seed=None) -> None:
        if ann == 'faiss':
            self.ann = ANNFaiss(dimension, n_neighbors)
        elif ann == 'annoy':
            self.ann = ANNAnnoy(dimension, n_neighbors)
        elif ann == 'hnswlib':
            self.ann = ANNHnswlib(dimension, n_neighbors, seed=seed)
        elif ann == 'nn':
            self.ann = SKNN(dimension, n_neighbors)
        
    def fit(self, samples):
        self.ann.fit(samples)
    def query(self, coor):
        return self.ann.query(coor)
    def nn_samples(self, coor):
        if self.ann.samples is None:
            return
        indices, _ = self.query(coor)
        return self.ann.samples[indices[0]]
    