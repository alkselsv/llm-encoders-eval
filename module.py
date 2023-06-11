import numpy as np
from sklearn.metrics import f1_score


def find_most_similar_vector_indices(vector_array):
    normalized_vectors = vector_array / \
        np.linalg.norm(vector_array, axis=1)[:, np.newaxis]
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
    np.fill_diagonal(similarity_matrix, -1)
    most_similar_indices = np.argmax(similarity_matrix, axis=1)
    return most_similar_indices


class Encoder():
    def __init__(self, X, y):
      self.X = X
      self.y = y
      self.X_embs = None

    def transform(self, embedder):
      self.X_embs = [embedder(x) for x in self.X]
      return self

    def eval(self):
      most_similar_indices = find_most_similar_vector_indices(self.X_embs)
      y_pred = self.y[most_similar_indices]
      result = f1_score(y_pred, self.y, average='micro')
      return result
