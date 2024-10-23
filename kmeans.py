import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None):
        """
        Initialize KMeans clustering algorithm.
        
        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters to form
        max_iters : int, default=100
            Maximum number of iterations for a single run
        tol : float, default=1e-4
            Tolerance for declaring convergence
        random_state : int, default=None
            Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        
    def fit(self, X):
        """
        Fit KMeans clustering to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        n_samples, n_features = X.shape
        
        # Randomly initialize centroids
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids_ = X[idx]
        
        for _ in range(self.max_iters):
            # Store old centroids
            old_centroids = self.centroids_.copy()
            
            # Assign points to nearest centroid
            distances = np.sqrt(((X - self.centroids_[:, None, :])**2).sum(axis=-1))
            self.labels_ = np.argmin(distances, axis=0)
            
            # Update centroids
            for k in range(self.n_clusters):
                if np.sum(self.labels_ == k) > 0:  # Avoid empty clusters
                    self.centroids_[k] = X[self.labels_ == k].mean(axis=0)
            
            # Check for convergence
            if np.sum((old_centroids - self.centroids_)**2) < self.tol:
                break
                
        # Calculate inertia (within-cluster sum of squares)
        self.inertia_ = 0
        for k in range(self.n_clusters):
            if np.sum(self.labels_ == k) > 0:
                self.inertia_ += np.sum((X[self.labels_ == k] - self.centroids_[k])**2)
                
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to predict
            
        Returns:
        --------
        labels : array of shape (n_samples,)
            Predicted cluster labels
        """
        distances = np.sqrt(((X - self.centroids_[:, None, :])**2).sum(axis=-1))
        return np.argmin(distances, axis=0)

def test_kmeans():
    """
    Test the KMeans implementation with a simple example.
    """
    # Generate sample data
    np.random.seed(42)
    X = np.concatenate([
        np.random.normal(0, 1, (100, 2)),
        np.random.normal(5, 1, (100, 2)),
        np.random.normal(10, 1, (100, 2))
    ])
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Basic assertions
    assert kmeans.centroids_.shape == (3, 2)
    assert len(kmeans.labels_) == 300
    assert kmeans.inertia_ > 0
    
    print("All tests passed!")

if __name__ == "__main__":
    test_kmeans()
