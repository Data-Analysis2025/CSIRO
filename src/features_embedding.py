"""Feature engineering specifically for embedding-based approaches."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class SupervisedEmbeddingEngine(BaseEstimator, TransformerMixin):
    def __init__(self, n_pca=0.98, n_pls=8, n_gmm=5, random_state=42):
        self.n_pca = n_pca
        self.n_pls = n_pls
        self.n_gmm = n_gmm
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca, random_state=random_state)
        # PLS is supervised, so we initialize it but fit only if y is provided
        self.pls = PLSRegression(n_components=n_pls, scale=False)
        self.gmm = GaussianMixture(n_components=n_gmm, covariance_type='diag', random_state=random_state)
        self.pls_fitted_ = False

    def fit(self, X, y=None):
        # X should be the embeddings matrix
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.gmm.fit(X_scaled)
        
        if y is not None:
            # y needs to be clean for PLS
            y_clean = y.values if hasattr(y, 'values') else y
            self.pls.fit(X_scaled, y_clean)
            self.pls_fitted_ = True
            
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        features = []
        
        # 1. PCA Features
        f_pca = self.pca.transform(X_scaled)
        features.append(f_pca)
        
        # 2. PLS Features (if fitted)
        if self.pls_fitted_:
            f_pls = self.pls.transform(X_scaled)
            features.append(f_pls)
            
        # 3. GMM Features (Probability)
        f_gmm = self.gmm.predict_proba(X_scaled)
        features.append(f_gmm)
        
        # Combine all features
        return np.hstack(features)