import torch
import torch.nn as nn
import numpy as np
from sklearn.covariance import LedoitWolf
from tqdm import tqdm

class DynaDetectAnomalyScorer:
    """
    Implements DynaDetect2.0 latent space anomaly scoring using Mahalanobis distance.
    This module extracts features from a CNN and computes the distance of each sample 
    to the class-conditional distribution of features.
    """
    def __init__(self, model, num_classes):
        self.model = model
        self.num_classes = num_classes
        self.class_means = {}
        self.class_cov_inv = {}
        
    def fit_distributions(self, dataloader, device):
        """
        Extracts features from the entire dataset and computes the mean 
        and inverse covariance matrix for each class.
        """
        self.model.eval()
        features_dict = {i: [] for i in range(self.num_classes)}
        
        print("Extracting features for Mahalanobis distribution modeling...")
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Feature Extraction"):
                inputs = inputs.to(device)
                # Use the extract_features method defined in our ResNet modification
                features = self.model.extract_features(inputs)
                
                for i in range(len(targets)):
                    if targets[i].dim() > 0:
                        active_classes = torch.where(targets[i] > 0.5)[0].tolist()
                        for c in active_classes:
                            features_dict[c].append(features[i].cpu().numpy())
                    else:
                        label = targets[i].item()
                        features_dict[label].append(features[i].cpu().numpy())
                    
        print("Computing class means and inverse covariance matrices...")
        for c in range(self.num_classes):
            if len(features_dict[c]) == 0:
                continue
                
            class_features = np.stack(features_dict[c])
            self.class_means[c] = np.mean(class_features, axis=0)
            
            # Compute empirical covariance and its inverse
            cov_estimator = LedoitWolf(assume_centered=False)
            cov_estimator.fit(class_features)
            
            # Use pseudo-inverse or add small ridge penalty for numerical stability
            try:
                self.class_cov_inv[c] = np.linalg.inv(cov_estimator.covariance_)
            except np.linalg.LinAlgError:
                # Add ridge penalty if singular
                reg_cov = cov_estimator.covariance_ + np.eye(cov_estimator.covariance_.shape[0]) * 1e-6
                self.class_cov_inv[c] = np.linalg.inv(reg_cov)

    def score_samples(self, dataloader, device):
        """
        Computes the Mahalanobis distance for each sample in the dataloader.
        Returns a list of distances and corresponding true labels.
        """
        self.model.eval()
        all_distances = []
        all_labels = []
        all_indices = [] # Assuming dataloader is un-shuffled to track exact dataset index
        
        current_idx = 0
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Anomaly Scoring"):
                inputs = inputs.to(device)
                features = self.model.extract_features(inputs).cpu().numpy()
                
                for i in range(len(targets)):
                    feat = features[i]
                    
                    if targets[i].dim() > 0:
                        active_classes = torch.where(targets[i] > 0.5)[0].tolist()
                        dists = []
                        for c in active_classes:
                            if c in self.class_means and c in self.class_cov_inv:
                                diff = feat - self.class_means[c]
                                dists.append(np.sqrt(np.dot(np.dot(diff, self.class_cov_inv[c]), diff.T)))
                        dist = np.mean(dists) if dists else float('inf')
                        label = active_classes[0] if active_classes else 0
                    else:
                        label = targets[i].item()
                        if label in self.class_means and label in self.class_cov_inv:
                            diff = feat - self.class_means[label]
                            dist = np.sqrt(np.dot(np.dot(diff, self.class_cov_inv[label]), diff.T))
                        else:
                            dist = float('inf')
                        
                    all_distances.append(dist)
                    all_labels.append(label)
                    all_indices.append(current_idx)
                    current_idx += 1
                    
        return np.array(all_distances), np.array(all_labels), np.array(all_indices)

    def get_flagged_indices(self, distances, threshold_percentile=95):
        """
        Flags samples whose distance exceeds a given percentile threshold.
        These are the samples suspected of being poisoned.
        """
        threshold = np.percentile(distances, threshold_percentile)
        flagged_indices = np.where(distances > threshold)[0]
        return flagged_indices
