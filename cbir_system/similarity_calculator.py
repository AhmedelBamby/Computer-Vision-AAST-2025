"""
Similarity calculation module for CBIR system
Implements different distance metrics: Cosine Similarity, Manhattan Distance, Euclidean Distance
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from pathlib import Path

class SimilarityCalculator:
    """Calculate similarity between image features using different metrics"""
    
    def __init__(self):
        self.available_methods = {
            'cosine': self.cosine_similarity,
            'euclidean': self.euclidean_distance,
            'manhattan': self.manhattan_distance
        }
    
    def cosine_similarity(self, query_features, database_features):
        """
        Calculate cosine similarity between query and database features
        
        Args:
            query_features: numpy array of query image features
            database_features: numpy array of database image features
            
        Returns:
            float: cosine similarity score (higher is more similar)
        """
        # Reshape features to 2D arrays for sklearn
        query_features = query_features.reshape(1, -1)
        database_features = database_features.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(query_features, database_features)[0][0]
        return similarity
    
    def euclidean_distance(self, query_features, database_features):
        """
        Calculate Euclidean distance between query and database features
        
        Args:
            query_features: numpy array of query image features
            database_features: numpy array of database image features
            
        Returns:
            float: euclidean distance (lower is more similar)
        """
        distance = euclidean(query_features, database_features)
        return distance
    
    def manhattan_distance(self, query_features, database_features):
        """
        Calculate Manhattan distance between query and database features
        
        Args:
            query_features: numpy array of query image features
            database_features: numpy array of database image features
            
        Returns:
            float: manhattan distance (lower is more similar)
        """
        distance = manhattan(query_features, database_features)
        return distance
    
    def normalize_features(self, features):
        """
        Normalize features to have zero mean and unit variance
        
        Args:
            features: numpy array of features
            
        Returns:
            numpy array: normalized features
        """
        if np.std(features) == 0:
            return features
        return (features - np.mean(features)) / np.std(features)
    
    def find_similar_images(self, query_features, features_database, method='cosine', top_k=3, normalize=True):
        """
        Find top-k most similar images from the database
        
        Args:
            query_features: numpy array of query image features
            features_database: dict containing image names and their features
            method: similarity method ('cosine', 'euclidean', 'manhattan')
            top_k: number of top similar images to return
            normalize: whether to normalize features before comparison
            
        Returns:
            list: list of tuples (image_name, similarity_score) sorted by similarity
        """
        if method not in self.available_methods:
            raise ValueError(f"Method {method} not available. Choose from {list(self.available_methods.keys())}")
        
        similarity_func = self.available_methods[method]
        similarities = []
        
        # Normalize query features if requested
        if normalize:
            query_features = self.normalize_features(query_features)
        
        for image_name, image_features in features_database.items():
            try:
                # Use combined features for comparison
                db_features = image_features['combined']
                
                # Normalize database features if requested
                if normalize:
                    db_features = self.normalize_features(db_features)
                
                # Calculate similarity/distance
                score = similarity_func(query_features, db_features)
                similarities.append((image_name, score))
                
            except Exception as e:
                print(f"Error calculating similarity for {image_name}: {e}")
                continue
        
        # Sort results based on method
        if method == 'cosine':
            # For cosine similarity, higher is better
            similarities.sort(key=lambda x: x[1], reverse=True)
        else:
            # For distances, lower is better
            similarities.sort(key=lambda x: x[1])
        
        return similarities[:top_k]
    
    def calculate_all_similarities(self, query_features, features_database, normalize=True):
        """
        Calculate similarities using all available methods
        
        Args:
            query_features: numpy array of query image features
            features_database: dict containing image names and their features
            normalize: whether to normalize features before comparison
            
        Returns:
            dict: results for each method
        """
        results = {}
        
        for method_name in self.available_methods.keys():
            try:
                similarities = self.find_similar_images(
                    query_features, 
                    features_database, 
                    method=method_name, 
                    normalize=normalize
                )
                results[method_name] = similarities
            except Exception as e:
                print(f"Error with method {method_name}: {e}")
                results[method_name] = []
        
        return results

class CBIRSystem:
    """Complete CBIR system combining feature extraction and similarity calculation"""
    
    def __init__(self, features_db_path):
        """
        Initialize CBIR system
        
        Args:
            features_db_path: path to the features database file
        """
        self.similarity_calculator = SimilarityCalculator()
        self.features_database = self.load_features_database(features_db_path)
        
    def load_features_database(self, features_db_path):
        """Load features database from file"""
        try:
            features_db = np.load(features_db_path, allow_pickle=True).item()
            print(f"Loaded features database with {len(features_db)} images")
            return features_db
        except Exception as e:
            print(f"Error loading features database: {e}")
            return {}
    
    def query_image(self, query_features, selected_methods=None, top_k=3):
        """
        Query the database with image features
        
        Args:
            query_features: numpy array of query image features
            selected_methods: list of methods to use (if None, use all)
            top_k: number of top results to return
            
        Returns:
            dict: results for each selected method
        """
        if not self.features_database:
            return {}
        
        if selected_methods is None:
            selected_methods = list(self.similarity_calculator.available_methods.keys())
        
        results = {}
        for method in selected_methods:
            if method in self.similarity_calculator.available_methods:
                similarities = self.similarity_calculator.find_similar_images(
                    query_features, 
                    self.features_database, 
                    method=method, 
                    top_k=top_k
                )
                results[method] = similarities
        
        return results
    
    def get_available_methods(self):
        """Get list of available similarity methods"""
        return list(self.similarity_calculator.available_methods.keys())
    
    def get_database_image_names(self):
        """Get list of all images in the database"""
        return list(self.features_database.keys())

if __name__ == "__main__":
    # Test the similarity calculator
    print("Testing similarity calculation...")
    
    # Load features database
    features_db_path = "features/features_db.npy"
    
    if Path(features_db_path).exists():
        cbir_system = CBIRSystem(features_db_path)
        print(f"Available methods: {cbir_system.get_available_methods()}")
        print(f"Database images: {cbir_system.get_database_image_names()}")
    else:
        print("Features database not found. Please run feature extraction first.")