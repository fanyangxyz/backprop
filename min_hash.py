import random
from typing import List, Set, Callable
import hashlib

class MinHash:
    def __init__(self, num_permutations: int = 100):
        """
        Initialize MinHash with specified number of hash functions
        
        Args:
            num_permutations: Number of hash functions to use
        """
        self.num_permutations = num_permutations
        self.hash_functions = self._generate_hash_functions()
        
    def _generate_hash_functions(self) -> List[Callable]:
        """Generate list of hash functions using different random seeds"""
        hash_functions = []
        for i in range(self.num_permutations):
            def hash_func(x, seed=i):
                return int(hashlib.md5(f"{seed}{x}".encode()).hexdigest(), 16)
            hash_functions.append(hash_func)
        return hash_functions
    
    def compute_signature(self, input_set: Set[str]) -> List[int]:
        """
        Compute MinHash signature for a set
        
        Args:
            input_set: Set of strings to compute signature for
            
        Returns:
            List of minimum hash values (signature)
        """
        signature = []
        for hash_func in self.hash_functions:
            min_hash = float('inf')
            for item in input_set:
                hash_value = hash_func(item)
                min_hash = min(min_hash, hash_value)
            signature.append(min_hash)
        return signature
    
    @staticmethod
    def estimate_similarity(sig1: List[int], sig2: List[int]) -> float:
        """
        Estimate Jaccard similarity between two sets using their MinHash signatures
        
        Args:
            sig1: MinHash signature of first set
            sig2: MinHash signature of second set
            
        Returns:
            Estimated Jaccard similarity
        """
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have equal length")
        
        matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
        return matches / len(sig1)

# Example usage
if __name__ == "__main__":
    # Create example sets
    set1 = {"apple", "banana", "orange", "pear", "grape"}
    set2 = {"apple", "banana", "orange", "kiwi", "melon"}
    
    # Initialize MinHash
    minhash = MinHash(num_permutations=100)
    
    # Compute signatures
    sig1 = minhash.compute_signature(set1)
    sig2 = minhash.compute_signature(set2)
    
    # Estimate similarity
    estimated_similarity = minhash.estimate_similarity(sig1, sig2)
    
    # Calculate actual Jaccard similarity for comparison
    actual_similarity = len(set1 & set2) / len(set1 | set2)
    
    print(f"Estimated similarity: {estimated_similarity:.3f}")
    print(f"Actual similarity: {actual_similarity:.3f}")
