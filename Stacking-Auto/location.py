import pandas as pd
import numpy as np

def extract_enhancer_region_dynamic(probabilities, fragment_size=200, overlap_size=50, min_region_length=3):
    """
    Dynamically determine all enhancer regions and merge nearby low-probability regions.
    
    Parameters:
    probabilities (list): The classification probability values for each fragment.
    fragment_size (int): The size of each fragment, default is 200 base pairs.
    overlap_size (int): The overlap size between fragments, default is 50 base pairs.
    min_region_length (int): The minimum number of consecutive fragments to avoid very small enhancer regions, default is 3.
    
    Returns:
    list: A list of start and end indices for all enhancer regions.
    """
    # Calculate the dynamic threshold
    mean_prob = np.mean(probabilities)
    std_prob = np.std(probabilities)
    threshold = mean_prob - 0.3 * std_prob  # Dynamic threshold
    print(f"Dynamic threshold: {threshold}")

    # Find regions with higher probabilities
    positive_fragments = []
    for idx, prob in enumerate(probabilities):
        if prob > threshold:
            positive_fragments.append(idx)
    
    if not positive_fragments:
        return []  # No fragments found that meet the condition
    
    # Merge consecutive fragments and remove small regions
    enhancer_regions = []
    start = positive_fragments[0]
    end = start
    
    for i in range(1, len(positive_fragments)):
        if positive_fragments[i] == positive_fragments[i - 1] + 1:
            end = positive_fragments[i]
        else:
            if (end - start + 1) >= min_region_length:
                enhancer_regions.append((start, end))
            start = positive_fragments[i]
            end = start
    
    if (end - start + 1) >= min_region_length:
        enhancer_regions.append((start, end))  # Add the last region
    
    return enhancer_regions

def merge_regions(enhancer_regions, fragment_size, overlap_size):
    """
    Merge partially overlapping or adjacent enhancer regions.
    
    Parameters:
    enhancer_regions (list): A list of enhancer regions, each represented as (start_idx, end_idx).
    fragment_size (int): The size of each fragment.
    overlap_size (int): The overlap size between fragments.
    
    Returns:
    list: A list of merged enhancer regions.
    """
    if not enhancer_regions:
        return []

    # Sort by starting index
    enhancer_regions.sort(key=lambda x: x[0])
    merged_regions = []
    current_start, current_end = enhancer_regions[0]

    for start, end in enhancer_regions[1:]:
        # Check for overlap or adjacency
        current_actual_end = fragment_size + current_end * overlap_size
        next_actual_start = start * overlap_size
        if next_actual_start <= current_actual_end:  # Overlapping or adjacent
            current_end = max(current_end, end)
        else:
            merged_regions.append((current_start, current_end))
            current_start, current_end = start, end

    # Add the last region
    merged_regions.append((current_start, current_end))
    return merged_regions

def process_samples(
    probabilities, 
    sample_size=77, 
    fragment_size=200, 
    overlap_size=50, 
    min_region_length=3, 
    verbose=True
):
    """
    Process classification probability values, split by 77 values per sample, 
    calculate enhancer regions for each sample, and output the top 3 highest probability regions for each sample.

    Parameters:
    probabilities (list): The classification probability values for all fragments.
    sample_size (int): The number of probability values per sample, default is 77.
    fragment_size (int): The size of each fragment, default is 200 base pairs.
    overlap_size (int): The overlap size between fragments, default is 50 base pairs.
    min_region_length (int): The minimum number of consecutive fragments to avoid very small enhancer regions, default is 3.
    verbose (bool): Whether to output detailed log information, default is True.

    Returns:
    list: A list of the top 3 enhancer regions for each sample.
    """
    results = []
    num_samples = len(probabilities) // sample_size

    for i in range(num_samples):
        # Get the probabilities for the current sample
        sample_probabilities = probabilities[i * sample_size: (i + 1) * sample_size]
        
        # Extract enhancer regions
        enhancer_regions = extract_enhancer_region_dynamic(
            sample_probabilities, 
            fragment_size=fragment_size, 
            overlap_size=overlap_size, 
            min_region_length=min_region_length
        )
        
        # Merge adjacent or overlapping regions
        merged_regions = merge_regions(enhancer_regions, fragment_size, overlap_size)
        
        # Sort regions by maximum probability within the region and take the top 3
        sorted_regions = sorted(
            merged_regions, 
            key=lambda region: max(sample_probabilities[region[0]:region[1] + 1]), 
            reverse=True
        )
        top_regions = sorted_regions[:3]
        
        # Save enhancer region information for each sample
        for region in top_regions:
            start_index, end_index = region
            actual_start = start_index * overlap_size
            actual_end = fragment_size + end_index * overlap_size
            
            results.append({
                'sample_index': i,
                'start': actual_start,
                'end': actual_end,
                'start_idx': start_index,
                'end_idx': end_index
            })
    
    if verbose:
        if results:
            print("All enhancer regions:")
            for region in results:
                print(f"Sample {region['sample_index']}:")
                print(f"  Enhancer Region (Position): {region['start']} to {region['end']}")
                print(f"  Enhancer Region (Index): {region['start_idx']} to {region['end_idx']}")
                print("---")
        else:
            print("No enhancer regions found.")
    
    return results

# Read the CSV file
csv_file = "predictions_with_proba.csv"  # Replace with the actual file path
data = pd.read_csv(csv_file)

# Extract the probability column, assuming the probability column name is '1'
probabilities = data['1'].values

# Process every 77 classification probability values as one sample and output the top 3 enhancer regions for each sample
enhancer_regions = process_samples(probabilities, sample_size=77, fragment_size=200, overlap_size=50)
