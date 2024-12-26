import glob
import json
import openai
from tqdm import tqdm
import numpy as np
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(list1, list2):
    # Convert lists to numpy arrays for vector operations
    vector1 = np.array(list1)
    vector2 = np.array(list2)

    # Compute cosine similarity
    similarity = dot(vector1, vector2) / (norm(vector1) * norm(vector2))
    return similarity

def batch_cosine_similarity(query_vector, item_matrix):
    # Convert query vector to numpy array
    query_vector = np.array(query_vector)
    
    # Compute dot products for all items at once
    dot_products = np.dot(item_matrix, query_vector)
    
    # Compute norms
    item_norms = np.linalg.norm(item_matrix, axis=1)  # Need norm for each item
    query_norm = np.linalg.norm(query_vector)         # Only one query norm needed
    
    # Compute similarities
    similarities = dot_products / (item_norms * query_norm)
    return similarities

def hyperbolic_tangent(list1, list2):
    # Convert lists to numpy arrays for vector operations
    vector1 = np.array(list1)
    vector2 = np.array(list2)
    
    # Compute cosine similarity
    similarity = dot(vector1, vector2)
    result = np.tanh(similarity)
    return result

def dot_product_similarity(list1, list2):
    # Convert lists to numpy arrays for vector operations
    vector1 = np.array(list1)
    vector2 = np.array(list2)
    
    # Compute cosine similarity
    similarity = dot(vector1, vector2)
    return similarity

def softmax(total, target):
    numerator = 0
    for value in total:
        numerator += np.exp(value)
    denominator = np.exp(target)
    return denominator / numerator

def calculate_rank(lst, value):
    sorted_list = sorted(lst)
    rank = sorted_list.index(value) + 1 if value in sorted_list else None
    return rank
                
# get the number of additional turns and its id
def get_additional_turn_num(sample_result_path):
    sample_result = json.load(open(sample_result_path))
    original_turn_num = sample_result['turn_id'] / 2
    extended_turn_num = len(sample_result['simulator_dialog']['context']) / 2
    # extended_turn_num = len(sample_result['simulator_dialog']) / 2
    
    return f'{sample_result["dialog_id"]}_{sample_result["turn_id"]}', extended_turn_num - original_turn_num

def get_result(sample_result_path):
    sample_result = json.load(open(sample_result_path))
    dialog = sample_result['simulator_dialog']['context']
    # dialog = sample_result['simulator_dialog']
    for idx, turn in enumerate(dialog[::-1]):
        if 'rec_success' in turn:
            rec_success = turn['rec_success']
            return f'{sample_result["dialog_id"]}_{sample_result["turn_id"]}', rec_success, idx