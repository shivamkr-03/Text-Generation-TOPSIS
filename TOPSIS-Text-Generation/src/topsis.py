import numpy as np

def normalize_matrix(matrix):
    norm = np.sqrt((matrix ** 2).sum(axis=0))
    return matrix / norm

def apply_weights(normalized_matrix, weights):
    return normalized_matrix * weights

def calculate_ideal_solutions(weighted_matrix, impacts):
    ideal_best = []
    ideal_worst = []

    for i in range(len(impacts)):
        if impacts[i] == "+":
            ideal_best.append(weighted_matrix[:, i].max())
            ideal_worst.append(weighted_matrix[:, i].min())
        else:
            ideal_best.append(weighted_matrix[:, i].min())
            ideal_worst.append(weighted_matrix[:, i].max())

    return np.array(ideal_best), np.array(ideal_worst)

def calculate_scores(weighted_matrix, ideal_best, ideal_worst):
    distance_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    scores = distance_worst / (distance_best + distance_worst)
    return scores
