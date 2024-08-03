import numpy as np


# Funkcja do wyświetlania wzorca w postaci macierzy, wyjustowanego zeby lepiej widać litere
def print_pattern(letter_pattern):
    matrix_pattern = letter_pattern.reshape(5, 7)
    for row in matrix_pattern:
        print(" ".join(map(lambda x: str(x).rjust(2), row)))


# Inicjalizacja wag na podstawie dostarczonych wzorców
def initialize_weights(patterns):
    pattern_size = len(patterns[0])
    weights = np.zeros((pattern_size, pattern_size))

    # Iteracja przez wzorce i tworzenie macierzy wag
    for pattern in patterns:
        # Tworzenie macierzy i ustawianie przekątnej na 0
        pattern_matrix = np.outer(pattern, pattern)
        np.fill_diagonal(pattern_matrix, 0)
        weights += pattern_matrix

    # Normalizacja macierzy wag
    return weights / pattern_size


# Aktualizacja stanu neuronów w sieci Hopfielda
def update_neurons(inputs, weights):
    net = np.dot(weights, inputs)
    # Aktualizacja wyjść na podstawie funkcji skokowej
    outputs = np.where(net > 0, 1, np.where(net == 0, inputs, -1))
    return outputs


# Implementacja algorytmu Hopfielda
def hopfield_network(input_pattern, weights, max_iterations=100, convergence_threshold=5):
    current_state = input_pattern.copy()
    consecutive_stable_iterations = 0

    # Iteracja przez maksymalną liczbę iteracji
    for iteration in range(max_iterations):
        new_state = update_neurons(current_state, weights)

        # Sprawdzenie, czy stan się ustabilizował
        if np.array_equal(current_state, new_state):
            consecutive_stable_iterations += 1
        else:
            consecutive_stable_iterations = 0

        current_state = new_state
        print(f"Iteration {iteration + 1} - Current State: ")
        print_pattern(current_state)

        # Przerwanie pętli, jeśli osiągnięto zbieżność
        if consecutive_stable_iterations == convergence_threshold:
            break

    return current_state


# Zakłócanie wzorca z zadanym współczynnikiem zakłóceń
def distort_pattern(pattern, distortion_ratio):
    distorted_pattern = pattern.copy()
    num_distorted_bits = int(distortion_ratio * len(pattern))
    # Losowe wybieranie indeksów do zakłócenia
    indices_to_distort = np.random.choice(len(pattern), num_distorted_bits, replace=False)
    distorted_pattern[indices_to_distort] *= -1
    return distorted_pattern


# Przykładowe wzorce
pattern_A = np.array([1,  1,  1,  1,  1,  1, 1,
                      1, -1, -1, -1, -1, -1, 1,
                      1, -1, -1, -1, -1, -1, 1,
                      1,  1,  1,  1,  1,  1, 1,
                      1, -1, -1, -1, -1, -1, 1])

pattern_C = np.array([1,  1,  1,  1,  1,  1,  1,
                      1, -1, -1, -1, -1, -1, -1,
                      1, -1, -1, -1, -1, -1, -1,
                      1, -1, -1, -1, -1, -1, -1,
                      1,  1,  1,  1,  1,  1,  1])

pattern_X = np.array([1, -1, -1, -1, -1, -1,  1,
                     -1,  1, -1, -1, -1,  1, -1,
                     -1, -1,  1,  1,  1, -1, -1,
                     -1,  1, -1, -1, -1,  1, -1,
                      1, -1, -1, -1, -1, -1,  1])

pattern_I = np.array([1,  1,  1,  1,  1,  1,  1,
                     -1, -1,  1,  1,  1, -1, -1,
                     -1, -1,  1,  1,  1, -1, -1,
                     -1, -1,  1,  1,  1, -1, -1,
                      1,  1,  1,  1,  1,  1,  1])

patterns = [pattern_A, pattern_C, pattern_X, pattern_I]

# Inicjalizacja wag
weights = initialize_weights(patterns)

# Zakłócenie wzorca
distorted_pattern = distort_pattern(pattern_A, distortion_ratio=0.2)

print("\nOriginal Pattern:")
print_pattern(pattern_A)
print("\nDistorted Pattern:")
print_pattern(distorted_pattern)

# Rekonstrukcja wzorca z zakłóconego wejścia
reconstructed_pattern = hopfield_network(distorted_pattern, weights)

print("\nReconstructed Pattern:")
print_pattern(reconstructed_pattern)
