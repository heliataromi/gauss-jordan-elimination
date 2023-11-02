import numpy as np


def gauss_jordan_elimination(A, b):
    """
    This function performs the Gauss-Jordan elimination method to solve a system of linear equations of the form Ax = b.

    Parameters:
    A (numpy array): The coefficient matrix of the system.
    b (numpy array): The constant matrix of the system, must have the same number of rows as A.

    Returns:
    x (numpy array): The solution matrix of the system, or None if the system is inconsistent or has infinitely many solutions.
    """

    # Check the dimensions of the matrices
    if A.shape[0] != b.shape[0]:
        print("The Coefficient Matrix dimensions do not match the Constant Matrix dimensions.")

    # Create the augmented matrix [A | b]
    augmented_matrix = np.hstack((A, b))

    # Find the number of equations
    n = augmented_matrix[:, :-1].shape[0]

    # Perform row operations to transform the augmented matrix into [I | x]
    for k in range(n):

        # Check if the matrix has an infinity of solutions
        if A.shape[1] > n:
            print("The system has an infinity of solutions.")
            return None
        if np.all(augmented_matrix[k] == 0):
            if np.count_nonzero(np.any(augmented_matrix, axis=1)) != augmented_matrix[:, :-1].shape[1]:
                print("The system has an infinity of solutions.")
                return None
            else:
                augmented_matrix = np.delete(augmented_matrix, k, 0)
                break

        # Check if the matrix is inconsistent
        if np.all(augmented_matrix[:, :-1][k] == 0) and augmented_matrix[k][-1] != 0:
            print("The system is inconsistent and isn't solvable.")
            return None

        # Normalize the pivot row
        if augmented_matrix[k, k] != 0:
            print(f'(1/{augmented_matrix[k, k]}) × R{k+1} → R{k+1}')
            augmented_matrix[k] = augmented_matrix[k] / augmented_matrix[k, k]
            print(augmented_matrix, '\n')

            # Eliminate the entries below and above the pivot
            for i in range(n):
                if i != k:
                    print(f'({-augmented_matrix[i, k]}) × R{k+1} + R{i+1} → R{i+1}')
                    augmented_matrix[i] = augmented_matrix[i] - augmented_matrix[k] * augmented_matrix[i, k]
                    print(augmented_matrix, '\n')

    # Extract the solution matrix x from the right part of the augmented matrix
    x = augmented_matrix[:, -1:]

    # Return the answer
    return x
