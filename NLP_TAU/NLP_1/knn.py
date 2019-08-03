import numpy as np

def get_cosine(vector,matrix):

    norm = np.linalg.norm(matrix,axis=1)
    matrix = matrix.view(np.ndarray)
    matrix = (matrix.T / norm).T
    vector = vector / np.linalg.norm(vector)
    return np.dot(matrix,vector)


def knn(vector, matrix, k=10):
    """
    Finds the k-nearest rows in the matrix with comparison to the vector.
    Use the cosine similarity as a distance metric.

    Arguments:
    vector -- A D dimensional vector
    matrix -- V x D dimensional numpy matrix.

    Return:
    nearest_idx -- A numpy vector consists of the rows indices of the k-nearest neighbors in the matrix
    """

    nearest_idx = []

    ### YOUR CODE HERE
    knn = get_cosine(vector,matrix)
    nearest_idx = np.argsort(-knn)[:k]
    ### END YOUR CODE
    return nearest_idx

def test_knn():
    """
    Use this space to test your knn implementation by running:
        python knn.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    mat = np.matrix([[1,2],[-1,7],[4,9],[2,3],[-1,-2]])
    vec = np.array([1,2])
    knn(vec,mat,3)
    ### END YOUR CODE

if __name__ == "__main__":
    test_knn()


