from kmeans import Kmeans

# Counts of clusters
K = 3
# The path of dataset file
DATA_PATH = './iris.data'
# The delimiter used in dataset file
DELIMITER = ','
# The columns to use in dataset file
USECOLS = (0, 1, 2, 3)

if __name__ == '__main__':
    # Create a new Kmeans object by giving a K
    kmeans = Kmeans(K)
    # Load dataset file with the path and read the columns split by the delimiter
    kmeans.loadtxt(DATA_PATH, delimiter=DELIMITER, usecols=USECOLS)
    # Calculate with the Kmeans algorithm
    kmeans.run()
    # Draw the diagram
    kmeans.show()
