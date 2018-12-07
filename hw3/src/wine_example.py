from kmeans import Kmeans

# Counts of clusters
K = 3
# The path of dataset file
DATA_PATH = './wine.data'
# The delimiter used in dataset file
DELIMITER = ','
# The columns to use in dataset file
USECOLS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

if __name__ == '__main__':
    # Create a new Kmeans object by giving a K
    kmeans = Kmeans(K)
    # Load dataset file with the path and read the columns split by the delimiter
    kmeans.loadtxt(DATA_PATH, delimiter=DELIMITER, usecols=USECOLS)
    # Calculate with the Kmeans algorithm
    kmeans.run()
    # Draw the diagram
    kmeans.show()
