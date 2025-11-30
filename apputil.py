#-----IMPORTS-----#
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from time import perf_counter
from typing import Tuple
from tqdm import tqdm

#-----DATASET LOAD-----#
#load diamonds dataset from seaborn
diamonds_database = sns.load_dataset("diamonds")

#select only columns that have numbers (carat, depth, table etc.)
NUMERIC_DIAMONDS = diamonds_database.select_dtypes(include=[np.number]).copy()


#-----FUNCTION DEFS-----#
#---EXERCISE #1---#
def kmeans(X: np.ndarray, k: int, *, random_state: int = 0, n_init: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform k-means clustering on a numerical numpy array

    Function Parameters:
    X : np.ndarray (2D numeric array of shape (n_samples, n_features) containing data points)
    k : int (mumber of clusters to form.)
    random_state : int, optional (random seed for reproducibility (default = 0))
    n_init : int, optional (number of initializations for scikit-learn’s KMeans (default = 10))

    Function returns:
    retruns centroids and labaels
    """


    #make sure that X is a numpy array
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    #also that X is also 2d
    if X.ndim != 2:
        raise ValueError("X must be a 2D numpy array (n_samples, n_features).")

    #create KMeans model with specified number of clusters and settings. Using randome
    model = KMeans(n_clusters = k, random_state = random_state, n_init = n_init)
    
    #fit model to the dataset
    model.fit(X)

    #extract the cluster centroid points and store
    centroids = model.cluster_centers_

    #extract the label assignments for each data sample and store too
    labels = model.labels_

    #return both extracted centroid points, and labels
    return centroids, labels


#---EXERCISE #2---#
def kmeans_diamonds(n: int, k: int, *, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run k-means clustering on the first n rows of the daimonds dataset

    Function parameters:
    n : int (number of rows (samples) from the dataset to use)
    k : int (number of clusters to form)
    random_state : int (random seed for reproducibility (default = 0))

    Function returns:
    return of shapes for centroids and labels
    """


    #ensure n is valid. n must both be positive and less than the length of column vector
    if n <= 0:
        raise ValueError("n must be positive.")
    if n > len(NUMERIC_DIAMONDS):
        raise ValueError(f"n cannot exceed {len(NUMERIC_DIAMONDS)}.")

    #convert the first n rows to numpy form
    X = NUMERIC_DIAMONDS.iloc[:n].to_numpy()

    #run kmeans on it
    return kmeans(X, k, random_state = random_state)


#---EXERCISE #3---#
def kmeans_timer(n: int, k: int, n_iterations: int = 5) -> float:
    """
    Run kmeans_diamonds(n, k) multiple times and return the average runtime in seconds

    Function parameters:
    n : int number of rows (samples) from the dataset to use
    k : int number of clusters to form
    n_iterations : int number of times to repeat the clustering for timing (default = 5)

    function returns:
    float the average runtime across all runs, measured in seconds
    """

    #validate iteration count
    if n_iterations <= 0:
        raise ValueError("Iterations of n must be positive.")

    #store the durations for each repeat
    durations = []

    #repeat the clustering n_iterations times
    for i in range(n_iterations):

        #timer start
        start = perf_counter()

        #run kmeans on diamonds using a random seed
        _ = kmeans_diamonds(n, k, random_state = i)

        # Stop timer and record duration
        durations.append(perf_counter() - start)

    #calculate and return average runtime
    return float(np.mean(durations))




#i am not using the notebook or the github codespaces so i had to figure out a way to basically recreate what the cells do
#but make it so that it works in visual studio on linux. I am hoping that this works with the autograder and fullfills the requirements
#that you said, meaning you can see the output graph
#----------------------------------------------------------------------------------------|
#---OUTPUT GRAPHS---#
def kmeans_performance_plots():
    """
    This function will output plots so that the TA can see the output of the notebooks cells. I am working in vsc on linux
    so the exact code in the notebook will not work and just breaks so I basically just made a new function to do the exact same thing.
    Hopefully the autograder will not break. Also I added a progress bar that appears on the terminal when I was debugging and 
    now I will just leave it because it looks cool.
    """
    sns.set_theme(style="whitegrid")

    #first graph
    n_values = np.arange(100, 20000, 2000)#lower iterations thsi is taking too long
    k5_times = [kmeans_timer(n, 5, 10) for n in tqdm(n_values, desc="Varying n (k=5)")]

    #second graph
    k_values = np.arange(2, 30)
    n10k_times = [kmeans_timer(5000, k, 5) for k in tqdm(k_values, desc="Varying k (n=10000)")]

    #create the plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("KMeans Time Complexity", y=1.05, fontsize=14)

    sns.lineplot(x=n_values, y=k5_times, ax=axes[0])
    axes[0].set_xlabel("Number of Rows (n)")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title("Increasing n for k = 5 Clusters")

    sns.lineplot(x=k_values, y=n10k_times, ax=axes[1])
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_title("Increasing k for n = 10k Samples")

    plt.tight_layout()
    plt.show()

    #confirmation output
    print("Performance plots generated.")
#----------------------------------------------------------------------------------------|



#-----MAIN-----#
if __name__ == "__main__":
    #main to trigger the plot creation function so that the TA can see the plots and grade them    
    print("\nGenerating plots, please hold this may take a bit...")
    #trigger the plot creation function
    kmeans_performance_plots()




"""

#<-----ONLY FOR DEBUGGING----->#



#-----MAIN-----#
def main():
    print("TEST: apputil.py output")
    print("-------------------------------------------")

    #test dataset
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    print("\nRunning test kmeans(X, k = 3)")
    centroids_1, labels_1 = kmeans(X, k = 3)
    print("Centroids:")
    print("Labels:")
    print(centroids_1)
    print(labels_1)

    #test with diamonds
    print("\nRunning test kmeans_diamonds(n = 1000, k = 5)")
    centroids_2, labels_2 = kmeans_diamonds(1000, 5)
    print("Centroids shape:", centroids_2.shape)
    print("Example centroid (rounded):", np.round(centroids_2[0], 3))
    print("Sample labels:", labels_2[:10])


    #timing test
    print("\nTiming test kmeans on diamond")
    average_time = kmeans_timer(n = 500, k = 3, n_iterations = 3)
    print(f"Average runtime over 3 runs: {average_time:.4f} seconds")


if __name__ == "__main__":
    main()

"""

