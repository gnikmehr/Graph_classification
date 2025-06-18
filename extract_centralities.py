import json
import os
import glob
import networkx as nx
import time
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import ijson  # Install ijson to enable memory-efficient JSON parsing
import numpy as np
import gc


# Function to process each post
def process_post(post, edge_dict, post_author):
    # Get the author of the main post
    post_author = post.get("author", "").strip()
    # Skip processing if post author is deleted or missing
    if not post_author or post_author == "[deleted]":
        return

    # Process only top-level comments in the post
    for comment in post.get("comments", []):
        commenter = comment.get("author", "").strip()
        
        # Add edge from post author to commenter if commenter is valid
        if commenter and commenter != "[deleted]":
            edge_dict[(commenter, post_author)] = edge_dict.get((commenter, post_author), 0) + 1

# Function to read large JSON files incrementally and process posts
def read_json_files(file_paths, process_func, u):
    edge_dict = {}
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            # Load the entire post JSON as a dictionary
            post = json.load(f)
            if isinstance(post, dict):  # Ensure the data is structured as expected
                # Call process_func on the complete post object
                process_func(post, edge_dict, u)
            else:
                print(f"Invalid structure in file: {file_path}, skipping")
    # print("Finished processing files")
    return edge_dict

def compute_path_counts(graph, weight='weight'):
        path_counts = {}
        for source in graph.nodes():
            # Use try-except to handle cases where a target is not reachable
            try:
                lengths, paths = nx.single_source_dijkstra(graph, source=source, weight=weight)
                path_counts[source] = {}
                for target in graph.nodes():
                    if target in paths:
                        # Count the paths which are equal to the shortest length
                        path_counts[source][target] = len(list(paths[target]))
                    else:
                        path_counts[source][target] = 0
            except nx.NetworkXNoPath:
                # If no path is found, set path count to zero for this target
                path_counts[source] = {target: 0 for target in graph.nodes()}
        return path_counts
def leverage_centrality(graph):
    centrality = {}
    for node in graph.nodes():
        ki = graph.degree(node)
        neighbor_degrees = [(graph.degree(neighbor), neighbor) for neighbor in graph.neighbors(node)]
        cl_sum = 0
        for kj, neighbor in neighbor_degrees:
            cl_sum += (ki - kj) / (ki + kj) if (ki + kj) != 0 else 0
        centrality[node] = cl_sum / ki if ki != 0 else 0
    
    # Normalize the centrality values
    max_centrality = max(centrality.values(), default=1)
    if max_centrality == 0:
        max_centrality = 1  # Avoid division by zero
    for node in centrality:
        centrality[node] /= max_centrality

    return centrality


def neighborhood_centrality(G, p, max_steps):
    centrality = {}
    for start_node in G.nodes():
        sum_centrality = 0
        path_lengths = nx.single_source_shortest_path_length(G, start_node, cutoff=max_steps)
        for node, length in path_lengths.items():
            weight = p ** length
            sum_centrality += weight * G.degree(node)
        centrality[start_node] = sum_centrality
    
    # Normalize centrality values
    max_centrality = max(centrality.values(), default=1)
    if max_centrality == 0:
        max_centrality = 1  # Avoid division by zero
    normalized_centrality = {node: val / max_centrality for node, val in centrality.items()}

    return normalized_centrality

# Function to calculate centralities for a specific user
def calculate_centralities(user_name, base_folder):
    # Prepare graph and read JSON data
    G = nx.DiGraph()  # Directed graph for user interactions
    post_files = glob.glob(os.path.join(base_folder, f'{user_name}/post_*_data.json'))
    comment_files = glob.glob(os.path.join(base_folder, f'{user_name}/comment_*_data.json'))
    

    # Process post and comment files
    edges = read_json_files(post_files, process_post, user_name)
    edges.update(read_json_files(comment_files, process_post, user_name))

    # Add aggregated edges to the graph
    for (u, v), weight in edges.items():
        G.add_edge(u, v, weight=weight)


    # Calculate centralities for the user
    node_name = user_name
    try:
        # Compute global centralities
        degree_centrality = nx.degree_centrality(G)
        in_degree_centrality = {node: G.in_degree(node, weight='weight') / (len(G.nodes) - 1) for node in G.nodes}
        out_degree_centrality = {node: G.out_degree(node, weight='weight') / (len(G.nodes) - 1) for node in G.nodes}
        betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
        closeness_centrality = nx.closeness_centrality(G, distance='weight')
        eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        pagerank = nx.pagerank(G, weight='weight')
        harmonic_centrality = nx.harmonic_centrality(G, distance='weight')
        # Normalize the centrality values
        max_centrality = max(harmonic_centrality.values())
        normalized_harmonic_centrality = {node: centrality / max_centrality for node, centrality in harmonic_centrality.items()}
        clustering_coefficient = nx.clustering(G, weight='weight')
        hits_hub, hits_authority = nx.hits(G)

        eigenvalues = np.linalg.eigvals(nx.to_numpy_array(G))
        phi = max(abs(eigenvalues))
        katz_centrality = nx.katz_centrality(G, 1 / phi - 0.01) #0.05

        leverageCentrality = leverage_centrality(G)

        # # Calculate Neighborhood Centrality
        p = 0.5  # Discount factor
        max_steps = 3  # Maximum steps to consider
        neighborhood = neighborhood_centrality(G, p, max_steps)

        # Extract centralities for the specified nodes
        centralities_for_node = {
            "User": node_name,
            "degree": degree_centrality.get(node_name),
            "in_degree": in_degree_centrality.get(node_name),
            "out_degree": out_degree_centrality.get(node_name),
            "betweenness": betweenness_centrality.get(node_name),
            "closeness": closeness_centrality.get(node_name),
            "eigenvector": eigenvector_centrality.get(node_name),
            "pagerank": pagerank.get(node_name),
            "harmonic": normalized_harmonic_centrality.get(node_name),
            "clustering_coefficient": clustering_coefficient.get(node_name),
            "hub_score": hits_hub.get(node_name),
            "authority_score": hits_authority.get(node_name),

            "katz": katz_centrality.get(node_name).real,
            "leverage": leverageCentrality.get(node_name),
            "neighborhood": neighborhood.get(node_name),

        }

        print(f"Centralities for node '{node_name}' finished")
        return centralities_for_node
    except Exception as e:
        print(f"Error calculating for {user_name}: {e}")
        return None

# Main function to process all users and calculate centralities
def main():
    base_folder = './user_data/'
    
    usernames = []

    print(len(usernames))
    start = time.time()

    # Parallel processing for multiple users
    data_centeralities = Parallel(n_jobs=-1)(delayed(calculate_centralities)(user_name, base_folder) for user_name in tqdm(usernames))
    
    # Filter out any None results due to errors
    data_centeralities = [data for data in data_centeralities if data is not None]

    # Save results to CSV
    df = pd.DataFrame(data_centeralities, columns=['User', 'degree', 'betweenness', 'closeness', 'eigenvector', 'pagerank', 'harmonic', 'clustering_coefficient', 'in_degree',
                                                   'out_degree', 'hub_score', 'authority_score', 'katz', 'leverage','neighborhood'])
    df.to_csv("nodes_centralities.csv", index_label="node")
    # print(df)
    print(f"Processing completed in {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()
