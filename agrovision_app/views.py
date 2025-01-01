

from django.shortcuts import render
from .forms import ClimateDataForm, CropRecommendationForm
from .models import ClimateData, CropRecommendation
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime
def create_graph_from_data(data):
    G = nx.Graph()
    for index, row in data.iterrows():
        G.add_node(row['climate'], type='Climate Label')

    # Add edges between climate data points (date information)
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            date1 = datetime.strptime(data.iloc[i]['date'], '%d-%m-%Y')
            date2 = datetime.strptime(data.iloc[j]['date'], '%d-%m-%Y')
            delta = date2 - date1
            G.add_edge(data.iloc[i]['climate'], data.iloc[j]['climate'], type='Date Connection', weight=delta.days)

    return G
# Define the graph outside the function
climate_data = pd.read_csv('datasets/climate.csv').sample(20)
G = create_graph_from_data(climate_data)
random_seed = 42
pos = nx.spring_layout(G, seed=random_seed)

def graph_operations(request, G=G, pos=pos):
    operation = request.GET.get('operation')
    if operation == 'spanning_tree':
        T = nx.random_spanning_tree(G)
        draw_graph_with_weights(T, "Spanning Tree", pos)
    elif operation == 'prim_algorithm':
        T = nx.minimum_spanning_tree(G, algorithm='prim')
        draw_graph_with_weights(T, "Prim's Algorithm Minimum Spanning Tree", pos)
    elif operation == 'kruskal_algorithm':
        T = nx.minimum_spanning_tree(G, algorithm='kruskal')
        draw_graph_with_weights(T, "Kruskal's Algorithm Minimum Spanning Tree", pos)
    elif operation == 'distance_centrality':
        centrality = nx.closeness_centrality(G)
        return render(request, 'agrovision_app/graph_operations.html', {'centrality': centrality})
    elif operation == 'isomorphic_graphs':
        crop_recommendation_data = pd.read_csv('datasets/crop_recommendation.csv').sample(20)
        crop_labels = list(crop_recommendation_data['label'])
        crop_graph = nx.Graph()
        for label in crop_labels:
            crop_graph.add_node(label, type='Crop Label')
        is_isomorphic = nx.is_isomorphic(G, crop_graph)
        return render(request, 'agrovision_app/graph_operations.html', {'is_isomorphic': is_isomorphic})
    elif operation == 'incidence_matrix':
        incidence_mat = nx.incidence_matrix(G, oriented=True).toarray()
        vertex_labels = sorted(G.nodes())
        edge_labels = [f"Edge {i + 1}" for i in range(len(G.edges()))]  # Generate edge labels
        incidence_df = pd.DataFrame(incidence_mat, index=vertex_labels, columns=edge_labels)
        draw_graph_with_weights(G, "Connected Graph of Climate Labels", pos)
        return render(request, 'agrovision_app/graph_operations.html', {'incidence_matrix': incidence_df})

    elif operation == 'adjacency_matrix':
        adjacency_mat = nx.adjacency_matrix(G).toarray()
        vertex_labels = sorted(G.nodes())
        adjacency_df = pd.DataFrame(adjacency_mat, index=vertex_labels, columns=vertex_labels)
        draw_graph_with_weights(G, "Connected Graph of Climate Labels", pos)
        return render(request, 'agrovision_app/graph_operations.html', {'adjacency_matrix': adjacency_df})
    elif operation == 'cutsets':
        cut_sets = nx.minimum_node_cut(G)
        # Create a new graph after removing vertices from cut set
        G_cut = G.copy()
        for node in cut_sets:
            G_cut.remove_node(node)
        draw_graph_with_weights(G_cut, "Graph after Removing Cut Set", pos)
        return render(request, 'agrovision_app/graph_operations.html', {'cut_sets': cut_sets})
    else:
        draw_graph_with_weights(G, "Connected Graph of Climate Labels", pos)

    return render(request, 'agrovision_app/graph_operations.html')

# Remaining functions remain unchanged...

def draw_graph_with_weights(G, title, pos):
    plt.figure(figsize=(10, 6))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Draw labels
    nx.draw_networkx_labels(G, pos)

    # Draw edge weights if the 'weight' attribute exists
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        if 'weight' in d:
            edge_labels[(u, v)] = d['weight']
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Add number of vertices
    num_vertices = len(G.nodes)
    plt.text(0.5, 0.95, f"Number of Vertices: {num_vertices}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.title(title)
    plt.show()


def draw_colored_graph(G, title, pos):
    plt.figure(figsize=(10, 6))

    # Initialize colors for nodes
    node_colors = {node: None for node in G.nodes}

    def is_valid_color(node, color):
        """Check if the color is valid for the given node."""
        for neighbor in G.neighbors(node):
            if node_colors[neighbor] == color:
                return False
        return True

    def get_valid_color(node, used_colors):
        """Get a valid color for the given node."""
        for color in range(1, len(used_colors) + 2):
            if color not in used_colors and is_valid_color(node, color):
                return color

    def assign_color(node):
        """Assign color to the given node."""
        neighbors = list(G.neighbors(node))
        used_colors = {node_colors[neighbor] for neighbor in neighbors if node_colors[neighbor] is not None}
        valid_color = get_valid_color(node, used_colors)
        node_colors[node] = valid_color

    # Determine colors for nodes
    for node in G.nodes:
        assign_color(node)

    # Draw nodes with colors
    nx.draw(G, pos, with_labels=True, node_size=700, node_color=[node_colors[node] for node in G.nodes], cmap=plt.cm.rainbow, vmin=0, vmax=max(node_colors.values()))

    plt.title(title)
    plt.show()



# Other functions remain unchanged...

def draw_graph(G, title):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_size=700)
    plt.title(title)
    plt.show()


def process_climate_data(climate_data):
    # Read climate dataset
    climate_df = pd.read_csv('datasets/climate.csv')

    # Drop unnecessary columns and handle missing values if any
    climate_df = climate_df.dropna()
    climate_df = climate_df.drop(columns=['date'])  # We won't use date for classification

    # Separate features and target variable
    X = climate_df.drop(columns=['climate'])
    y = climate_df['climate']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN classifier
    knn = KNeighborsClassifier()
    knn.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = knn.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Predict climate classification for input data
    climate_classification = knn.predict([list(climate_data.values())])

    return climate_classification[0], accuracy
def process_crop_recommendation_data(crop_data):
    # Load the dataset
    crop_df = pd.read_csv('datasets/crop_recommendation.csv')

    # Prepare data
    X = crop_df.drop(columns=['label'])
    y = crop_df['label']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN classifier
    knn = KNeighborsClassifier()
    knn.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = knn.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Prepare crop data for prediction
    crop_data_array = np.array([[crop_data['N'], crop_data['P'], crop_data['K'], crop_data['temperature'], crop_data['humidity'], crop_data['ph'], crop_data['rainfall']]])

    # Predict crop recommendation for input data
    crop_recommendation = knn.predict(crop_data_array)

    return crop_recommendation, accuracy

def home(request):
    if request.method == 'POST':
        climate_form = ClimateDataForm(request.POST)
        crop_form = CropRecommendationForm(request.POST)
        if climate_form.is_valid() and crop_form.is_valid():
            climate_instance = climate_form.save(commit=False)
            crop_instance = crop_form.save(commit=False)

            climate_data = {
                'precipitation': climate_instance.precipitation,
                'temp_max': climate_instance.temp_max,
                'temp_min': climate_instance.temp_min,
                'wind': climate_instance.wind
            }

            # Process climate data
            climate_classification, climate_accuracy = process_climate_data(climate_data)

            crop_data = {
                'N': crop_instance.N,
                'P': crop_instance.P,
                'K': crop_instance.K,
                'temperature': climate_instance.temp_max,  # Use temp_max for crop recommendation
                'humidity': crop_instance.humidity,
                'ph': crop_instance.ph,
                'rainfall': crop_instance.rainfall
            }

            # Process crop recommendation data
            crop_recommendation, crop_accuracy = process_crop_recommendation_data(crop_data)

            climate_instance.save()
            crop_instance.save()

            return render(request, 'agrovision_app/result.html', {
                'climate_classification': climate_classification,
                'climate_accuracy': climate_accuracy,
                'crop_recommendation': crop_recommendation,
                'crop_accuracy': crop_accuracy
            })
    else:
        climate_form = ClimateDataForm()
        crop_form = CropRecommendationForm()
    return render(request, 'agrovision_app/home.html', {'climate_form': climate_form, 'crop_form': crop_form})