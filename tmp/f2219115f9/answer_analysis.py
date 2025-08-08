import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load the edges data
edges_df = pd.read_csv('/home/archer/projects/grasper/tmp/f2219115f9/edges.csv')

# Create a graph from the edge list
G = nx.from_pandas_edgelist(edges_df, 'source', 'target')

# 1. How many edges are in the network?
edge_count = G.number_of_edges()

# 2. Which node has the highest degree?
degrees = dict(G.degree())
highest_degree_node = max(degrees, key=degrees.get)

# 3. What is the average degree of the network?
average_degree = float(sum(degrees.values()) / len(G.nodes()))

# 4. What is the network density?
density = float(nx.density(G))

# 5. What is the length of the shortest path between Alice and Eve?
shortest_path_alice_eve = None
if 'Alice' in G and 'Eve' in G:
    try:
        shortest_path_alice_eve = int(nx.shortest_path_length(G, source='Alice', target='Eve'))
    except nx.NetworkXNoPath:
        shortest_path_alice_eve = None # No path exists

# 6. Draw the network graph and encode as base64 PNG
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=10, font_weight='bold')
plt.title('Network Graph')
buf = BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
buf.seek(0)
network_graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

# 7. Plot the degree distribution as a bar chart with green bars and encode as base64 PNG
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
degree_counts = pd.Series(degree_sequence).value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.bar(degree_counts.index, degree_counts.values, color='green')
plt.title('Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.xticks(degree_counts.index)
buf = BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
buf.seek(0)
degree_histogram_base64 = base64.b64encode(buf.read()).decode('utf-8')
plt.close()

# Prepare the final JSON object
result = {
    'edge_count': edge_count,
    'highest_degree_node': highest_degree_node,
    'average_degree': average_degree,
    'density': density,
    'shortest_path_alice_eve': shortest_path_alice_eve,
    'network_graph': network_graph_base64,
    'degree_histogram': degree_histogram_base64
}

print(result)