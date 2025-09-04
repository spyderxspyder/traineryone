import networkx as nx
import matplotlib.pyplot as plt

# Define graph
G = nx.DiGraph()

# Nodes (variables and operations)
G.add_node("x", label="x")
G.add_node("y", label="y")
G.add_node("mul", label="*")
G.add_node("const3", label="3")
G.add_node("add1", label="+")
G.add_node("add2", label="+")
G.add_node("z", label="z")

# Edges (flow of computation)
G.add_edges_from([
    ("x", "mul"),
    ("y", "mul"),
    ("mul", "add2"),
    ("y", "add1"),
    ("const3", "add1"),
    ("add1", "add2"),
    ("add2", "z")
])

# Draw
pos = nx.spring_layout(G, seed=42)
labels = nx.get_node_attributes(G, "label")

plt.figure(figsize=(8,6))
nx.draw(G, pos, with_labels=True, labels=labels,
        node_size=2000, node_color="lightblue",
        arrowsize=20, font_size=12, font_weight="bold")
plt.title("Computational Graph: z = (x * y) + (y + 3)")
plt.show()
