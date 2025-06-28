import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("⚡️ Generalized SCED Simulator")

st.markdown("Select **number of nodes**, define **adjacency matrix**, then set demand, cost, and generator/line limits.")

# === Select number of nodes ===
num_nodes = st.number_input("Number of Nodes", min_value=2, max_value=10, value=3, step=1)

# === Adjacency matrix input ===
st.markdown("### 🔗 Adjacency Matrix")
def_adj = np.triu(np.ones((num_nodes, num_nodes)),k=1)
adj_df = st.data_editor(pd.DataFrame(def_adj), num_rows="fixed", key="adj_matrix")
adj_matrix = adj_df.values
adj_matrix = np.triu(adj_matrix) + np.triu(adj_matrix, 1).T

# === Build graph ===
G = nx.from_numpy_array(adj_matrix)
mapping = {i: chr(65 + i) for i in range(num_nodes)}
G = nx.relabel_nodes(G, mapping)
nodes = list(G.nodes)
edges = list(G.edges)

col1, col2 = st.columns(2)

# === Demand, Cost, Generator Limits ===
demand = []
costs = []
Pmin = []
Pmax = []

st.markdown("### 🚧 Load Limits (MW)")
for i in range(num_nodes):
    demand.append(st.slider(f"Demand at Node {nodes[i]} (MW)", 0, 100, 0 if i != 1 else 100))

st.markdown("### 🚧 Costs of Generation ($/MWh)")
for i in range(num_nodes):
    costs.append(st.slider(f"Cost at Node {nodes[i]} ($/MWh)", 0, 100, 10 + 10*i))

st.markdown("### 🚧 Generator Capacity (MWh)")
for i in range(num_nodes):
    low, high = st.slider(f"Production Limits {nodes[i]} (MW)", 0, 100, (0, 100), step=1)
    Pmin.append(low)
    Pmax.append(high)

demand = np.array(demand)
costs = np.array(costs)
Pmin = np.array(Pmin)
Pmax = np.array(Pmax)
total_demand = np.sum(demand)

st.success(f"✅ Total system demand: {total_demand:.2f} MW")

# === Line limits ===
st.markdown("### 🚧 Line Flow Limits (MW)")
Fmax = []
for u, v in edges:
    limit = st.slider(f"Line {u}-{v} Limit", 10, 100, 100)
    Fmax.append(limit)
Fmax = np.array(Fmax)

# === Compute PTDF matrix ===

# === Compute PTDF matrix ===
def compute_PTDF( nodes, edges, slack_node=0, reactance=0.1):
    n = len(nodes)
    m = len(edges)
    node_index = {node: i for i, node in enumerate(nodes)}

    # Incidence matrix B (m x n)
    B = np.zeros((m, n))
    for i, (u, v) in enumerate(edges):
        B[i, node_index[u]] = 1
        B[i, node_index[v]] = -1

    # Reactance matrix X (m x m)
    X_diag = np.eye(m) * (1 / reactance)

    # Laplacian matrix (n x n)
    L = B.T @ X_diag @ B

    # Remove slack node
    keep = [i for i in range(n) if i != slack_node]
    L_red = L[np.ix_(keep, keep)]
    L_red_inv = np.linalg.inv(L_red)



    # Compute PTDF (m x n)
    PTDF = X_diag @ B @ np.zeros((n, n))
    for idx, j in enumerate(keep):
        e = np.zeros(n - 1)
        e[idx] = 1
        v = L_red_inv @ e
        PTDF[:, j] = X_diag @ B @ np.insert(v, slack_node, 0)

    return PTDF

PTDF = compute_PTDF(nodes, edges, slack_node=0, reactance=0.1)

# === Solve SCED ===
P = cp.Variable(num_nodes)

constraints = [
    P >= Pmin,
    P <= Pmax,
    cp.sum(P) == total_demand,
]

net_injection = P - demand
line_flows = PTDF @ net_injection

constraints += [
    line_flows <= Fmax,
    line_flows >= -Fmax,
]

problem = cp.Problem(cp.Minimize(costs @ P), constraints)
problem.solve()

# === Output ===
st.subheader("🔢 SCED Results")
if P.value is not None:
    st.write("**Optimal Generation (MW):**")
    st.write({f"Node {n}": round(p, 2) for n, p in zip(nodes, P.value)})

    st.write("**Line Flows (MW):**")
    for i, (u, v) in enumerate(edges):
        color = "red" if abs(line_flows.value[i]) > Fmax[i] - 1e-3 else "black"
        st.markdown(
            f"<span style='color:{color}'>{u} → {v}: {line_flows.value[i]:.2f} MW (limit = {Fmax[i]} MW)</span>",
            unsafe_allow_html=True
        )

    lambda_val = -constraints[2].dual_value
    mu_plus = constraints[3].dual_value
    mu_minus = constraints[4].dual_value
    lmp = lambda_val + PTDF.T @ (mu_plus - mu_minus)

    st.write("**System lambda (marginal cost):**")
    st.write(f"${lambda_val:.2f} /MWh")
else:
    st.error("❌ Optimization failed. Try different settings.")

# === Visualization ===
st.subheader("🖼️ Grid Visualization")
pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(6, 4))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1000, ax=ax, edge_color="gray")

if P.value is not None:
    for i, node in enumerate(nodes):
        coord = pos[node]
        g = P.value[i]
        d = demand[i]
        price = lmp[i]
        label = f"G: {g:.1f} MW\nD: {d:.1f} MW\nLMP: {price:.2f} $/MWh"
        ax.text(coord[0], coord[1] + 0.1, label, ha='center', fontsize=10, color='black',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1))

    for i, (u, v) in enumerate(edges):
        f = line_flows.value[i]
        mid = (np.array(pos[u]) + np.array(pos[v])) / 2
        ax.text(mid[0], mid[1]+0.1, f"{f:.1f} MW", ha='center', fontsize=9, color='darkred')

col_ptdf1, col_ptdf2 = st.columns(2)
with col_ptdf1:
    st.pyplot(fig)

with col_ptdf2:
    fig_ptdf, ax_ptdf = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        PTDF,
        annot=True,
        cmap="coolwarm",
        xticklabels=[f"Node {n}" for n in nodes],
        yticklabels=[f"Line {u}-{v}" for u, v in edges],
        center=0,
        fmt=".2f",
        cbar_kws={'label': 'PTDF Value'},
        ax=ax_ptdf
    )
    ax_ptdf.set_title("PTDF Matrix Heatmap")
    st.pyplot(fig_ptdf)
