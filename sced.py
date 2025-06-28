import streamlit as st
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt

import seaborn as sns

st.set_page_config(layout="wide")
st.title("⚡️ 3-Node SCED Simulator")

st.markdown("Set **demand**, **generation costs**, and **line limits**. Generation is optimized to meet demand.")

#test
#test two
col1, col2 = st.columns(2)

# === Demand input ===
with col1:
    st.markdown("### 🚧  Load")
    demand = np.array([
        st.slider("Demand at Node A (MW)", 0, 100, 0),
        st.slider("Demand at Node B (MW)", 0, 100, 100),
        st.slider("Demand at Node C (MW)", 0, 100, 0),
    ])

total_demand = np.sum(demand)
st.success(f"✅ Total system demand: {total_demand:.2f} MW")

# === Generation costs ($/MWh) ===
with col1:
    st.markdown("### 💰 Generation Costs")
    cost_A = st.slider("Cost at Node A", 0, 100, 10)
    cost_B = st.slider("Cost at Node B", 0, 100, 20)
    cost_C = st.slider("Cost at Node C", 0, 100, 30)

c = np.array([cost_A, cost_B, cost_C])

# === Line limits ===
with col2:
    st.markdown("### 🚧 Line Flow Limits (MW)")
    limit_AB = st.slider("Line AB Limit", 10, 100, 100)
    limit_BC = st.slider("Line BC Limit", 10, 100, 100)
    limit_AC = st.slider("Line AC Limit", 10, 100, 100)

Fmax = np.array([limit_AB, limit_BC, limit_AC])


# === Generator limits ===
with col2:
    st.markdown("### 🚧 Generator Limits (MW)")

    # Create a range slider
    Pmin_A, Pmax_A = st.slider(
        "Production Limits A",
        min_value=0,
        max_value=100,
        value=(0, 100),  # default selected range
        step=1
    )
    # Create a range slider
    Pmin_B, Pmax_B = st.slider(
        "Production Limits B",
        min_value=0,
        max_value=100,
        value=(0, 100),  # default selected range
        step=1
    )
    # Create a range slider
    Pmin_C, Pmax_C = st.slider(
        "Production Limits C",
        min_value=0,
        max_value=100,
        value=(0, 100),  # default selected range
        step=1
    )


# === Generator limits ===
Pmin = np.array([Pmin_A, Pmin_B, Pmin_C])
Pmax = np.array([Pmax_A, Pmax_B, Pmax_C])


# === PTDF matrix (rows: lines AB, BC, AC; cols: nodes A, B, C) ===
PTDF = np.array([
    [ 0.5, -0.5,  0.0],  # AB
    [ 0.0,  0.5, -0.5],  # BC
    [ 0.5,  0.0, -0.5],  # AC
])

# === Solve SCED ===
P = cp.Variable(3)  # Generation at each node

# Constraints
constraints = [
    P >= Pmin,
    P <= Pmax,
    cp.sum(P) == total_demand,
]

net_injection = P - demand  # net injection = generation - demand
line_flows = PTDF @ net_injection

constraints += [
    line_flows <= Fmax,
    line_flows >= -Fmax,
]

problem = cp.Problem(cp.Minimize(c @ P), constraints)
problem.solve()

# === Output ===
st.subheader("🔢 SCED Results")

if P.value is not None:
    st.write("**Optimal Generation (MW):**")
    st.write({f"Node {n}": round(p, 2) for n, p in zip(["A", "B", "C"], P.value)})

    st.write("**Line Flows (MW):**")
    for i, (a, b) in enumerate([("A", "B"), ("B", "C"), ("A", "C")]):
        color = "red" if abs(line_flows.value[i]) > Fmax[i] - 1e-3 else "black"
        st.markdown(
            f"<span style='color:{color}'>{a} → {b}: {line_flows.value[i]:.2f} MW (limit = {Fmax[i]} MW)</span>",
            unsafe_allow_html=True
        )

    # === Compute LMPs ===
    lambda_val = -constraints[2].dual_value  # dual of power balance
    mu_plus = constraints[3].dual_value     # duals of line_flow <= Fmax
    mu_minus = constraints[4].dual_value    # duals of line_flow >= -Fmax
    lmp = lambda_val + PTDF.T @ (mu_plus - mu_minus)

    st.write("**System lambda (marginal cost):**")
    st.write(f"${lambda_val:.2f} /MWh")

else:
    st.error("❌ Optimization failed. Try different settings.")

# === Grid Visualization ===
st.subheader("🖼️ Grid Visualization")
G = nx.Graph()
G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
pos = {"A": (0, 1), "B": (1, 1), "C": (0.5, 0)}

fig, ax = plt.subplots(figsize=(6, 4))
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1000, ax=ax, edge_color="gray")

if P.value is not None:
    for i, node in enumerate(["A", "B", "C"]):
        coord = pos[node]
        g = P.value[i]
        d = demand[i]
        price = lmp[i]

        label = (
            f"G: {g:.1f} MW\n"
            f"D: {d:.1f} MW\n"
            f"LMP: {price:.2f} $/MWh"
        )
        ax.text(
            coord[0], coord[1] + 0.1,
            label,
            ha='center', fontsize=10, color='black',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1)
        )
    # Add line flow labels at edge midpoints
    for i, (u, v) in enumerate([("A", "B"), ("B", "C"), ("A", "C")]):
        f = line_flows.value[i]
        mid = (np.array(pos[u]) + np.array(pos[v])) / 2
        ax.text(mid[0], mid[1]+0.1, f"{f:.1f} MW", ha='center', fontsize=9, color='darkred')



col_ptdf1, col_ptdf2 = st.columns(2)

# Reuse existing Grid Visualization in left column
with col_ptdf1:
    st.pyplot(fig)


with col_ptdf2:
    fig_ptdf, ax_ptdf = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        PTDF,
        annot=True,
        cmap="coolwarm",
        xticklabels=["Node A", "Node B", "Node C"],
        yticklabels=["Line AB", "Line BC", "Line AC"],
        center=0,
        fmt=".2f",
        cbar_kws={'label': 'PTDF Value'},
        ax=ax_ptdf
    )
    ax_ptdf.set_title("PTDF Matrix Heatmap")
    st.pyplot(fig_ptdf)