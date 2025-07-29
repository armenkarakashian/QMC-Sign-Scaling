import sys

def label_spins(N):
    bottom = list(range(1, N, 2))
    top = list(range(2, N+1, 2))
    return bottom, top

def generate_triangular_ladder_edges(N):
    bottom, top = label_spins(N)
    edges = []
    n_rungs = len(bottom)
    # Vertical "rungs"
    for i in range(n_rungs):
        edges.append((bottom[i], top[i]))
    # Horizontal "legs" (bottom)
    for i in range(n_rungs-1):
        edges.append((bottom[i], bottom[i+1]))
    # Horizontal "legs" (top)
    for i in range(n_rungs-1):
        edges.append((top[i], top[i+1]))
    # Diagonal "zig-zags"
    for i in range(n_rungs-1):
        edges.append((bottom[i], top[i+1]))
    # Sort by lowest node, then by highest node
    edges = sorted(edges, key=lambda x: (min(x), max(x)))
    return edges

def select_rung_edges(N):
    return [(i, i+1) for i in range(1, N, 2)]

def select_defect_rungs(N, Nd):
    rungs = select_rung_edges(N)
    L = len(rungs)
    if Nd == 1:
        indices = [L // 2]
    else:
        indices = [int(round(i * (L - 1) / (Nd - 1))) for i in range(Nd)]
    return [rungs[i] for i in indices]

def generate_heisenberg_input(N, Nd, filename=None):
    N = int(N)
    Nd = int(Nd)
    if N % 2 != 0:
        raise ValueError("N must be even for a triangular ladder.")
    if filename is None:
        filename = f"TriHeis_defects_N={N}_Nd={Nd}.txt"
    edge_list = generate_triangular_ladder_edges(N)
    defect_edges = select_defect_rungs(N, Nd)
    paulis = ['X', 'Y', 'Z']
    lines = []
    for (i, j) in edge_list:
        coeff = 1.0 if (i, j) in defect_edges or (j, i) in defect_edges else -1.0
        for p in paulis:
            lines.append(f"{coeff:.1f} {i} {p} {j} {p}")
    with open(filename, "w") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"File '{filename}' generated. Spins: {N}, Edges: {len(edge_list)}, Defects: {Nd}")
    print("Defect edges:", defect_edges)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_heisenberg_input.py N Nd [filename]")
        sys.exit(1)
    N = int(sys.argv[1])
    Nd = int(sys.argv[2])
    filename = sys.argv[3] if len(sys.argv) > 3 else None
    generate_heisenberg_input(N, Nd, filename)

