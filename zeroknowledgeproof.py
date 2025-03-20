import colorsys
from abc import ABC, abstractmethod
import random
import hashlib
from ortools.sat.python import cp_model

# Bokeh imports
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, HoverTool


class BaseChallenger(ABC):
    @abstractmethod
    def challenge(self):
        """
        Produce a new 'challenge' – in this case, two numbers to be added.
        """
        pass


class BaseSolver(ABC):
    @abstractmethod
    def solve(self, challenge):
        """
        Given a challenge (e.g., two numbers), produce a solution (sum).
        Possibly faulty or correct, depending on implementation.
        """
        pass


class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, challenge, solution):
        """
        Given the challenge and a solution, verify correctness.
        Return True if correct, False otherwise.
        """
        pass

#helper for the graph coloring visualisation
def generate_color_palette(k):
    """
    Generates k distinct colors around the HSV color circle.
    Returns a list of hex color strings, e.g. ['#cc33cc', '#33cccc', ...]
    """
    palette = []
    for i in range(k):
        hue = i / max(k, 1)
        saturation = 0.6
        value = 0.9
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        palette.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return palette

class GraphColorChallenger(BaseChallenger):
    """
    Challenger that accepts a predefined graph (edges, num_nodes) and a color limit (k).
    """
    def __init__(self, edges, num_nodes, num_colors):
        self.edges = edges
        self.num_nodes = num_nodes
        self.num_colors = num_colors

    def challenge(self):
        """
        Returns a dictionary describing the graph coloring challenge.
        """
        return {
            "num_nodes": self.num_nodes,
            "edges": self.edges,
            "num_colors": self.num_colors
        }


class GraphColorSolver(BaseSolver):
    """
    - solve(challenge): use OR-Tools CP-SAT to find a valid coloring with up to k colors.
    - shuffle_coloring(): permute color labels for zero-knowledge.
    - commit_coloring(): produce a cryptographic commitment for each node's color.
    - open_edge_colors(edge): reveal the color + salt for two endpoints of that edge.
    """

    def __init__(self):
        self.graph_info = None
        self.original_coloring = None  # final integer color assignment for each node
        self.permuted_labels = None
        self.k = 0

        # For ZKP commitments
        self.commitments = []
        self.salts = []

    def solve(self, challenge):
        """
        Solve the graph k-coloring feasibility problem using CP-SAT:
         - Each node has an integer var in [0..k-1]
         - For each edge (u,v), color[u] != color[v]
         - If feasible, store the coloring. Otherwise, raise an error.
        """
        self.graph_info = challenge
        self.k = challenge["num_colors"]
        n = challenge["num_nodes"]
        edges = challenge["edges"]

        coloring = self._solve_with_cp_sat(n, edges, self.k)
        if coloring is None:
            raise ValueError(f"No valid {self.k}-coloring found for the given graph.")
        self.original_coloring = coloring

        # Initialize a random permutation of color labels
        self._generate_new_label_permutation()

    def _solve_with_cp_sat(self, n, edges, k):
        """
        Use OR-Tools CP-SAT to check if the graph can be colored with up to k colors.
        - We only do a feasibility check: color[i] in [0..k-1]
        - For each edge (u,v), color[u] != color[v]
        - If feasible, return a valid coloring list. Otherwise, return None.
        """
        model = cp_model.CpModel()

        # Create integer variables for each node in [0..k-1]
        color_vars = [model.NewIntVar(0, k - 1, f'color_{i}') for i in range(n)]

        # Adjacency constraints
        for (u, v) in edges:
            model.Add(color_vars[u] != color_vars[v])

        # Solve
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
            return [solver.Value(color_vars[i]) for i in range(n)]
        else:
            return None

    def shuffle_coloring(self):
        """Randomly permute color labels for zero-knowledge each round."""
        self._generate_new_label_permutation()

    def commit_coloring(self):
        """
        Produce a commitment (SHA-256 hash of random salt + color) for each node
        under the current permuted labeling.
        """
        n = self.graph_info["num_nodes"]
        self.commitments = []
        self.salts = []

        for node in range(n):
            real_color = self.original_coloring[node]
            perm_color = self.permuted_labels[real_color]
            # Random salt
            salt = random.getrandbits(128).to_bytes(16, 'big')
            # Create hash input
            hash_input = salt + perm_color.to_bytes(4, 'big')
            commit_hash = hashlib.sha256(hash_input).hexdigest()

            self.commitments.append(commit_hash)
            self.salts.append(salt)

        return self.commitments

    def open_edge_colors(self, edge):
        """
        Reveal (nodeIndex, permuted_color, salt) for both endpoints of the edge.
        """
        (u, v) = edge
        color_u = self._get_permuted_color(u)
        color_v = self._get_permuted_color(v)
        salt_u = self.salts[u]
        salt_v = self.salts[v]
        return (u, color_u, salt_u), (v, color_v, salt_v)

    def _generate_new_label_permutation(self):
        label_list = list(range(self.k))
        random.shuffle(label_list)
        self.permuted_labels = label_list

    def _get_permuted_color(self, node):
        real_c = self.original_coloring[node]
        return self.permuted_labels[real_c]
    
    def visualize_coloring(self, use_permuted=False):
        """
        Returns an ipycytoscape widget showing each node in its color.
        
        By default, it uses the *original* color assignment.
        If 'use_permuted=True', it applies the currently stored 'permuted_labels'
        to each node's original color (which might have been generated in a shuffle).
        
        Works around older ipycytoscape versions by assigning node colors 
        *after* loading the graph.
        """
        import networkx as nx
        import ipycytoscape
        import colorsys

        def generate_color_palette(k):
            """
            Generates k distinct colors around the HSV color circle.
            Returns a list of hex color strings, e.g. ['#cc33cc', '#33cccc', ...].
            """
            palette = []
            for i in range(k):
                hue = i / max(k, 1)
                saturation = 0.6
                value = 0.9
                r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
                palette.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
            return palette

        n = self.graph_info["num_nodes"]
        edges = self.graph_info["edges"]

        # Create a NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)

        # Generate a distinct color palette for k colors
        color_palette = generate_color_palette(self.k)

        # Assign each node's color to a node attribute
        for node in G.nodes:
            original_color = self.original_coloring[node]
            
            if use_permuted:
                # If we haven't shuffled or 'permuted_labels' doesn't exist, handle gracefully
                if not hasattr(self, 'permuted_labels') or self.permuted_labels is None:
                    raise ValueError("Cannot visualize permuted labels; 'permuted_labels' not set. Call shuffle_coloring() first.")
                permuted_label = self.permuted_labels[original_color]
                assigned_color = color_palette[permuted_label]
            else:
                assigned_color = color_palette[original_color]
            
            G.nodes[node]['color'] = assigned_color

        # Create the Cytoscape widget
        cyto_widget = ipycytoscape.CytoscapeWidget()
        cyto_widget.graph.add_graph_from_networkx(G)

        # Manually copy each node's 'color' attribute from G into the ipycytoscape node data
        for node in cyto_widget.graph.nodes:
            node_id_str = node.data['id']       # e.g. "0"
            node_id_int = int(node_id_str)      # your original node index is an integer
            node.data['color'] = G.nodes[node_id_int]['color']

        # Define a style to color each node background using 'data(color)'
        style = [
            {
                'selector': 'node',
                'style': {
                    'content': 'data(id)',
                    'background-color': 'data(color)',
                    'color': '#fff',  # node label color
                    'text-valign': 'center'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'line-color': '#999'
                }
            }
        ]
        cyto_widget.set_style(style)

        return cyto_widget

class GraphColorVerifier(BaseVerifier):
    """
    - verify(...) remains a placeholder for a single-step check.
    - We'll do multiple rounds: pick an edge, check commitments, etc.
    """

    def __init__(self):
        self.last_commitments = []  # store solver's commitments each round
        self.graph_info = None

    def verify(self, challenge, solution):
        # Not used in partial ZKP approach. Always True as a placeholder.
        return True

    def pick_edge_for_check(self):
        edges = self.graph_info["edges"]
        return random.choice(edges)

    def store_commitments(self, commitments):
        self.last_commitments = commitments

    def verify_opened_edge(self, opened_u, opened_v):
        """
        Check that:
         1) The committed hash matches the revealed color+salt for each node.
         2) The two revealed colors differ.
        """
        node_u, color_u, salt_u = opened_u
        node_v, color_v, salt_v = opened_v

        commit_u = self._compute_hash(salt_u, color_u)
        commit_v = self._compute_hash(salt_v, color_v)

        # Check if re-hashed values match stored commitments
        if commit_u != self.last_commitments[node_u]:
            return False
        if commit_v != self.last_commitments[node_v]:
            return False

        # The two colors must differ
        return (color_u != color_v)

    def _compute_hash(self, salt, color):
        hash_input = salt + color.to_bytes(4, 'big')
        return hashlib.sha256(hash_input).hexdigest()


# 5. Multi-Round ZKP Demo with Commitments
def zero_knowledge_proof_demo_with_commitments(challenger, solver, verifier, rounds=5):
    """
    - solver commits to a shuffled coloring
    - verifier picks an edge
    - solver opens that edge
    - verifier checks the commitment & difference
    Repeated for 'rounds' times => high confidence in correctness.
    """
    # Challenger gives the challenge
    challenge_data = challenger.challenge()
    verifier.graph_info = challenge_data
    edges = challenge_data["edges"]
    E = len(edges)

    print(f"CHALLENGER: Graph has {challenge_data['num_nodes']} nodes, "
          f"{E} edges, with k={challenge_data['num_colors']}.")

    # Solver solves with CP-SAT
    solver.solve(challenge_data)
    print(f"SOLVER: Found a valid {solver.k}-coloring (kept secret).")

    success_count = 0
    for i in range(rounds):
        # 1) Shuffle & commit
        solver.shuffle_coloring()
        commitments = solver.commit_coloring()
        verifier.store_commitments(commitments)

        # 2) Verifier picks an edge
        edge = verifier.pick_edge_for_check()

        # 3) Solver opens that edge
        opened_u, opened_v = solver.open_edge_colors(edge)

        # 4) Verifier checks
        if verifier.verify_opened_edge(opened_u, opened_v):
            print(f"[Round {i+1}] Edge={edge} => PASS")
            success_count += 1
        else:
            print(f"[Round {i+1}] Edge={edge} => FAIL")

    # Probability analysis (if there's at least one bad edge in the coloring)
    # the chance of never picking it in 'rounds' random checks is (1 - 1/E)^rounds
    if success_count == rounds:
        print(f"\nVERIFIER: All {rounds} checks passed. High confidence in solver's correctness.")
    else:
        print(f"\nVERIFIER: {success_count}/{rounds} checks passed. Possibly solver is cheating or unlucky.")

if __name__ == "__main__":
    random.seed(0)  # For reproducibility
    edges = [(0,1), (0,2), (1,3), (2,3), (3,4), (4,5), (3,5), (3,6), (4,6)]
    num_nodes = 7
    k = 3  # We want to see if there's a 3-coloring

    # Instantiate with graph
    challenger = GraphColorChallenger(edges, num_nodes, k)
    solver = GraphColorSolver()
    verifier = GraphColorVerifier()
    zero_knowledge_proof_demo_with_commitments(challenger, solver, verifier, rounds=50)
