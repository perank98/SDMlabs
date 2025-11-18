import random

from collections import defaultdict
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import orjson
from tqdm import tqdm
from datetime import timezone, datetime


def load_graph(path: str, verbose: bool = False):
    G = ig.Graph.Read_GraphML(path)
    # currently 'process_tweets' saves a directed graph, thus it is converted here
    G_undirected = G.as_undirected(combine_edges=None)

    if verbose:
        print("Graph\n")
        print(G.summary())
        print("##################\n")
        print("Simplified Graph\n")
        simplified_G = G.simplify()
        print(simplified_G.summary())

    return G_undirected


def summarise_network(g: ig.Graph, name:str = "Graph"):
    order = g.vcount()
    size = g.ecount()
    num_components = len(g.connected_components())
    p = g.density(loops=False)
    transitivity = g.transitivity_undirected()
    deg = g.degree()

    # num_samples = min(1000, g.vcount())
    # sampled_vertices = random.sample(g.vs.indices, num_samples)
    # betweenness = g.betweenness(vertices=sampled_vertices, cutoff=5)

    # use log-log axes (?)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hist(deg, max(deg), histtype="barstacked")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(f"Degree Distribution (log–log) for {name}")

    

    summary = {
        "name": name,
        "order": order,
        "size": size,
        "degrees": deg,
        "num_components": num_components,
        "density": p,
        "transitivity": transitivity,
        "degree_plot": fig,
        # "betweenness": betweenness
    }

    return summary


def print_summary(graph_summary, to_file: bool = False):

    lines = [
        f"Name: {graph_summary["name"]}",
        f"Order (number of vertices): {graph_summary["order"]}",
        f"Size (number of edges): {graph_summary["size"]}",
        f"Number of components: {graph_summary["num_components"]}",
        f"Density: {graph_summary["density"]}",
        f"Clustering coefficient / Transitivity: {graph_summary["transitivity"]}",
        # f"Betweenness: {np.mean(graph_summary["betweenness"])}",
    ]

    # optionally save to file
    if to_file:
        path = f"./summaries/{graph_summary["name"]}"
        with open(path + ".txt", "w") as f:
            for line in lines:
                f.write(line + "\n")

        graph_summary["degree_plot"].savefig(path + ".png")

    else:
        # print to console
        for line in lines:
            print(line)

    return


def draw_graph(g: ig.Graph, output: str, scale_with_degree: bool = True):
    layout = g.layout_fruchterman_reingold()

    g.vs["color"] = "rgba(30,144,255,0.8)"
    g.es["color"] = "rgba(0,0,0,0.5)"

    if scale_with_degree:
        g.vs["size"] = [2 + d * 0.1 for d in g.degree()]

    else:
        betweenness = g.betweenness(cutoff=5)
        g.vs["size"] = [2 + 10 * (b / max(betweenness)) for b in betweenness]

    plt = ig.plot(
        g,
        layout=layout,
        vertex_label="",
        target=output,
        vertex_frame_width=0,
        edge_width=0.5,
    )

    return None


def extract_top10_actors(g: ig.Graph, graph_summary: dict, df):
    degrees = graph_summary["degrees"]
    degree_indeces = np.argmax(degrees)
    degree_indeces = np.argpartition(degrees, -10)[-10:]

    # betweenness = graph_summary["betweenness"]
    # betweenness_indeces = np.argmax(betweenness)
    # betweenness_indeces = np.argpartition(betweenness, -10)[-10:]

    # Get corresponding account IDs
    degree_ids = [g.vs[i]["account_id"] for i in degree_indeces]
    # betweenness_ids = [g.vs[i]["account_id"] for i in betweenness_indeces]
    
    degree_df = df[df["author_id"].isin(degree_ids)]
    # betweenness_df = df[df["author_id"].isin(betweenness_ids)]

    return degree_df   # , betweenness_df)


def random_walk_graph(g: ig.Graph, num_iter=1000):
    n = g.vcount()
    counts = np.zeros(n, dtype=int)
    actor = random.randrange(0, n)

    for _ in range(num_iter):
        counts[actor] += 1
        neighbors = g.neighbors(actor)
        actor = random.choice(neighbors)

    return counts


def information_diffusion(g: ig.Graph, num_iter: int = 1000, p: float = 0.1):
    for vertex in g.vs:
        vertex["polarity"] = random.choice([0, 1])

    for _ in tqdm(range(num_iter)):
        for node in g.vs:
            if node["polarity"] == 1:
                for neighbor in g.neighbors(node):
                    if random.random() < p:
                        g.vs[neighbor]["polarity"] = 1

    return len([vertex for vertex in g.vs if vertex["polarity"] == 1])


def opinion_diffusion(
    g: ig.Graph, num_positive=50, num_iter: int = 1000, opinion_change_th=0.5
):
    n = g.vcount()
    for node in g.vs:
        node["opinion"] = 0

    infected_indeces = random.sample(range(n), num_positive)
    for i in infected_indeces:
        g.vs[i]["opinion"] = 1

    for _ in range(num_iter):
        for node in g.vs:
            neighbor_count = 0
            local_positive_opinion = 0
            for neighbor in g.neighbors(node):
                neighbor_count += 1
                opinion_plus = 0

                if g.vs[neighbor]["opinion"] == 1:
                    opinion_plus += 1

                local_positive_opinion = opinion_plus / neighbor_count

            if local_positive_opinion > opinion_change_th:
                node["opinion"] = 1
            else:
                node["opinion"] = 0

    return len([vertex for vertex in g.vs if vertex["opinion"] == 1])


def plot_histogram(counts, outfile):
    plt.figure(figsize=(7, 5))
    plt.hist(
        counts,
        bins=range(min(counts), max(counts) + 2),
        edgecolor="black",
        align="left",
    )
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Integer Values")
    plt.xticks(range(min(counts), max(counts) + 1))
    plt.savefig(
        "./plots/random_walk_histograms/" + outfile + "_token_passing.svg",
        bbox_inches="tight",
    )  # can also use .pdf, .svg, etc.
    plt.close()


def load_tweets_jsonl(path):
    """Return list of dicts: {account_id:int, ts:int (unix seconds), urls:list[str], id:str}"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            t = orjson.loads(line)
            acc_id = int(t["account"]["id"])
            ts = int(datetime.fromisoformat(t["date"]).replace(tzinfo=timezone.utc).timestamp())
            urls = t.get("urls", []) or []
            out.append({"id": t.get("id"), "account_id": acc_id, "ts": ts, "urls": urls})
    return out


def get_coaction_dict(tweets:list[dict], s:int=1, s_lower:int=10000):
    url_index = defaultdict(list)
    for tweet in tweets:
        for url in tweet.get("urls", []):
            url_index[url].append((tweet["ts"], tweet["account_id"]))
    edges = {}
    # Process each URL separately
    for url, entries in tqdm(url_index.items()):
        # Sort by timestamp to allow sliding window
        entries.sort()  # sorts by timestamp automatically
        # Sliding window over sorted timestamps
        start = 0
        for end in range(len(entries)):
            t2, acc2 = entries[end]
            # Slide start pointer until window condition (≤ 2 seconds) is satisfied
            while t2 - entries[start][0] > s:
                start += 1
            # Compare each pair within the small sliding window
            for i in range(start, end):
                t1, acc1 = entries[i]
                # Count edge
                edges[(acc1, acc2)] = edges.get((acc1, acc2), 0) + 1

    return edges


def get_graph_from_coaction_dict(edges, r:int=5)-> ig.Graph:
        graph_edges = []
        for key, value in edges.items():
            if key[0] == key[1]:
                continue
            if value >= r:
                edge = (
                        key[0],
                        key[1],
                        value,
                    )
                graph_edges.append(edge)
        
        bot_graph = ig.Graph.TupleList(
            graph_edges,
            vertex_name_attr="account_id",
            edge_attrs=["weight", "tweet_id", "possible_sensitive"],
        )
        return bot_graph