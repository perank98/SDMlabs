import igraph as ig
import matplotlib.pyplot as plt
import numpy as np


def load_graph(path: str, verbose: bool = False):
    # Load the graph
    G = ig.Graph.Read_GraphML(path)
    # currently 'process_tweets' saves a directed graph; thus it is converted here
    G_undirected = G.as_undirected(combine_edges=None)

    if verbose:
        print("Graph\n")
        print(G.summary())
        print("##################\n")
        print("Simplified Graph\n")
        simplified_G = G.simplify()
        print(simplified_G.summary())

    return G_undirected


def summarize_network(g: ig.Graph):
    # TODO
    order = g.vcount()
    size = g.ecount()

    num_components = len(g.connected_components())

    p = g.density(loops=False)

    # clustering coefficient/transitivity
    transivity = g.transitivity_undirected()

    # plot the degree of dist.
    # generate random graph per Erdos-Renyi model
    er = ig.Graph.Erdos_Renyi(n=order, p=p, directed=False, loops=False)

    deg = g.degree()
    er_deg = er.degree()

    betweenness = g.betweenness()

    # use log-log axes (?)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.hist(deg, max(deg), histtype="barstacked")
    ax.hist(er_deg, max(er_deg), histtype="barstacked")

    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.show()

    summary = {
        "name": "graph",
        "order_n": order,
        "size_m": size,
        "degrees": deg, 
        "num_components": num_components,
        "density": p,
        "transitivity": transivity,
        "betweenness": betweenness
    }
    print(summary)

    return summary


def draw_graph(g:ig.Graph, output:str="plot.pdf"):
    layout = retweet_graph.layout_davidson_harel()

    g.vs["color"] = "rgba(30,144,255,0.8)"
    g.es["color"] = "rgba(0,0,0,0.5)"

    # approximate for speed
    # TODO
    def fast_betweenness(g: ig.Graph, samples=500):
        return g.betweenness(vertices=np.random.choice(g.vs.indices, samples, replace=False))

    # scale node size with n_degree
    # g.vs["size"] = [2 + d * 0.3 for d in g.degree()]

    betweenness = fast_betweenness(g)
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


if __name__ == "__main__":
    # reply_graph = load_graph("./data/500_reply_network.graphml")
    retweet_graph = load_graph("./data/500_retweet_network.graphml")
    
    summary = summarize_network(retweet_graph)

    # draw_graph(retweet_graph, output="retweets.pdf")
    # # draw_graph(reply_graph, output="replies.pdf")

    # top 10 actors, in measures of degree and betweenness
    degrees = summary["degrees"]
    indeces = np.argmax(degrees)
    indeces = np.argpartition(degrees, -10)[-10:]
    print(indeces)

    betweens = summary["betweenness"]
    indeces_b = np.argmax(betweens)
    indeces_b = np.argpartition(betweens, -10)[-10:]

    for i in indeces:
        print(int(retweet_graph.vs[i]["account_id"]))
        print(int(retweet_graph.vs[i]["account_id"]))