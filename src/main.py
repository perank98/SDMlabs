from coaction_analysis import *
from utils import *

if __name__ == "__main__":
    # graph = load_graph("./sampled_data/20000_retweet.graphml")
    # graph = ig.read("./sampled_data/all_retweets.graphml")
    # summary = summarise_network(graph, name="all_retweets")
    # print_summary(summary, to_file=True)

    # # draw_graph(graph, output="./plots/20000_retweets_betweenness.png", scale_with_degree=False)
    # # draw_graph(graph, output="./plots/20000_retweets_degree.png", scale_with_degree=True)
    

    # # # top 10 actors, in measures of degree and betweenness
    # top_accounts = pd.read_csv("./data/accounts.tsv", sep="\t")
    
    # degree_actors = extract_top10_actors(graph, summary, top_accounts)

    # with open("./top_actors/all_retweets.txt", "w") as f:
    #     f.write(degree_actors[["Type", "Stance", "Lang"]].to_string() + "\n")
    #     # f.write("#######################################\n")
    #     # f.write(betweenness_actors[["Type", "Stance", "Lang"]].to_string())

    # # task 1.7 and onwards
    # gc = graph.connected_components().giant()
    # gc_summary = summarise_network(gc, name="Largest Component Only")

    # # task 2.1
    # order = gc_summary["order"]
    # p = gc.density(loops=False)
    # er1 = ig.Graph.Erdos_Renyi(n=order, p=p, directed=False, loops=False)
    # # er2 = ig.Graph.Erdos_Renyi(n=order, p=p, directed=False, loops=False)
    # # er3 = ig.Graph.Erdos_Renyi(n=order, p=p, directed=False, loops=False)
    # # er_list = [er1, er2, er3]

    # # er_summaries = [summarise_network(er, name=f"Erdos Renyi {ind+1}") for ind, er in enumerate(er_list)]
    # #for summary in er_summaries:
    # #    print_summary(summary, to_file=True)


    # # task 2.2: produce 3 BA networks, same order, same ap matrix as retweet network
    # # n = len(gc.vs)
    # # s = len(gc.es)
    # # m = int(s/n)
    # ba1 = ig.Graph.Barabasi(n=order, m=gc.degree(), directed=False)
    # # ba2 = ig.Graph.Barabasi(n=order, m=gc.degree(), directed=False)
    # # ba3 = ig.Graph.Barabasi(n=order, m=gc.degree(), directed=False)
    # # ba_list = [ba1, ba2, ba3]

    # # ba_summaries = [summarise_network(ba, name=f"Barabasi {ind+1}") for ind, ba in enumerate(ba_list)]
    # # for summary in ba_summaries:
    # #     print_summary(summary, to_file=True)

    # # task 2.3
    # p = gc.average_path_length() / 7
    # cc = gc.transitivity_avglocal_undirected()
    # ws1 = ig.Graph.Watts_Strogatz(
    #     dim=1, size=100, nei=4, p=p, allowed_edge_types="multi"
    # )
    # # ws2 = ig.Graph.Watts_Strogatz(
    # #     dim=1, size=100, nei=4, p=p, allowed_edge_types="multi"
    # # )
    # # ws3 = ig.Graph.Watts_Strogatz(
    # #     dim=1, size=100, nei=4, p=p, allowed_edge_types="multi"
    # # )

    # # ws_list = [ws1, ws2, ws3]

    # # ws_summaries = [summarise_network(ws, name=f"Watts-Strogatz {ind+1}") for ind, ws in enumerate(ws_list)]
    # # for summary in ws_summaries:
    # #     print_summary(summary, to_file=True)

    # # task 2.4
    # # starting network (retweet)
    # rewired_graph = graph.connected_components().giant().copy()
    # # rewire edges. using keeping_degseq to preserve degree distr
    # num_swaps = 10 * rewired_graph.ecount()

    # # avoid multiple edges and loops
    # rewired_graph.rewire(num_swaps, allowed_edge_types="simple")

    # # get and see stats after rewiring
    # # rewired_summary = summarise_network(rewired_graph, name="Rewired Graph")
    # # print_summary(rewired_summary, to_file=True)

    # # task 3.1
    # for graph, name in zip([graph, er1, ba1, ws1, rewired_graph],["original", "erdos_renyi", "barabasi", "watts_strogatz","rewired_graph"]):
    #     counts = random_walk_graph(graph)
    #     plot_histogram(counts, name)
    
    # # task 3.2
    # for graph, name in zip([graph, er1, ba1, ws1, rewired_graph],["original", "erdos_renyi", "barabasi", "watts_strogatz","rewired_graph"]):
    #     positive_actors = information_diffusion(graph, num_iter=5, p=0.01)
    #     print(name, ":\t", positive_actors)

    # # task 3.3
    # for graph, name in zip([graph, er1, ba1, rewired_graph],["original", "erdos_renyi", "barabasi","rewired_graph"]):
    #     positive_actors = [opinion_diffusion(graph, num_positive=100, num_iter=10, opinion_change_th=0.8),
    #                        opinion_diffusion(graph, num_positive=250, num_iter=10, opinion_change_th=0.8),
    #                        opinion_diffusion(graph, num_positive=100, num_iter=10, opinion_change_th=0.5),
    #                        opinion_diffusion(graph, num_positive=250, num_iter=10, opinion_change_th=0.5)]
    #     print(name, ":\t", positive_actors)

    tweets = load_tweets_jsonl("./sampled_data/2260916_only_tweets.jsonl")

    bot_edges = get_coaction_dict(tweets)
    ideology_edges = get_coaction_dict(tweets, s=600, s_lower=5)

    bot_graph = get_graph_from_coaction_dict(bot_edges)
    ideology_graph = get_graph_from_coaction_dict(ideology_edges, r=25)

    bot_summary = summarise_network(bot_graph, name="Bot Network")
    ideology_summary = summarise_network(ideology_graph, name="Ideology Network")
    print_summary(bot_summary, to_file=True)
    print_summary(ideology_summary, to_file=True)

    draw_graph(bot_graph, output="./plots/bot_graph.svg")
    draw_graph(ideology_graph, output="./plots/ideology_graph.svg")
    