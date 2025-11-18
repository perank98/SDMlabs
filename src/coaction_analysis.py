from collections import defaultdict
from tqdm import tqdm
import igraph as ig

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