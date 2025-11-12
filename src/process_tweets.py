# for graph plotting
import igraph as ig
import matplotlib.pyplot as plt

# faster than standard json package
import orjson
import pandas as pd
from tqdm import tqdm


def process_tweets(
    tweets: str = "./data/tweets.dat",
    authors: str = "./data/accounts.tsv",
    sample: int = 2260916,
):

    authors_df = pd.read_csv(authors, sep="\t")
    author_meta = authors_df.set_index("author_id")[["Lang", "Type", "Stance"]].to_dict(
        orient="index"
    )

    # buffer to not write each line to output individually
    processed = 0
    buffer = []
    buffer_size = 10000
    retweet_edges = []
    reply_edges = []
    # count number of tweets with multiple references (might be worth keeping them in; currently ~400 removed)
    multiple_references = 0
    # with open("tweets_output.jsonl", "w") as out_file, open("../data/first1000_tweets.dat", "r") as in_file:
    with open(f"./sampled_data/{sample}_tweets.dat", "w") as out_file, open(
        tweets, "r"
    ) as in_file:
        for line in tqdm(in_file, total=2260916):
            row = orjson.loads(line)
            tweet = {
                "id": row["id"],
                "text": row["text"],
                "date": row["created_at"][:10],
            }
            author_id = int(row["author_id"])
            if author_id in author_meta:
                tweet["account"] = {
                    "id": author_id,
                    "language": author_meta[author_id]["Lang"],
                    "type": author_meta[author_id]["Type"],
                    "stance": author_meta[author_id]["Stance"],
                }
            buffer.append(orjson.dumps(tweet).decode())
            if len(buffer) >= buffer_size:
                out_file.write("\n".join(buffer) + "\n")
                buffer.clear()

            refs = row.get("referenced_tweets")
            if not isinstance(refs, list):
                processed += 1
                if processed == sample:
                    break
                continue
            if len(refs) != 1:
                multiple_references += 1
                continue
            tweet_type = refs[0]["type"]
            if tweet_type in ["retweeted", "replied_to"]:
                # Extract target account
                entities = row.get("entities", {})
                mentions = entities.get("mentions", [])
                if not mentions:
                    continue
                target = int(mentions[0]["id"])

                # Add edge
                edge = (
                    author_id,
                    target,
                    row["public_metrics"]["retweet_count"],
                    row["id"],
                    row["possibly_sensitive"],
                )
                if tweet_type == "retweeted":
                    retweet_edges.append(edge)
                else:
                    reply_edges.append(edge)

                processed += 1
                if processed == sample:
                    break

        # add tweets still remaining in buffer
        if buffer:
            out_file.write("\n".join(buffer) + "\n")

    # create a reply & a retweet graph
    reply_graph = ig.Graph.TupleList(
        reply_edges,
        vertex_name_attr="account_id",
        edge_attrs=["weight", "tweet_id", "possible_sensitive"],
    )
    retweet_graph = ig.Graph.TupleList(
        retweet_edges,
        vertex_name_attr="account_id",
        edge_attrs=["weight", "tweet_id", "possible_sensitive"],
    )

    print("Number of multiple references", multiple_references)

    # write the graphs to .graphml files
    reply_graph.write_graphml(f"./sampled_data/{sample}_reply.graphml")
    retweet_graph.write_graphml(f"./sampled_data/{sample}_retweet.graphml")


if __name__ == "__main__":
    process_tweets(sample=20000)
