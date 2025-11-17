"""
Tweet processing and interaction graph construction.

This script reads a newline-delimited JSON tweet dataset, extracts relevant
metadata, stores simplified tweet records, and constructs directed interaction
graphs (retweets and replies) using igraph.

Two GraphML files are produced:
    - `<sample>_reply.graphml`
    - `<sample>_retweet.graphml`

Additionally, a simplified tweet dataset is written to:
    - `./sampled_data/<sample>_tweets.dat`

The script can be invoked from the command line with an optional `-n` argument
to process only the first N tweet lines (otherwise the full dataset is used).
"""

# for graph plotting
import igraph as ig
import matplotlib.pyplot as plt
import argparse

# faster than standard json package
import orjson
import pandas as pd
from tqdm import tqdm


def process_tweets(
    tweets: str = "./data/tweets.dat",
    authors: str = "./data/accounts.tsv",
    sample: int = 2260916,
)-> None:
    """
    Process raw tweet data, extract metadata, and build reply/retweet graphs.

    Parameters
    ----------
    tweets : str, optional
        Path to the newline-delimited JSON file containing tweet objects.
    authors : str, optional
        Path to a TSV file mapping IDs of popular accounts to their language, 
        type, and stance.
    sample : int, optional
        Maximum number of tweet lines to process. Defaults to 2,260,916
        (full dataset). If smaller, the function stops after reading `sample`
        valid tweet entries.

    Notes
    -----
    This function performs several operations:

    1. Loads author metadata into a dictionary for fast lookup.
    2. Iterates over tweet lines, extracting:
        - tweet ID
        - tweet text
        - timestamp (ISO-8601, truncated to seconds)
        - author metadata
        - expanded URLs (if present)
    3. Writes simplified tweets into a `.jsonl` file in JSONL format.
    4. Parses interaction types (retweets and replies) and constructs
       edge lists with associated properties.
    5. Builds two directed igraph graphs:
        - reply_graph
        - retweet_graph
    6. Outputs the graphs in GraphML format.

    ------------
    Writes files to disk:
        - ./sampled_data/<sample>_tweets.dat
        - ./sampled_data/<sample>_reply.graphml
        - ./sampled_data/<sample>_retweet.graphml
    """

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
    with open(f"./sampled_data/{sample}_tweets.jsonl", "w") as out_file, open(
        tweets, "r"
    ) as in_file:
        for line in tqdm(in_file, total=2260916):
            processed += 1
            if processed > sample:
                break
            row = orjson.loads(line)
            tweet = {
                "id": row["id"],
                "text": row["text"],
                "date": row["created_at"][:19],
            }
            author_id = int(row["author_id"])
            tweet["account"] = {"id": author_id}
            if author_id in author_meta:
                tweet["account"].update({
                    "language": author_meta[author_id]["Lang"],
                    "type": author_meta[author_id]["Type"],
                    "stance": author_meta[author_id]["Stance"]})
    
            if row.get("entities", {}).get("urls", []):
                tweet["urls"] = [url["expanded_url"] for url in row["entities"]["urls"]]
            buffer.append(orjson.dumps(tweet).decode())
            if len(buffer) >= buffer_size:
                out_file.write("\n".join(buffer) + "\n")
                buffer.clear()

            refs = row.get("referenced_tweets")
            if not isinstance(refs, list):
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
    parser = argparse.ArgumentParser(description="Process tweet dataset.")
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=None,
        help="Number of tweets to process (default: all tweets)",
    )

    args = parser.parse_args()

    if args.num is None:
        process_tweets() # processes full dataset
    else:
        process_tweets(sample=args.num)
