from utils import *
import pandas as pd
import krippendorff

if __name__ == "__main__":
    # tweet_dict = {}
    # with open(f"./sampled_data/2260916_only_tweets.jsonl", "r") as file:
    #     for line in file:
    #         row = orjson.loads(line)
    #         tweet_type = row["account"].get("type", None)
    #         if tweet_type is not None:
    #             tweet_dict.setdefault(tweet_type, []).append(row["text"])

    # print(tweet_dict.keys())

    df = pd.read_excel("data/train.xlsx", sheet_name="CODER1", usecols="B:D")
    df = df.dropna()
    matrix = df.to_numpy()
    matrix_T = matrix.T
    krippendorff.alpha(matrix_T)

    df = pd.read_excel("data/train.xlsx", sheet_name="CODER2", usecols="A:C")
    df = df.dropna()
    print(df.head)
    # matrix = df.to_numpy()
    # matrix_T = matrix.T
    # krippendorff.alpha(matrix_T)
