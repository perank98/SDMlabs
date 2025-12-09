from utils import *
import krippendorff
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import precision_score, recall_score

if __name__ == "__main__":
    # Path to your JSONL
    file_path = "./sampled_data/2260916_only_tweets.jsonl"

    # Prepare lists to collect data
    ids = []
    types = []
    texts = []
    stances = []

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            row = orjson.loads(line)
            lang = row["account"].get("language", None)
            if lang == "en":
                ids.append(row["id"])
                texts.append(row["text"])
                types.append(row["account"]["type"])
                stances.append(row["account"]["stance"])

    # Build DataFrame
    df_analysis = pd.DataFrame({
        "ID": ids,
        "TEXT": texts,
        "TYPE": types,
        "STANCE": stances
    })

    df_train = pd.read_excel("data/train.xlsx", sheet_name="CODER1", usecols="B:D")
    df_train = df_train.dropna()
    matrix = df_train.to_numpy()
    matrix_T = matrix.T
    krippendorff.alpha(matrix_T)

    df_test = pd.read_excel("data/test.xlsx", sheet_name="CODER2", usecols="A:C")
    df_test = df_test[["ID", "TEXT", "CODE"]].dropna()
    df_test["CODE"] = df_test["CODE"].astype(int)

    for df in [df_test, df_analysis]:
        analyzer = SentimentIntensityAnalyzer()
        lexicon = load_dictionary("data/dictionary.csv")
        vader_sentiments = []
        dictionary_sentiments = []
        for sentence in df["TEXT"]:
            dictionary_sentiments.append(annotate_with_lexicon(sentence, lexicon))
            vs = analyzer.polarity_scores(sentence)
            sentiment = 0
            if vs["compound"] >= 0.5:
                sentiment = 1
            elif vs["compound"] <= -0.5:
                sentiment = -1
            vader_sentiments.append(sentiment)

        df["DICTIONARY"] = dictionary_sentiments
        df["VADER"] = vader_sentiments

    # sentiments = pd.read_csv("data/gpt.tsv", sep="\t")
    # df_test["ID"] = df_test["ID"].astype(str)
    # df_analysis["ID"] = df_analysis["ID"].astype(str)
    # sentiments["ID"] = sentiments["ID"].astype(str)
    # df_test = df_test.merge(sentiments, on="ID", how="left")
    # df_analysis = df_analysis.merge(sentiments, on="ID", how="left")

    df_test = df_test.dropna()
    df_analysis = df_analysis.dropna()
    # df_analysis["SENTIMENT"] = df_analysis["SENTIMENT"].astype(int)

    print(df_analysis)
    avg_per_type = df_analysis.groupby(["TYPE", "STANCE"])["VADER"].mean()
    print(avg_per_type)


    # for text in df_analysis.loc[df_analysis["TYPE"] == "Scientific actors", "TEXT"]:
    #     print(text)

    # for metric in ["DICTIONARY", "VADER", "SENTIMENT"]:
    #     precision = precision_score(df_test["CODE"], df_test[metric], average="macro")
    #     recall = recall_score(df_test["CODE"], df_test[metric], average="macro")

    #     print(f"Precision: {precision:.3f}")
    #     print(f"Recall: {recall:.3f}")
        