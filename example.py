import pandas as pd

from intent_agent import IntentAgent


def read_data(is_parquet, filename="./data/BaseTable_v1.parquet"):
    if is_parquet:
        return pd.read_parquet(filename)
    else:
        return pd.read_csv(filename)


def driver():
    original_df = read_data(True)  # initial df needs to be read client side
    agent = IntentAgent()
    params = {"df": original_df}  # optional params
    while (
        True
    ):  # could be a running loop for queries with loading spinner - error handling done in agent itself
        prompt = input("Q: ")
        response = agent.query(prompt, params)
        print(f"> {response}\n")
        print("-" * 25)
        print("")


if __name__ == "__main__":
    driver()
