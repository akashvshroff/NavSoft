import pandas as pd

from dataframe_agent import DataframeAnalysisAgent


def read_data(is_parquet, filename="./data/BaseTable_v1.parquet"):
    if is_parquet:
        return pd.read_parquet(filename)
    else:
        return pd.read_csv(filename)


def driver():
    df = read_data(True)
    agent = DataframeAnalysisAgent(df, False)

    while True:
        prompt = input("Please enter your query below? \n\n")
        response = agent.query(prompt)
        print(f"> {response}\n")
        print("-" * 25)


if __name__ == "__main__":
    driver()
