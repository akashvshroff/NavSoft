import pandas as pd

from dataframe_agent import DataframeAnalysisAgent

from interface_agent import InterfaceAgent


def read_data(is_parquet, filename="./data/BaseTable_v1.parquet"):
    if is_parquet:
        return pd.read_parquet(filename)
    else:
        return pd.read_csv(filename)


def driver():
    # df = read_data(True)
    # df needs to be read in client side and then passed in to agent
    # create analysis agent using dataframe - singleton class so only instantiates once, to change df use load_new_df
    # TODO: might need to keep loading new dfs every time the forecast is regenerated - but probably a good idea to avoid agent creation time
    agent = InterfaceAgent()

    while (
        True
    ):  # could be a running loop for queries with loading spinner - error handling done in agent itself
        prompt = input("Please enter your query below? \n\n")
        response = agent.query(prompt)
        print(f"> {response['response']}\n")
        print("-" * 25)


if __name__ == "__main__":
    driver()
