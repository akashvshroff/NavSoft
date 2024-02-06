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

        """
        Response object has the following structure:
        {
            status: 0 for success, 1 for clarification, 2 for error (could maybe give a colour indication for error), 
            response: to be printed for the user, either result of analysis, loading message for forecast or error message, 
            intent: forecast or analysis, in case no error - outlines what the user wants to do
            feature: parameter that the user wants to edit in case of forecast
            change: float value indicating percent change that user wants to make (+ve for increase, -ve for decrease)
        }

        In case of intent==forecast, use the feature and change value to make new prediction and then update the df passed in params to query
        so that the user has the new df for analysis
        """

        print(f"> {response}\n")
        print("-" * 25)
        print("")


if __name__ == "__main__":
    driver()
