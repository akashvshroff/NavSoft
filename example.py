import pandas as pd
from intent_agent import IntentAgent
from IPython.display import display, Markdown


def read_data(is_parquet, filename="./data/all_data.csv"):
    if is_parquet:
        return pd.read_parquet(filename)
    else:
        return pd.read_csv(filename)


def driver():
    original_df = read_data(
        False, "./data/all_data.csv"
    )  # initial df needs to be read client side
    original_df = original_df[original_df["data_type"] == "forModel"]
    agent = IntentAgent()
    params = {"df": original_df}  # params - df and features
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
            intent: forecast, analysis or simulation, in case no error - outlines what the user wants to do
            feature: parameter that the user wants to edit in case of forecast
            change: float value indicating percent change that user wants to make (+ve for increase, -ve for decrease)
        }

        In case of intent==forecast, use the feature and change value to make new prediction and then update the df passed in params to query
        so that the user has the new df for analysis
        """

        # if response["intent"] == "forecast":
        #     feature = response["feature"]
        #     change = response["change"]
        #     df = model.make_prediction({feature: change})
        #     params["df"] = df  # update parameters for analysis
        #     # or break after new query

        # if response["intent"] == "simulation"
        # run the model 5 times changing response["feature"] (by default only "discount_percentage") from 0 to 5

        # for both forecast and simulation, pass the results to the LLM to analyze using the function agent.analyze_results
        # you will need to pass it both the original user_input and the resultant dataframe.

        print(f"> {response}\n")
        print("-" * 25)
        print("")


if __name__ == "__main__":
    driver()
