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
    conversation_history = []  # keep appending while the clarification loop is on
    last_intent = ""
    while True:
        prompt = input("Q: ")
        params = {"df": original_df, "conversation_history": conversation_history}
        response = agent.query(prompt, params)

        # changes made to parameters
        if response["status"] != 1:  # if no clarification then reset memory
            conversation_history = []
            last_intent = ""
        else:
            if "intent" in response and response["intent"] == last_intent:
                conversation_history.append(
                    {"user_input": prompt, "response": response["response"]}
                )
            else:
                conversation_history = [
                    {"user_input": prompt, "response": response["response"]}
                ]
                last_intent = response.get("intent", "")

            # actions based on response

        print(f"> {response}\n")
        print("-" * 25)
        print(conversation_history)
        print(f"last intent: {last_intent}")
        print("-" * 25)
        print("")


if __name__ == "__main__":
    driver()
