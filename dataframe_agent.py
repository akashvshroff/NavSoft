import os
import pandas as pd
import secret

# langchain imports
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI, OpenAI

os.environ["OPENAI_API_KEY"] = secret.OPENAI_KEY


def read_data(is_parquet, filename="./data/BaseTable_v1.parquet"):
    if is_parquet:
        return pd.read_parquet(filename)
    else:
        return pd.read_csv(filename)


def create_agent(df, model="gpt-3.5-turbo-0613", temp=0):
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=temp, model=model),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    return agent


def driver():
    df = read_data(True)
    agent = create_agent(df)

    while True:
        prompt = input("Please enter your query below? \n")
        try:
            agent.run(prompt)
        except Exception as e:
            print(f"There was an error: {e}. Please try again.")
        print("-" * 25)


if __name__ == "__main__":
    driver()
