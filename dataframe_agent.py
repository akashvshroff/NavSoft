import os, logging
import pandas as pd
import secret

# langchain imports
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = secret.OPENAI_KEY
logging.basicConfig(
    filename="df_agent_errors.txt", encoding="utf-8", level=logging.WARNING
)


class DataframeAnalysisAgent(object):
    def __new__(cls, df, gpt4=True):
        """
        DataframeAnalysisAgent is a singleton class so that we can avoid recreating agents
        Use load_new_df to make any changes to the df and agent
        """
        if not hasattr(cls, "instance"):
            cls.instance = super(DataframeAnalysisAgent, cls).__new__(cls)
            cls.instance.df = df
            cls.instance.gpt4 = gpt4
        return cls.instance

    def __init__(self, df, gpt4=True):
        self.df = df
        self.model = "gpt-4-0125-preview" if gpt4 else "gpt-3.5-turbo-1106"
        self.agent = self.create_agent(0.1)

    def create_agent(self, temp=0.1):
        """
        Create chat agent with given df and model.
        """
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=temp, model=self.model),
            self.df,
            verbose=False,  # set to true if debugging
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        return agent

    def load_new_df(self, df):
        """
        Function to add new dataframe and update agent
        """
        self.df = df
        self.agent = self.create_agent(0.1)

    def query(self, user_prompt):
        """
        Runs the query against the agent and returns response (or appropriate error)
        """
        prompt = f"""You are a helpful data analyst that will solve the provided user request using the dataframe appropriately.
        It is important that you answer accurately. If you do not understand the question, or cannot answer it, be clear and ask for clarifications.
        Remember that you should only answer questions about the dataframe. If a user request pertains to something apart from the dataframe, reply saying that 
        the query is out of your domain.

        User Request: {user_prompt}
        """
        try:
            response = self.agent.invoke(prompt)
            return response["output"]
        except Exception as e:
            logging.error(str(e))
            return "An unknown error occured. Please try again later."
