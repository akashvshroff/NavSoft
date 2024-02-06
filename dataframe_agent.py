import os, logging
import secret
from datetime import datetime

# langchain imports
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


os.environ["OPENAI_API_KEY"] = secret.OPENAI_KEY
logging.basicConfig(
    filename="agent_errors.txt", encoding="utf-8", level=logging.WARNING
)


class Analysis(BaseModel):
    status: int = Field(
        description="status code for query, 0 for success, 1 for clarification, 2 for out of domain questions."
    )
    response: str = Field(
        description="string output for query, either analysis, clarification question or error message"
    )


class DataframeAnalysisAgent(object):
    def __new__(cls, df=None, gpt4=True):
        """
        DataframeAnalysisAgent is a singleton class so that we can avoid recreating agents
        Use load_new_df to make any changes to the df and agent
        """
        if not hasattr(cls, "instance"):
            cls.instance = super(DataframeAnalysisAgent, cls).__new__(cls)
        return cls.instance

    def __init__(self, df=None, gpt4=True):
        self.model = "gpt-4-0125-preview" if gpt4 else "gpt-3.5-turbo-1106"
        self.parser = JsonOutputParser(pydantic_object=Analysis)
        if df is not None:
            self.load_new_df(df)

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
        if not hasattr(self, "df") or not self.df.equals(df):
            self.df = df
            self.agent = self.create_agent(0.1)

    def query(self, user_prompt):
        """
        Runs the query against the agent and returns response (or appropriate error)
        """

        prompt_template = PromptTemplate(
            template="""You are a helpful data analyst that will solve the provided user request using the dataframe appropriately.

            It is important that you answer accurately. If you do not understand the question, or cannot answer it, be clear and ask for clarifications. Remember that you should only answer questions about the dataframe. 
            
            If a user request pertains to something apart from the dataframe, reply saying that the query is out of your domain. \n{format_instructions}\n{query}
            
            In all cases, make sure you adhere to the given output format instructions. If you need more information from users, return status = 1. If the question is out of domain, return status = 2.""",
            input_variables=["query"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

        try:
            prompt = prompt_template.format(query=user_prompt)
            response = self.agent.invoke(prompt)
            raw_obj = response["output"]
            response_obj = self.parser.parse(raw_obj)
            assert isinstance(response_obj, dict)
            return response_obj

        except Exception as e:
            logging.error(f"{datetime.now()} Dataframe Agent Error: {str(e)}")
            return {
                "status": 2,
                "response": "An unknown error occurred. Please try again later.",
            }
