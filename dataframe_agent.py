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
        description="markdown string output for query, either analysis, clarification question or error message"
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

    def create_agent(self, temp=0.0):
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
            Your job is to help the user, so even if their question is incorrectly phrased, make assumptions about what they might have intended and answer those. However, state your assumptions clearly in your response.

            Here are some example user inputs, and a potential approach you could use to solving the problem:
            Eg: user_input: What products have the highest sales volume? approach: Sort products by units sold in descending order and find top sellers.
            Eg: user_input: Which products yield highest profit margin? approach: Calculate the margin for each product and rank them to identify the ones with the highest margins.
            Eg: user_input: What is the inventory turnover rate for high-margin products? approach: Calculate inventory turnover by comparing units sold to average inventory levels for high-margin items.
            Eg: user_input: What is the week-over-week growth in sales for new products? approach: Ask the user for clarification about what new means and ask them to re-enter their query. 
            Eg: user_input: How do unit sales correlate with the level of discount offered? approach: Perform regression analysis to understand the correlation between discount levels and unit sales.
            Eg: user_input: What is the return on investment (ROI) for each product? approach: combine margin data with investment costs for each product to compute ROI and return in the response, don't consider the product categories themselves, only specific products.
            
            If a user request pertains to something apart from the dataframe, reply saying that the query is out of your domain.
            
            \n{format_instructions}\n{query}

            You must strictly adhere to the given format with the JSON object of response and status.
            Your response field must be a string in markdown format. If the user requires some data in the form of a table or a list, make sure you leverage the markdown tools for formatting data in that manner. 
            If you use latex in your markdown response string, use double backslashes to avoid any JSON encoding errors. 
                                   
            Make sure that you provide intermediate results in the response that you provide. Don't provide code or possible steps but rather results from intermediate steps that you have completed. The user should be confident
            that your working is correct. 

            If you do not understand what the user is asking or cannot answer it clearly given the dataframe, make sure you return status = 1 and in your response explain what clarification you need.
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
