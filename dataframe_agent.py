import os, logging
import secret
from datetime import datetime

# variables
import prompt_variables

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


class DataframeAnalysisAgent:
    def __init__(self, df=None, model="gpt-4-turbo"):
        self.model = model
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

    def query(self, user_input, conversation_history):
        """
        Runs the query against the agent and returns response (or appropriate error)
        """

        prompt_template = PromptTemplate(
            template="""
            You are a helpful data analyst that will solve the provided user request using the given dataframe appropriately.

            It is important that you answer accurately. If you do not understand the question, or cannot answer it, be clear and ask for clarifications. Remember that you should only answer questions about the dataframe. 
            Your job is to help the user, and they may not know how to instruct you appropriately. In this case, make sure you ask clarifying questions about what analysis to conduct. 

            Here are some example user inputs, and potential approaches you could use to solving the problem:
            {analysis_few_shot}
            
            If a user request pertains to something apart from the dataframe, reply saying that the query is out of your domain.
            
            Make sure you strictly adhere to the following format instructions.
            {format_instructions}

            You must strictly adhere to the given format with the JSON object of response and status.
            Your response field must be a string in markdown format. If the user requires some data in the form of a table or a list, make sure you leverage the markdown tools for formatting data in that manner. 
            If you use latex for any equations in your markdown response string, use double backslashes to avoid any JSON encoding errors. Make sure that you adhere to this. 
                                   
            Make sure that you provide intermediate results in the response that you provide. Don't provide code or possible steps but rather results from intermediate steps that you have completed. The user should be confident
            that your working is correct. 

            If you do not understand what the user is asking or cannot answer it clearly given the dataframe, make sure you return status = 1 and pose a clarifying question.
            In all cases, make sure you adhere to the given output format instructions. If the question is out of domain, return a polite error message and status = 2.

            You must also use the chat history (if any) to understand the user request in case they are clarifying any previous questions. 
            {conversation_history}
            
            {user_input}""",
            input_variables=["user_input", "conversation_history"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
                "analysis_few_shot": prompt_variables.analysis_drinks_few_shot,
            },
        )

        try:
            prompt = prompt_template.format(
                user_input=user_input, conversation_history=conversation_history
            )
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
