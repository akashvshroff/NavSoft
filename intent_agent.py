import os, logging
import secret
from datetime import datetime

# agents
from interface_agent import InterfaceAgent
from dataframe_agent import DataframeAnalysisAgent

# langchain imports
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

os.environ["OPENAI_API_KEY"] = secret.OPENAI_KEY

logging.basicConfig(
    filename="agent_errors.txt", encoding="utf-8", level=logging.WARNING
)


class Intent(BaseModel):
    intent: str = Field(
        "the task that the user is trying to accomplish, forecast, analysis or error."
    )


class IntentAgent:
    def __init__(self, gpt4=True):
        """
        Creates llm agent to recognize user intent and forward to respective agent.
        """
        self.gpt4 = gpt4
        model_name = "gpt-4-0125-preview" if gpt4 else "gpt-3.5-turbo-1106"
        self.model = ChatOpenAI(model=model_name, temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=Intent)
        self.chain = None
        self.create_chain()

    def create_chain(self):
        prompt = PromptTemplate(
            template="""You are a helpful data scientist who is trying to understand user intent. Based on a given user input, you have to determine whether the
            user is trying to run analysis on a dataframe, generate a forecast or neither. If they are trying to do neither, you must simply return error.

            Eg: user_input: Increase inflation by 5 percent? intent: forecast
            Eg: user_input: What are the top selling items? intent: analysis
            Eg: user_input: How can I write mergesort? intent: error
            Eg: user_input: What happens if I increase average temperature? intent: forecast
            Eg: user_input: Write me a new df? intent: error

            If you are unsure about what the user is trying to do, pick the most appropriate option. If their statement does not relate to analysis or forecast, mark it as an error.
            \n{format_instructions}\n{user_input}
            """,
            input_variables=["user_input"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.chain = prompt | self.model | self.parser

    def query(self, user_input, df=None):
        """
        Recognizes user intent and calls on the appropriate agent to handle the query.
        """
        try:
            response_obj = self.chain.invoke({"user_input": user_input})
            assert isinstance(response_obj, dict)
            intent = response_obj["intent"]
            agent = None
            if intent == "error":
                return {
                    "status": 2,
                    "response": "Sorry, I can only help you with queries relating to forecasting or analysis of data.",
                }

            elif intent == "forecast ":
                agent = InterfaceAgent(self.gpt4)

            elif intent == "analysis":
                agent = DataframeAnalysisAgent(df, self.gpt4)
                if (
                    df is not None
                ):  # load in new df - need to do it when a forecast is done
                    agent.load_new_df(df)
            else:
                return {
                    "status": 2,
                    "response": "An unknown error occured. Please try again later.",
                }

            agent_response_obj = agent.query(user_input)
            print(agent_response_obj)
            agent_response_obj["intent"] = intent
            return agent_response_obj

        except Exception as e:
            logging.error(f"{datetime.now()} Intent Agent Error: {str(e)}")
            return {
                "status": 2,
                "response": "An unknown error occured. Please try again later.",
            }
