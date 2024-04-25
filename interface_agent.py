import os, logging
import secret
from datetime import datetime

# variables
import prompt_variables

# langchain imports
from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.memory import ConversationSummaryBufferMemory

os.environ["OPENAI_API_KEY"] = secret.OPENAI_KEY

logging.basicConfig(
    filename="agent_errors.txt", encoding="utf-8", level=logging.WARNING
)


class Forecast(BaseModel):
    status: int = Field(
        description="status code for query inference, 0 for success, 1 for clarifications or incomplete prompts, 2 for errors."
    )
    feature: str = Field(
        description="feature that is going to be modified in the forecast"
    )
    change: float = Field(
        description="percentage amount that the feature should change by"
    )
    response: str = Field(description="response message for user")


class InterfaceAgent:
    def __init__(self, model_name="gpt-4-turbo"):
        self.model = ChatOpenAI(model=model_name, temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=Forecast)
        self.chain = None
        self.create_chain()

    def create_chain(self):
        prompt = PromptTemplate(
            template="""
            You are a data scientist chatbot whose role is to understand a user request.
            The user is instructing the system on what parameters of a forecast to modify. Your role is to identify the values of these parameters.

            Here is a list of features that the user can modify:
            {features}. 
            
            Given a user string, you need to identify the most appropriate feature from the given list that they are trying to modify as well as the percentage change (as a float, positive indicating increase and negative indicating decrease).

            Here are some examples for user_input and appropriate outputs.

            {forecast_few_shot}
            
            {forecast_instructions}

            Remember the feature can only be a value from the above list, if the user indicates a feature outside the list, either pick the most appropriate one or clarify by giving them the available options.
            If the user has not specified these two parameters, you need to clarify and ask them to indicate both the feature and the percent change.
            
            You may need to help the user set up a forecast itself. If the user input doesn't contain information about feature and change (they are missing both parameters), respond with a clarification (status = 1). You need to respond informing users of available features they can modify (from above) and encouraging them to provide the necessary parameters. Be helpful and polite.
            If the user is missing only one parameter, acknowledge the parameter that they have passed (only if they have explicitly passed one), note it (by populating either feature or change) and respond with a clarification (status = 1), and information about the parameter they are missing and ask them how they want to proceed.
            You must ignore any additional user request for analysis or to generate plots. Your role is to only help users generate forecasts, and to infer features and change values.
            If the user input is valid, use the response field to indicate that their forecast will be generated shortly. Your responses must be helpful, polite and in complete sentences.

            Make sure you adhere to the given output format instructions. 
            {format_instructions}

            You must also use the chat history (if any) to understand the user request in case they are clarifying any previous questions. Make sure to pay close attention to their responses and use that to fill in information about the feature and change. If you can determine both of these, then the request is successful (status = 0). You must adhere to the above formatting instructions strictly.
            {conversation_history}
            {user_input}
            """,
            input_variables=["user_input", "conversation_history"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
                "features": prompt_variables.interface_drinks_features,
                "forecast_few_shot": prompt_variables.interface_drinks_few_shot,
                "forecast_instructions": prompt_variables.interface_drinks_instructions,
            },
        )
        self.chain = prompt | self.model | self.parser

    def query(self, user_input, conversation_history):
        """
        Parses user input to extract relevant forecasting information
        """

        try:

            response_obj = self.chain.invoke(
                {"user_input": user_input, "conversation_history": conversation_history}
            )
            assert isinstance(response_obj, dict)
            return response_obj

        except Exception as e:
            logging.error(f"{datetime.now()} Interface Agent Error: {str(e)}")
            return {
                "status": 2,
                "response": "An unknown error occured. Please try again later.",
            }
