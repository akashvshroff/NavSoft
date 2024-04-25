import os, logging
import secret
from datetime import datetime

# variables
import prompt_variables

# langchain imports
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

os.environ["OPENAI_API_KEY"] = secret.OPENAI_KEY

logging.basicConfig(
    filename="agent_errors.txt", encoding="utf-8", level=logging.WARNING
)


class Simulation(BaseModel):
    status: int = Field(
        description="status code for query inference, 0 for success, 1 for clarifications or incomplete prompts, 2 for errors."
    )
    feature: str = Field(
        description="feature that is going to be modified in the forecast"
    )
    response: str = Field(description="response message for user")


class SimulationAgent:
    def __init__(self, model_name="gpt-4-turbo"):
        self.model = ChatOpenAI(model=model_name, temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=Simulation)
        self.chain = None
        self.create_chain()

    def create_chain(self):
        prompt = PromptTemplate(
            template="""
            You are a data scientist chatbot whose role is to understand a user request.
            The user is trying to instruct the system on the parameters of a simulation that they want to run. 

            Here is a list of features that the user can modify:
            {features}. 
            
            Given a user string, you need to identify the most appropriate feature from the given list that they are trying to modify. If the user gives you a feature outside the list, clarify the available features and ask them again.

            Here are some examples for user_input and appropriate outputs.

            {simulate_few_shot}
            
            Remember the feature can only be a value from the above list, if the user indicates a feature outside the list, either pick the most appropriate one or clarify by giving them the available options.
            If the user has not specified the feature, you need to clarify and ask them to indicate the feature.
            
            You may need to help the user set up a simulation itself. If the user input doesn't contain information about feature, respond with a clarification (status = 1). You need to respond informing users of available features they can modify (from above) and encouraging them to provide the necessary parameters. Be helpful and polite.
            You must ignore any additional user request for analysis or to generate plots. Your role is to only help users generate simulations, and to infer features.
            If the user input is valid, use the response field to indicate that their simulation will be generated shortly. Your responses must be helpful, polite and in complete sentences.

            Make sure you adhere to the given output format instructions. 
            {format_instructions}

            You must also use the chat history (if any) to understand the user request in case they are clarifying any previous questions. Make sure to pay close attention to their responses and use that to fill in information about the feature for the simulation. If you can determine the feature, then the request is successful (status = 0). You must adhere to the above formatting instructions strictly.
            {conversation_history}
            {user_input}
            """,
            input_variables=["user_input", "conversation_history"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
                "features": prompt_variables.simulation_drinks_features,
                "simulate_few_shot": prompt_variables.simulation_drinks_few_shot,
            },
        )
        self.chain = prompt | self.model | self.parser

    def query(self, user_input, conversation_history):
        """
        Parses user input to extract relevant simulation information
        """

        try:

            response_obj = self.chain.invoke(
                {"user_input": user_input, "conversation_history": conversation_history}
            )
            assert isinstance(response_obj, dict)
            return response_obj

        except Exception as e:
            logging.error(f"{datetime.now()} Simulation Agent Error: {str(e)}")
            return {
                "status": 2,
                "response": "An unknown error occured. Please try again later.",
            }
