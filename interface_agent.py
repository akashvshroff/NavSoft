import os, logging
import secret

# langchain imports
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

os.environ["OPENAI_API_KEY"] = secret.OPENAI_KEY

logging.basicConfig(
    filename="llm_interface_errors.txt", encoding="utf-8", level=logging.WARNING
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
    response: str = Field(description="response for user depending on status")


class InterfaceAgent:
    def __new__(
        cls,
        gpt4=True,
        features=["Discount_Perc", "gas_prices", "cpi", "inflation", "tavg", "snow"],
    ):
        """
        Interface Agent is a singleton class so that we don't need to reload the parser and model chain.
        """
        if not hasattr(cls, "instance"):
            cls.instance = super(InterfaceAgent, cls).__new__(cls)
            cls.instance.gpt4 = gpt4
        return cls.instance

    def __init__(
        self,
        gpt4=True,
        features=["Discount_Perc", "gas_prices", "cpi", "inflation", "tavg", "snow"],
    ):
        model_name = "gpt-4-0125-preview" if gpt4 else "gpt-3.5-turbo-1106"
        self.model = ChatOpenAI(model=model_name, temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=Forecast)
        self.chain = None
        self.features = features
        self.create_chain()

    def create_chain(self):
        prompt = PromptTemplate(
            template="""
            You are a helpful data scientist that is helping understand some provided user request. The user is trying to instruct the system on what parameters of a forecast to modify. You are going to help the system understand what parameters can be modified.

            Here is a list of features that the user can modify: {features}. The user must also specify a percent increase or decrease in value for the feature. Given a user string, you need to identify the most appropriate feature from the given list that they are trying to modify as
            well as the percentage change (as a float, positive indicating increase and negative indicating decrease). Remember the feature can only be a value from the above list, if the user indicates a feature outside the list, pick the most appropriate one from the list itself.
            If the user has not specified these two parameters, you need to ask them to indicate both the feature and the percent change.

            Eg: user_input: Increase inflation by 5 percent? response: feature = cpi, float = +5.0
            Eg: user_input: Decrease by 5 percent? response: What feature do you want to change? Please indicate both feature and change percentage.
            Eg: user_input: Decrease discount by 2? response: feature = Discount_Perc, float = -2.0
            Eg: user_input: Decrease price by 2? response: Price is not a feature you can modify, please pick an appropriate feature.

            If a user is trying to modify multiple features indicate to them what is wrong with their request. If the user request isn't related to modifying features, return an error (status = 2) saying that I can only help with queries relating to changes in the forecast.
            \n{format_instructions}\n{user_input}

            Make sure you adhere to the given output format instructions. The feature can only be one of the features from the list that the user can modify. If the user input is missing data, use status = 1 and ask for clarifications in the response field. For errors or actions beyond the scope, use status = 2 and indicate
            why in the response section. If the user input is valid, use the response field to indicate that their forecast will be generated shortly.
            """,
            input_variables=["features", "user_input"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.chain = prompt | self.model | self.parser

    def query(self, user_input):
        """
        Parses user input to extract relevant forecasting information
        """

        try:
            response_obj = self.chain.invoke(
                {"features": ", ".join(self.features), "user_input": user_input}
            )
            assert isinstance(response_obj, dict)
            print(response_obj)
            return response_obj

        except Exception as e:
            logging.error(str(e))
            return {
                "status": 2,
                "response": "An unknown error occured. Please try again later.",
            }
