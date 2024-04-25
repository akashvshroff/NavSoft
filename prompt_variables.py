# IntentAgent

intent_options = """
intent: analysis, means that the user is trying to run analytical queries on the dataframe or learn specific results from the dataframe
intent: forecast, means that the user is trying to change parameters and rerun the forecast 
intent: simulation, means that the user is trying to schedule multiple forecasts to determine the optimum value for some parameter
intent: conversation, means that the user is trying to converse or is unsure about capabilities
"""

intent_few_shot = """
user_input: Increase inflation by 5%? intent: forecast
user_input: What are the top selling items? intent: analysis
user_input: How can I write mergesort? intent: error
user_input: What can you do? intent: conversation
user_input: What is the best discount value to maximize revenue? intent: simulation
user_input: What happens if I increase average temperature? intent: forecast
user_input: Write me a new df? intent: error
user_input: What is the dataframe about? intent: analysis
user_input: Can I ask you about the highest value items? intent: conversation
user_input: What is the optimum discount for the highest sales? intent: simulation
user_input: That is incorrect. intent: conversation
"""

# InterfaceAgent

interface_drinks_features = """[
    "discount_percentage",
    "gas_price",
    "consumer_price_index",
    "inflation",
    "average_temperature",
    "average_snow",
]"""

interface_drinks_instructions = """
Here, please consider price and discount_percentage as synonyms. If the user wants to change the price, they can only do so by changing discount_percentage in the inverse manner (i.e by - change given).
Ignore any product names or product categories that the user mentions to you. You are only concerned with features from the above list (and specified synonyms) and change values.
If a user is trying to modify multiple features (remember this doesn't mean multiple products) indicate to them what is wrong with their request and clarify their intentions. 
"""

interface_drinks_few_shot = """
user_input: What happens to sales if I increase discount by 0.0001? response: feature = discount_percentage, change = +0.0001
user_input: Increase inflation by 5%? response: feature = inflation, change = +5.0
user_input: Increase temperature by 10? response: feature = temperature, change=+10.0
user_input: Decrease by 5 percent? response: What feature do you want to change? Please indicate both feature and change percentage.
user_input: Decrease discount by 2? response: feature = discount_percentage, change = -2.0
user_input: Decrease price by 2? response: feature = discount_percentage, change=+2.0
user_input: What happens to sales if I decrease inflation by 0? response: Please specify what non-zero value to change inflation by.
user_input: What happens to sales if I increase discount for pepsi by 10%: feature = discount_percentage, change = +10.0
user_input: Increase pepsi price by 10 and generate plots. response: feature = discount_percentage, change = -10.0
user_input: What happens to sales for mountain dew and coke if inflation decreases by 20%: feature = inflation, change = -20.0
user_input: What happens to sales and revenue for drinks and food if we increase price by 10?: feature = discount_percentage, change = -10
user_input: What happens to sales if inflation increased? response: Please specify what value to change inflation by.
"""

# AnalysisAgent

analysis_drinks_few_shot = """
user_input: What products have the highest sales volume? approach: Sort products by units sold in descending order and find top sellers.
user_input: Which products yield highest profit margin? approach: Calculate the margin for each product and rank them to identify the ones with the highest margins.
user_input: What is the inventory turnover rate for high-margin products? approach: Calculate inventory turnover by comparing units sold to average inventory levels for high-margin items.
user_input: What is the week-over-week growth in sales for new products? approach: Ask the user for clarification about what new means and ask them to re-enter their query. 
user_input: How do unit sales correlate with the level of discount offered? approach: Perform regression analysis to understand the correlation between discount levels and unit sales.
user_input: What is the return on investment (ROI) for each product? approach: combine margin data with investment costs for each product to compute ROI and return in the response, don't consider the product categories themselves, only specific products.
"""

# SimulationAgent

simulation_drinks_features = """[
    "discount_percentage",
]"""

simulation_drinks_few_shot = """
user_input: What is the optimal discount for my sales? feature: discount_percentage
user_input: What is the optimal? response: Please specify what feature to run the simulation for.
user_input: What is the best value for discount? feature: discount_percentage
"""
