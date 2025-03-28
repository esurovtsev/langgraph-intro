from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
import operator
from typing import Annotated
from pydantic import BaseModel
import yfinance as yf
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, START

class InvestmentAdvisorState(TypedDict):
    financial_area: str
    stock_number: int
    stock_tickers: list[str]
    stock_details: Annotated[list, operator.add]
    recommendation: str

llm = ChatOpenAI(model="gpt-4o-mini")


class StockTickers(BaseModel):
    stock_tickers: list[str]


def generate_list_of_stocks(state: InvestmentAdvisorState):
    prompt = f"""
    You are an experienced financial analyst. 
    Based on current market trends and public data, suggest the top {state["stock_number"]} publicly traded companies in the area of "{state["financial_area"]}".

    Return only a list of their stock tickers (symbols), without any explanations.
    """
    
    response = llm.with_structured_output(StockTickers).invoke(prompt)
    return {"stock_tickers": response.stock_tickers}


class StockState(TypedDict):
    ticker: str

def fetch_stock_details(state: StockState) -> dict:
    period = "1mo"
    stock_symbol = state["ticker"]
    try:
        stock = yf.Ticker(stock_symbol)

        # Retrieve general stock info and historical market data
        stock_info = stock.info  # Basic company and stock data
        stock_history = stock.history(period=period).to_dict()  # Historical OHLCV data

        # Combine both into a single dictionary
        stock = {
            "stock_symbol": stock_symbol,
            "info": stock_info,
            "history": stock_history
        }
        
        return {"stock_details": [str(stock)]}
        
    except Exception as e:
        return {"error": f"Error fetching stock data for {stock_symbol}: {str(e)}"}


def generate_stock_recommendations(state: InvestmentAdvisorState):
    financial_data = "\n\n\n".join(state["stock_details"])

    prompt = f"""
    You are a professional investment advisor.

    Below is a list of companies with their financial data:

    {financial_data}

    Your task is to:
    1. Analyze each company based on the provided technical information (consider only companies from the list above).
    2. Write a short paragraph for each company that includes:
    - The company name and ticker
    - A brief description
    - Why it may or may not be a good investment
    3. Rank the companies from best to worst in terms of investment potential.
    4. Return the result as a sorted list (highest priority first), where each item includes:
    - Rank (starting from 1)
    - Ticker
    - Company name
    - Short description
    - Reason for its ranking
    """

    recommendation = llm.invoke([HumanMessage(content=prompt)])

    return {"recommendation": recommendation.content}



builder = StateGraph(InvestmentAdvisorState)

builder.add_node("generate_list_of_stocks", generate_list_of_stocks)
builder.add_node("fetch_stock_details", fetch_stock_details)
builder.add_node("generate_stock_recommendations", generate_stock_recommendations)


builder.add_edge(START, "generate_list_of_stocks")


# Map - Reducing
from langgraph.constants import Send
def continue_to_details(state: InvestmentAdvisorState):
    return [Send("fetch_stock_details", {"ticker": ticker}) for ticker in state["stock_tickers"]]

builder.add_conditional_edges("generate_list_of_stocks", continue_to_details, ["fetch_stock_details"])



builder.add_edge("fetch_stock_details", "generate_stock_recommendations")
builder.add_edge("generate_stock_recommendations", END)

graph = builder.compile(interrupt_before=["fetch_stock_details"])