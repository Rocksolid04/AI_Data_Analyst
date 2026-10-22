
import asyncio
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

# Sample dataset for testing tool calling
df = pd.DataFrame({
    "Product": ["Laptop", "Phone", "Tablet"],
    "Sales": [1000, 2000, 3000],
})


@tool
def calculate_column_statistic(
    column: str,
    operation: str,
) -> dict:
    """Calculate a statistic for a numeric column in the loaded dataset.

    Args:
        column: Name of the numeric column.
        operation: One of mean, sum, min, max, or median.
    """
    allowed_operations = {
        "mean": "mean",
        "sum": "sum",
        "min": "min",
        "max": "max",
        "median": "median",
    }

    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}

    if operation not in allowed_operations:
        return {"error": "Unsupported operation."}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric."}

    result = getattr(df[column], operation)()

    return {
        "column": column,
        "operation": operation,
        "result": float(result),
    }


async def main():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY was not loaded from .env")

    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_retries=2,
    )

    tools = [calculate_column_statistic]
    tools_by_name = {item.name: item for item in tools}
    model_with_tools = model.bind_tools(tools)

    messages = [
        (
            "system",
            "You are a data analyst. Use the provided tool for "
            "calculations. Never invent calculation results.",
        ),
        HumanMessage(
            content="What is the average Sales in the dataset?"
        ),
    ]

    # Ask the LLM which tool to call
    ai_message = await model_with_tools.ainvoke(messages)
    messages.append(ai_message)

    if not ai_message.tool_calls:
        print("The model did not request a tool.")
        print(ai_message.content)
        return

    # Execute the requested tool using Python
    for tool_call in ai_message.tool_calls:
        tool_function = tools_by_name.get(tool_call["name"])

        if tool_function is None:
            raise ValueError("The model requested an unknown tool.")

        result = await tool_function.ainvoke(tool_call["args"])

        print("Tool called:", tool_call["name"])
        print("Tool result:", result)

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    # Let the LLM explain the real calculation result
    final_response = await model.ainvoke(messages)

    print("\nAgent answer:")
    print(final_response.content)


if __name__ == "__main__":
    asyncio.run(main())
