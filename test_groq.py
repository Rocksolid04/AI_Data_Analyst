
import asyncio
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

async def main():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not loaded. Check the .env file."
        )

    model = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0,
        max_retries=2,
    )

    response = await model.ainvoke(
        [
            HumanMessage(
                content="Reply with exactly: Groq connection successful"
            )
        ]
    )

    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
