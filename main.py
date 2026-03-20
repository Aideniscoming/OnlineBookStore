# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain.tools import tool
# from langchain_core.messages import HumanMessage
# from langgraph.prebuilt import create_react_agent
# import requests
# # ---------------------------
# # Load API key from .env file
# # ---------------------------
# load_dotenv()

# # Module-level conversation so tools (e.g. `save_conversation`) can access it.
# conversation = []

# # ---------------------------
# # Define tools
# # ---------------------------
# search_tool = DuckDuckGoSearchRun()


# @tool
# def BookSearch(query: str) -> str:
#     """
#     Search for books using the Google Books API and provides a librarian-style recommendation.
    
#     Args:
#         query (str): The search term or user interest (e.g., 'quantum physics' or 'cooking').
        
#     Returns:
#         str: A formatted recommendation with book details and suitability analysis.
#     """
#     # Google Books API Endpoint for searching volumes
#     url = "https://www.googleapis.com/books/v1/volumes"
#     params = {
#         "q": query,
#         "maxResults": 1,  # We only want the best match
#         "printType": "books"
#     }

#     try:
#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         data = response.json()
        
#         if "items" not in data:
#             return f"I'm sorry, I couldn't find any books matching '{query}' in our digital archives."

#         # Extract the first book's data
#         book_info = data["items"][0]["volumeInfo"]
        
#         title = book_info.get("title", "Unknown Title")
#         authors = ", ".join(book_info.get("authors", ["Unknown Author"]))
#         pub_date = book_info.get("publishedDate", "N/A")
#         description = book_info.get("description", "No description available.")

#         # Constructing the Librarian's Analysis
#         # We use a snippet of the description to justify the choice
#         analysis_preview = description[:200] + "..." if len(description) > 200 else description
        
#         recommendation = (
#             f"## 📚 Librarian's Choice: {title}\n"
#             f"**Author(s):** {authors}\n"
#             f"**Published:** {pub_date}\n\n"
#             f"--- \n"
#             f"### Why this fits your interest:\n"
#             f"Based on your search for *\"{query}\"*, I've selected this volume. It covers significant ground, "
#             f"specifically focusing on: \n\n> {analysis_preview}\n\n"
#             f"It is a highly relevant resource for someone looking to dive deeper into this subject."
#         )
        
#         return recommendation

#     except requests.exceptions.RequestException as e:
#         return f"Error connecting to the library service: {str(e)}"

# # --- Example Call ---
# # result = BookSearch("modern architecture in Japan")
# # print(result)

    

# @tool
# def calculator(a: float, b: float, operation: str = "add") -> str:
#     """
#     Perform basic math operations. 
#     Supported operations: 'add', 'subtract', 'multiply', 'divide'.
#     """
#     try:
#         op = operation.lower()
#         if op == "add":
#             return str(a + b)
#         elif op in ["subtract", "minus"]:
#             return str(a - b)
#         elif op in ["multiply", "times"]:
#             return str(a * b)
#         elif op in ["divide"]:
#             if b == 0:
#                 return "Error: Division by zero."
#             return str(a / b)
#         else:
#             return f"Unsupported operation: {operation}. Please use add, subtract, multiply, or divide."
#     except Exception as e:
#         return f"Error: {e}"


# # ---------------------------
# # Tool to save conversation
# # ---------------------------
# @tool
# def save_conversation(filename: str = "conversation.txt") -> str:
#     """
#     Save the current conversation to a text file inside 'history_conversation' folder.
    
#     Parameters:
#     - filename (str): Name of the file to save conversation
#     """
#     try:
#         global conversation
#     except NameError:
#         return "No conversation found to save."

#     # Ensure folder exists
#     folder = "history_conversation"
#     os.makedirs(folder, exist_ok=True)

#     # Full file path
#     filepath = os.path.join(folder, filename)

#     # Write conversation to the file
#     with open(filepath, "w", encoding="utf-8") as f:
#         for message in conversation:
#             role = getattr(message, "type", "message").upper()
#             f.write(f"{role}: {getattr(message, 'content', '')}\n")
    
#     return f"Conversation saved to {filepath}"

# tools = [calculator, search_tool, save_conversation, BookSearch]

# # ---------------------------
# # Create agent
# # ---------------------------
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# system_instructions = (
#     "You are a helpful AI assistant. "
#     "Use the calculator tool for math problems and DuckDuckGo for word definitions or facts. "
#     "Give concise, direct answers."
# )

# agent = create_react_agent(llm, tools=tools, prompt=system_instructions)

# # ---------------------------
# # Chat loop with memory
# # ---------------------------
# def main():
#     print("--- AI Math & Word Assistant ---")
#     print("Type 'exit' or 'quit' to stop.\n")

#     # Conversation history (messages only; the system prompt is supplied via `create_react_agent(prompt=...)`)
#     global conversation
#     conversation = []

#     while True:
#         query = input("You: ").strip()
#         if query.lower() in ["exit", "quit"]:
#             print("Goodbye!")
#             break
#         if not query:
#             continue

#         try:
#             # Add user input to conversation
#             conversation.append(HumanMessage(content=query))

#             # Send full conversation to agent
#             response = agent.invoke({"messages": conversation})

#             # Update local conversation state with agent results (AI + tool messages).
#             conversation = response["messages"]
#             final_message = conversation[-1]
#             print(f"AI: {final_message.content}")

#         except Exception as e:
#             print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()

import os
import requests
from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Conversation memory
# ---------------------------
conversation = []

# ---------------------------
# Tools
# ---------------------------
search_tool = DuckDuckGoSearchRun()

@tool
def BookSearch(query: str) -> str:
    print(f"[Tool] BookSearch called with query: {query}")

    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": query, "maxResults": 1, "printType": "books"}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "items" not in data:
            return f"No books found for '{query}'"

        book_info = data["items"][0]["volumeInfo"]
        title = book_info.get("title", "Unknown Title")

        print(f"[Tool] Found book: {title}")
        return f"Recommended book: {title}"

    except Exception as e:
        print(f"[Tool ERROR] {e}")
        return f"Error: {str(e)}"


@tool
def calculator(a: float, b: float, operation: str = "add") -> str:
    print(f"[Tool] Calculator: {a} {operation} {b}")

    if operation == "add":
        return str(a + b)
    elif operation == "subtract":
        return str(a - b)
    elif operation == "multiply":
        return str(a * b)
    elif operation == "divide":
        return str(a / b) if b != 0 else "Division by zero error"

    return "Invalid operation"


@tool
def save_conversation(filename: str = "conversation.txt") -> str:
    global conversation

    folder = "history_conversation"
    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for message in conversation:
            role = getattr(message, "type", "MESSAGE").upper()
            f.write(f"{role}: {getattr(message, 'content', '')}\n")

    print(f"[Tool] Conversation saved to {filepath}")
    return f"Saved to {filepath}"


tools = [calculator, search_tool, save_conversation, BookSearch]

# ---------------------------
# LLM + Agent
# ---------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_instructions = (
    "You are a helpful AI assistant. "
    "Use tools when necessary. Be concise and direct."
)

agent = create_react_agent(llm, tools=tools, prompt=system_instructions)

# ---------------------------
# Request schema
# ---------------------------
class ChatRequest(BaseModel):
    message: str
    history: list = []

# ---------------------------
# Chat endpoint
# ---------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    global conversation

    print("\n========== NEW REQUEST ==========")
    print(f"[1] Incoming message: {req.message}")
    print(f"[2] History length: {len(req.history)}")

    try:
        # Load history from frontend
        conversation = req.history
        print("[3] History loaded")

        # Append new message
        conversation.append(HumanMessage(content=req.message))
        print("[4] User message appended")

        # Invoke agent
        print("[5] Invoking agent...")
        response = agent.invoke({"messages": conversation})

        conversation = response["messages"]
        final_message = conversation[-1].content

        print("[6] Agent response generated")
        print(f"[AI] {final_message}")
        print("================================\n")

        return {"reply": final_message}

    except Exception as e:
        print("[ERROR]", str(e))
        return {"reply": "Error processing request"}