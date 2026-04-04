import os
import re
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
if not os.getenv("OPENAI_API_KEY") and os.getenv("OPEN_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_API_KEY", "")

# Google Books API key — set GOOGLE_BOOKS_API_KEY in your Render environment variables
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY", "")

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Conversation memory
# ---------------------------
conversation = []

# ---------------------------
# Helpers
# ---------------------------
def _stringify_ai_content(content) -> str:
    """Safely convert LangChain AI message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def enrich_reply_with_book_links(reply: str) -> str:
    """Verify [[BOOK:...]] tokens survived the LLM response. Logs if missing."""
    if re.search(r'\[\[BOOK:[^\]|]+\|[^\]]+\]\]', reply):
        print("[enrich] Tokens present — OK.")
    else:
        print("[enrich] WARNING: No [[BOOK:...]] tokens in reply.")
        print("[enrich] Reply:", repr(reply[:400]))
    return reply

# ---------------------------
# Tools
# ---------------------------
search_tool = DuckDuckGoSearchRun()


@tool
def BookSearch(query: str) -> str:
    """
    Search Google Books API for books matching the user's query.
    Returns up to 4 results each tagged with [[BOOK:volumeId|Title]] tokens.
    MUST be called for any book-related request.

    Args:
        query (str): Keywords from the user's message.

    Returns:
        str: Formatted book list with [[BOOK:id|Title]] tokens.
    """
    print(f"[BookSearch] query='{query}' | API key set: {bool(GOOGLE_BOOKS_API_KEY)}")

    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": query,
        "maxResults": 4,
        "printType": "books",
        "orderBy": "relevance",
    }

    # Always include the API key — required for reliable access on hosted servers
    if GOOGLE_BOOKS_API_KEY:
        params["key"] = GOOGLE_BOOKS_API_KEY

    try:
        response = requests.get(url, params=params, timeout=8)
        print(f"[BookSearch] HTTP {response.status_code}")
        response.raise_for_status()
        data = response.json()

        if "items" not in data or not data["items"]:
            print("[BookSearch] No items returned.")
            return f"[BookSearch] No results found for '{query}'. Ask the user to retry."

        results = []
        for item in data["items"][:4]:
            info = item.get("volumeInfo", {})
            volume_id = item.get("id", "unknown")
            title = info.get("title", "Unknown Title")
            authors = ", ".join(info.get("authors", ["Unknown Author"]))
            raw_desc = info.get("description", "")

            if len(raw_desc) > 220:
                trimmed = raw_desc[:220]
                cutoff = max(
                    trimmed.rfind(". "),
                    trimmed.rfind("! "),
                    trimmed.rfind("? "),
                )
                short_desc = (trimmed[:cutoff + 1] if cutoff > 80 else trimmed) + "…"
            else:
                short_desc = raw_desc if raw_desc else "No description available."

            results.append(f"- [[BOOK:{volume_id}|{title}]] by {authors} — {short_desc}")

        output = "\n".join(results)
        print(f"[BookSearch] Returning {len(results)} books.")
        return output

    except requests.exceptions.Timeout:
        print("[BookSearch] Timed out.")
        return "[BookSearch] Request timed out. Ask the user to retry."
    except requests.exceptions.RequestException as e:
        print(f"[BookSearch] Network error: {e}")
        return f"[BookSearch] Network error: {str(e)}"
    except Exception as e:
        print(f"[BookSearch] Unexpected error: {e}")
        return f"[BookSearch] Unexpected error: {str(e)}"


@tool
def calculator(a: float, b: float, operation: str = "add") -> str:
    """Perform basic arithmetic: add, subtract, multiply, or divide."""
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
    """Save the current in-memory conversation to a text file."""
    global conversation
    folder = "history_conversation"
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        for message in conversation:
            role = getattr(message, "type", "MESSAGE").upper()
            f.write(f"{role}: {getattr(message, 'content', '')}\n")
    return f"Saved to {filepath}"


tools = [BookSearch, calculator, search_tool, save_conversation]

# ---------------------------
# LLM + Agent
# ---------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_instructions = (
    "You are a helpful AI book assistant for an online bookstore. "

    "RULE 1 — ALWAYS USE TOOL: You MUST call the BookSearch tool for ANY request "
    "involving books, genres, authors, or recommendations. No exceptions. "
    "You are completely forbidden from naming, listing, or describing books from memory. "
    "If BookSearch returns a string starting with '[BookSearch]', that is an error — "
    "tell the user the search is temporarily unavailable and ask them to retry. "
    "Do NOT fall back to books from memory under any circumstance. "

    "RULE 2 — FORMAT: After BookSearch returns results, reply exactly as follows:\n"
    "  Line 1: One short sentence introducing the results.\n"
    "  Lines 2+: Copy each '- [[BOOK:id|Title]] by Author — description' line from "
    "the tool output VERBATIM. Do not rewrite, reformat, or remove the [[ ]] tokens.\n"
    "  Last line: One short follow-up question.\n"

    "RULE 3 — TOKENS: Never modify, remove, or invent [[BOOK:id|Title]] tokens. "
    "Copy them exactly as returned by the tool."
)

agent = create_react_agent(llm, tools=tools, prompt=system_instructions)

# ---------------------------
# Request schema
# ---------------------------
class ChatRequest(BaseModel):
    message: str
    history: list = []

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "AI_assistance API", "chat_endpoint": "/chat"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# Diagnostic: test BookSearch directly without going through the LLM
# Usage: GET /test-booksearch?query=horror
@app.get("/test-booksearch")
def test_booksearch(query: str = "horror"):
    result = BookSearch.func(query)
    return {
        "query": query,
        "api_key_set": bool(GOOGLE_BOOKS_API_KEY),
        "has_tokens": "[[BOOK:" in result,
        "result": result,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    global conversation

    print("\n========== NEW REQUEST ==========")
    print(f"[1] Message: {req.message}")

    try:
        conversation = req.history
        conversation.append(HumanMessage(content=req.message))

        response = agent.invoke({"messages": conversation})
        conversation = response["messages"]

        # Log message chain to confirm BookSearch was called
        print(f"[2] Message chain ({len(conversation)} messages):")
        for i, msg in enumerate(conversation):
            print(f"    [{i}] {type(msg).__name__}: {repr(getattr(msg, 'content', ''))[:200]}")

        raw_content = conversation[-1].content
        final_message = _stringify_ai_content(raw_content)
        final_message = enrich_reply_with_book_links(final_message)

        # Split last non-empty line off as the follow-up prompt
        lines = final_message.strip().split("\n")
        non_empty = [(i, l) for i, l in enumerate(lines) if l.strip()]

        if len(non_empty) >= 2:
            last_idx, prompt_line = non_empty[-1]
            body = "\n".join(lines[:last_idx]).strip()
        else:
            body = final_message.strip()
            prompt_line = ""

        print(f"[3] Body: {repr(body[:400])}")
        print(f"[4] Prompt: {repr(prompt_line)}")
        print("=================================\n")

        return {"reply": body, "prompt": prompt_line}

    except Exception as e:
        import traceback
        print(f"[ERROR] {str(e)}")
        print(traceback.format_exc())
        return {"reply": "Error processing request", "prompt": ""}