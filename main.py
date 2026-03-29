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

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Conversation memory
# ---------------------------
conversation = []

# ---------------------------
# Helper: safely convert AI message content to a plain string.
# LangChain sometimes returns a list of content blocks instead of a string
# (e.g. [{"type": "text", "text": "..."}]) — this handles both cases.
# ---------------------------
def _stringify_ai_content(content) -> str:
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

# ---------------------------
# Helper: ensure [[BOOK:id|Title]] tokens from the BookSearch tool
# are preserved verbatim in the final reply.
# The LLM sometimes reformats them (e.g. wraps in markdown bold or
# drops the brackets) — this function restores any that got mangled
# by scanning the raw tool output stored in the conversation.
# If the tokens are already intact, this is a no-op.
# ---------------------------
def enrich_reply_with_book_links(reply: str) -> str:
    # Tokens are already intact — nothing to do
    TOKEN_RE = re.compile(r'\[\[BOOK:[^\]|]+\|[^\]]+\]\]')
    if TOKEN_RE.search(reply):
        print("[enrich] Tokens already present in reply — no enrichment needed.")
        return reply

    # Tokens missing — log a warning so we can diagnose
    print("[enrich] WARNING: No [[BOOK:...]] tokens found in reply.")
    print("[enrich] Reply text:", repr(reply[:300]))
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

    Args:
        query (str): Keywords from the user's message.

    Returns:
        str: Formatted book list with [[BOOK:id|Title]] tokens.
    """
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": query,
        "maxResults": 4,
        "printType": "books",
        "orderBy": "relevance",
    }

    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()

        if "items" not in data or not data["items"]:
            return (
                f"[BookSearch] No results found for '{query}'. "
                "Search unavailable or no matches. Ask the user to retry."
            )

        results = []
        for item in data["items"][:4]:
            info = item.get("volumeInfo", {})
            volume_id = item.get("id", "unknown")
            title = info.get("title", "Unknown Title")
            authors = ", ".join(info.get("authors", ["Unknown Author"]))
            raw_desc = info.get("description", "")

            # Trim description to a clean sentence boundary
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

            book_token = f"[[BOOK:{volume_id}|{title}]]"

            # One clean line per book: token + author + description
            results.append(f"- {book_token} by {authors} — {short_desc}")

        tool_output = "\n".join(results)
        print(f"[BookSearch] Tool output:\n{tool_output}")
        return tool_output

    except requests.exceptions.Timeout:
        return "[BookSearch] Request timed out. Ask the user to retry."
    except requests.exceptions.RequestException as e:
        return f"[BookSearch] Network error: {str(e)}"
    except Exception as e:
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

    "You MUST call the BookSearch tool for ANY request involving books, genres, authors, "
    "recommendations, or examples — no exceptions. "
    "You are FORBIDDEN from listing, naming, or describing any books from memory. "
    "If BookSearch returns an error or no results, say the search is temporarily unavailable "
    "and ask the user to retry — do NOT substitute books from your own knowledge. "

    "After calling BookSearch, format your reply EXACTLY as follows:\n"
    "1. One short conversational sentence introducing the results.\n"
    "2. Copy each book line from the tool output EXACTLY as-is — do NOT rewrite, "
    "   reformat, or remove the [[BOOK:volumeId|Title]] tokens. Each book is already "
    "   formatted as: - [[BOOK:id|Title]] by Author — description. Keep it exactly that way.\n"
    "3. End with one short follow-up question on its own line.\n"

    "Never nest bullet points. Never add sub-bullets. Each book must be exactly one line. "
    "Never invent or modify volume IDs. Never remove the [[ or ]] brackets."
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

        raw_content = conversation[-1].content
        print(f"[2] Raw AI content type: {type(raw_content)}")
        print(f"[3] Raw AI content: {repr(raw_content)[:500]}")

        final_message = _stringify_ai_content(raw_content)
        print(f"[4] Stringified: {repr(final_message)[:500]}")

        final_message = enrich_reply_with_book_links(final_message)
        print(f"[5] After enrich: {repr(final_message)[:500]}")

        # Split the last non-empty line off as the follow-up prompt
        lines = final_message.strip().split("\n")
        non_empty = [(i, l) for i, l in enumerate(lines) if l.strip()]

        if len(non_empty) >= 2:
            last_idx, prompt_line = non_empty[-1]
            body = "\n".join(lines[:last_idx]).strip()
        else:
            body = final_message.strip()
            prompt_line = ""

        print(f"[6] Body: {repr(body)[:400]}")
        print(f"[7] Prompt: {repr(prompt_line)}")
        print("================================\n")

        return {"reply": body, "prompt": prompt_line}

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {"reply": "Error processing request", "prompt": ""}