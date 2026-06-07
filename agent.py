import json
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, MAX_TOOL_ROUNDS
from tools import lookup_plant, get_seasonal_conditions

_client = Groq(api_key=GROQ_API_KEY)

# ──────────────────────────────────────────────
# Tool definitions
#
# These are the schemas that tell the LLM what tools are available and how to
# call them. The LLM reads these descriptions and decides when (and how) to use
# each tool. They're already complete — your job is to implement the tool
# functions in tools.py and the agent loop below.
# ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_plant",
            "description": (
                "Look up care information for a specific houseplant by name. "
                "Returns detailed watering, light, humidity, and temperature requirements. "
                "Use this whenever the user asks about a specific plant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_name": {
                        "type": "string",
                        "description": "The plant name to look up. Can be a common name, scientific name, or nickname (e.g., 'pothos', 'devil's ivy', 'Monstera deliciosa').",
                    }
                },
                "required": ["plant_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seasonal_conditions",
            "description": (
                "Get seasonal care adjustments for houseplants. "
                "Returns guidance on watering, fertilizing, light, and pests for the current or specified season. "
                "Use this when a user asks a season-specific question, or to complement plant care advice with seasonal context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {
                        "type": "string",
                        "description": "The season to get care conditions for. If omitted, the current season is detected automatically.",
                        "enum": ["spring", "summer", "fall", "winter"],
                    }
                },
                "required": [],
            },
        },
    },
]

# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable and friendly plant care advisor. "
    "Help users care for their houseplants by looking up specific plant information "
    "and current seasonal conditions using your available tools.\n\n"
    "Always use your tools to look up plant-specific information before answering — "
    "don't rely on your general knowledge alone.\n\n"
    "When lookup_plant returns found: False, do NOT invent specific care numbers "
    "(watering frequencies, temperature ranges, etc.) as if they came from the "
    "database. Instead: (1) clearly tell the user the plant isn't in your database, "
    "(2) offer general guidance based on the plant type or what they described, and "
    "(3) point them to a trusted source for specifics. Acknowledge the gap rather "
    "than papering over it.\n\n"
    "Keep your advice practical and specific. Cite the source of your information "
    "when you have it (e.g., 'According to the care data for your monstera...')."
)

# ──────────────────────────────────────────────
# Tool dispatch
#
# This is already complete. It routes tool calls from the LLM to the actual
# Python functions in tools.py, and returns results as JSON strings (which is
# what the Groq API expects for tool results).
# ──────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """Route a tool call to the correct function and return the result as a JSON string."""
    print(f"  → Tool call: {tool_name}({tool_args})")
    if tool_name == "lookup_plant":
        result = lookup_plant(tool_args["plant_name"])
    elif tool_name == "get_seasonal_conditions":
        result = get_seasonal_conditions(tool_args.get("season"))
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    print(f"  ← Result: {json.dumps(result)[:120]}{'...' if len(json.dumps(result)) > 120 else ''}")
    return json.dumps(result)


# ──────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────

def run_agent(user_message: str, history: list) -> str:
    """
    Run the plant care agent for one user turn and return its response.

    This is the loop that makes Plant Advisor an agent rather than a chatbot: we
    hand the LLM the conversation plus the tool schemas, let it decide what to call,
    feed the results back, and repeat until it has enough to answer. MAX_TOOL_ROUNDS
    is our safety valve so the loop can never run away.

    The one ordering rule that matters: the assistant message (the one holding the
    tool_calls) goes into `messages` BEFORE the tool results, because each result
    points back at its request via tool_call_id.
    """
    # 1. Build the messages list: system prompt + replayed history + new user turn.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": user_message})

    # 2. Tool-calling loop, capped by MAX_TOOL_ROUNDS so a misbehaving tool can't
    #    spin forever.
    for _ in range(MAX_TOOL_ROUNDS):
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message

        # Exit condition (a): no tool calls means the LLM has a final answer.
        if not assistant_message.tool_calls:
            return assistant_message.content or (
                "🌱 I wasn't able to put together a response. Could you rephrase your question?"
            )

        # Append the assistant message FIRST — each tool result must reference the
        # tool_call recorded here via tool_call_id.
        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            # Arguments arrive as a JSON string. For a no-arg call the LLM may send
            # "", "null", or "{}" — normalize all of those to an empty dict so
            # dispatch_tool always receives a real mapping.
            raw_args = tool_call.function.arguments or "{}"
            tool_args = json.loads(raw_args) or {}
            tool_result = dispatch_tool(tool_name, tool_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

    # Exit condition (b): hit the round cap. Ask the LLM for a final answer one
    # last time with tools disabled, so it summarizes what it has gathered.
    final = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tool_choice="none",
    )
    return final.choices[0].message.content or (
        "🌱 I gathered some information but ran out of steps before finishing. "
        "Could you narrow down your question?"
    )
