# Spec: `run_agent()`

**File:** `agent.py`
**Status:** Partially pre-filled — complete the two blank fields before implementing

---

## Purpose

Orchestrate a single conversational turn for the Plant Advisor agent. Given a user message and the conversation history, call the LLM with available tools, execute any tool calls the LLM requests, and return the final text response.

This is the core of what makes Plant Advisor an *agent* rather than a simple chatbot: the ability to decide which tools to call, use their results to inform its response, and loop until it has everything it needs.

---

## Input / Output Contract

**Inputs:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_message` | `str` | The user's current message |
| `history` | `list` | Gradio conversation history — list of `[user_msg, assistant_msg]` pairs |

**Output:** `str`

The agent's final text response for this turn. Should never be empty — if something goes wrong, return a user-readable fallback message.

---

## Design Decisions

*Read `specs/system-design.md` (especially the "How the Groq Tool Calling API Works" section) before reviewing these. Complete the two blank fields before writing any code.*

---

### Messages list structure

The messages list must start with the system prompt, then replay the conversation
history, then add the new user message. Gradio history is a list of `[user, assistant]`
pairs — convert each pair to two API-format dicts:

```python
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

for user_msg, assistant_msg in history:
    messages.append({"role": "user", "content": user_msg})
    if assistant_msg:
        messages.append({"role": "assistant", "content": assistant_msg})

messages.append({"role": "user", "content": user_message})
```

---

### Initial LLM call

Pass the model, the messages list, the tool definitions, and `tool_choice="auto"`
so the LLM can decide whether to call a tool or respond directly:

```python
response = client.chat.completions.create(
    model=LLM_MODEL,
    messages=messages,
    tools=TOOL_DEFINITIONS,
    tool_choice="auto",
)
```

---

### Detecting tool calls in the response

The response object has a `choices` list. Index 0 gives the assistant message.
Check its `tool_calls` attribute — if it's truthy, the LLM wants to call tools:

```python
assistant_message = response.choices[0].message

if not assistant_message.tool_calls:
    # No tool calls — LLM has a final answer
    ...
```

---

### Appending the assistant message

When there are tool calls, append the full assistant message object to `messages`
**before** appending any tool results. The API requires this ordering — a tool
result message must immediately follow the assistant message that requested it:

```python
messages.append(assistant_message)  # must come first
```

---

### Executing and appending tool results

For each tool call, extract the name and arguments, call `dispatch_tool()`, and
append the result as a `"tool"` role message. The `tool_call_id` links this result
back to the specific tool call that requested it:

```python
for tool_call in assistant_message.tool_calls:
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    tool_result = dispatch_tool(tool_name, tool_args)

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result,
    })
```

---

### Loop termination conditions

*The loop should stop when: (a) the LLM returns a response with no tool calls, OR (b) the MAX_TOOL_ROUNDS limit is reached. Describe how you will detect each condition and what you will return in each case.*

```
The loop is a `for _ in range(MAX_TOOL_ROUNDS)` rather than `while True`, so the
round cap is structural — it can never spin forever.

(a) No tool calls: after each create() call, check
    `if not assistant_message.tool_calls`. When falsy, the LLM has produced a
    final answer — return assistant_message.content immediately (with a
    user-readable fallback string if content is somehow empty/None).

(b) MAX_TOOL_ROUNDS reached: if the for-loop runs to completion, the LLM was
    still asking for tools on the last allowed round. I make ONE final create()
    call with tool_choice="none" to force a text answer from the context already
    gathered, and return its content (again with a fallback if empty). This exits
    gracefully with a real answer instead of crashing or returning "".

Edge cases handled: empty content -> fallback string; the function never returns
None or "" (contract says output is never empty); no risk of appending a tool
result without its preceding assistant message because the assistant message is
appended before the tool-result loop on every iteration.
```

---

### Extracting the final text response

*Once the loop exits because there are no more tool calls, how do you extract the text content from the response object? What field holds the string you should return?*

```
response.choices[0].message.content

The response has a `choices` list; index 0 is the relevant completion. Its
`.message` is the assistant message object, and `.content` is the generated text
string. (When the message is a tool-call request instead, .content is None and
.tool_calls is populated — which is exactly the branch the loop uses to decide
whether to keep looping or return.) I return `assistant_message.content` directly,
falling back to a fixed user-readable string only if it is empty.
```

---

## Implementation Notes

*Fill this in after implementing and testing.*

**Trace of a working agent turn (what tools were called and in what order):**

```
Query: "How should I water my monstera this time of year?"
Round 1 tool call: lookup_plant({'plant_name': 'monstera'})  -> found: True
Round 1 tool call: get_seasonal_conditions({})  -> Summer (auto-detected)
                   (both calls came back in the SAME assistant turn)
Round 2: no tool calls -> final answer returned
Final response: Cites the monstera's "every 1-2 weeks / top 2 inches dry"
               watering data and ties it to summer (water more frequently,
               watch for overwatering signs). Both data sources are reflected.
```

**What happens when you ask about a plant that isn't in the database?**

```
"How do I care for my bird of paradise?" -> lookup_plant returns found: False
with the instructional not-found message. The agent then (a) states the plant
isn't in its database, (b) offers general guidance (bright indirect light,
moderate watering, temp/humidity range), and (c) redirects to a trusted source
(RHS/AHS). It does NOT fabricate specific database-style care numbers. This is
the graceful-degradation behavior — driven by the not-found message + system
prompt, not the loop logic.
```

**One thing about the tool call API that surprised you:**

```
For a no-argument tool call, the LLM sends the arguments as the JSON string
"null" (not "{}"), so json.loads(...) returns None — which then crashed
dispatch_tool's tool_args.get("season"). I had to normalize the parsed arguments
to {} before dispatching. The model can also request multiple tool calls in a
SINGLE assistant message (lookup_plant AND get_seasonal_conditions together),
so the result-appending loop has to handle every tool_call in the list, each
with its own tool_call_id, before calling the LLM again.
```
