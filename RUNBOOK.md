# Practice guide: rebuilding the project from scratch

**English** · [فارسی](PRACTICE.fa.md)

This file is your work plan. The code in `src/` is the **answer**. It is not
there for you to copy.

## How to use it

```bash
mkdir my_agent                # your own code goes here
cp -r tests my_agent/tests    # borrow the tests
cp -r data  my_agent/data
```

The rule for each phase:

1. Read the "Goal" and "What to write" sections.
2. Write the code yourself **before you look at `src/`**.
3. Run the tests for that phase.
4. Are they green? Now open the reference file and compare. Understand where
   the two differ. Your solution may be better.

The `fake` model (`--provider fake`) lets you run the whole graph without
spending a single token. Until Phase 9 you need almost no API key.

Approximate total time: **10 to 16 hours** of real work. Do not rush. Phases 4
and 7 are the parts that really teach you something new.

---

## Phase 0: Setup (15 minutes)

```bash
python -m venv .venv && source .venv/bin/activate
pip install langgraph langgraph-checkpoint-sqlite langchain-core \
            langchain-openai langchain-anthropic \
            pandas matplotlib pydantic python-dotenv pytest
```

**Important:** The course was recorded with LangGraph 0.x. This project uses
1.x. The table of differences is in `README.md`, section "LangGraph
version". Read it once. Then you will not be confused when `MemorySaver` is
not found.

---

## Phase 1: state and loader, without any LLM

**Goal:** Learn what a reducer is and why it matters.

**What to write:**

- `state.py`: a `TypedDict` with the keys `messages`, `csv_path`, `question`,
  `profile`, `findings`, `charts`, `errors`, `failure_count`, `step_count`,
  `failed_node`.
  For `messages`, use `add_messages`. For `findings` and `charts`, write a
  reducer that removes duplicates. For `errors` and the numbers, use
  `operator.add`.
- `dataset.py`: `load_df(path)` with `lru_cache`, and `profile_df(df)`, which
  returns a small, readable dict.
- `nodes/loader.py`: a node that reads the CSV, builds the profile, and adds
  an initial `HumanMessage` to `messages`.

**Acceptance:**

```bash
pytest tests/test_state_and_tools.py -k "append_unique or profile"
```

**A point that is easy to miss:** Why does `profile` have no reducer while
`findings` has one? Because only one node writes `profile`. If two parallel
nodes write a key that has no reducer, LangGraph complains with
`InvalidUpdateError`. It is right to do so: it is not clear which one should
win.

**Challenge:** The profile must be *small*, because all of it goes into the
prompt. A full `describe()` on 50 columns is several thousand tokens. Decide
what you really need.

---

## Phase 2: planner with structured output

**Goal:** Turn the model's answer from "text" into "typed data".

**What to write:**

- `schemas.py`: a `Plan` class with `steps: list[str]` and `rationale: str`.
- `llm.py`: a `get_llm()` function that builds the model based on the
  environment variable `LLM_PROVIDER`. Wrap it with `lru_cache`.
- `prompts.py`: the planner's system prompt.
- `nodes/planner.py`: one call
  `get_llm().with_structured_output(Plan).invoke([...])`.

**Acceptance:** Call the node with `--provider fake` and check that `plan` is
filled in the state.

**Tip:** Put the prompts in a separate file. Do not mix them into the logic.
When you want to compare two versions of a prompt, you will understand why.

---

## Phase 3: the minimal graph that works

**Goal:** The conditional edge and the supervisor pattern.

**What to write:**

- `schemas.py`: a `Route` class with
  `next: Literal["analyst","charter","reporter","FINISH"]`.
- `nodes/supervisor.py`: a structured call that returns a `Route`, and a
  function `route_from_supervisor(state)` that returns the target string.
- `nodes/reporter.py`: a simple call that writes Markdown and saves the
  file.
- `graph.py`: only `loader → planner → supervisor → reporter → supervisor → END`.
  The other workers come later.

**Acceptance:** The graph runs to the end and produces a `.md` file.

```bash
python draw_graph.py     # look at the map
```

**The main point of this phase:** The routing function **must not** call the
LLM. The supervisor puts its decision in `state["next_worker"]`. The routing
function only reads that value. If you mix the two, you cannot test the
routing.

**Trap:** `add_conditional_edges` needs a mapping dict. If your function
returns a string that is not in the dict, you get a runtime error. You do not
get it at compile time.

---

## Phase 4: the sandbox (the hardest and most instructive part)

**Goal:** Let the model run code without handing it your laptop.

**What to write:** `sandbox.py` with three layers:

1. **Static validation with the AST.** Build the tree with `ast.parse` and
   walk it with `ast.walk`. Reject: `ast.Import`, `ast.ImportFrom`, every
   `ast.Attribute` whose `attr` starts with `__`, and every `ast.Name` that is
   on the blacklist (`eval`, `open`, `__import__`, ...).

   **Do not write a blacklist for file access.** The temptation is to block
   `{"to_csv", "to_excel", "to_parquet"}`. But pandas adds new IO methods in
   every version, and your list goes stale: `to_markdown('/tmp/x')`,
   `to_xml`, `to_hdf`, `to_latex`, `to_clipboard` all slip through your
   fingers. The correct rule is the reverse. Reject every `to_*` **unless**
   it is on the whitelist of in-memory converters (`to_dict`, `to_string`,
   `to_numpy`, `to_frame`, ...), and reject every `read_*` completely. The
   general rule: when one of two sets grows fast and the other grows slowly,
   **enumerate the slow set**.
2. **Restricted namespace.** `exec(code, {"__builtins__": ALLOWED, "df": df.copy(),
   "pd": pd, "np": np})`. Write the `ALLOWED` list by hand. Do not forget to
   include `next`, `iter`, `type` and the **exception classes** (`ValueError`,
   `KeyError`, ...). If you leave them out, no snippet can contain
   `except ValueError`, which is ordinary defensive code. The test
   `test_ordinary_pandas_is_not_wrongly_rejected` checks exactly these.
3. **Timeout.** Use `signal.setitimer(signal.ITIMER_REAL, n)` and a handler
   that raises an exception. Build a context manager.

The contract with the model: the snippet must put the answer in a variable
named `result`.

**Acceptance: all sandbox tests.**

```bash
pytest tests/test_sandbox.py -v
```

**Two things you must discover yourself:**

- Why `df.copy()`? The test `test_snippet_cannot_mutate_the_original_df` has
  the answer. A snippet that modifies `df` poisons all later steps.
- Why is `().__class__.__bases__[0].__subclasses__()` dangerous? It is the
  standard way to escape a Python sandbox. With it you can reach classes that
  open files. This is why every attribute with `__` is rejected.

**Be honest with yourself:** This sandbox is not real security. You build it
because it stops accidental mistakes by the model, and because thinking about
its layers makes you a better engineer. For untrusted input in production,
you need a container.

---

## Phase 5: tools and the ReAct loop

**Goal:** The core pattern of every tool-using agent.

**What to write:**

- `tools/analysis.py`: five tools with `@tool`. `run_pandas` is general. The
  other four (`group_aggregate`, `value_counts`, `correlations`,
  `column_stats`) are narrow and typed.
- For `csv_path`, use `Annotated[dict, InjectedState]`. This argument is
  removed from the **tool schema**, so the model never builds a wrong path.
  `ToolNode` fills it from the state by itself.

  Be precise about what you gain: `InjectedState` removes the argument from
  the schema. It does not remove the value from the prompt. In this project,
  the loader node puts the file path in the initial message on purpose, so
  the model *does see* that string. If a value must never reach the model,
  injection is necessary but not sufficient. Keep the value out of the
  transcript too.
- `nodes/analyst.py`: `get_llm().bind_tools(TOOLS).invoke(...)`, and the
  routing function:

  ```
  analyst --(has tool_calls)--> analyst_tools --> analyst
  analyst --(none)------------> supervisor
  ```

- In `graph.py`: `builder.add_node("analyst_tools", ToolNode(TOOLS,
  handle_tool_errors=True))`.

**Acceptance:**

```bash
pytest tests/test_state_and_tools.py -k "tool or pandas or group or value or correl"
pytest tests/test_graph.py -k "tool_errors"
```

**A question to ask yourself:** If `run_pandas` can do everything, why the
four other tools? Because the model can hardly get the typed signature of
`group_aggregate` wrong, while free-form code breaks in a hundred creative
ways. Give specific tools for the common 80%, and keep free-form code for the
rest.

**Output tip:** A tool must not return 5000 rows to the model. Set a limit and
say "425 rows, first 20 rows shown". The model must *know* that the output was
cut.

---

## Phase 6: charts

**Goal:** Learn to trust the tool's output instead of what the model says.

**What to write:**

- `tools/charts.py`: a `make_chart` tool with `matplotlib.use("Agg")`
  **before** you import pyplot. Sanitize the file name so that the model
  cannot escape from `outputs/` with `../../etc/x.png`.
- `nodes/charter.py`: the same loop shape as in Phase 5, with one tool.

**Acceptance:**

```bash
pytest tests/test_state_and_tools.py -k "chart"
```

**The most important lesson of this phase:** Take the chart's file path from
the **tool's return value**. Do not take it from the model's text. The model
will calmly say "I saved the chart to `revenue.png`" without calling any
tool. The only reliable source is the `ToolMessage`.

---

## Phase 7: human in the loop, with a durable pause

**Goal:** The concept that separates a toy agent from a usable agent.

**What to write:**

- `nodes/approval.py`: a node that calls `answer = interrupt({...})` and sets
  `approved` and `human_feedback` based on the answer.
- `graph.py`: `make_checkpointer()`, which builds a `SqliteSaver`:
  ```python
  conn = sqlite3.connect(path, check_same_thread=False)
  return SqliteSaver(conn)
  ```
- Routing: `approval → planner` if rejected, `approval → supervisor` if
  approved.
- In the planner, if `human_feedback` is set, add it to the prompt.

**Acceptance:**

```bash
pytest tests/test_graph.py -k "interrupt or rejected"
```

And by hand, in two separate processes:

```bash
python main.py data/sales.csv "..." --provider fake --thread-id demo
# Press Ctrl+C, or just leave it at the approval question and close the window
python main.py --resume --thread-id demo --provider fake
```

**What you must understand deeply:** `interrupt()` raises a special
exception. The runtime catches it, the checkpointer saves the exact position,
and the process can die. Later, `graph.invoke(Command(resume=x), config)`
makes the same `interrupt()` **return** the value `x`, as if nothing had
happened.

**And its dangerous consequence:** Everything you wrote in that node *before*
`interrupt()` **runs again** after the resume. So sending an email, writing a
file, and every other side effect must come **after** `interrupt()`. See this
once in practice so that you never forget it: put one `print` before and one
`print` after `interrupt()`, and resume.

**Without a checkpointer, `interrupt()` does not work.** The pause has no
place to be saved.

---

## Phase 8: error handling and guardrails

**Goal:** An agent that does not die when something fails and does not burn
your API credit.

**What to write:**

- `errors.py`: a decorator `safe_node(name)` that catches the exception and
  returns `{"errors": [...], "failure_count": 1, "failed_node": name}`.
  Wrap all nodes with it. **And on the success path, return
  `failed_node: ""`.** The reason is in lesson 1 of
  [Eighteen lessons you only learn from failure](#eighteen-lessons-you-only-learn-from-failure).

  **But it has a serious trap.** `interrupt()` does its job by raising an
  exception that the runtime must catch, and that exception is a subclass of
  `Exception`. So your plain `except Exception` swallows it, and your human
  pause turns into an error recorded in the log. The graph silently
  continues and never asks anyone. See it for yourself first:

  ```python
  from langgraph.errors import GraphInterrupt
  print([c.__name__ for c in GraphInterrupt.__mro__])
  # ['GraphInterrupt', 'GraphBubbleUp', 'Exception', ...]
  ```

  The correct way: put an `except GraphBubbleUp: raise` **before**
  `except Exception`. Then you can put the decorator on any node without
  worry.
- In the supervisor, **before you spend tokens**:
  - if `report_path` is set → `FINISH`
  - if `failed_node == "reporter"` → end (read the reason for this further
    down)
  - if `step_count >= MAX_STEPS` or `failure_count > MAX_FAILURES` → `reporter`
    to write a partial report
- And **after** the model's answer, enforce the rules in code: turn a
  `FINISH` without a report into `reporter`, and turn a `charter` without any
  finding into `analyst`.

**Acceptance:**

```bash
pytest tests/test_graph.py -k "budget or finish or eager or failing"
```

**A rule worth memorizing:** Do not leave to the prompt any rule that you can
check in code. The prompt *asks*. The code *guarantees*. A good model follows
the rule 95% of the time, and that 5% is exactly what hits you in
production.

**The message window part:** Write `messages.py` here too. The simplest method
("the last N messages") breaks the conversation, because an `AIMessage` with
`tool_calls` and the `ToolMessage` that answers it are an inseparable pair.
Cut between the two and the provider complains with
`tool_use ids must have matching tool_result blocks`. The test
`test_window_never_orphans_a_tool_message` checks exactly this.

---

## Phase 9: streaming and the CLI

**Goal:** User experience. An agent that stays silent for 90 seconds looks
broken.

**What to write:** In `main.py`:

```python
for chunk in graph.stream(payload, config=config, stream_mode="updates"):
    for node, update in chunk.items():
        if node == "__interrupt__":
            ...   # catch the interrupt and ask the user
        else:
            describe(node, update)
```

`stream_mode="updates"` gives one dict for each node that finishes. Make the
outer loop a `while True`, so that the stream continues after a resume.

**Acceptance:** Live, readable output, with the line
`python main.py --resume --thread-id ...` at the end.

**Extra exercise:** Try `stream_mode="messages"`. It gives output token by
token. It looks nice for the final report that the reporter writes.

---

## Phase 10: testing with a scripted model

**Goal:** A skill that few people have and that pays off the most.

**What to write:** `tests/fake_llm.py`:

- A class that inherits from `BaseChatModel` and whose `_generate` takes the
  next answer from a queue.
- Override `bind_tools()` so that it returns itself (and records the tool
  names, so that you can test that the right tools were bound).
- Override `with_structured_output(schema)` so that it returns an object with
  an `.invoke()` method that gives the next answer in the queue. It checks
  that the answer's type matches `schema`.

**Acceptance:** The whole suite, with no network and no key:

```bash
pytest
```

**Why this matters:** An agent is 90% *plumbing*: reducers, routing, tool
wiring, message pairing. You can test all of this deterministically if you
replace the model with a queue of prepared answers. Then you need the real
model only to measure *answer quality*, which is a separate, smaller test
suite.

**Trick:** When the model takes something from the queue whose type does not
match, raise an error immediately. It means your test script is not aligned
with the real path through the graph. It is better to find out at that moment
than three tests later.

---

## Eighteen lessons you only learn from failure

These were real bugs in this project. In the first version the tests were green and
the happy path worked without a flaw. Each one showed up only when someone
asked: "What happens if this node fails?". If you plan to read one section of this
file again, read this one.

### 1. The sticky flag

`safe_node` sets `failed_node`, and the routing functions read it to find out
"did the node that just ran fail?". If you do not **clear it on the success
path** anywhere, it no longer answers that question. It answers "did it fail
once in the past?". The result in practice: one transient 429 on the supervisor, and after that

- if you had no finding: the run ends silently with empty hands, and the retry budget
  is not even touched;
- if you had findings: `supervisor → reporter → supervisor → reporter …` up to the
  recursion limit, and every round is one paid reporter call.

On the analyst it is worse: every later analyst turn is diverted away from the tool loop, even
when it has `tool_calls`. So your agent piles up `AIMessage`s with unanswered tool calls,
and the next provider call breaks with `tool_use ids must have matching
tool_result blocks`. That is the same error you learned elsewhere to blame on the
message window.

**Lesson:** every sticky flag in the state has one clear meaning, and it must be *set* and
also *cleared*. Write `result.setdefault("failed_node", "")`. One line.

### 2. Name the counter for every cycle you add

Only the supervisor increased `step_count`. But `analyst ↔ analyst_tools` is a
closed cycle that **never passes through the supervisor**, so the supervisor's step
budget puts no limit on it. With `MAX_STEPS=1` and a model that keeps calling
tools: 101 LLM calls until the recursion limit. With the real default limit of
10000, about five thousand paid calls.

**Lesson:** every time you add a cycle to the graph, ask: "Which counter breaks
this?" If the answer is "none", the cycle is unbounded. Check the budget inside
`route_from_analyst`, and not only in the supervisor.

### 3. The last exit needs its own exit

Both fallback branches of the supervisor go to `reporter`, and the only way to end
is that `report_path` gets set. That is exactly what a broken reporter does not
do. So a faulty reporter means `supervisor → reporter → supervisor → …`
forever. Because the model call happens *before* the failure, you get thousands of paid
calls before the crash.

Two fixes are needed, and you should do both: (a) in the supervisor, if
`failed_node == "reporter"` → end; (b) in the reporter, put the model call in a
`try`, and in the `except` build a **mechanical** report from the findings, without
an LLM. Then the reporter practically cannot fail.

**Lesson:** trace your fallback path until it reaches END. "On error we go to X"
is an answer only if X can succeed.

### 4. An untagged sentence is not evidence

This line is tempting:

```python
statement = parse_tagged(text, "FINDING") or text.strip()
```

It is wrong. With that fallback, "Sorry, I cannot answer with these columns"
becomes a `Finding` with `source="analyst"`. That finding satisfies the
supervisor's condition "do we have a finding?", it reaches the reporter, and the
reporter's prompt says that findings are proven facts. So a line you wrote for
convenience has built a production line for fake evidence.

**Lesson:** in any agent that collects findings, the boundary between "the model said
something" and "the model proved something" must be in the **code**, not in the prompt. Untagged? Not a finding.

---

### 5. A test that reads your `.env` is not a test

This was the fifth real bug in the project, and more insidious than the others. The test suite was green on my
machine, because I had no `.env`. On the user's machine three tests failed:

```
assert 'llama3.1' == 'claude-sonnet-5'
```

The reason: the user's `.env` had `LLM_MODEL=llama3.1`, and my tests read it. So the
test did not say "is the code correct?". It said "is your config the same as mine?".
A test like that cannot tell a real regression from a difference in settings.

**The fix has two parts:**

1. An `autouse` fixture that clears the environment variables, builds a fresh `Settings`
   and puts it in place in all modules (`tests/conftest.py`).

2. And the subtler point: in a test file, **do not write `from src.config import settings`.**
   That statement binds the object that was built from the real `.env` at import time
   to a local name, and no fixture can reach that name. Get the settings from the
   fixture:

   ```python
   def test_something(clean_env):
       assert clean_env.resolved_model("anthropic") == "claude-sonnet-5"
   ```

   Then the test **declares** which settings it needs, instead of inheriting whatever
   is on the machine.

**Lesson:** after you write the test suite, create a `.env` with strange values and
run `pytest`. If the color changes, your tests measure the config, not the code.

---

### 6. An empty report is the worst place to trust the prompt

The sixth real bug, and the most expensive in its consequences. A real run on local `llama3.1`
ended with **zero findings**, and the reporter wrote a complete, good-looking report.
It named the wrong region as the winner and gave it 99% of the file's total revenue.

The fake numbers had a signature: all sums ended in `119.19` and all
averages in `62.18`. The model had produced a *good-looking* descending series, not a calculation.

And `REPORTER_SYSTEM` already said "never make up a number". The model
ignored it. What did the code do? This:

```python
findings_block = "\n".join(...) or "(no findings were produced -- say so plainly)"
```

So with zero findings, it gave the same prompt to the model and **hoped** it would obey.

**The fix is one condition:**

```python
if not findings:
    path.write_text(_no_findings_report(state))   # no model call at all
    return {"report_path": str(path), ...}
```

When there is nothing to write, do not ask the model. Now the report says honestly that nothing
was found, and *why*.

**Lesson:** wherever the model's output will be treated as "fact", ask: "What happens if the input
is empty?". If the answer is "we trust the prompt", you have built a
factory for fake evidence. And note that this is Lesson 4 again, one layer
higher. I applied it in the analyst and forgot it in the reporter. You must apply patterns
at **all** boundaries, not only where you first thought of them.

---

### 7. A table whose meaning depends on visual alignment

The seventh bug, and the subtlest of them. With `qwen2.5` the agent gave a report whose numbers
were **real**. The fix from Lesson 6 worked and nothing was made up. But one of the numbers was
assigned to the wrong region: East revenue was reported as West revenue.

The cause was in my code, not in the model. The `group_aggregate` tool returned the result with
`Series.to_string()`, and on a MultiIndex pandas **drops the repeated
level**:

```
region  order_date
East    2025-03-24    41573.82
        2025-10-25    40579.02     <- region is empty
```

A human reads that blank as "same as above". The model put "West" there.

**Lesson:** never give the model a table where the meaning of a row depends on **looking up**.
Visual alignment is the first thing that is lost in a token stream.
Every row must carry its own keys:

```
  region=East | order_date=2025-10-25 -> 40,579.02
```

**And the second half of the same bug:** the report said "there is only one date in the data, so
a trend cannot be computed". In fact there were 366 region/date combinations. The tool
had returned the first 15 rows, and the model drew its conclusion from that slice. The fix: in the header,
say **how many groups exist**, not only how many are shown:

```
sum(revenue) grouped by ['region', 'order_date'] -- showing top 4 of 366 groups:
  ...
  ... [362 more groups not shown. These are NOT totals per region; each row
       is one region/order_date combination.]
```

**General rule:** also say what you *left out*, not only what you
show. And if your tool output can be read in two ways, one day it will be read the
wrong way.

**The third root cause:** the plan said "group by `region` and `order_date`".
That splits the question "which region?" into hundreds of small groups. This was added to
`PLANNER_SYSTEM`: "in each step, group by only **one** dimension".

---

### 8. The supervisor does not know which step is done

The eighth bug, and the most classic design mistake in the supervisor/worker pattern.

In a real run, Step 1 of the plan produced a **correct** finding
(`North: $1,373,177.21`). What did the supervisor say on its next turn?

```
Supervisor -> analyst: Calculate the total revenue for each region. (Not yet done.)
```

The same step again, with the reason "not yet done". The step budget was burned on repeats, and
the run ended with an incomplete report.

**Cause:** my prompt showed the plan and also showed the findings, but
**nothing connected the two**. The model had to guess on its own which finding
belonged to which step, and it guessed wrong.

**The fix, in three pieces:**

1. `Route` got a `step_index` field: the supervisor must say which numbered step
   this task belongs to.
2. The state got an `addressed_steps`. When the analyst produces a finding,
   it marks the current step number, **only on success**, not when the step is
   dispatched.
3. `_progress()` shows each step as `[done]` or `[pending]`, and the code redirects
   a step that is `[done]` to the first `[pending]` step.

```
Plan:
  1. [done] Revenue per region
  2. [pending] Monthly trend
  3. [pending] Report
```

**Lesson:** in any agent that has a "plan", progress must be **data in the state**,
not something the model infers from the history. If you give the model two lists and
expect it to work out the mapping between them, one day it will not.

And note that the fix is the same pattern as Lesson 4, one layer higher: **mark a step when it is
finished, not when it starts.** If I had marked it in the supervisor, an
analyst that produced nothing would have left the step "done" forever.

---

### 9. If you need a shape, ask for a typed object, not a text format

The ninth bug, and it completes Lesson 4.

`ANALYST_SYSTEM` asked for the `FINDING:` / `EVIDENCE:` format, with a full example.
Hosted models follow it. Local `qwen2.5` did not: it called the tool,
**got the correct numbers**, and then answered in plain prose. Without a tag, no finding
was recorded (correctly, per Lesson 4), the step stayed `[pending]`, and the supervisor sent it
again five times until the budget ran out. An empty report, with a correct calculation that was thrown
away.

**The fix is the same principle this project has used for routing from the start:**
when you need a specific shape from the model, **do not describe the shape and hope**.
Ask for a **typed object**. A schema is enforced; a text format is only a request.

So when the tag is missing, a second, cheap call with
`with_structured_output(FindingOut)` extracts the same information from what the model *actually*
said plus the tool output it *actually* got:

```python
class FindingOut(BaseModel):
    has_finding: bool   # only when a tool has produced a number
    statement: str = ""
    evidence: str = ""
```

**And note that this second door is not a door for fabrication.** `has_finding` must be False unless
a number came from a tool, and the extractor prompt says it explicitly: "When in doubt,
False. A missed finding costs one turn; a fake finding corrupts
the report". The test `test_the_extractor_still_refuses_an_ungrounded_claim`
guarantees exactly this.

**General lesson:** wherever your agent expects a "text format" from the model (a tag,
a separator, hand-written JSON), that point breaks on weaker models. If the shape matters,
use a mechanism that **guarantees** the shape.

---

### 10. A node that runs silently cannot be debugged

A small but expensive lesson. In the same run, the output looked like this:

```
[supervisor] Supervisor -> analyst (step 1): ...
[supervisor] Supervisor -> analyst (step 1): ...
```

Two supervisor lines in a row. It looked as if the supervisor was talking to itself.
In fact the analyst **had run** between the two. But because it neither called a tool nor
produced a finding, my `describe()` function printed nothing.

So exactly in the case where we most needed to see output, we saw
nothing. Now:

```
[analyst] no result. said: 'It looks like North is clearly the biggest region here.'
```

And that one line gives the full diagnosis.

**Lesson:** in an agent's log, "nothing happened" is also an event and must be
printed. The success case shows itself; it is the failure case that needs a voice.

### 11. Read the API error, not the model name

Two runs in a row, on two OpenAI models, died on two parameters, not on the analysis:

```
gpt-6-luna     'temperature' does not support 0.0 with this model.
gpt-5.6-terra  Function tools with reasoning_effort are not supported ...
               set reasoning_effort to 'none'.
```

Read the second one carefully: this project **never** sends `reasoning_effort`. The model
applied its own default and then rejected the combination of that default with tools. No amount of care in
building the request could have predicted this.

The tempting solution is a list of model names and their habits. langchain-openai
itself has exactly such a list (`_model_prefers_responses_api`), and
`gpt-5.6-terra` is not in it, because it was released after the list was written. A list of model
habits rots silently.

But the 400 itself is structured and names the parameter at fault
(`error.param`). `src/llm.py` reads it, fixes the request and
sends it again once. Three details that matter:

- **At the bottom layer, not at each call.** The graph reaches the model through four doors (`invoke`,
  structured output, `bind_tools`, stream). A fix in `_generate`/`_stream`
  covers all of them.
- **With a scope.** The terra problem happened only with tools; the planner and the supervisor worked with
  reasoning. So `reasoning_effort=none` goes only on calls that carry
  tools.
- **With a limit.** No request is sent twice without a change.

Its tests (`tests/test_openai_client.py`) run the **real** client against a fake
server (`tests/fake_openai.py`) and check the JSON that went over the wire. The first
version of these tests monkeypatched `_generate`. It proved the retry logic
and nothing about the request itself.

**Lesson:** when an external system tells you where the problem is, read what it says.
Predicting its behavior in your own code is a copy of its knowledge that gets older every day.

### 12. An error that a retry cannot fix is not the supervisor's job

`safe_node` turns every exception into data so the supervisor can route around it. For
a 429, a timeout or pandas code that failed, this is right: the world changes, or the model writes
different code.

But in the terra run this happened:

```
[analyst] ! Function tools with reasoning_effort are not supported ...
[supervisor] -> analyst: Remove duplicate order_id rows ...
[analyst] ! Function tools with reasoning_effort are not supported ...
[supervisor] -> analyst: Using pandas with reasoning_effort disabled, drop ...
```

The supervisor read the error text and started to write "turn off reasoning_effort" into
its **instruction**. An LLM that tries to change an HTTP parameter with prose. Four
times, until the error budget ran out.

A broken key, a wrong model name, a parameter the model does not accept: none of these are in the conversation,
so no node can change them. Now these (`UNFIXABLE_BY_RETRY` in
`src/errors.py`, based on the neutral classes in `langchain_core.exceptions` that both
providers use) **stop** the run, with a readable message.

And here the checkpointer pays for itself: the stop is saved right before the faulty node,
with `failure_count=0`. After you fix `.env`,
`--resume --thread-id ...` continues from that same node.

**Lesson:** "turn the error into data" is not a universal rule. It is for errors that
someone in the graph can do something about. Before you give an error to the LLM,
ask: if I send the same request again, can the result be different? If not,
stop.

### 13. A limit you document but do not check is a delayed bug

In one round, two bugs were found that had the same root cause.

**The sandbox.** From day one, the docstring of the `time_limit` function said "SIGALRM only on
Unix and only in the main thread". The code checked only the first half. All tests called it from the
main thread and were green. But `ToolNode` in LangGraph runs tools
in a worker thread. So on Linux and macOS **every**
`run_pandas` call inside a real run failed with
`signal only works in main thread`. On Windows it was not visible,
because there the missing SIGALRM took the other branch earlier.

**The dates.** `load_df` detected a text column with `dtype == "object"`.
pandas 3 reads text with dtype `str`, the comparison became False, and no date was
parsed any more. Nothing raised an error. But the planner prompt still told the model
"date columns are already datetime". A library upgrade turned a true sentence in the
prompt into a lie.

**Lesson:** every "only if..." in a docstring or a prompt is a claim that a test
must check, and that test must take **the same path** that the real program
takes: from inside a thread, and on the real result (is `order_date` really
datetime?), not on the code we think ran.

### 14. A label is not evidence

When I connected the chart tool to the cleaned data, its test checked that the output
said `drawn from cleaned working data`. The test was green. Then I deliberately switched the tool back to the
raw file to see whether the test would fail. **It did not.** The output was:

```
drawn from cleaned working data; ...; peak 2025-11 = 690,712.31
```

The label said "cleaned" and the number said "raw" (the same fake November spike). The label was built from the
state, not from the data that was actually plotted. The test saw only the label.

Now the tool reports the peak of the series it **actually plotted**, and the test checks that number
(`peak 2025-12 = 363,111.86`). This also helps the model: the charter
can write the `CHART:` line from the real numbers.

**Lesson:** a test that only checks "what the system said it did" does not prove that it did
it. Measure the outcome. And try the test by breaking the code on purpose. A test that
has never failed has not proved anything yet.

### 15. Split the shared budget, and ask the interpreter for the list of "all places"

A real run: the supervisor dispatched the analyst only **twice**, and then
`step budget exhausted`. The first two steps of the plan (check duplicates, check the formula) used up the whole
budget. The analyst spun in the `analyst ⇄ analyst_tools` loop as much as it wanted,
and its only brake was the total budget. Region, trend and chart never got a
turn. Worse, the budget guardrail was checked before the chart rule, so that
rule never ran.

Now the budget is split (`src/budget.py`): each step has at most `MAX_TOOL_ROUNDS`
rounds, and after that the finding is extracted from the last tool output. As long as we still owe
a chart, the last four steps are reserved for the charter.

**But the bigger lesson was in the tests.** The new round-limit test failed for no clear reason. The cause:
the `override_settings` fixture changed the settings only in a hand-written list of five
modules, and `src.nodes.analyst` was not in it. So from day one, the tests that
changed `max_steps` changed the value everywhere **except in the code that
ran it**. An old test ("the tool loop must stop exactly at the budget") was green
only because the analyst consumed the report message as its fifth round and the script
happened to end.

Now the fixture asks `sys.modules` which modules have `settings`.

**Lesson:** a list of "all the places where X is used" always misses one place, and the day it misses one
it gives no error. If the machine can build the list, do not write it by
hand. And a test that is green "by accident" is worse than a test that does not exist: it gives you
a confidence that nobody built.

### 16. A test whose script was not followed tests nothing

When the scripted model and the graph disagree about the order of work, the graph does not
crash: `safe_node` turns the `script order mismatch` error into data, and
the run limps on. A test that checks only a few fields stays
green.

After Lesson 15 I added a guard to the tests: if the script was not followed, the test
must fail loudly and clearly. **Nine tests failed.** To find out which of them were caused by
today's change, I ran the same tests on the code of the previous version: all nine failed
there too. So from version 0.11.1 on, since the rule "every report gets one chart"
arrived, these tests had been green **by accident**. Their scripts had been written before that rule,
and the charter consumed the message that was meant for the reporter. The script "ended" at exactly
the right moment, and every assertion held.

**Lesson:** a test proves something only when you are sure that the path you think it takes
was really taken. The claim "136 tests green" in 0.11.1 was true and meaningless. Every test
that builds its input on the assumption of a specific path must check that this path was taken;
in this project that means `assert_script_matched`.

(There was also a smaller lesson in the same round: the round limit that I added in 0.12.0
used, for finding extraction, the same function that had been written for the analyst's prose answer.
Its prompt says "there is a finding only when the analyst's *answer* says something about the
numbers", and I sent an answer that said nothing. So every time it returned "no
finding". A helper function written for a different question does not answer
your question.)

### 17. Review tools are code too, and they have just as many bugs

The full review of 0.13.0 found thirty-one bugs. But on the same day, the **review tools** made
mistakes three times:

- **The verification script** still said "the bug is there" after the code was fixed. Its condition
  was `"0.00" in text`, and the correct "0.004" also contains "0.00". A substring assertion,
  which compared characters instead of values.
- **A mutation** reported "the test does not catch this fix". In fact my replacement pattern
  had changed the first of three identical lines, which is a different branch of the code
  that the test never reached.
- **The mutation run was killed by a time limit** in the middle of a mutation. The `finally` that restored the file
  never ran. The real `sandbox.py` stayed broken and the test suite
  hung, until the cause was found. Now mutations run only on a **copy**.

**Lesson:** a script that checks whether code is correct is itself code whose correctness has not been
checked. Apply the same principles to it: compare values, not text; make sure
you changed exactly what you think you changed; and do not let any tool
change the real code in place. If the result of a check looks strange, do not trust the
check itself first.

### 18. "One person" was an assumption written down nowhere

For the web app, the agent code barely changed. But in three places it silently assumed that only one person
works at a time:

- **The model chooses the chart file name** (`revenue_by_region.png`), and everyone wrote into
  one `outputs/` folder. Two concurrent runs, charts with the same name, and the first run's report
  showed the second run's chart, without any error. Now each run
  has its own `outputs/runs/<thread>/`, and the path is in the state.
- **pyplot** keeps its list of figures global. Drawing a chart now has a lock.
- **The web page must not run the analysis itself.** Streamlit re-runs the page script on every click
  and interrupts the previous run; closing the tab does the same.
  The run happens in `src/jobs.py` in its own thread, and the page only watches it.

The concurrency test (`test_two_runs_at_once_keep_their_own_charts`) starts two runs at exactly the same time with
`threading.Barrier`. If you switch `charts.py` back to the
shared folder, this test turns red.

And one more that only Windows revealed: all of this was green on Linux. The first
`pytest` on Windows gave a `PermissionError: [WinError 5]`. Windows does not let you
replace a file that another thread is reading, and Linux does. The error was in the run thread,
**outside** the `try`, so the thread died silently
and the page stayed on "Working" forever. Now `tests/test_windows_files.py` simulates the same Windows
behavior on any system.

**Lesson:** "one user, one run" is an assumption you do not see until the user's second day. Wherever
a fixed name (a file, a folder, a global variable) is combined with a value that each run
creates on its own, ask what happens if two runs happen at the same time. And write the code of a thread
so that no error can kill it silently: everything that runs in the thread
must be inside a `try` and must record its status before it does anything else.

---

## Extra exercises, by difficulty

**Warm-up**

1. Add a tool `outlier_rows(column, z=3)` that returns the outlier rows.
   Try it on `revenue`. Does it find the row that is a hundred times too large?
2. Make the reporter write the report in two languages (Persian + English) and save
   both.
3. Add a `--max-charts` flag to the CLI and apply it in the charter prompt.

**Intermediate**

4. **A `validator` node.** Put a node between analyst and supervisor. It re-checks each finding
   with an independent calculation and marks the findings it cannot confirm.
   The report quotes only confirmed findings.
5. **Several CSV files.** Give the state a `csv_paths: list[str]` and change the tools
   so they can also join. `InjectedState` is very useful here.
6. **Cache tool results.** Wrap the tools so that a repeated call with the same
   arguments gets its answer from the cache. What percentage of tokens did you save?
7. **Excel output.** Add an `exporter` node that writes the findings to a `.xlsx`
   file with `openpyxl` and puts the charts on a separate sheet.

**Hard**

8. **Parallelism.** Instead of one worker, the supervisor calls several independent workers
   at once (in LangGraph, by returning several `Send` objects from the routing function). This is where
   your reducers face a real test. Without a correct reducer,
   you get `InvalidUpdateError`.
9. **Memory across sessions.** With the same `thread_id`, the agent remembers earlier analyses
   of the same file: "Last time you said North was ahead. Is it still?"
10. **Web interface, second version.** `app.py` is written with Streamlit. Rewrite it with FastAPI
    so it streams `stream_mode="messages"` over Server-Sent Events
    (token by token, instead of node by node). Reuse `src/runner.py` and `src/jobs.py`.
    If you can't, they depend on Streamlit somewhere they should not.
11. **Evaluation (eval).** Prepare 10 questions with correct answers written by hand. Write a script
    that runs the agent on all of them and reports the percentage of correct answers. Then
    change the prompt and see whether the number goes up or down. **This step is the line between
    "I wrote an agent" and "I made the agent better".**

---

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `InvalidUpdateError: at key 'x'` | Two parallel nodes wrote to a key that has no reducer | Add a reducer |
| `tool_use ids must have matching tool_result blocks` | The message window has split a tool pair | `messages.py:window` |
| The graph loops forever between two nodes | You have no guard rail | `MAX_STEPS` in the supervisor |
| `interrupt()` does nothing | The graph was compiled without a checkpointer | `compile(checkpointer=...)` |
| After resume, earlier work runs again | The side effect comes before `interrupt()` | Move it |
| The `interrupt` turned into a recorded error | `except Exception` swallowed its exception | Add `except GraphBubbleUp: raise` |
| One transient error breaks the whole run | `failed_node` is not cleared on the success path | `result.setdefault("failed_node", "")` |
| The tool loop runs up to the recursion limit | The step budget is only checked in the supervisor | Check it in `route_from_analyst` too |
| Endless `supervisor ↔ reporter` | The reporter is the only exit, and the reporter itself is broken | Mechanical fallback + an end branch |
| The report has a number that no tool produced | An unlabelled message was recorded as a finding | Record only messages with the `FINDING:` label |
| Tests pass on your machine and fail on another | The tests read the local `.env` | `clean_env` fixture in conftest |
| Correct number, assigned to the wrong key | `to_string()` drops the repeated MultiIndex level | Give each row its own keys |
| "The data has no trend" when it does | The model drew a conclusion from a truncated slice | State the total number of groups in the header |
| The supervisor repeats the same step | Plan progress is not in the state | `step_index` + `addressed_steps` + `[done]/[pending]` |
| The model calls the tool, but no finding is recorded | It did not follow the text format | Extract with `with_structured_output` |
| Two supervisor lines in a row, with no explanation | A node with no result prints nothing | Print "no result. said: ..." |
| Two reports overwrite each other | Timestamp with one-second resolution + `write_text` | Build a unique name |
| The model invents a column | The profile is missing from the prompt or incomplete | Give the column names explicitly |
| Token usage explodes | Every message goes to every node | Window + cap on tool output |
| The tool works, but the model does not see the result | `ToolMessage` was not added to the state | `add_messages` reducer on `messages` |

---

## Delivery checklist

When you can say "yes" to all of these, the project is done:

- [ ] `pytest` passes, without any API key
- [ ] It runs with both providers (OpenAI and Claude), only by changing env
- [ ] Human approval works, and rejecting the plan really produces a *different* plan
- [ ] Stopping in one process and resuming in another works
- [ ] A broken CSV, a missing column, and an unrelated question do not cause a crash
- [ ] `MAX_STEPS=2` leads to an incomplete report instead of an endless loop. You have also tested this
      with a model that keeps calling tools, and not only with the happy path
- [ ] A transient error (a `RuntimeError` in the model script) does not stop the run
- [ ] If the reporter itself fails, a report is still written and the graph finishes
- [ ] An unlabelled analyst answer is not recorded as a finding
- [ ] The sandbox rejects `import os`, `open()` and `to_csv`
- [ ] The generated report lists the data quality problems of `sales.csv` in the Caveats section
- [ ] You can draw the graph on paper from memory
- [ ] You can explain in three sentences why `interrupt()` needs a checkpointer

The last one is the most important. If you can explain the architecture to someone who does not know LangGraph,
you have learned the project. You have not just typed it.
