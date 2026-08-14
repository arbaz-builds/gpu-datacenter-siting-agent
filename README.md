# GPU Datacenter Siting Agent

**The problem:** Picking a GPU data center site is currently a multi-week,
five-figure-consultant process — power availability, terrain, water,
hazards, and electricity pricing all live in different databases, and
someone has to manually stitch them together for every candidate address
before a developer can even start comparing options.

**What this does:** an autonomous LangGraph agent that takes a US address
(or a state/city), fetches real siting data and electricity pricing, and
returns an explicit REJECT / SHORTLIST / SELECT decision per candidate —
in minutes, not weeks. It combines:

- **[Mireye](https://www.mireye.com) physical-world site data APIs** — terrain, power, water, hazards, and other siting facts for any US address or coordinate.
- **EIA (U.S. Energy Information Administration)** electricity pricing data — the other half of the cost equation Mireye doesn't cover.
- **NVIDIA NIM** (`openai/gpt-oss-120b`) for the reasoning and structured decision-making layer.

**Who this is for:** data center developers, colocation operators, and
hyperscalers doing early-stage site screening — the people who today pay
consultants to rule sites in or out before committing capital to a formal
feasibility study.

Built for the [Mireye Build Challenge](https://www.mireye.com) hackathon.

## How it works

The agent is a multi-step LangGraph pipeline:

```
START
  → intro_router          (understand the user's request; ask follow-ups if unclear)
    → planner               (turn a state/city into 5 candidate locations to evaluate)
      → llm_tool             (route to the site-data tool; makes no decisions itself)
        → tools               (get_datacenter_site_data: Mireye + EIA)
          → structure_llm      (normalize raw site JSON into a compact schema)
          → llm_tool            (loop back, now with structured data in context)
        → decision            (REJECT / SHORTLIST / SELECT — one business decision per candidate)
          → action              (deterministic: map decisions to next steps)
            → answer              (explain the decision and actions to the user)
END
```

- **`intro_router`** — classifies the user's message as `gpu_search` (enough
  detail to proceed) or `unclear` (needs more info), using structured output.
- **`planner_llm`** — turns a state (e.g. "Texas") into the 5 best candidate
  cities, or a city into nearby industrial areas — without scoring or
  analyzing them yet. Planning is kept separate from evaluation so the agent
  never scores a location it hasn't actually looked at.
- **`llm_tool_node`** — a tool-routing loop (max 3 iterations) that calls
  `get_datacenter_site_data` for each candidate that still needs data. It is
  explicitly forbidden from scoring, ranking, or deciding anything — that's
  the decision node's job.
- **`get_datacenter_site_data`** — geocodes the address, extracts the US
  state, fetches Mireye's `data_center_siting` preset, and pulls EIA's latest
  state electricity price. Includes retry logic for transient API failures.
- **`structure_llm_node`** — strips metadata and normalizes the raw Mireye/EIA
  JSON into a clean schema for the decision node.
- **`decision_node`** — makes the actual business call for every candidate:
  `REJECT`, `SHORTLIST`, or `SELECT`. Enforced in code, not just prompted:
  a candidate with a blocking issue can never be `SELECT`, and if the model
  ever returns more than one `SELECT`, only the first is kept and the rest
  are demoted to `SHORTLIST`.
- **`action_node`** — a plain deterministic mapping from decisions to next
  steps (e.g. `SELECT` → start due diligence, `REJECT` → exclude,
  `SHORTLIST` → keep as backup). No LLM call — nothing to hallucinate here.
- **`answer_node`** — explains the decision and actions to the user in plain
  language. Treats `decision_node`'s output as authoritative and refuses to
  fabricate a report if no real site data was ever retrieved.

State is persisted per-conversation via **Postgres** (`AsyncPostgresSaver`),
so a `thread_id` can resume a prior conversation.

## Project structure

```
gpu-datacenter-siting-agent/
├── main.py       # entry point — _invoke(query_text, thread_id)
├── config.py     # LLM client + API keys (all via environment variables)
├── state.py      # LangGraph State TypedDict + structured-output schemas
├── tools.py      # Mireye/EIA API calls + get_datacenter_site_data tool
├── nodes.py      # graph node functions (intro_router, decision_node, action_node, etc.)
├── graph.py      # StateGraph wiring (nodes + edges)
├── prompts.py    # all system prompts
└── requirements.txt
```

## Setup

1. **Clone and install dependencies**

   ```bash
   git clone https://github.com/arbaz-builds/gpu-datacenter-siting-agent.git
   cd gpu-datacenter-siting-agent
   pip install -r requirements.txt
   ```

2. **Set environment variables**

   | Variable          | Description                                              |
   |-------------------|-----------------------------------------------------------|
   | `NVIDIA_API_KEY`  | NVIDIA NIM API key                                        |
   | `NVIDIA_BASE_URL` | Optional — defaults to `https://integrate.api.nvidia.com/v1` |
   | `MIREYE_TOKEN`    | Mireye API bearer token                                    |
   | `EIA_API_KEY`     | U.S. EIA API key (free — register at eia.gov/opendata) |
   | `DATABASE_URL`    | Postgres connection string (for conversation checkpointing) |

   ```bash
   export NVIDIA_API_KEY="nvapi-..."
   export MIREYE_TOKEN="..."
   export EIA_API_KEY="..."
   export DATABASE_URL="postgresql://user:pass@host/dbname?sslmode=require"
   ```

3. **Run the API**

   ```bash
   uvicorn main:app --reload
   ```

   Then either open the interactive docs at `http://127.0.0.1:8000/docs`,
   or call it directly:

   ```bash
   curl -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "Looking for an H100 GPU data center site near Austin, TX"}'
   ```

   `GET /health` is a plain liveness check that doesn't touch Postgres or
   any external API — useful for deployment health checks.

4. **Or run it as a standalone script** (no HTTP server)

   ```python
   import asyncio
   from langchain_core.messages import HumanMessage
   from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
   from config import DATABASE_URL
   from graph import agent

   async def run():
       async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as cp:
           await cp.setup()
           result = await agent.compile(checkpointer=cp).ainvoke(
               {"messages": [HumanMessage(content="Looking for an H100 GPU data center site near Austin, TX")]},
               config={"configurable": {"thread_id": "1"}},
           )
       print(result["messages"][-1].content)

   asyncio.run(run())
   ```

## Example

**Input:** `"Looking for an H100 GPU data center site near Austin, TX"`

The agent plans 5 candidate locations, geocodes each one, pulls Mireye's
`data_center_siting` fields (power, terrain, water, hazards) and EIA's
latest Texas electricity price, then makes an explicit decision for every
candidate — `SELECT`, `SHORTLIST`, or `REJECT` — with reasons and any
blocking issues cited. Exactly one candidate can be `SELECT`ed, and a
candidate with a blocking issue is never selected, even if the model tries
to. The final answer explains the selected site, the backups, the rejected
sites and why, and the recommended next action (e.g. "begin preliminary
due diligence for Houston"). If any of those lookups fail, the agent says
so explicitly instead of guessing.

Each decision also carries a screening-level annual electricity cost
estimate (assumed load in MW x 8,760 hours x the EIA retail price for
that state), so a SELECT vs. REJECT decision reads as a dollar figure,
not just a label — e.g. *"Houston: ~$24.1M/year vs. Dallas: ~$32.9M/year
— roughly $8.8M/year lower at Houston."* The MW figure is a stated
screening assumption (50 MW by default), not a measured site capacity.

## Testing

`test_decision_guards.py` unit-tests the validation/repair layer in
`decision_node` directly — no LLM call, no Mireye/EIA call, no network
access. It proves that invalid or dangerous model output gets caught by
deterministic code rather than relying on the prompt being followed:

```
PASS  test_normal_selection
PASS  test_blocking_issue_rejected
PASS  test_multiple_select_guard
PASS  test_all_candidates_rejected
PASS  test_electricity_cost_is_recomputed_not_trusted
PASS  test_electricity_cost_is_none_when_price_missing

6 passed, 0 failed
```

Run it with:
```bash
python3 test_decision_guards.py
# or, with pytest installed:
python3 -m pytest test_decision_guards.py -v
```

## Notes

- No credentials are hardcoded — everything is read from environment variables.
- The `get_datacenter_site_data` tool includes automatic retries (with
  backoff) for transient network failures against both Mireye and EIA.
- `answer_node` will not generate a scored report unless real site data was
  successfully retrieved during the conversation — it fails safe rather than
  hallucinating a plausible-looking result.
