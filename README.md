# GPU Center Location Agent

**The problem:** Picking a GPU data center site is currently a multi-week,
five-figure-consultant process — power availability, terrain, water,
hazards, and electricity pricing all live in different databases, and
someone has to manually stitch them together for every candidate address
before a developer can even start comparing options.

**What this does:** an autonomous LangGraph agent that takes a US address
(or a state/city), fetches real siting data and electricity pricing, and
returns a scored, sourced recommendation — in minutes, not weeks. It
combines:

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
      → llm_tool             (decide whether to call the site-data tool)
        → tools               (get_datacenter_site_data: Mireye + EIA)
          → structure_llm      (normalize raw site JSON into a compact schema)
          → llm_tool            (loop back, now with structured data in context)
        → answer              (final scored recommendation, once data is in hand)
END
```

- **`intro_router`** — classifies the user's message as `gpu_search` (enough
  detail to proceed) or `unclear` (needs more info), using structured output.
- **`planner_llm`** — turns a state (e.g. "Texas") into the 5 best candidate
  cities, or a city into nearby industrial areas — without scoring or
  analyzing them yet. Planning is kept separate from evaluation so the agent
  never scores a location it hasn't actually looked at.
- **`llm_tool_node`** — a decision-tool loop (max 3 iterations) that calls
  `get_datacenter_site_data` when it needs real site data.
- **`get_datacenter_site_data`** — geocodes the address, extracts the US
  state, fetches Mireye's `data_center_siting` preset, and pulls EIA's latest
  state electricity price. Includes retry logic for transient API failures.
- **`structure_llm_node`** — strips metadata and normalizes the raw Mireye/EIA
  JSON into a clean schema for the decision LLM.
- **`answer_node`** — produces the final scored recommendation (JSON) across
  power, cooling, land, water, hazards, environment, and residential impact.
  Refuses to fabricate a report if no real site data was ever retrieved.

State is persisted per-conversation via **Postgres** (`AsyncPostgresSaver`),
so a `thread_id` can resume a prior conversation.

## Project structure

```
gpu-center-location-agent/
├── main.py       # entry point — _invoke(query_text, thread_id)
├── config.py     # LLM client + API keys (all via environment variables)
├── state.py      # LangGraph State TypedDict + IntroDecision schema
├── tools.py      # Mireye/EIA API calls + get_datacenter_site_data tool
├── nodes.py      # graph node functions (intro_router, llm_tool_node, etc.)
├── graph.py      # StateGraph wiring (nodes + edges)
├── prompts.py    # all system prompts
└── requirements.txt
```

## Setup

1. **Clone and install dependencies**

   ```bash
   git clone https://github.com/arbaz-builds/gpu-center-location-agent.git
   cd gpu-center-location-agent
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

3. **Run**

   ```python
   import asyncio
   from main import _invoke

   result = asyncio.run(_invoke("Looking for an H100 GPU data center site near Austin, TX"))
   print(result["messages"][-1].content)
   ```

## Example

**Input:** `"Looking for an H100 GPU data center site near Austin, TX"`

The agent geocodes the address, pulls Mireye's `data_center_siting` fields
(power, terrain, water, hazards) and EIA's latest Texas electricity price,
then returns a scored breakdown — each category scored 0–10, with the
underlying data source and timestamp cited for every claim, plus an
overall recommendation and the trade-offs behind it. If any of those
lookups fail, the agent says so explicitly instead of guessing.

## Notes

- No credentials are hardcoded — everything is read from environment variables.
- The `get_datacenter_site_data` tool includes automatic retries (with
  backoff) for transient network failures against both Mireye and EIA.
- `answer_node` will not generate a scored report unless real site data was
  successfully retrieved during the conversation — it fails safe rather than
  hallucinating a plausible-looking result.
