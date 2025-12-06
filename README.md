## Agent Routing with Deep Dive Logprobs

CLI demo that routes a user query to the best-fitting helper agent and prints a probability-aware audit trail. It uses LangChain with OpenAI chat logprobs to show exactly why the router said YES or NO for each agent. A companion notebook contrasts naive “hallucinated” confidence scores with logprob-based probabilities.

### Prerequisites
- Python 3.13+
- uv installed (`pip install uv` if needed)
- An OpenAI API key available as `OPENAI_API_KEY` (put it in a local `.env`)

### Install and run (CLI)
```bash
# optional: create .env with your key
echo OPENAI_API_KEY=sk-... > .env

# run the built-in demo with several queries
uv run python main.py

# route a custom query
uv run python main.py ask --query "Write a short startup pitch for a coffee app."
```

### Notebook walkthrough
- Open `agent_routing_demo.ipynb` and run the cells top-to-bottom.
- It loads `.env` for `OPENAI_API_KEY` and shows:
  - Naive confidence scores (0–100) as pure text generations (hallucinated).
  - Logprob-based YES/NO probabilities with a token-level deep dive.
- Alternatively, use the scriptified notebook cells in `router_cells.py` (supports `#%%` cells in VS Code) and run `python router_cells.py` to see both naive and logprob routers.

### What you will see
- A summary table sorted by `p(YES)` per agent.
- A deep-dive section per agent with every considered token, its logprob, and a normalized YES/NO split.
- Example screenshot:  
![Logprob visualization](output.png)

In the deep-dive, tokens normalized to YES/NO show which way the model leaned; the final line reports the normalized YES vs NO percentages for that agent. Use the summary table to pick the top agent, and the deep-dive to debug or tune the routing prompt.

- Idea first read: https://cookbook.openai.com/examples/using_logprobs
