# ai-visibility-scan

**What the answer engines say about a brand, measured per engine, with the sources.**

Shoppers now ask Perplexity, Gemini, ChatGPT and the rest which bank, hotel or trainer to
pick. The brand gets no click and no log line. This tool asks the engines the questions a
shopper would type, keeps every answer and every cited source, and scores them so a brand
team gets numbers it can act on and re-run next month.

Built as the prototype for a take-home exercise (DEPT®, Nov 2026). Worked example: Monzo in
UK current accounts. Swap `brands/monzo.yaml` for any brand.

```
brief (brand, competitors, a watchlist of brands counted by name only, attributes,
       verified facts, 24 shopper questions: unbranded = names no tracked brand)
   │
   ▼
engines ──► no key, read through Chrome signed out, as a shopper sees them:
            perplexity-web · gemini-web · brave-web (Ask Brave) · duckai-web (Duck.ai)
            opt-in, on your own login (aivs login grok): grok-web
            optional, keyed, the providers' APIs with web search on:
            claude · chatgpt · gemini · perplexity · grok  each answer + sources cached as JSON
   │
   ▼
score ────► rules: is each brand named? which domains were cited? (deterministic)
            judge: recommended or just mentioned · order · sentiment per attribute ·
                   claims about the client vs its verified facts
                   (Claude Sonnet 5.5 via structured output when ANTHROPIC_API_KEY is set,
                    six calls in parallel; keyword heuristic otherwise, and the report says which)
   │
   ▼
metrics ──► presence · recommendation rate · share of voice · sentiment by attribute ·
            accuracy · own-site citation share · consistency across runs, per engine,
            plus a rollup per journey stage (discover, compare, choose, trust, switch)
   │
   ▼
report.html  headline tiles · by engine (with access badge) · by question (the unbranded set,
             per engine) · shortlist winners · also-named watchlist · sentiment ·
             sources (with citation counts per engine) · fix list (rule-derived, each item with an action built from what those answers
             cited; contradicted claims grouped by fact) · every answer verbatim ·
             method & limits
```

## Run it

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/) and Google Chrome (or run
`.venv/bin/playwright install chromium` and set `AIVS_BROWSER_CHANNEL=chromium`).
Every command below is run from the repository root. `.venv/bin/aivs` is the tool.

### Three-minute demo (no keys, nothing to configure)

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev,engines]"
.venv/bin/aivs scan --brief brands/monzo.yaml --engines gemini-web --limit 1 --scan-id demo
```

A Chrome window opens, types the first shopper question into Gemini as a signed-out user,
waits for the answer, opens the source chips, saves the answer and its sources, scores it and
writes `runs/demo/report.html`. Open that file. Without `ANTHROPIC_API_KEY` the judge is the
keyword heuristic and the report says so in a banner.

### Reproduce the Monzo run end to end

1. **Keys (optional).** `cp .env.example .env` and fill in what you have. `ANTHROPIC_API_KEY`
   turns on the Claude judge and the `claude` engine; `OPENAI_API_KEY`, `GEMINI_API_KEY`,
   `PERPLEXITY_API_KEY`, `XAI_API_KEY` add the other API engines. `.venv/bin/aivs engines`
   shows what will run.
2. **API engines (minutes, no Chrome).**
   `.venv/bin/aivs scan --brief brands/monzo.yaml --engines api --runs 2 --scan-id monzo-live`
   Every engine with a key, 24 questions, two runs each. Progress prints one line per answer.
3. **Browser engines (about 90 minutes of wall time, Chrome types in its own window).**
   `.venv/bin/aivs scan --brief brands/monzo.yaml --engines web --runs 2 --scan-id monzo-live`
   Same scan id, so the report covers both groups. Perplexity and Gemini also work headless
   (`--engines perplexity-web,gemini-web --headless`). If Perplexity shows its sign-in wall,
   the scan records those questions as failed; run the same command again later and only the
   failed ones are retried. Interrupt and resume at any time: nothing is asked twice.
4. **Facts.** Edit `brands/monzo.yaml` so every fact you can confirm is `status: verified`,
   then `.venv/bin/aivs rescore runs/monzo-live`. It re-reads the YAML and re-judges only
   what changed; it asks no engine anything.
5. **Check the judge.** `.venv/bin/aivs verify runs/monzo-live` walks 30 unbranded answers in
   the terminal (Enter = agree, r / m = correct, s = skip, q = stop), saves the labels, writes
   `runs/monzo-live/judge_check.json` and re-renders the report with the result in its header.
   Blind alternative: `.venv/bin/aivs label-template runs/monzo-live`, fill `human_stance` in
   `evals/labels.csv`, then `.venv/bin/python evals/run_evals.py runs/monzo-live evals/labels.csv`.
6. **Report.** `runs/monzo-live/report.html` is rewritten by every step above;
   `.venv/bin/aivs report runs/monzo-live` re-renders it from the saved scan at any time.

Other useful forms: `--engines api,gemini-web,perplexity-web --headless` mixes routes;
`--limit N` asks only the first N questions; `--judge heuristic` scores without a key;
`.venv/bin/aivs demo` runs offline on synthetic fixture answers, labelled as such.

## How the browser engines behave

- One persistent Chrome profile (`~/.aivs/chrome-profile`), signed out, locale en-GB,
  timezone Europe/London. One question at a time, a pause between questions
  (`AIVS_BROWSER_PACE_S`, default 4 s).
- Perplexity signed out has an undocumented quota: on 21 Nov 2026 it answered 13 of 24
  questions in one sitting, then showed a sign-in wall. The engine detects the wall and
  fails that question at once; resume the same scan id later (failed questions are retried),
  or `aivs login perplexity` to run it on your own account, labelled "signed in".
- Gemini signed out sometimes answers without searching: no source chips, no footnotes,
  nothing to click (9 of 24 Monzo questions on 21 Nov 2026). The answer is kept with a note
  saying so and the report counts "answers with no sources" per engine. It is what the
  shopper saw, so it is never retried or filled in from elsewhere.
- Bot checks are never solved. In a visible window the scan waits up to two minutes for a
  human to clear it; headless it records the question as failed. Every failure is listed
  in the report.
- Observed 21 Nov 2026 from London: Perplexity and Gemini answer headless. Ask Brave and
  Duck.ai show a bot check headless and answer in a visible window, so the default is a
  visible window. Google AI Mode shows a CAPTCHA on the first request, ChatGPT signed out a
  Cloudflare challenge, Copilot has no guest mode from the UK, Grok shows a sign-up card instead of an
  answer; those are not included and the report says so.
- Each engine keeps its selectors at the top of its class in `src/aivs/engines/browser.py`.
  When a site changes its interface, that is the only place to look.

## Check the judge before you quote it

```bash
# simplest: verify in the terminal (Enter = agree, r / m = correct, s = skip, q = stop)
.venv/bin/aivs verify runs/<scan_id>
# or a sheet: 30 unbranded answers, spread across engines, judge hidden (blind) ...
.venv/bin/aivs label-template runs/<scan_id> --sample 30
# ... or with the judge's call shown (write ok, or the corrected word)
.venv/bin/aivs label-template runs/<scan_id> --sample 30 --show-judge
.venv/bin/python evals/run_evals.py runs/<scan_id> evals/labels.csv
```

Only unbranded questions are sampled, because that is where the recommendation rate is
measured. Blind is the stronger claim; verify is faster. Both write
`runs/<scan_id>/judge_check.json`, which the report shows in its header as "N answers verified
by hand" or "labelled blind", so one is never presented as the other.

Presence and sources are rules and need no checking. Recommended-vs-mentioned is a judgement,
so it is measured against hand labels (target 85%+ agreement) before the number goes in front
of a client. The judge asks for a closed JSON schema (`output_config` structured output), not a
forced tool call: Claude Sonnet 5.5 and Opus 5.5 reject forced `tool_choice` with a 400. A score
whose judge call failed is retried on the next `aivs rescore`; good scores are never re-bought.

## What it does not do (on purpose)

- It does not work around bot checks or sign in with anyone's account. Surfaces that answer
  a signed-out browser are read as they are; the others go through the providers' APIs with
  web search on, labelled as API, not app; Google AI Overviews / AI Mode are out.
- It does not fix anything. The fix list says where to work (own pages unreadable, forums
  driving sentiment, contradicted facts, engine-specific gaps); the work is the next engagement.
- It does not audit the brand's site for agent-readiness (feeds, llms.txt, checkout
  protocols). That is a separate offering with a different buyer.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Offline: fake engine on fixtures, heuristic judge (attribute matching, fee negation), full
pipeline, caching, one report across several invocations, metrics and the stage rollup,
report badges and indicative markers, source rules with path prefixes, and the browser
engines' pure parts (bot-check detection, link cleaning, chip-label parsing, registry).
Live behaviour is checked with `--engines web --limit 1`.
