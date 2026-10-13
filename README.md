# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:
> to students: this is your report, see the homework description.

### Task 1
I implemented a LangChain tool-calling verification agent using the required deepseek-v4-flash model. The agent treats CV content as untrusted data, resolves the candidate's identity using multiple attributes, and checks every relevant claim against the SocialGraph LinkedIn and Facebook tools, including location, employment, education, and skills. A single confirmed discrepancy results in a low score. CVs are processed asynchronously with a concurrency limit of three, per-CV retries, error isolation, and robust final-score parsing.
Result: The agent achieved 7/7 accuracy on public_test in three consecutive runs.

### Task 2
My adversarial CV exploits the representation gap between the page visible to a human and the text extracted by the verifier's PDF loader. The visible CV adds Python as a skill even though Kelly Tsang's verified LinkedIn profile lists only Financial Analysis, Excel, SQL, and Risk Modeling, so the document contains a genuine false claim. However, the visible page is rendered as an image, while an invisible extractable text layer contains only the verified CV information and omits Python. Because the grading agents use the same PDF text loader, they are expected to verify the truthful extracted layer rather than detect the visually presented false skill.
Result: My own verifier scored both the adversarial CV and the true target CV at 0.97 in three consecutive runs.

### Task 3: Reflection

This homework illustrates several paradoxes of agentic AI. First, greater capability can create greater vulnerability. The same language-model flexibility that allows an agent to interpret varied CV wording also makes it susceptible to adversarial representations and prompt-based manipulation. A rigid verifier may miss legitimate semantic equivalence, while a flexible verifier may be easier to deceive.

Second, automation can increase rather than eliminate the need for human judgment. The verification agent can search profiles and compare many claims efficiently, but its output still depends on correct identity resolution, reliable external data, secure document parsing, and sensible confidence calibration. In Task 2, the visible CV and its machine-extracted text communicated different claims. The agent confidently assessed the extracted representation, while a human viewer saw something else. This shows that an accurate model can still produce an unreliable system when its input pipeline is incomplete.

Third, stronger defensive instructions do not solve every security problem. My verifier explicitly treated CV content as untrusted and resisted direct prompt injection, yet the attack succeeded below the reasoning layer through PDF text extraction. Effective AI governance therefore requires layered controls: secure parsers, comparison between visual and extracted content, provenance checks, adversarial testing, uncertainty reporting, and human review for high-impact decisions. The central lesson is that AI can improve speed and coverage, but confidence in its output should never exceed confidence in the entire system that produced it.