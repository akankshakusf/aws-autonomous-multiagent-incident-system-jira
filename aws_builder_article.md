# Building an Autonomous Multi-Agent System for AWS Incident Response: From Alert Fatigue to Self-Learning JIRA Automation

*How I built a production-grade agentic workflow on Amazon Bedrock that detects, diagnoses, and resolves AWS incidents — without anyone touching a ticket.*

---

## The itch that started this

In a previous role, I used to stare at our ops dashboard and wonder the same thing every week: **why am I writing the same JIRA ticket I wrote three months ago?**

The pattern was painfully familiar. CloudWatch fires an alarm. Someone pages the on-call. The on-call pulls up CloudTrail, scrolls through logs, remembers that "oh, we saw this exact thing in Q2," digs through old tickets, finds the fix, writes a new ticket, copy-pastes the remediation steps, assigns it to the right team. An hour gone. Multiply by a dozen alerts a week. Multiply by every engineer on the team.

The ridiculous part? The knowledge was already there. We had *solved* these problems. We just kept re-solving them manually because no system connected "this alert" to "the thing we did last time."

I used to think an automated fix for this was wishful thinking. Fast forward to my work at the current organization, and I finally built the system I'd been sketching in my head for years: an **Autonomous Multi-Agent Incident Response Pipeline on AWS** that monitors AWS in real time, diagnoses problems by searching its own memory first, falls back to web search for novel incidents, and creates fully contextual JIRA tickets — all without a human in the loop.

This post is a walkthrough of how I built it, what each agent does, and the architectural decisions I'd recommend if you're thinking about doing something similar.

📦 **Repo:** [https://github.com/akankshakusf/aws-autonomous-multiagent-incident-system-jira](https://github.com/akankshakusf/aws-autonomous-multiagent-incident-system-jira)

---

## What the system actually does

At a high level: you point it at your AWS account, tell it which services to watch (IAM, S3, EC2, Bedrock, Elastic Beanstalk, whatever), and it runs a four-agent loop every time something interesting happens in the logs.

1. **Supervisor Agent** orchestrates the whole workflow and tracks state across handoffs through langgraph
2. **Monitoring Agent** pulls logs from CloudWatch and CloudTrail, detects anomalies
3. **Diagnosis Agent** does RAG against a vector store of past incidents; if no match, escalates to Tavily web search
4. **Resolution Agent** writes a detailed JIRA ticket with root cause and remediation, then pushes a notification to the team

The twist that makes it more than a glorified script: **every resolved incident gets written back to the knowledge base.** The system gets smarter with every run. Month one, it's searching the web for half the incidents. Month three, 80% of incidents match something in memory and resolve in seconds.

### The architecture

![System Architecture](https://github.com/akankshakusf/Autonomous-MultiAgent-AWS-Incident-Jira-Automation/blob/main/images/arch%202.png?raw=true)

---

## The stack

Nothing exotic here — I deliberately stayed on well-supported services so this is production-viable, not a toy demo:

- **Amazon Bedrock** — Claude and Amazon Nova (Pro for reasoning-heavy diagnosis, Micro for routing and ticket creation)
- **LangGraph** — state machine orchestration for the agent graph
- **LangChain** — ReAct agent primitives and tool wrappers
- **HuggingFace embeddings + Chroma** — vector store for the incident memory
- **Tavily API** — web search fallback when RAG misses
- **AWS CloudWatch, CloudTrail, boto3** — the actual log sources
- **Atlassian Python API** — JIRA ticket creation
- **LangSmith** — tracing and debugging across the whole multi-agent graph

One thing worth calling out: I use **different models for different agents based on the cognitive load of the task**. Monitoring and resolution are largely deterministic — fetch data, format output, call an API — so Nova Micro handles them cheaply. Diagnosis needs real reasoning and synthesis, so it gets Nova Pro. This kind of per-agent model selection is one of the biggest wins in multi-agent design, and it's something I'd push harder on in a v2.

---

## Walking through each agent

### 1. The Supervisor: orchestration and state

The Supervisor is the traffic cop. It doesn't do any reasoning about AWS or incidents — its job is to look at the current workflow state and decide which agent should run next.

I defined the state as a `TypedDict` so LangGraph can validate handoffs:

```python
class WorkflowState(TypedDict, total=False):
    current_step: str
    monitoring_complete: bool
    diagnosis_complete: bool
    resolution_complete: bool
    jira_ticket_created: bool

class AWSMonitoringState(TypedDict):
    messages: list
    workflow: WorkflowState
```

The supervisor node inspects these flags and routes accordingly. Monitoring done but diagnosis not? Route to diagnosis. Diagnosis done? Route to resolution. Ticket created? End the workflow.

This is a simple pattern but it matters a lot. Without an explicit state object, multi-agent systems drift — one agent re-runs, another gets skipped, and you spend an afternoon debugging why the same incident got three tickets. Making state first-class and enforcing transitions through the supervisor is what keeps the graph sane.

### 2. The Monitoring Agent: pulling the right logs

The Monitoring Agent is a ReAct agent with one tool: `fetch_cloudwatch_logs_for_service`. Given a service name (like `iam`,`s3`, `EC2`, `Bedrock`, `Elastic Beanstalk`), a lookback window in days, and an optional filter pattern, it pulls the relevant log streams via boto3.

```python
fetch_logs_tool = StructuredTool.from_function(
    func=fetch_cloudwatch_logs_for_service,
    name="fetch_cloudwatch_logs_for_service",
    description="Fetch CloudWatch logs for a specified AWS service. "
                "You can provide days and optional filter patterns..."
)

monitoring_toolkit = [fetch_logs_tool]
```

The agent itself runs on Nova Micro with a system prompt that tells it to act as a CloudWatch monitoring specialist — parse the user's intent, figure out which service to query, pick a sensible time window, and summarize what it finds.

This is where a lot of "agentic" projects go wrong, in my opinion. People throw fifteen tools at an agent and hope it figures out which one to call. I'd rather have **one well-described tool with clear input constraints** and let the LLM focus on interpretation, not tool selection. Nova Micro handles this perfectly well, and the latency stays low.

### 3. The Diagnosis Agent: RAG first, web second, memory always

This is the most interesting agent in the system, and the one I iterated on the most.

The flow is:

1. Take the monitoring summary
2. Embed it and search the Chroma vector store (the "incident memory")
3. If we get a strong match → use the stored remediation
4. If we don't → fall back to Tavily web search
5. Either way → write the diagnosis back to memory so future runs can retrieve it

The vector store is built from a `memory.md` file that grows over time. Initial seed:

```python
loader = TextLoader("memory.md")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(docs, embeddings)
```

The RAG + web-search-fallback pattern is wrapped in a single tool the agent calls:

```python
@tool
def remediation_strategy_from_monitoring(state: Dict) -> Dict:
    """
    Extracts issues from monitoring agent's response,
    retrieves remediation steps via RAG or Tavily fallback,
    and appends results to messages.
    """
    # 1. Pull issues from the last message
    # 2. Similarity search against vector store
    # 3. If weak match, invoke Tavily
    # 4. Return structured diagnostic report
```

This agent uses **Nova Pro** because the diagnosis step is where the real reasoning happens. It has to look at raw log events like "DeleteUser attempt from user `malicious.actor`" and produce a structured Security & Compliance Diagnostic Report with key findings, severity, affected area, and step-by-step remediation guidance.

Here's the kind of output it produces:

```
Security & Compliance Diagnostic Report:

Key Findings:
1. Repeated Log Filtering Operations: The IAM user `aktest1` has performed
   multiple log filtering operations within the past week...

Severity: Medium

Potential Risk or Affected Area:
- Security: The repeated log filtering operations could potentially be a
  sign of malicious activity if they are not legitimate...

Remediation Guidance:
- Investigate the Operations: Review the log filtering operations...
- Implement Monitoring: Set up additional monitoring and alerting...
- Review IAM Policies: Ensure that the IAM policies associated with the
  user are properly configured...
```

This is the format that goes straight into the JIRA ticket — no reformatting needed downstream.

### 4. The Resolution Agent: turning diagnosis into action

The Resolution Agent is the simplest of the four. It takes the diagnostic report, calls the JIRA API, and fires a push notification. That's it.

```python
@traceable(run_type="tool", name="Search Tool")
def create_jira_issue(summary: str, description: str,
                      project_key: str = 'BTS',
                      issue_type: str = "Task",
                      assignee: str = None):
    jira = JiraAPIWrapper()
    issue_fields = {
        "summary": summary,
        "description": description,
        "issuetype": {"name": issue_type},
        "project": {"key": project_key}
    }
    # ... create issue, return URL
```

But there's a guardrail I added that matters: **the resolution node has an explicit skip list.** If the diagnosis comes back saying "no critical errors or exceptions were found" or "no significant security or compliance events detected," the agent short-circuits and doesn't create a ticket. Nothing kills trust in an automated system faster than it filing JIRA tickets for "everything looks fine."

```python
def resolution_node(state: Dict, config=None) -> Dict:
    messages = state["messages"]
    last_msg = messages[-1].content if messages else "No diagnosis provided"

    if (
        "no critical errors or exceptions were found" in last_msg.lower()
        or "no significant security or compliance events" in last_msg.lower()
        or "no critical security failures" in last_msg.lower()
    ):
        # Skip ticket creation, end workflow cleanly
        ...
```

One of those unglamorous details that makes the difference between a demo and something you'd actually run in production.

---

## The glue: observability with LangSmith

Multi-agent systems are hard to debug because the failure mode is rarely a stack trace — it's "the agent made a weird decision three steps ago and now the state is wrong."

I wrapped the whole graph with LangSmith tracing using `@traceable` decorators on each node. Every agent invocation, every tool call, every model response is captured as a span. When something breaks, I can literally open a waterfall view and see exactly which agent routed where, what it reasoned about, and what tokens it consumed.

If you're building agentic systems and you don't have tracing set up, stop and do that first. It is not optional. I lost a lot of time debugging by print statement before I wired LangSmith in.

---

## What I'd do differently in v2

A few honest reflections after running this for a while — and one of them changed how I think about RAG in general.

**1. Rethink the memory layer: compile, don't chunk.**

This is the big one, and it connects to something I wrote about recently on LinkedIn after reading an Andrej Karpathy post that genuinely rearranged how I think about retrieval.

In this project, `memory.md` is essentially a growing log of raw incident reports that I chunk and embed. That's the default RAG pattern almost everyone uses: dump raw docs into a vector store, chunk them, embed them, retrieve top-k at query time. It works — until your corpus gets big enough that the retrieved chunks are noisy, redundant, or contradictory.

Karpathy's insight was that **the bottleneck was never retrieval — it was always what you were retrieving from.** Instead of chunking raw sources, you have an LLM *compile* those sources into a structured wiki first: concept articles with cross-links, summaries, backlinks, auto-maintained index files. Retrieval then runs over clean, structured, interlinked knowledge — not raw mess.

For this incident response system, the v2 I'm imagining looks like this: every time the Diagnosis Agent resolves an incident, a separate "Librarian Agent" takes that resolution and *integrates* it into the knowledge base — merging it with related past incidents, updating cross-references (e.g., "this IAM issue is related to these three prior EC2 permission issues"), writing a clean concept article for the incident category if one doesn't exist, and flagging obsolete entries when infrastructure changes. The vector store then indexes the compiled wiki, not the raw incident log.

That single architectural shift would fix most of the problems I ran into — stale entries, duplicate retrievals, drift — because the knowledge base would be self-curating instead of self-accumulating.

**2. Replace `memory.md` with a structured metadata store.**

Even within the "compile, don't chunk" approach, markdown as the persistence layer doesn't scale. A production v2 would store incidents and compiled concept articles in DynamoDB with structured metadata (service, severity, resolution type, timestamp, related-incident IDs) and do **hybrid search** — keyword filtering on metadata plus vector similarity on the compiled content. Pure embedding similarity is a blunt instrument once the corpus is big enough.

**3. Strands Agents SDK would simplify the agent code.**

I wrote this project before I was deep in Strands. If I rebuilt it today, I'd use the AWS Strands Agents SDK for cleaner agent definitions and native Bedrock integration, and swap in Bedrock Guardrails for the Diagnosis Agent to prevent it from ever producing bad remediation advice.

**4. The supervisor could be smarter.**

Right now it routes based on boolean flags. A more sophisticated supervisor would route conditionally based on incident severity — e.g., skip diagnosis entirely and page a human for anything marked `CRITICAL`, or run diagnosis and resolution in parallel for independent issues in the same batch.

**5. Push notifications should be multi-channel.**

I only wired Slack, but production incident response usually wants a ladder: Slack for low severity, PagerDuty for medium, phone call for critical.

---

## Why this pattern is bigger than incident response

The real takeaway from this project — and the reason I keep thinking about it — is that **supervisor-orchestrated multi-agent workflows with RAG-backed memory are a general-purpose template for automating any knowledge-work loop.**

Swap the Monitoring Agent for a "Pull Request Reviewer" agent and the Resolution Agent for a "Create GitHub Issue" agent, and you have an automated code review system. Swap in a "Customer Email Classifier" and a "Draft Response" agent, and you have a Tier-1 support system. The architecture — supervisor state machine, specialized agents, self-learning memory, web fallback, observability — is the thing that generalizes.

Any time you catch yourself or your team doing the same multi-step reasoning task over and over, there's probably a multi-agent system hiding in it. Start with the state machine. Pick models per agent based on cognitive load. Make the memory self-updating. Trace everything.

---

## Try it yourself

The full code is on GitHub, including the Jupyter notebook that walks through every agent end-to-end, a Streamlit UI, and the memory.md seed file:

👉 [github.com/akankshakusf/Autonomous-MultiAgent-AWS-Incident-Jira-Automation](https://github.com/akankshakusf/Autonomous-MultiAgent-AWS-Incident-Jira-Automation)

You'll need AWS credentials with Bedrock access, a Tavily API key, a JIRA project, and optionally a LangSmith account for tracing.

If you build on this, I'd love to hear about it — especially if you adapt the pattern to a domain other than incident response. That's where this gets interesting.

---

*If you're working on agentic systems, AWS AI/ML tooling, or just want to nerd out about multi-agent architectures, I'd love to connect. You can find me on LinkedIn or read more of my work on Substack.*

**#AWS #AmazonBedrock #Claude #MultiAgent #RAG #LangGraph #AIEngineering #CloudOps**
