/**
 * Performance Optimizer — Karpathy-style iterative agent.
 *
 * Conversational + autonomous. Answers questions naturally.
 * When asked to optimize, enters an iterative loop:
 * edit → test → keep/revert → report → next.
 *
 * Streams tool uses back to the host for Claude Code-style logging.
 */
import { createDeepAgent, LocalShellBackend } from "deepagents";
import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";
import { createInterface } from "readline";

const WORKSPACE = process.env.WORKSPACE || process.cwd();

const SYSTEM_PROMPT = `You are the Performance Optimizer. You live in a workspace at ${WORKSPACE}.

## Personality
You are conversational. If the user asks a question, answer it directly. If they ask you to look at something, look at it and tell them what you see. Only enter the optimization loop when explicitly asked to optimize, improve performance, or fix slow code.

## Optimization Loop
When asked to optimize, work like an autonomous researcher:

LOOP until all performance tests pass or you run out of ideas:
1. Explore the codebase — ls, read_file, glob, grep
2. Run the full test suite to see the baseline: \`python -m pytest -v 2>&1\`
3. Pick ONE slow function to optimize
4. Read it, identify the bottleneck, implement a focused fix
5. Run correctness tests: \`python -m pytest test_correctness.py -v 2>&1\`
6. If correctness FAILS → \`git checkout -- .\` and try a different approach
7. Run performance tests: \`python -m pytest test_performance.py -v 2>&1\`
8. If perf FAILS → \`git checkout -- .\` and try a different approach
9. If BOTH PASS → \`git add -A && git commit -m "opt: description"\`
10. Report what you optimized, the technique, and the speedup
11. Move to the next slow function — don't stop, don't ask

## Rules
- ONE function per iteration. Never batch changes.
- Always test correctness BEFORE performance.
- If correctness fails, your change is WRONG — revert immediately.
- Read files before editing them. Use edit_file for surgical changes.
- When all tests pass, give a final summary table with before/after/speedup.
- Python is \`python\` or \`python3\`. Git is available.
- NEVER STOP to ask "should I continue?" — keep going until done.`;

// --- NDJSON Protocol ---

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + "\n");
}

// --- Persistent stdin reader for multi-turn ---

class MessageReader {
  constructor() {
    this._rl = createInterface({ input: process.stdin });
    this._queue = [];
    this._resolve = null;
    this._closed = false;

    this._rl.on("line", (line) => {
      line = line.trim();
      if (!line) return;
      try {
        const msg = JSON.parse(line);
        if (msg.type === "shutdown") {
          this._closed = true;
          if (this._resolve) { this._resolve(null); this._resolve = null; }
          this._rl.close();
          return;
        }
        if (this._resolve) {
          const r = this._resolve;
          this._resolve = null;
          r(msg);
        } else {
          this._queue.push(msg);
        }
      } catch { /* ignore malformed */ }
    });

    this._rl.on("close", () => {
      this._closed = true;
      if (this._resolve) { this._resolve(null); this._resolve = null; }
    });
  }

  next() {
    if (this._queue.length > 0) return Promise.resolve(this._queue.shift());
    if (this._closed) return Promise.resolve(null);
    return new Promise((resolve) => { this._resolve = resolve; });
  }
}

// --- Content extraction ---

function extractText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join("\n");
  }
  return String(content);
}

// --- Tool activity summarizer ---

function summarizeTool(name, input) {
  if (!input) return name;
  if (name === "read_file") return `read ${input.path || ""}`;
  if (name === "write_file") return `write ${input.path || ""}`;
  if (name === "edit_file") return `edit ${input.path || ""}`;
  if (name === "execute") return `$ ${(input.command || "").slice(0, 120)}`;
  if (name === "ls") return `ls ${input.path || "."}`;
  if (name === "glob") return `glob ${input.pattern || ""}`;
  if (name === "grep") return `grep ${input.pattern || ""}`;
  if (name === "write_todos") return "update todos";
  return name;
}

// --- Agent (created once, reused across messages) ---

let backend = null;
let agent = null;

async function ensureAgent() {
  if (agent) return agent;

  backend = await LocalShellBackend.create({
    rootDir: WORKSPACE,
    virtualMode: false,
    inheritEnv: true,
    timeout: 120,
  });

  agent = createDeepAgent({
    model: new ChatAnthropic({ model: "claude-opus-4-6", temperature: 0 }),
    systemPrompt: SYSTEM_PROMPT,
    backend,
  });

  return agent;
}

async function handleMessage(content, onActivity) {
  const ag = await ensureAgent();

  // Stream events for tool-use logging (Claude Code-style)
  let finalContent = "";

  try {
    const eventStream = ag.streamEvents(
      { messages: [new HumanMessage(content)] },
      { version: "v2", recursionLimit: 200 }
    );

    for await (const event of eventStream) {
      // Tool calls → activity events
      if (event.event === "on_tool_start") {
        const toolName = event.name || "tool";
        const input = event.data?.input || {};
        const desc = summarizeTool(toolName, input);
        onActivity(toolName, desc);
      }

      // Capture final response from the graph
      if (event.event === "on_chain_end" && event.name === "LangGraph") {
        const output = event.data?.output;
        if (output?.messages?.length) {
          const last = output.messages[output.messages.length - 1];
          if (last.content) {
            finalContent = extractText(last.content);
          }
        }
      }
    }
  } catch (err) {
    // Fallback: if streamEvents isn't supported, use invoke
    if (err.message?.includes("streamEvents") || err.message?.includes("not a function")) {
      onActivity("agent", "Streaming not available, using invoke...");
      const result = await ag.invoke(
        { messages: [new HumanMessage(content)] },
        { recursionLimit: 200 }
      );
      const messages = result.messages || [];
      if (messages.length > 0) {
        const last = messages[messages.length - 1];
        if (last.content) {
          finalContent = typeof last.content === "string"
            ? last.content
            : JSON.stringify(last.content);
        }
      }
    } else {
      throw err;
    }
  }

  return finalContent || "Agent completed.";
}

// --- Entry Points ---

async function runPrimordial() {
  const reader = new MessageReader();
  send({ type: "ready" });

  // Multi-turn conversation loop
  while (true) {
    const msg = await reader.next();
    if (!msg) break;

    const mid = msg.message_id || "";
    const content = msg.content || "";

    try {
      const result = await handleMessage(content, (tool, desc) => {
        send({ type: "activity", tool, description: desc, message_id: mid });
      });

      send({
        type: "response",
        content: result,
        message_id: mid,
        done: true,
      });
    } catch (err) {
      send({
        type: "error",
        error: `Agent failed: ${err.message || err}`,
        message_id: mid,
      });
    }
  }
}

async function runStandalone() {
  const content = process.argv.slice(2).join(" ");
  if (!content) {
    console.log("Usage: node agent.js 'optimize the code'");
    process.exit(1);
  }

  console.error("Starting agent...");
  const result = await handleMessage(content, (tool, desc) => {
    console.error(`  › ${desc}`);
  });
  console.log("\n" + result);
}

// Detect mode
if (process.argv.length > 2) {
  runStandalone().catch(console.error);
} else if (process.stdin.isTTY) {
  console.log("Performance Optimizer — Karpathy-style iterative agent");
  console.log("Usage: node agent.js 'optimize the code'");
  console.log("   Or pipe via Primordial NDJSON protocol.");
} else {
  runPrimordial().catch((err) => {
    send({ type: "error", error: `Fatal: ${err.message || err}` });
    process.exit(1);
  });
}
