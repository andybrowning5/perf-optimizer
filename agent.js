/**
 * Coding agent — Karpathy-style iterative worker.
 *
 * Conversational + autonomous. Answers questions naturally.
 * When given a task, enters an iterative loop:
 * edit → test → keep/revert → report → next.
 *
 * Language-agnostic: detects the project's language, test runner,
 * and build system automatically.
 *
 * Streams tool uses back to the host for Claude Code-style logging.
 */
import { createDeepAgent, LocalShellBackend } from "deepagents";
import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";
import { createInterface } from "readline";

const WORKSPACE = process.env.WORKSPACE || process.cwd();

const SYSTEM_PROMPT = `You are a coding agent. You live in a workspace at ${WORKSPACE}.

## Personality
You are conversational. If the user asks a question, answer it. If they ask you to look at something, look and report back. When given a task (optimize, fix bugs, refactor, add features), enter the work loop below.

## Work Loop
When given a coding task, work like an autonomous researcher:

1. **Explore** — ls, read_file, glob, grep to understand the project structure, language, and tooling.
2. **Detect** — Figure out the language, test runner, build system, and how to verify changes:
   - Python: \`python -m pytest -v 2>&1\`, \`python -m unittest discover -v 2>&1\`
   - Node.js: \`npm test 2>&1\`, \`npx jest --verbose 2>&1\`, \`npx vitest run 2>&1\`
   - Go: \`go test ./... -v 2>&1\`
   - Rust: \`cargo test 2>&1\`
   - Or whatever the project uses — check package.json scripts, Makefile, CI config, etc.
3. **Baseline** — Run the tests to see what passes and what fails.
4. **LOOP** — For each change:
   a. Make ONE focused change (one function, one bug, one improvement).
   b. Run the tests.
   c. If tests PASS → \`git add -A && git commit -m "description"\`
   d. If tests FAIL → \`git checkout -- .\` and try a different approach.
   e. Report what you changed and the result.
   f. Move to the next issue — don't stop, don't ask.

## Rules
- ONE change per iteration. Never batch unrelated changes.
- Always run tests after each change. If they fail, revert immediately.
- Read files before editing them. Use edit_file for surgical changes.
- When done, give a summary of everything you changed and the results.
- Git is available. Commit after each successful change so reverts are clean.
- NEVER STOP to ask "should I continue?" — keep going until the task is done.
- If you run out of ideas or everything passes, summarize and stop.`;

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
          finalContent = extractText(last.content);
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
  console.log("Coding Agent — Karpathy-style iterative worker");
  console.log("Usage: node agent.js 'fix the failing tests'");
  console.log("   Or pipe via Primordial NDJSON protocol.");
} else {
  runPrimordial().catch((err) => {
    send({ type: "error", error: `Fatal: ${err.message || err}` });
    process.exit(1);
  });
}
