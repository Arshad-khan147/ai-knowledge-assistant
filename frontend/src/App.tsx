
import { useState } from "react";
import {
  Bot,
  FileText,
  Plus,
  Send,
  Paperclip,
  Sparkles,
  LoaderCircle,
} from "lucide-react";

type Source = {
  id: string;
  source: string;
  page: number | string;
  score: number;
  text: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

const API_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function App() {
  const [apiKey, setApiKey] = useState("");
  const [keyInput, setKeyInput] = useState("");
  const [connected, setConnected] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [documents, setDocuments] = useState<
    { filename: string; size_bytes: number }[]
  >([]);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  async function connect() {
    if (!keyInput.trim()) {
      setError("Please enter your API key.");
      return;
    }

    try {
      const response = await fetch(`${API_URL}/documents`, {
        headers: { "X-API-Key": keyInput.trim() },
      });

      if (!response.ok) {
        throw new Error(
          response.status === 401
            ? "Invalid API key."
            : `Connection failed: ${response.status}`
        );
      }

      const data = await response.json();

      setApiKey(keyInput.trim());
      setDocuments(data.documents ?? []);
      setConnected(true);
      setError("");
    } catch (err) {
      setConnected(false);
      setError(
        err instanceof Error
          ? err.message
          : "Could not connect to FastAPI."
      );
    }
  }

  async function uploadPdf(file?: File) {
    if (!file || !apiKey) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Please select a PDF file.");
      return;
    }

    setUploading(true);
    setError("");

    try {
      const form = new FormData();
      form.append("file", file);

      const response = await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        headers: { "X-API-Key": apiKey },
        body: form,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "PDF upload failed.");
      }

      setMessages((old) => [
        ...old,
        {
          role: "assistant",
          content: `Successfully processed ${file.name}. Added ${data.chunks_added ?? 0} new chunks.`,
        },
      ]);

      const listResponse = await fetch(`${API_URL}/documents`, {
        headers: { "X-API-Key": apiKey },
      });

      if (listResponse.ok) {
        const listData = await listResponse.json();
        setDocuments(listData.documents ?? []);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function askQuestion() {
    const prompt = question.trim();
    if (!prompt || !connected || busy) return;

    setQuestion("");
    setError("");
    setBusy(true);

    const assistantIndex = messages.length + 1;

    setMessages((old) => [
      ...old,
      { role: "user", content: prompt },
      { role: "assistant", content: "", sources: [] },
    ]);

    const updateAssistant = (update: Partial<Message>) => {
      setMessages((old) =>
        old.map((message, index) =>
          index === assistantIndex
            ? { ...message, ...update }
            : message
        )
      );
    };

    try {
      const response = await fetch(`${API_URL}/ask/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": apiKey,
        },
        body: JSON.stringify({ question: prompt, top_k: 3 }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(
          data.detail ||
            (response.status === 429
              ? "Rate limit reached. Please wait and retry."
              : `Request failed: ${response.status}`)
        );
      }

      if (!response.body) throw new Error("Streaming is unavailable.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const event of events) {
          const line = event
            .split("\n")
            .find((item) => item.startsWith("data: "));

          if (!line) continue;

          const data = JSON.parse(line.slice(6));

          if (data.type === "sources") {
            updateAssistant({ sources: data.sources ?? [] });
          } else if (data.type === "token") {
            setMessages((old) =>
              old.map((message, index) =>
                index === assistantIndex
                  ? { ...message, content: message.content + data.content }
                  : message
              )
            );
          } else if (data.type === "error") {
            throw new Error(data.message || "Answer generation failed.");
          }
        }
      }
    } catch (err) {
      updateAssistant({
        content: err instanceof Error ? err.message : "Something went wrong.",
      });
    } finally {
      setBusy(false);
    }
  }

  function newChat() {
    if (!busy) setMessages([]);
    setError("");
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[#111111] text-gray-200">
      <aside className="hidden w-72 shrink-0 flex-col border-r border-white/10 bg-[#171717] md:flex">
        <div className="flex items-center gap-3 p-5">
          <div className="rounded-xl bg-emerald-400/15 p-2 text-emerald-300">
            <Sparkles size={22} />
          </div>
          <div>
            <h1 className="font-semibold">Knowledge AI</h1>
            <p className="text-xs text-gray-500">Your PDF assistant</p>
          </div>
        </div>

        <button
          onClick={newChat}
          disabled={busy}
          className="mx-3 flex items-center gap-3 rounded-xl border border-white/10 p-3 text-sm hover:bg-white/5"
        >
          <Plus size={17} /> New chat
        </button>

        <div className="mt-7 flex-1 overflow-y-auto px-4">
          <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">
            Your documents ({documents.length})
          </p>

          {documents.length === 0 ? (
            <p className="text-sm leading-6 text-gray-500">
              Connect your API and upload a PDF to get started.
            </p>
          ) : (
            documents.map((doc) => (
              <div
                key={doc.filename}
                className="flex gap-2 rounded-lg px-2 py-3 text-sm hover:bg-white/5"
              >
                <FileText size={17} className="shrink-0 text-emerald-300" />
                <span className="break-words">{doc.filename}</span>
              </div>
            ))
          )}
        </div>

        <div className="space-y-2 border-t border-white/10 p-4">
          <p className="text-xs text-gray-400">
            {connected ? "API connected" : "Connect your backend"}
          </p>
          <input
            type="password"
            value={keyInput}
            onChange={(event) => setKeyInput(event.target.value)}
            placeholder="Enter API access key"
            className="w-full rounded-lg border border-white/10 bg-[#222] px-3 py-2 text-sm outline-none focus:border-emerald-400"
          />
          <button
            onClick={connect}
            className="w-full rounded-lg bg-emerald-400 px-3 py-2 text-sm font-semibold text-black hover:bg-emerald-300"
          >
            Connect to API
          </button>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-16 shrink-0 items-center justify-between border-b border-white/10 px-5 sm:px-8">
          <div className="flex items-center gap-2">
            <Bot size={22} className="text-emerald-300" />
            <span className="font-medium">AI Knowledge Assistant</span>
          </div>

          <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-white/10 px-3 py-2 text-sm hover:bg-white/5">
            {uploading ? (
              <LoaderCircle size={16} className="animate-spin" />
            ) : (
              <Paperclip size={16} />
            )}
            Upload PDF
            <input
              type="file"
              accept=".pdf,application/pdf"
              disabled={!connected || uploading}
              className="hidden"
              onChange={(event) => uploadPdf(event.target.files?.[0])}
            />
          </label>
        </header>

        {error && (
          <div className="mx-auto mt-4 w-full max-w-3xl rounded-lg border border-red-400/20 bg-red-400/10 p-3 text-sm text-red-200">
            {error}
          </div>
        )}

        <section className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="mx-auto flex h-full max-w-3xl flex-col justify-center px-6 py-12">
              <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-400/10 text-emerald-300">
                <Sparkles size={28} />
              </div>
              <h2 className="text-3xl font-semibold sm:text-4xl">
                What would you like to know?
              </h2>
              <p className="mt-3 leading-7 text-gray-400">
                Ask questions about your PDFs and get answers grounded in
                your documents, with source references.
              </p>

              <div className="mt-8 grid gap-3 sm:grid-cols-2">
                {[
                  "Summarize my uploaded documents",
                  "Explain the main concepts",
                  "Explain a topic step by step",
                  "Find relevant source pages",
                ].map((text) => (
                  <button
                    key={text}
                    onClick={() => setQuestion(text)}
                    className="rounded-xl border border-white/10 p-4 text-left text-sm hover:border-emerald-400/40 hover:bg-white/[0.03]"
                  >
                    {text}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-8 px-5 py-8">
              {messages.map((message, index) => (
                <article key={index} className="flex gap-4">
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-white/10">
                    {message.role === "user" ? (
                      <span className="text-xs">You</span>
                    ) : (
                      <Bot size={18} className="text-emerald-300" />
                    )}
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="mb-2 text-sm font-semibold">
                      {message.role === "user" ? "You" : "Knowledge AI"}
                    </p>
                    <div className="whitespace-pre-wrap break-words text-sm leading-7 text-gray-300">
                      {message.content ||
                        (busy && index === messages.length - 1
                          ? "Searching your documents..."
                          : "")}
                    </div>

                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-4 flex flex-wrap gap-2">
                        {message.sources.map((source) => (
                          <details
                            key={source.id}
                            className="max-w-full rounded-lg border border-white/10 px-3 py-2 text-xs"
                          >
                            <summary className="cursor-pointer text-emerald-300">
                              [{source.id}] {source.source} · Page {source.page}
                            </summary>
                            <p className="mt-3 whitespace-pre-wrap leading-5 text-gray-300">
                              {source.text}
                            </p>
                            <p className="mt-2 text-gray-500">
                              Score: {source.score}
                            </p>
                          </details>
                        ))}
                      </div>
                    )}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <footer className="shrink-0 px-4 pb-5 pt-3 sm:px-6">
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void askQuestion();
            }}
            className="mx-auto max-w-3xl rounded-2xl border border-white/15 bg-[#1c1c1c] p-3 focus-within:border-white/30"
          >
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void askQuestion();
                }
              }}
              disabled={!connected || busy}
              placeholder={
                connected
                  ? "Ask anything about your documents..."
                  : "Connect your API to start chatting..."
              }
              rows={2}
              className="w-full resize-y bg-transparent px-2 py-1 text-sm leading-6 outline-none placeholder:text-gray-500"
            />
            <div className="flex items-center justify-between pt-2">
              <p className="text-xs text-gray-500">
                Answers are based on your indexed PDFs
              </p>
              <button
                type="submit"
                disabled={!connected || !question.trim() || busy}
                className="rounded-xl bg-emerald-400 p-2.5 text-black hover:bg-emerald-300"
              >
                {busy ? (
                  <LoaderCircle className="animate-spin" size={18} />
                ) : (
                  <Send size={18} />
                )}
              </button>
            </div>
          </form>
          <p className="mt-3 text-center text-xs text-gray-600">
            Verify important answers against your original documents.
          </p>
        </footer>
      </main>
    </div>
  );
}
