import React, { useEffect, useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import "./chat-widget.css";

/**
 * ChatWidget (plain CSS version)
 *
 * Props:
 * - apiUrl (string): backend endpoint (POST) that accepts { message, history } and returns { reply } or streaming text body
 * - authToken (string|null): optional Bearer token
 * - headerTitle (string)
 * - placeholder (string)
 */
function uid(prefix = "id") {
  return `${prefix}_${Date.now().toString(36)}_${Math.random()
    .toString(36)
    .slice(2, 8)}`;
}

export default function ChatWidget({
  // apiUrl = "https://9c028ce5410f.ngrok-free.app/query",
  apiUrl = import.meta.env.VITE_API_URL, //"https://9c028ce5410f.ngrok-free.app/query",//"http://localhost:8000/query",
  authToken = null,
  initialMessages = [],
  headerTitle = "Chat",
  placeholder = "Type a message and press Enter...",
  maxMessages = 200,
  showAvatar = true,
  minimized = false,
  onError = null,
}) {
  const [messages, setMessages] = useState(() =>
    (initialMessages || []).map((m) => ({ ...m, id: m.id || uid("msg") }))
  );
  const [text, setText] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [visible, setVisible] = useState(!minimized);
  const [error, setError] = useState(null);
  const containerRef = useRef(null);
  const controllerRef = useRef(null);
  const [sessionId, setSessionId] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, visible]);

  useEffect(() => {
    return () => {
      if (controllerRef.current) controllerRef.current.abort();
    };
  }, []);

  useEffect(() => {
    const id = uuidv4();
    setSessionId(id);
    console.log("Session ID:", id);
  }, []);

  function pushMessage(msg) {
    setMessages((prev) => {
      const next = [...prev, msg].slice(-maxMessages);
      return next;
    });
  }

  function updateMessage(id, patch) {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...patch } : m))
    );
  }

  function formatOutgoing(userText) {
    return {
      query: userText,
      session_id: sessionId,
      // history: history.map((h) => ({ role: h.role, text: h.text, id: h.id, createdAt: h.createdAt })),
    };
  }

  async function sendMessage() {
    const trimmed = text.trim();
    if (!trimmed || isSending) return;
    setError(null);

    const userMsg = {
      id: uid("u"),
      role: "user",
      text: trimmed,
      createdAt: new Date().toISOString(),
    };
    pushMessage(userMsg);
    setText("");

    // This is the bot's temporary message bubble for the typing indicator
    const typingMsg = {
      id: uid("typing"),
      role: "assistant",
      text: "...", // Or you can use an SVG or animated GIF in a separate component
      isTypingIndicator: true, // A flag to identify this message
      createdAt: new Date().toISOString(),
    };
    pushMessage(typingMsg);

    setIsSending(true);

    const controller = new AbortController();

    controllerRef.current = controller;

    try {
      const outgoing = formatOutgoing(trimmed);
      const headers = { "Content-Type": "application/json" };
      if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
      console.log("Sending msg to backend: ", outgoing);
      
      const res = await fetch(apiUrl, {
        method: "POST",
        headers,
        body: JSON.stringify(outgoing),
        signal: controller.signal,
      });

      if (!res.ok) {
        const textErr = await res.text();
        throw new Error(`Server ${res.status}: ${textErr}`);
      }

      const data = await res.json();
      const reply = data?.answer ?? JSON.stringify(data);

      // Remove the typing indicator message before adding the real reply
      setMessages((prev) => prev.filter((m) => m.id !== typingMsg.id));

      const botMsg = {
        id: uid("b"),
        role: "assistant",
        text: String(reply),
        createdAt: new Date().toISOString(),
      };
      pushMessage(botMsg);
    } catch (err) {
      // Also remove the typing indicator on error
      setMessages((prev) => prev.filter((m) => m.id !== "typing"));
      const message = err?.message ?? String(err);
      const errorMsg = {
        id: uid("error"),
        role: "assistant",
        text: `Error: ${message}`,
        error: true,
        createdAt: new Date().toISOString(),
      };
      pushMessage(errorMsg);
      setError(message);
      if (typeof onError === "function") onError(err);
    } finally {
      setIsSending(false);
      controllerRef.current = null;
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
    if (e.key === "Escape") {
      setVisible(false);
    }
  }

  function clearChat() {
    setMessages([]);
  }

  function retryLast() {
    const lastAssistant = [...messages]
      .reverse()
      .find(
        (m) =>
          m.role === "assistant" && (m.error || !m.text || m.text.length < 1)
      );
    if (!lastAssistant) return;
    const idx = messages.findIndex((m) => m.id === lastAssistant.id);
    const prev = messages[idx - 1];
    if (!prev || prev.role !== "user") return;
    setMessages((prevArr) => prevArr.filter((m) => m.id !== lastAssistant.id));
    setText(prev.text);
    setTimeout(() => sendMessage(), 50);
  }

  return (
    <div className="chat-widget-root" role="region" aria-label="Chat widget">
      <div className="chat-widget-card">
        <div className="chat-widget-header">
          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <svg
                className="chat-icon"
                viewBox="0 0 24 24"
                width="22"
                height="22"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M12 2a10 10 0 100 20 10 10 0 000-20z"
                  stroke="currentColor"
                  strokeWidth="1.2"
                />
              </svg>
            </div>
            <div>
              <div className="chat-widget-title">{headerTitle}</div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 8 }}>
            <button
              className="icon-btn"
              onClick={() => setVisible((v) => !v)}
              aria-label={visible ? "Minimize chat" : "Open chat"}
            >
              {visible ? "—" : "+"}
            </button>
            <button
              className="icon-btn"
              onClick={clearChat}
              aria-label="Clear chat"
            >
              🗑
            </button>
          </div>
        </div>

        {visible && (
          <div className="chat-widget-body" role="log" aria-live="polite">
            <div ref={containerRef} className="chat-widget-messages">
              {messages.length === 0 && (
                <div className="chat-empty">
                  Say hello 👋 — ask about your product, docs, or anything.
                </div>
              )}

              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`msg-row ${
                    msg.role === "user" ? "msg-user" : "msg-assistant"
                  }`}
                >
                  <div className="msg-bubble-wrapper">
                    {showAvatar && (
                      <div className="avatar">
                        {msg.role === "user" ? <span>U</span> : <span>B</span>}
                      </div>
                    )}
                    <div>
                      <div className="msg-bubble">
                        {msg.isTypingIndicator ? (
                          <span className="typing-indicator">...</span>
                        ) : (
                          msg.text
                        )}
                      </div>
                      <div className="msg-meta">
                        {new Date(msg.createdAt).toLocaleString()}
                      </div>
                      {msg.error && (
                        <div className="msg-error">
                          There was an error.{" "}
                          <button onClick={retryLast} className="link-btn">
                            Retry
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="chat-widget-footer">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  sendMessage();
                }}
                className="chat-input-row"
              >
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={placeholder}
                  className="chat-textarea"
                  aria-label="Type your message"
                />
                <div>
                  <button
                    type="submit"
                    className="chat-send-btn"
                    disabled={isSending}
                  >
                    {isSending ? "Sending..." : "Send"}
                  </button>
                </div>
              </form>

              {error && <div className="error-text">{error}</div>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
