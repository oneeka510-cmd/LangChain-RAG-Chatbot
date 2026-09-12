const $ = (selector) => document.querySelector(selector);
const conversation = $("#conversation");
const question = $("#question");
const sessionId = localStorage.getItem("rag-session") || crypto.randomUUID();
localStorage.setItem("rag-session", sessionId);

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value;
  return node.innerHTML;
}

function addMessage(role, text, result = null) {
  $("#empty")?.remove();
  const article = document.createElement("article");
  article.className = `message ${role}`;

  if (role === "user") {
    article.textContent = text;
  } else {
    article.innerHTML = `<div class="answer">${escapeHtml(text)}</div>`;
    if (result?.sources.length) {
      article.innerHTML += `<div class="meta">${result.sources.length} sources</div>`;
      const sources = document.createElement("div");
      sources.className = "sources";
      result.sources.forEach((source) => {
        sources.innerHTML += `<details class="source"><summary>[${source.citation}] ${escapeHtml(source.source)} · page ${source.page}</summary><p>${escapeHtml(source.preview)}…</p></details>`;
      });
      article.appendChild(sources);
    }
  }

  conversation.appendChild(article);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
  return article;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Request failed");
  return data;
}

$("#chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text) return;

  addMessage("user", text);
  question.value = "";
  const loading = addMessage("assistant", "Searching your documents…");
  loading.classList.add("loading");
  $("#send").disabled = true;

  try {
    const result = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ question: text, session_id: sessionId }),
    });
    loading.remove();
    addMessage("assistant", result.answer, result);
  } catch (error) {
    loading.querySelector(".answer").textContent = error.message;
    loading.classList.remove("loading");
  } finally {
    $("#send").disabled = false;
    question.focus();
  }
});

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#chatForm").requestSubmit();
  }
});

document.querySelectorAll(".suggestions button").forEach((button) => {
  button.addEventListener("click", () => {
    question.value = button.textContent;
    $("#chatForm").requestSubmit();
  });
});

$("#clear").addEventListener("click", async () => {
  await api("/api/clear", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId }),
  });
  location.reload();
});

