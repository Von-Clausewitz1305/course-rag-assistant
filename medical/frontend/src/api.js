const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8080";

async function postJson(path, body = {}) {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.detail || "请求失败");
  }
  return data;
}

export function askQuestion(question, apiKey, model) {
  return postJson("/api/ask", {
    question,
    api_key: apiKey,
    model,
  });
}

export function rebuildIndex() {
  return postJson("/api/rebuild-index");
}
