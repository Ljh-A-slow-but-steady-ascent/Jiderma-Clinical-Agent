const API_BASE = import.meta.env.VITE_API_URL || "";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export async function request(path, options = {}, token = "") {
  const headers = new Headers(options.headers || {});
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!response.ok) {
    let detail = "请求失败";
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      detail = (await response.text()) || detail;
    }
    throw new ApiError(detail, response.status);
  }
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : response;
}

export const authApi = {
  login: (username, password) => request("/v1/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  }),
  me: (token) => request("/v1/auth/me", {}, token),
  logout: (token) => request("/v1/auth/logout", { method: "POST" }, token),
};

export async function fetchBlob(path, token) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!response.ok) throw new ApiError("资源读取失败", response.status);
  return response.blob();
}
