import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity, Archive, ArrowUp, Check, ChevronLeft, Copy, Download, Eye,
  FilePenLine, FileText, FileUp, ImagePlus, LoaderCircle, LogOut, Menu, MessageSquare,
  MoreHorizontal, PanelLeftClose, PanelLeftOpen, Pencil, Plus, Search,
  ShieldCheck, Sparkles, Square, Stethoscope, Trash2, X,
} from "lucide-react";
import { ApiError, authApi, fetchBlob, request } from "./api";

const TOKEN_KEY = "jiderma_auth_token";
const LABELS = {
  nevus: "色素痣", melanoma: "黑色素瘤", "basal cell carcinoma": "基底细胞癌",
  "squamous cell carcinoma": "鳞状细胞癌", "actinic keratosis": "日光性角化病",
  "seborrheic keratosis": "脂溢性角化病", dermatofibroma: "皮肤纤维瘤",
};
const LEVELS = {
  RETAKE: "建议重新拍摄", LOCAL_REVIEW: "基层继续评估", REFERRAL: "建议专科转诊",
  PRIORITY_REFERRAL: "建议优先转诊",
};

const uid = () => crypto.randomUUID?.() || `thread-${Date.now()}-${Math.random()}`;
const now = () => new Date().toISOString();
const threadKey = (userId) => `jiderma_threads_v2_${userId}`;

function freshThread() {
  return { id: uid(), title: "新对话", caseId: null, intakeDraft: "", preparedIntake: null, intakeState: "IDLE", createdAt: now(), updatedAt: now(), messages: [] };
}

function loadThreads(userId) {
  try {
    const parsed = JSON.parse(localStorage.getItem(threadKey(userId)) || "[]");
    if (Array.isArray(parsed) && parsed.length) return parsed;
  } catch { /* use a clean workspace */ }
  return [freshThread()];
}

function titleFrom(text) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  return clean ? clean.slice(0, 24) : "图像分析";
}

function recoverIntakeNarrative(messages) {
  const commandOnly = /^(?:开始分析|开始|继续|继续分析|按上文|按照上文|按上文记忆|按照上文记忆|执行|重新提交|再次提交|确认)$/;
  return (messages || [])
    .filter((item) => item.role === "user")
    .map((item) => String(item.content || "").trim())
    .filter((content) => content && !content.startsWith("[已附加") && !commandOnly.test(content.replace(/[\s。，,！!？?]/g, "")))
    .slice(-12)
    .join("\n补充信息：")
    .slice(-6000);
}

function looksLikeClinicalIntakeDocument(content) {
  const text = String(content || "");
  const signals = [
    /年龄\s*[：:]/, /性别\s*[：:]/, /病灶部位\s*[：:]/,
    /病灶尺寸\s*[：:]/, /瘙痒\s*[：:]/, /疼痛\s*[：:]/,
    /出血\s*[：:]/, /近期增大\s*[：:]/, /近期变化\s*[：:]/,
    /皮肤癌病史\s*[：:]/,
  ];
  return signals.filter((pattern) => pattern.test(text)).length >= 3;
}

function formatTime(value) {
  const date = new Date(value);
  const today = new Date();
  if (date.toDateString() === today.toDateString()) return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  return date.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

function Login({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event) {
    event.preventDefault(); setError(""); setLoading(true);
    try { await onLogin(await authApi.login(username.trim(), password)); }
    catch (err) { setError(err.message || "登录失败"); }
    finally { setLoading(false); }
  }

  return <main className="login-page">
    <section className="login-brand">
      <div className="brand-mark large">霁</div>
      <p className="eyebrow">霁肤 · JIDERMA</p>
      <h1>让临床证据<br />更清晰地浮现。</h1>
      <p>皮肤病灶多模态分析、医学证据检索与病例报告，在一个医生工作台中完成。</p>
      <div className="trust-row"><span><ShieldCheck size={16} />本地图像处理</span><span><Stethoscope size={16} />医生辅助决策</span></div>
    </section>
    <section className="login-panel">
      <form className="login-card" onSubmit={submit}>
        <div className="brand-mark">霁</div><h2>登录医生工作台</h2><p>使用已配置的院内账号继续</p>
        <label>账号<input autoFocus autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} /></label>
        <label>密码<input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
        {error && <div className="form-error">{error}</div>}
        <button className="primary-button" disabled={loading || !username || !password}>{loading ? <LoaderCircle className="spin" size={17} /> : null}{loading ? "正在验证" : "进入工作台"}</button>
      </form>
    </section>
  </main>;
}

function Sidebar({ open, threads, activeId, user, onToggle, onNew, onSelect, onRename, onDelete, onWorkspace, onLogout }) {
  const [query, setQuery] = useState("");
  const visible = threads.filter((item) => item.title.toLowerCase().includes(query.toLowerCase()));
  return <aside className={`sidebar ${open ? "open" : "closed"}`}>
    <div className="sidebar-head"><div className="brand-mark">霁</div>{open && <strong>霁肤</strong>}<button className="icon-button push" onClick={onToggle}>{open ? <PanelLeftClose /> : <PanelLeftOpen />}</button></div>
    {open && <>
      <button className="new-chat" onClick={onNew}><Plus size={17} />新对话</button>
      <div className="side-nav">
        <button onClick={() => onWorkspace("cases")}><Archive size={17} />病例库</button>
        <button onClick={() => onWorkspace("reports")}><FileText size={17} />报告库</button>
        <button onClick={() => onWorkspace("documents")}><FilePenLine size={17} />文档库</button>
      </div>
      <div className="side-label">最近对话</div>
      <div className="sidebar-search"><Search size={14} /><input placeholder="搜索对话" value={query} onChange={(e) => setQuery(e.target.value)} /></div>
      <div className="thread-list">{visible.map((thread) => <div key={thread.id} className={`thread-item ${thread.id === activeId ? "active" : ""}`} onClick={() => onSelect(thread.id)}>
        <MessageSquare size={15} /><div className="thread-copy"><strong>{thread.title}</strong><span>{thread.caseId ? `已关联 ${thread.caseId}` : formatTime(thread.updatedAt)}</span></div>
        <div className="thread-actions"><button title="重命名" onClick={(e) => { e.stopPropagation(); onRename(thread.id); }}><Pencil size={13} /></button><button title="删除" onClick={(e) => { e.stopPropagation(); onDelete(thread.id); }}><Trash2 size={13} /></button></div>
      </div>)}</div>
      <div className="profile"><div className="profile-avatar">{(user.full_name || user.username || "医")[0]}</div><div><strong>{user.full_name || user.username}</strong><span>临床医生</span></div><button title="退出登录" onClick={onLogout}><LogOut size={16} /></button></div>
    </>}
  </aside>;
}

function SecureMarkdownImage({ src, alt, token, ...props }) {
  const [url, setUrl] = useState("");
  const isProtected = /^\/v1\/cases\/[^/]+\/image$/.test(String(src || ""));
  useEffect(() => {
    let active = true; let objectUrl = "";
    if (!isProtected || !token) return undefined;
    fetchBlob(src, token).then((blob) => { if (!active) return; objectUrl = URL.createObjectURL(blob); setUrl(objectUrl); }).catch(() => setUrl(""));
    return () => { active = false; if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [src, token, isProtected]);
  if (isProtected && !url) return <span className="secure-image-loading">正在安全加载病灶图片…</span>;
  return <img src={isProtected ? url : src} alt={alt || "文档图片"} {...props} />;
}

function Markdown({ children, token }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
    a: ({ ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
    table: ({ children }) => <div className="table-scroll"><table>{children}</table></div>,
    img: ({ ...props }) => <SecureMarkdownImage {...props} token={token} />,
  }}>{children}</ReactMarkdown>;
}

function ExecutionCard({ run, onStop }) {
  const events = run?.events || [];
  return <div className="execution-card">
    <div className="execution-title">{run?.running ? <LoaderCircle className="spin" size={16} /> : <Check size={16} />}<strong>{run?.running ? "Agent 正在执行" : "执行完成"}</strong>{run?.running && <button onClick={onStop}><Square size={12} />停止等待</button>}</div>
    <div className="event-list">{events.length ? events.slice(-8).map((event, index) => <div key={`${event.sequence}-${index}`} className={index === events.slice(-8).length - 1 && run.running ? "current" : "done"}><i></i><span>{event.message}</span></div>) : <div className="current"><i></i><span>正在理解任务并选择工具…</span></div>}</div>
  </div>;
}

function ApprovalCard({ action, onConfirm, onReject, busy }) {
  return <div className={`approval-card ${action.destructive ? "destructive" : ""}`}><div className="approval-icon"><ShieldCheck /></div><div className="approval-body"><strong>{action.title}</strong><p>{action.summary}</p><details><summary>查看操作预览</summary><pre>{JSON.stringify(action.preview || {}, null, 2)}</pre></details><div className="approval-actions"><button onClick={onReject} disabled={busy}>取消</button><button className="approve" onClick={onConfirm} disabled={busy}>{busy && <LoaderCircle className="spin" size={14} />}{action.destructive ? "确认删除" : "确认执行"}</button></div></div></div>;
}

function ArtifactCard({ item, token, onOpen }) {
  async function download() {
    const blob = await fetchBlob(`/v1/agent/documents/${encodeURIComponent(item.document_id)}/download`, token);
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = item.filename || "agent-document.md"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 500);
  }
  return <div className="artifact-card"><div className="artifact-icon"><FilePenLine size={19} /></div><div><strong>{item.title}</strong><span>{item.filename} · 第 {item.revision || 1} 版</span></div><button onClick={() => onOpen({ type: "document", data: item })}>编辑</button><button onClick={download}><Download size={14} />下载</button></div>;
}

function Message({ message, token, onOpen, onApprove, approving }) {
  const copy = () => navigator.clipboard.writeText(message.content || "");
  return <article className={`message ${message.role}`}>
    {message.role === "assistant" && <div className="assistant-avatar">霁</div>}
    <div className="message-column"><div className="message-content"><Markdown token={token}>{message.content}</Markdown></div>
      {message.run && <ExecutionCard run={{ ...message.run, running: false }} />}
      {message.pendingAction && <ApprovalCard action={message.pendingAction} busy={approving} onConfirm={() => onApprove(message, true)} onReject={() => onApprove(message, false)} />}
      {message.artifacts?.map((item) => <ArtifactCard key={item.document_id} item={item} token={token} onOpen={onOpen} />)}
      {message.caseId && <button className="inline-workspace-link" onClick={() => onOpen({ type: "case", id: message.caseId })}><Stethoscope size={15} />打开病例工作台<span>{message.caseId}</span></button>}
      {message.reportId && <button className="inline-workspace-link" onClick={() => onOpen({ type: "report", id: message.reportId })}><FilePenLine size={15} />打开可编辑报告<span>{message.reportId}</span></button>}
      <div className="message-meta"><span>{formatTime(message.createdAt)}</span>{message.role === "assistant" && <button onClick={copy}><Copy size={12} />复制</button>}</div>
    </div>
  </article>;
}

function EmptyChat({ onPrompt }) {
  return <div className="empty-chat"><div className="empty-orb"><Sparkles /></div><h1>随时可以开始。</h1><p>询问医学证据，或附加病灶图片与患者描述建立分析任务。</p><div className="prompt-grid">
    {["我需要准备哪些病例信息？", "检索色素痣与黑色素瘤的鉴别要点", "创建新病例并执行联合分析"].map((text) => <button key={text} onClick={() => onPrompt(text)}>{text}</button>)}
  </div></div>;
}

const INTAKE_STATUS = {
  COLLECTING: "已记住现有资料，请继续补充缺失项",
  READY: "病例资料已完整，可直接开始分析",
  WAITING_CONFIRM: "分析任务已准备，等待医生确认",
};

function Composer({ value, onChange, onSend, onDocument, attachment, setAttachment, intakeState, busy, models, model, setModel }) {
  const fileRef = useRef(null);
  const documentRef = useRef(null);
  const previewUrl = useMemo(() => attachment ? URL.createObjectURL(attachment) : "", [attachment]);
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);
  function keyDown(event) { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); onSend(); } }
  return <div className="composer-zone">{attachment && <div className="attachment-preview"><img src={previewUrl} alt="待分析病灶" /><div><strong>{attachment.name}</strong><span>{Math.round(attachment.size / 1024)} KB · {INTAKE_STATUS[intakeState] || "将结合当前描述分析"}</span></div><button onClick={() => setAttachment(null)}><X size={15} /></button></div>}
    <div className="composer"><textarea rows={1} placeholder="输入问题；附图时请直接描述患者身份、症状和病程" value={value} onChange={(e) => onChange(e.target.value)} onKeyDown={keyDown} /><div className="composer-tools"><input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(e) => { setAttachment(e.target.files?.[0] || null); e.target.value = ""; }} /><button className="tool-button" title="附加病灶图片" onClick={() => fileRef.current?.click()}><ImagePlus size={18} /></button><input ref={documentRef} type="file" accept=".md,.txt,.json,.csv" hidden onChange={(e) => { const file = e.target.files?.[0]; if (file) onDocument(file); e.target.value = ""; }} /><button className="tool-button" title="上传可编辑文档" onClick={() => documentRef.current?.click()}><FileUp size={18} /></button><select value={model} onChange={(e) => setModel(e.target.value)}>{models.map((item) => <option key={item.id} value={item.id} disabled={!item.available}>{item.label}{item.available ? "" : " · 不可用"}</option>)}</select><button className={`send-button ${busy ? "stop" : ""}`} onClick={onSend} disabled={!busy && !value.trim() && !attachment}>{busy ? <Square size={14} /> : <ArrowUp size={18} />}</button></div></div>
    <p className="disclaimer">人工智能仅用于辅助筛查和分级转诊，不构成医学诊断</p>
  </div>;
}

function CaseView({ id, token, onDeleted }) {
  const [data, setData] = useState(null); const [error, setError] = useState("");
  const [editing, setEditing] = useState(false); const [form, setForm] = useState(null); const [saving, setSaving] = useState(false);
  useEffect(() => { let url = ""; Promise.all([
    request(`/v1/cases/${id}/patient`, {}, token), request(`/v1/cases/${id}/visual`, {}, token),
    request(`/v1/cases/${id}/triage`, {}, token), request(`/v1/cases/${id}/care-pathway`, {}, token),
    fetchBlob(`/v1/cases/${id}/image`, token),
  ]).then(([patient, visual, triage, care, blob]) => { url = URL.createObjectURL(blob); setData({ patient, visual, triage, care, url }); }).catch((err) => setError(err.message)).finally(() => {}); return () => { if (url) URL.revokeObjectURL(url); }; }, [id, token]);
  if (error) return <div className="drawer-state error">{error}</div>;
  if (!data) return <div className="drawer-state"><LoaderCircle className="spin" />正在读取病例…</div>;
  async function saveCase() {
    setSaving(true); setError("");
    try { const patient = await request(`/v1/cases/${id}`, { method: "PUT", body: JSON.stringify(form) }, token); setData((old) => ({ ...old, patient })); setEditing(false); }
    catch (err) { setError(err.message); } finally { setSaving(false); }
  }
  async function removeCase() {
    const confirmation = window.prompt(`删除病例会同时删除其图片、分析和报告。\n请输入 ${id} 确认：`);
    if (confirmation !== id) return;
    try { await request(`/v1/cases/${id}`, { method: "DELETE", body: JSON.stringify({ confirm_case_id: id }) }, token); onDeleted?.(); }
    catch (err) { setError(err.message); }
  }
  function startEdit() { setForm({ name: data.patient.name, sex: data.patient.sex, age: data.patient.age, outpatient_no: data.patient.outpatient_no, phone: data.patient.phone || "", medical_history: data.patient.medical_history || "", allergy_history: data.patient.allergy_history || "", lesion_site: data.patient.lesion_site, chief_complaint: data.patient.chief_complaint, clinical_profile: data.patient.clinical_profile }); setEditing(true); }
  if (editing) return <div className="case-edit"><div className="editor-meta"><div><p className="eyebrow">编辑病例源记录</p><h2>{id}</h2><span>修改患者和结构化病例信息</span></div><div><button onClick={() => setEditing(false)}>取消</button><button className="primary" disabled={saving} onClick={saveCase}>{saving ? <LoaderCircle className="spin" size={14} /> : <Check size={14} />}保存</button></div></div><div className="case-form">
    <label>患者姓名<input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></label><label>门诊号<input value={form.outpatient_no} onChange={(e) => setForm({ ...form, outpatient_no: e.target.value })} /></label><label>性别<select value={form.sex} onChange={(e) => setForm({ ...form, sex: e.target.value, clinical_profile: { ...form.clinical_profile, gender: e.target.value === "男" ? "MALE" : "FEMALE" } })}><option>男</option><option>女</option></select></label><label>年龄<input type="number" value={form.age} onChange={(e) => setForm({ ...form, age: Number(e.target.value), clinical_profile: { ...form.clinical_profile, age: Number(e.target.value) } })} /></label><label>电话<input value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} /></label><label>病灶部位<input value={form.lesion_site} onChange={(e) => setForm({ ...form, lesion_site: e.target.value })} /></label><label className="wide">主诉<textarea value={form.chief_complaint} onChange={(e) => setForm({ ...form, chief_complaint: e.target.value })} /></label><fieldset className="wide"><legend>结构化症状</legend>{[["itch","瘙痒"],["hurt","疼痛"],["bleed","出血"],["grew","近期增大"],["changed","近期变化"],["skin_cancer_history","皮肤癌病史"],["cancer_history","其他癌症病史"]].map(([key,label]) => <label className="check" key={key}><input type="checkbox" checked={Boolean(form.clinical_profile?.[key])} onChange={(e) => setForm({ ...form, clinical_profile: { ...form.clinical_profile, [key]: e.target.checked } })} />{label}</label>)}</fieldset><label>长径（mm）<input type="number" step="0.1" value={form.clinical_profile?.diameter_1 || ""} onChange={(e) => setForm({ ...form, clinical_profile: { ...form.clinical_profile, diameter_1: Number(e.target.value) } })} /></label><label>短径（mm）<input type="number" step="0.1" value={form.clinical_profile?.diameter_2 || ""} onChange={(e) => setForm({ ...form, clinical_profile: { ...form.clinical_profile, diameter_2: Number(e.target.value) } })} /></label>
  </div>{error && <div className="form-error">{error}</div>}</div>;
  return <div className="case-view"><div className="case-visual"><img src={data.url} alt="病灶" /><span>原始图像仅在当前工作台受权访问</span><div className="case-actions"><button onClick={startEdit}><Pencil size={14} />编辑病例</button><button className="danger" onClick={removeCase}><Trash2 size={14} />删除病例</button></div>{error && <div className="form-error">{error}</div>}</div><div className="case-details">
    <div className="detail-card identity"><p className="eyebrow">患者与就诊信息</p><h2>{data.patient.name}</h2><div className="identity-grid"><span>门诊号<strong>{data.patient.outpatient_no}</strong></span><span>性别 / 年龄<strong>{data.patient.sex} · {data.patient.age} 岁</strong></span><span>病灶部位<strong>{data.patient.lesion_site}</strong></span><span>病例编号<strong>{id}</strong></span></div><p>{data.patient.chief_complaint}</p></div>
    <div className="detail-card"><p className="eyebrow">联合分析候选</p><div className="prediction-list">{data.visual.top_predictions?.map((item, index) => <div key={item.label}><span>{index + 1}</span><strong>{LABELS[item.label?.toLowerCase()] || item.label}</strong><i><b style={{ width: `${Math.max(2, item.score * 100)}%` }}></b></i><em>{(item.score * 100).toFixed(1)}%</em></div>)}</div></div>
    <div className="detail-card risk-card"><p className="eyebrow">风险与照护路径</p><h3>{LEVELS[data.triage.level] || data.triage.level}</h3><p>{data.care.action}</p><ul>{data.care.recommended_next_steps?.map((item) => <li key={item}>{item}</li>)}</ul></div>
  </div></div>;
}

function Library({ type, token, onOpen }) {
  const [items, setItems] = useState(null); const [query, setQuery] = useState(""); const [folders, setFolders] = useState([]); const [folder, setFolder] = useState(""); const [version, setVersion] = useState(0); const uploadRef = useRef(null);
  const [notice, setNotice] = useState("");
  const [filters, setFilters] = useState({ sex: "", triage_level: "", diagnosis: "", min_age: "", max_age: "", lesion_site: "" });
  const resourceType = type === "cases" ? "case" : type === "reports" ? "report" : "document";
  useEffect(() => { request("/v1/workspace/folders", {}, token).then(setFolders).catch(() => setFolders([])); }, [token, version]);
  useEffect(() => {
    const params = new URLSearchParams(); if (folder) params.set("folder_id", folder);
    if (type === "cases") Object.entries(filters).forEach(([key,value]) => { if (value !== "") params.set(key,value); });
    const endpoint = type === "cases" ? "/v1/cases" : type === "reports" ? "/v1/reports" : "/v1/agent/documents";
    request(`${endpoint}?${params}`, {}, token).then(setItems).catch(() => setItems([]));
  }, [type, token, folder, filters, version]);
  const filtered = (items || []).filter((item) => JSON.stringify(item).toLowerCase().includes(query.toLowerCase()));
  const folderNames = Object.fromEntries(folders.map((item) => [item.folder_id, item.name]));
  async function createFolder() { const name = window.prompt("文件夹名称"); if (!name?.trim()) return; await request("/v1/workspace/folders", { method: "POST", body: JSON.stringify({ name: name.trim(), parent_id: null }) }, token); setVersion((n) => n + 1); }
  async function deleteFolder(item) { if (!window.confirm(`删除文件夹“${item.name}”？其中的病例、报告和文档会保留并回到“全部内容”。`)) return; await request(`/v1/workspace/folders/${item.folder_id}`, { method: "DELETE" }, token); if (folder === item.folder_id) setFolder(""); setVersion((n) => n + 1); }
  async function move(item, folderId) { if (folderId === "__choose") return; const id = item.case_id || item.report_id || item.document_id; try { await request("/v1/workspace/items", { method: "PUT", body: JSON.stringify({ resource_type: resourceType, resource_id: id, folder_id: folderId === "__root" ? null : folderId }) }, token); setNotice(folderId === "__root" ? "已移回根目录" : `已移入文件夹“${folderNames[folderId]}”`); setVersion((n) => n + 1); } catch (err) { setNotice(`移动失败：${err.message}`); } }
  async function remove(item, event) { event.stopPropagation(); const id = item.case_id || item.report_id || item.document_id; if (type === "cases") { const confirmation = window.prompt(`删除病例会同时删除其图片、分析和报告。\n请输入 ${id} 确认：`); if (confirmation !== id) return; } else if (!window.confirm(`确定删除“${item.title || item.filename || id}”？此操作不可撤销。`)) return; const endpoint = type === "cases" ? `/v1/cases/${id}` : type === "reports" ? `/v1/reports/${id}` : `/v1/agent/documents/${id}`; const options = type === "cases" ? { method: "DELETE", body: JSON.stringify({ confirm_case_id: id }) } : { method: "DELETE" }; try { await request(endpoint, options, token); setNotice("删除完成"); setVersion((n) => n + 1); } catch (err) { setNotice(`删除失败：${err.message}`); } }
  async function upload(event) { const file = event.target.files?.[0]; if (!file) return; const form = new FormData(); form.append("file", file); form.append("conversation_id", "workspace"); form.append("folder_id", folder); await request("/v1/agent/documents/upload", { method: "POST", body: form }, token); event.target.value = ""; setVersion((n) => n + 1); }
  const descriptions = { cases: "病例是包含患者、图像、结构化症状、模型结果与证据的源记录。", reports: "报告是从病例生成的可编辑、可版本化临床文档。", documents: "文档是医生上传或 Agent 创建的工作资料，可关联病例并在线修改。" };
  return <div className="library-layout"><aside className="folder-panel"><button className={!folder ? "active" : ""} onClick={() => setFolder("")}><Archive size={15} />全部内容</button>{folders.map((item) => <div className={`folder-entry ${folder === item.folder_id ? "active" : ""}`} key={item.folder_id}><button onClick={() => setFolder(item.folder_id)}><span>📁</span>{item.name}</button><button title="删除文件夹" onClick={() => deleteFolder(item)}><X size={12} /></button></div>)}<button className="new-folder" onClick={createFolder}><Plus size={14} />新建文件夹</button></aside><div className="library"><p className="library-description">{descriptions[type]}</p>{notice && <div className="library-notice"><Check size={14} />{notice}</div>}<div className="library-toolbar"><div className="library-search"><Search size={16} /><input placeholder={`搜索${type === "cases" ? "病例" : type === "reports" ? "报告" : "文档"}`} value={query} onChange={(e) => setQuery(e.target.value)} /></div>{type === "documents" && <><input hidden ref={uploadRef} type="file" accept=".md,.txt,.json,.csv" onChange={upload} /><button className="upload-document" onClick={() => uploadRef.current?.click()}><Plus size={15} />上传文档</button></>}</div>{type === "cases" && <div className="structured-filters"><select value={filters.sex} onChange={(e) => setFilters({ ...filters, sex: e.target.value })}><option value="">全部性别</option><option>男</option><option>女</option></select><input placeholder="最小年龄" type="number" value={filters.min_age} onChange={(e) => setFilters({ ...filters, min_age: e.target.value })} /><input placeholder="最大年龄" type="number" value={filters.max_age} onChange={(e) => setFilters({ ...filters, max_age: e.target.value })} /><input placeholder="病灶部位" value={filters.lesion_site} onChange={(e) => setFilters({ ...filters, lesion_site: e.target.value })} /><input placeholder="Top-1 候选" value={filters.diagnosis} onChange={(e) => setFilters({ ...filters, diagnosis: e.target.value })} /><select value={filters.triage_level} onChange={(e) => setFilters({ ...filters, triage_level: e.target.value })}><option value="">全部风险</option><option value="LOCAL_REVIEW">基层评估</option><option value="REFERRAL">建议转诊</option><option value="PRIORITY_REFERRAL">优先转诊</option></select></div>}{items === null ? <div className="drawer-state"><LoaderCircle className="spin" />正在读取…</div> : <div className="library-list">{filtered.map((item) => { const id = item.case_id || item.report_id || item.document_id; const location = item.folder_id ? folderNames[item.folder_id] : "未归档"; return <div className="library-row" key={id} onClick={() => onOpen({ type: type === "cases" ? "case" : type === "reports" ? "report" : "document", id, data: item })}><span className="library-icon">{type === "cases" ? <Stethoscope /> : type === "reports" ? <FileText /> : <FilePenLine />}</span><span><strong>{item.patient_name || item.title || item.filename || item.case_id}</strong><small>{type === "cases" ? `${item.outpatient_no || "未关联门诊号"} · ${item.lesion_site || "未填写部位"} · ${LABELS[item.top_diagnosis] || item.top_diagnosis || "未分析"}` : type === "reports" ? `${item.outpatient_no} · ${item.status}` : `${item.filename} · 第 ${item.revision} 版`}<b className="folder-badge">📁 {location}</b></small></span><select value={item.folder_id || "__root"} onClick={(e) => e.stopPropagation()} onChange={(e) => move(item,e.target.value)}><option value="__root">未归档</option>{folders.map((f) => <option value={f.folder_id} key={f.folder_id}>{f.name}</option>)}</select><em>{formatTime(item.updated_at || item.created_at)}</em><button className="row-delete" title="删除" onClick={(event) => remove(item, event)}><Trash2 size={14} /></button></div>; })}</div>}</div></div>;
}

function ReportEditor({ id, token, onDeleted }) {
  const [report, setReport] = useState(null); const [content, setContent] = useState(""); const [state, setState] = useState("");
  const [viewMode, setViewMode] = useState("preview");
  useEffect(() => { request(`/v1/reports/${id}`, {}, token).then((item) => { setReport(item); setContent(item.content); }).catch((err) => setState(err.message)); }, [id, token]);
  async function save() { setState("正在保存…"); try { const item = await request(`/v1/reports/${id}`, { method: "PUT", body: JSON.stringify({ content, status: report.status || "draft" }) }, token); setReport(item); setState("已保存新版本"); } catch (err) { setState(err.message); } }
  async function download() { const blob = await fetchBlob(`/v1/reports/${id}/download`, token); const url = URL.createObjectURL(blob), a = document.createElement("a"); a.href = url; a.download = report?.filename || `${id}.md`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 500); }
  async function remove() { if (!window.confirm(`确定删除报告“${report.filename}”？病例和图片将保留。`)) return; try { await request(`/v1/reports/${id}`, { method: "DELETE" }, token); onDeleted?.(); } catch (err) { setState(err.message); } }
  if (!report) return <div className="drawer-state">{state || <><LoaderCircle className="spin" />正在读取报告…</>}</div>;
  return <div className="editor-view"><div className="editor-meta"><div><p className="eyebrow">在线报告</p><h2>{report.filename}</h2><span>{report.patient_name} · {report.outpatient_no} · {report.status}</span></div><div><span className="editor-tabs"><button className={viewMode === "edit" ? "active" : ""} onClick={() => setViewMode("edit")}><Pencil size={14} />编辑</button><button className={viewMode === "preview" ? "active" : ""} onClick={() => setViewMode("preview")}><Eye size={14} />预览</button></span><button onClick={download}><Download size={15} />下载</button><button className="danger-button" onClick={remove}><Trash2 size={14} />删除</button><button className="primary" onClick={save}><Check size={15} />保存修改</button></div></div>{viewMode === "edit" ? <textarea value={content} onChange={(e) => setContent(e.target.value)} /> : <div className="markdown-preview"><Markdown token={token}>{content}</Markdown></div>}<div className="save-state">{state}</div></div>;
}

function DocumentEditor({ item, token, onUpdated, onDeleted }) {
  const [title, setTitle] = useState(item.title); const [content, setContent] = useState(item.content); const [state, setState] = useState("");
  const [viewMode, setViewMode] = useState("preview");
  async function save() { setState("正在保存…"); try { const saved = await request(`/v1/agent/documents/${item.document_id}`, { method: "PUT", body: JSON.stringify({ title, content }) }, token); setState(`已保存第 ${saved.revision} 版`); onUpdated?.(saved); } catch (err) { setState(err.message); } }
  async function remove() { if (!window.confirm(`确定删除文档“${item.filename}”？关联病例和报告将保留。`)) return; try { await request(`/v1/agent/documents/${item.document_id}`, { method: "DELETE" }, token); onDeleted?.(); } catch (err) { setState(err.message); } }
  return <div className="editor-view"><div className="editor-meta"><div><p className="eyebrow">AGENT 文档</p><input className="editor-title" value={title} onChange={(e) => setTitle(e.target.value)} /><span>{item.filename} · 第 {item.revision || 1} 版</span></div><div><span className="editor-tabs"><button className={viewMode === "edit" ? "active" : ""} onClick={() => setViewMode("edit")}><Pencil size={14} />编辑</button><button className={viewMode === "preview" ? "active" : ""} onClick={() => setViewMode("preview")}><Eye size={14} />预览</button></span><button className="danger-button" onClick={remove}><Trash2 size={14} />删除</button><button className="primary" onClick={save}><Check size={15} />保存新版本</button></div></div>{viewMode === "edit" ? <textarea value={content} onChange={(e) => setContent(e.target.value)} /> : <div className="markdown-preview"><Markdown token={token}>{content}</Markdown></div>}<div className="save-state">{state}</div></div>;
}

function WorkspaceDrawer({ view, token, onClose, onOpen }) {
  if (!view) return null;
  const titles = { cases: "病例库", reports: "报告库", documents: "文档库", case: "病例工作台", report: "报告编辑器", document: "Agent 文档" };
  return <div className="workspace-drawer"><header><button onClick={onClose}><ChevronLeft size={18} /></button><strong>{titles[view.type]}</strong><span>{view.id || view.data?.document_id || ""}</span><button className="drawer-x" onClick={onClose}><X size={18} /></button></header><main>
    {view.type === "case" && <CaseView id={view.id} token={token} onDeleted={onClose} />}
    {(view.type === "cases" || view.type === "reports" || view.type === "documents") && <Library type={view.type} token={token} onOpen={onOpen} />}
    {view.type === "report" && <ReportEditor id={view.id} token={token} onDeleted={onClose} />}
    {view.type === "document" && <DocumentEditor item={view.data} token={token} onDeleted={onClose} />}
  </main></div>;
}

function answerFrom(payload) {
  const result = payload.result || {};
  let text = payload.message || "任务已完成。";
  if (result.kind === "chat" && result.chat?.response) text = result.chat.response;
  if (result.kind === "analysis" && result.analysis) {
    const analysis = result.analysis;
    const candidates = (analysis.visual?.top_predictions || []).slice(0, 3).map((item, index) => `${index + 1}. **${LABELS[item.label?.toLowerCase()] || item.label}** · ${(item.score * 100).toFixed(1)}%`).join("\n");
    const evidence = (analysis.guidelines || []).slice(0, 4).map((item) => `- ${item.title}`).join("\n");
    text += `\n\n### 联合分析结果\n${candidates || "暂无候选"}\n\n**风险分级**：${LEVELS[analysis.triage?.level] || analysis.triage?.level || "待复核"}\n\n**医学证据**\n${evidence || "当前未检索到相关证据"}`;
  }
  return text;
}

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY) || "");
  const [user, setUser] = useState(null); const [checking, setChecking] = useState(Boolean(token));
  const [threads, setThreads] = useState([]); const [activeId, setActiveId] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true); const [view, setView] = useState(null);
  const [input, setInput] = useState(""); const [attachment, setAttachment] = useState(null);
  const [models, setModels] = useState([]); const [model, setModel] = useState("auto");
  const [liveRun, setLiveRun] = useState(null); const [approving, setApproving] = useState(false);
  const abortRef = useRef(null); const bottomRef = useRef(null);
  const active = threads.find((item) => item.id === activeId) || threads[0];

  useEffect(() => { if (!token) { setChecking(false); return; } authApi.me(token).then(setUser).catch(() => { localStorage.removeItem(TOKEN_KEY); setToken(""); }).finally(() => setChecking(false)); }, [token]);
  useEffect(() => { if (!user) return; const loaded = loadThreads(user.id); setThreads(loaded); setActiveId(loaded[0].id); request("/v1/chat/models", {}, token).then((items) => { setModels(items); const first = items.find((item) => item.available); if (first) setModel(first.id); }).catch(() => setModels([{ id: "auto", label: "自动选择", available: true }])); }, [user, token]);
  useEffect(() => { if (user && threads.length) localStorage.setItem(threadKey(user.id), JSON.stringify(threads)); }, [threads, user]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [active?.messages, liveRun]);

  async function handleLogin(payload) { localStorage.setItem(TOKEN_KEY, payload.access_token); setToken(payload.access_token); setUser(payload.user); }
  async function logout() { try { await authApi.logout(token); } catch { /* local logout still succeeds */ } localStorage.removeItem(TOKEN_KEY); setToken(""); setUser(null); setThreads([]); }
  function updateActive(updater) { setThreads((items) => items.map((item) => item.id === active.id ? updater(item) : item)); }
  function addMessage(message) { updateActive((item) => ({ ...item, title: item.messages.length === 0 && message.role === "user" ? titleFrom(message.content) : item.title, updatedAt: now(), messages: [...item.messages, { id: uid(), createdAt: now(), ...message }] })); }
  function newChat() { const item = freshThread(); setThreads((items) => [item, ...items]); setActiveId(item.id); setView(null); setAttachment(null); setInput(""); }
  function renameThread(id) { const item = threads.find((thread) => thread.id === id); const title = window.prompt("输入新的对话名称", item?.title || ""); if (title?.trim()) setThreads((items) => items.map((thread) => thread.id === id ? { ...thread, title: title.trim(), updatedAt: now() } : thread)); }
  function deleteThread(id) { if (!window.confirm("删除该对话？病例和报告记录不会被删除。")) return; const next = threads.filter((item) => item.id !== id); if (!next.length) next.push(freshThread()); setThreads(next); if (activeId === id) setActiveId(next[0].id); }

  async function pollRun(runId, signal) {
    for (let index = 0; index < 600; index += 1) {
      if (signal.aborted) throw new DOMException("已停止", "AbortError");
      const payload = await request(`/v1/agent/runs/${runId}`, { signal }, token);
      setLiveRun({ ...payload, running: payload.status === "RUNNING" });
      if (payload.status !== "RUNNING") return payload;
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    throw new Error("任务执行超时");
  }

  async function uploadDocument(file) {
    if (!active || !file) return;
    const form = new FormData();
    form.append("file", file);
    form.append("conversation_id", active.id);
    form.append("case_id", active.caseId || "");
    try {
      const document = await request("/v1/agent/documents/upload", { method: "POST", body: form }, token);
      const isClinicalIntake = looksLikeClinicalIntakeDocument(document.content);
      if (isClinicalIntake && !active.caseId) {
        updateActive((item) => {
          const block = `[病例资料文件：${file.name}]\n${document.content}`;
          const intakeDraft = [item.intakeDraft, block].filter(Boolean).join("\n\n").slice(-6000);
          return { ...item, intakeDraft, preparedIntake: null, intakeState: "COLLECTING", updatedAt: now() };
        });
      }
      addMessage({ role: "user", content: `[已上传文档] ${file.name}` });
      addMessage({
        role: "assistant",
        content: isClinicalIntake && !active.caseId
          ? "已读取并记住文档中的病例结构化信息。附加病灶图片后可以直接输入“开始分析”；系统只会继续询问文档里真正缺失的字段。"
          : "文档已存入当前对话，可以直接打开编辑或下载。",
        artifacts: [document],
      });
    } catch (err) {
      addMessage({ role: "assistant", content: `文档上传失败：${err.message || "网络异常"}` });
    }
  }

  async function send() {
    if (liveRun?.running) { abortRef.current?.abort(); return; }
    const text = input.trim(); if ((!text && !attachment) || !active) return;
    const file = attachment; const history = active.messages.slice(-12).map(({ role, content }) => ({ role, message: content }));
    setInput(""); addMessage({ role: "user", content: text || "[已附加病灶图片和病例描述]" });
    const controller = new AbortController(); abortRef.current = controller; setLiveRun({ running: true, events: [] });
    try {
      let intake = null;
      if (file) {
        const command = text.replace(/[\s。，,！!？?]/g, "");
        const resumesPreparedCase = Boolean(active.preparedIntake)
          && /^(?:开始分析|开始|继续|继续分析|按上文|按照上文|按上文记忆|按照上文记忆|执行|重新提交|再次提交|确认)$/.test(command);
        if (resumesPreparedCase) {
          intake = active.preparedIntake;
          updateActive((item) => ({ ...item, intakeState: "READY", updatedAt: now() }));
        } else {
          const rememberedNarrative = active.intakeDraft || recoverIntakeNarrative(active.messages);
          const accumulatedNarrative = [rememberedNarrative, text].filter(Boolean).join("\n补充信息：");
          if (accumulatedNarrative.length < 10) throw new Error("附图时请同时描述患者姓名、门诊号、性别、年龄、病灶部位与症状变化。");
          const extracted = await request("/v1/agent/intake/extract", { method: "POST", body: JSON.stringify({ narrative: accumulatedNarrative, model }) }, token);
          if (!extracted.complete) {
            updateActive((item) => ({ ...item, intakeDraft: accumulatedNarrative, preparedIntake: null, intakeState: "COLLECTING", updatedAt: now() }));
            addMessage({ role: "assistant", content: `我已合并保留前面的病例信息，不需要重复填写。请只补充：**${(extracted.missing_field_labels || extracted.missing_fields).join("、")}**` });
            setLiveRun(null); return;
          }
          intake = extracted.intake;
          updateActive((item) => ({ ...item, intakeDraft: accumulatedNarrative, preparedIntake: intake, intakeState: "READY", updatedAt: now() }));
        }
      }
      const form = new FormData(); form.append("message", text || "创建病例并执行分析"); form.append("case_id", active.caseId || ""); form.append("model", model); form.append("detail_level", "detailed"); form.append("conversation_id", active.id); form.append("history_json", JSON.stringify(history));
      if (file) { form.append("image", file); form.append("intake_json", JSON.stringify(intake)); }
      const initial = await request("/v1/agent/runs/dispatch", { method: "POST", body: form, signal: controller.signal }, token);
      const payload = await pollRun(initial.run_id, controller.signal);
      const artifacts = payload.result?.kind === "chat" ? payload.result.chat?.artifacts || [] : [];
      const reportId = payload.result?.kind === "report" ? payload.result.report?.report_id : null;
      addMessage({ role: "assistant", content: answerFrom(payload), run: payload, pendingAction: payload.pending_action, runId: payload.run_id, caseId: payload.case_id, reportId, artifacts });
      if (payload.case_id) updateActive((item) => ({ ...item, caseId: payload.case_id }));
      if (payload.status === "WAITING_CONFIRM") updateActive((item) => ({ ...item, intakeState: "WAITING_CONFIRM" }));
      if (file && payload.status !== "WAITING_CONFIRM") {
        setAttachment(null);
        updateActive((item) => ({ ...item, intakeDraft: "", preparedIntake: null, intakeState: payload.case_id ? "CREATED" : "IDLE" }));
      }
    } catch (err) {
      if (err.name !== "AbortError") addMessage({ role: "assistant", content: `操作未完成：${err.message || "网络异常"}` });
    } finally { setLiveRun(null); abortRef.current = null; }
  }

  async function approve(message, confirmed) {
    setApproving(true);
    try {
      const payload = await request(`/v1/agent/runs/${message.runId}/${confirmed ? "confirm" : "reject"}`, { method: "POST", body: confirmed ? JSON.stringify({ action_id: message.pendingAction.action_id }) : undefined }, token);
      updateActive((item) => ({ ...item, caseId: payload.case_id || item.caseId, messages: item.messages.map((entry) => entry.id === message.id ? { ...entry, pendingAction: null } : entry) }));
      addMessage({ role: "assistant", content: confirmed ? answerFrom(payload) : "已取消该操作，未写入临床数据。", run: payload, caseId: payload.case_id, reportId: payload.result?.report?.report_id });
      if (confirmed && message.pendingAction.kind === "CREATE_ANALYZE_CASE") {
        setAttachment(null);
        updateActive((item) => ({ ...item, intakeDraft: "", preparedIntake: null, intakeState: "CREATED" }));
      } else if (!confirmed && message.pendingAction.kind === "CREATE_ANALYZE_CASE") {
        updateActive((item) => ({ ...item, intakeState: "READY" }));
      }
    } catch (err) { addMessage({ role: "assistant", content: `确认操作失败：${err.message}` }); }
    finally { setApproving(false); }
  }

  if (checking) return <div className="app-loading"><div className="brand-mark large">霁</div><LoaderCircle className="spin" /></div>;
  if (!token || !user) return <Login onLogin={handleLogin} />;
  return <div className="app-shell">
    <Sidebar open={sidebarOpen} threads={threads} activeId={active?.id} user={user} onToggle={() => setSidebarOpen((value) => !value)} onNew={newChat} onSelect={(id) => { setActiveId(id); setView(null); }} onRename={renameThread} onDelete={deleteThread} onWorkspace={(type) => setView({ type })} onLogout={logout} />
    <section className="chat-stage">
      <header className="topbar"><button className="mobile-menu" onClick={() => setSidebarOpen((value) => !value)}><Menu /></button><div><strong>{active?.title || "新对话"}</strong><span>{active?.caseId ? `关联病例 ${active.caseId}` : "Agent 工作区"}</span></div><button className="top-workspace" onClick={() => setView({ type: active?.caseId ? "case" : "cases", id: active?.caseId })}><Activity size={16} />{active?.caseId ? "病例工作台" : "查看病例"}</button></header>
      <div className="chat-scroll"><div className="message-stream">{active?.messages?.length ? active.messages.map((message) => <Message key={message.id} message={message} token={token} onOpen={setView} onApprove={approve} approving={approving} />) : <EmptyChat onPrompt={setInput} />}{liveRun && <ExecutionCard run={liveRun} onStop={() => abortRef.current?.abort()} />}<div ref={bottomRef} /></div></div>
      <Composer value={input} onChange={setInput} onSend={send} onDocument={uploadDocument} attachment={attachment} setAttachment={setAttachment} intakeState={active?.intakeState} busy={Boolean(liveRun?.running)} models={models} model={model} setModel={setModel} />
      <WorkspaceDrawer view={view} token={token} onClose={() => setView(null)} onOpen={setView} />
    </section>
  </div>;
}
