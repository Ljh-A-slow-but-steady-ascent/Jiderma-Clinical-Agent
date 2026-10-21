# 霁肤 · Jiderma

面向基层医生的皮肤病灶辅助决策与病例报告工作台。系统将病灶图像、患者自然语言描述、本地多模态分类、风险分级、本地医学知识检索、受约束 Agent 和报告归档组织为一条完整工作流。

> 本系统用于辅助筛查、医学资料检索和转诊决策参考，不替代医生诊断或病理检查。

## 当前工作流

```text
医生登录
  → 创建患者病例并填写自然语言描述
  → PrivFlow 脱敏与固定字段抽取
  → 医生确认结构化临床信息
  → 病灶图像 + 临床文本进入本地多模态模型
  → 输出疾病 Top-3 与本地风险等级
  → Agent 通过 MCP 检索本地医学知识库
  → 医生查看可追溯证据并综合判断
  → LLM 生成结构化报告字段
  → 本地回填患者身份信息
  → 在线编辑、保存和归档报告
```

系统采用单一医生工作台，必须登录后使用。患者姓名、联系方式、原始图像和本地文件路径不会发送给云端模型；云模型只接收通过 PrivFlow 检查的脱敏结构化文本。

## 系统结构

![霁肤系统架构](./系统架构图.png)

![霁肤技术路线](./技术路线图.png)

当前版本只保留一条线上主链：`React 医生工作台 → FastAPI 服务层 → 本地多模态分析 / 医学检索 / Agent Harness → 病例、报告与文档数据库`。旧 Streamlit 界面、旧固定轮次 Agent 和二次校准链路已移除。

核心边界：

- 本地多模态模型负责生成候选结果，LLM 不修改或重排 Top-3。
- 风险等级和照护路径由本地确定性逻辑产生，LLM 无权下调风险。
- LLM 主要负责固定字段抽取、检索规划、证据整理和报告语言生成。
- REST API 与 MCP Tools 共用 Service Layer，不复制业务逻辑。
- Dense Retriever 不可用时自动切换本地词法检索；Deep Agent 由已连接的云端模型驱动。

## Deep Agents / LangGraph Agent Harness

聊天框由两层开源 Runtime 组成：官方 `deepagents` 负责多轮上下文理解、任务清单、工具循环、子任务委派和虚拟工作区；LangGraph `StateGraph` 负责确定性的任务路由、病例写入、人工确认、暂停恢复和事件持久化。两层共享现有的多模态 Provider、MCP 工具、RAG、报告和数据库服务，统一构成 Deep Agent Harness。普通读取、检索、计算、临时规划和会话文档草稿由Agent自主执行，病例及正式报告的创建、修改和删除进入医生批准门。聊天窗默认实时展示可审计的任务计划、工具动作、执行状态和结果依据，不保存或展示模型私有思维链：

```text
医生指令 / 图片 / 自然语言病例
  → route_request
  ├─ collect_materials      等待补充信息
  ├─ prepare_case          生成病例写入预览
  ├─ prepare_update        生成病例修改预览
  ├─ analyze_case          本地多模态分析
  ├─ open_source_agent     Deep Agents + 任务计划 + 委派 + MCP 工具循环
  ├─ generate_report       生成并归档报告草稿
  └─ pause                 保存状态并暂停
```

病例新建和修改必须再经过独立的 `route_confirmation` 确认图。医生未点击确认时，
Agent 只保存待执行操作，不会写入患者或病例数据。`agent_runs` 和
`agent_events` 保存运行状态、事件时间线和暂停/恢复信息；原始图片路径和完整身份资料不出现在公开的 Agent 状态中。

聊天框中点击“＋”只选择病灶图片，患者信息与症状直接写在同一个对话输入框。只有当前消息实际附图时，系统才会抽取结构化字段并准备本地多模态分析；缺少字段会在对话中追问。医生确认写入预览后才执行联合分析，并在对话中显示 Top-3、风险分级、医学证据和可点击的病例工作台入口。React 前端通过页内工作区打开病例、报告和 Agent 文档，不会因页面跳转丢失登录状态。

对话交互使用任务工作台形式：提交后立即返回任务编号，前端持续读取已持久化的 Agent 事件。Deep Agent 自主判断是否需要拆分任务、委派子任务、使用虚拟草稿、本地医学 RAG、病例证据、计算器或公开检索；普通问题不再被默认送去联网搜索。会话线程保留最近上下文，可正常理解“它、这个、为什么、继续”等追问。

React 工作台支持左侧会话管理、新建、切换、重命名、删除、停止运行和移动端响应式布局。Agent 通过
`create_document / read_document / rewrite_document / list_documents` 创建和改写会话文档；
文档以登录医生和会话为边界保存在本地数据库，并在聊天消息中显示为可编辑文件卡片。医生可直接保存新版本或下载到本地；病例与报告通过页内工作区打开。

Agent 还可通过 `search_clinical_archive / read_clinical_archive_resource`
查询当前登录医生有权访问的报告库和文档库。查询在本地执行；患者姓名、门诊号和电话在云端调用前替换为临时别名，工具返回的文档摘要会再次经过 EgressGuard。医学解释会附带工具验证的来源，并区分“本地知识库/报告库/文档库（已存库）”和“外部公开资料（未存入本地知识库）”。

## 本地动态 RAG 与 Agent

病例分析阶段会根据医生描述、Top-3、阳性症状、风险等级和检索主题自动构造动态 Query，检索本地结构化医学资料。结果按照风险、鉴别、检查、转诊、冲突和信息缺口组织，并保留文档、机构、年份、章节、页面位置和版本信息。

ChatBox 使用 Deep Agents 驱动的 Agentic RAG。检索步数由任务目标、工具观察和证据缺口动态决定；任务复杂时先维护任务清单，再自主决定是否继续检索：

```text
理解医生问题
  → 判断证据缺口
  → 生成检索 Query
  → 调用 search_guideline MCP Tool
  → 观察证据覆盖与来源
  → 必要时改写 Query 再检索
  → 基于证据生成回答或提出待确认操作
```

本地 Dense Retrieval 使用 `weights/rag/IlhaEmbed/` 中的轻量医学文本编码器。Dense 不可用时自动退化为本地词法检索。低于阈值的结果会标记为“当前知识库证据不足”，不会要求 LLM 自行补充医学事实。

MCP Server 提供 6 个受控工具：

- `get_visual_evidence`
- `get_triage_result`
- `search_guideline`
- `get_care_pathway`
- `get_privacy_report`
- `web_search`（仅用于简单非医学公开资料）

工具输出在进入云端前必须通过 EgressGuard；图片字节、Base64、本地路径和患者身份字段会被阻止并写入审计记录。

## PAD-UFES-20 多模态实验

当前业务视觉 Provider 使用冻结的图像与文本编码器，将 RGB 临床图像和经医生确认的 PAD 格式病例文本融合后输出候选结果。实现复用 DermFM-Zero 官方组件，不修改 `third_party/DermFM-Zero`：

```text
224×224 RGB 图像 → 冻结视觉编码器 ───────┐
                                         ├→ 官方融合模块 → 分类头 → 6 类概率
结构化信息 → PAD Prompt → 冻结文本编码器 ─┘
```

三种实验模式使用作者提供的同一组 `train.csv / val.csv / test.csv`，不自行重新划分：

| 模式 | 最佳轮次 | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| `image_only` | 17 | 0.8000 | 0.7080 | 0.7253 |
| `text_only` | 20 | 0.6957 | 0.5714 | 0.5911 |
| `image_text` | 30 | **0.8348** | **0.8402** | **0.8040** |

最佳 checkpoint 由验证集 Macro-F1 选择，测试集仅在模型选定后评估。作者划分存在跨集合患者 ID 重复，因此该结果用于验证 MVP 链路和同划分模式比较，不直接代表独立患者泛化性能。

训练：

```bash
bash scripts/train_pad_multimodal_mvp.sh \
  --mode all --epochs 30 --batch-size 4 \
  --accumulation-steps 8 --precision bf16 --device cuda
```

单样本推理：

```bash
bash scripts/infer_pad_multimodal_mvp.sh \
  --image data/pad_ufes20/images/PAT_980_1849_295.png \
  --metadata examples/pad_patient_example.json \
  --model-checkpoint artifacts/training/pad_multimodal_mvp/image_text/best.pt \
  --device cuda
```

实验结果、预测 CSV、混淆矩阵、训练曲线、日志与 checkpoint 保存在 `artifacts/training/pad_multimodal_mvp/`。

## HAM10000 视觉线性头归档

项目保留早期视觉 Encoder 下游实验，用于复现冻结视觉特征上的轻量任务头训练；它不属于当前 Web 推理链。

- 冻结视觉 Encoder，只训练 `Linear(768, 7)`。
- 官方下游划分：训练 8,012、验证 500、测试 1,503。
- 可训练参数：5,383。
- 测试 Accuracy：89.29%。
- 测试 Balanced Accuracy：79.34%。
- 测试 Macro-F1：80.98%。

代码和结果：

- `src/dermfm_provider.py`：视觉特征提取。
- `scripts/train_dermfm_linear_head.py`：任务头训练。
- `scripts/visualize_training_results.py`：从日志重绘结果。
- `artifacts/training/ham7_linear_head/`：权重、日志、配置、特征缓存和测试结果。

## 启动项目

依赖环境：

- `dermaguard`（Python 3.11+）：FastAPI、RAG、Deep Agent 与报告服务；同时提供 Node.js 22 构建 React 前端。
- `dermfm-zero`：本地多模态 Provider 和训练推理。

所有直接依赖已集中在根目录 `requirements.txt`。该文件通过 Python
版本标记适配现有两个环境：Python 3.11 安装 Web/API/MCP/Deep Agents 依赖，
Python 3.9 安装上游兼容的模型训练依赖。

```bash
conda run -n dermaguard python -m pip install -r requirements.txt
conda run -n dermfm-zero python -m pip install -r requirements.txt
```

首次构建 React 前端时执行：

```bash
conda install -n dermaguard -c conda-forge nodejs=22
PATH=/home/lu/miniforge3/envs/dermaguard/bin:$PATH npm --prefix frontend install
```

一键启动网站：

```bash
cd '/media/lu/CE387FD6387FBC51/科研项目/医疗agent'
bash scripts/run_mock_demo.sh
```

访问地址：

- React 医生工作台：<http://127.0.0.1:8501>
- FastAPI 文档：<http://127.0.0.1:8000/docs>

独立启动 MCP stdio Server：

```bash
bash scripts/run_mcp_server.sh
```

MCP 标准输出是 JSON-RPC 通道，因此直接运行后安静等待属于正常状态。

## 云端模型配置

不配置 API Key 时可以使用本地 Mock。启用云端模型时，只将密钥保存在本机 `.env`，不要提交到 Git：

```bash
cp .env.example .env
chmod 600 .env
```

支持的 Provider 和模型白名单以 `.env.example` 及 ChatBox 模型列表为准。所有云端请求仍需经过 PrivFlow 和 EgressGuard。

## 测试

```bash
/home/lu/miniforge3/envs/dermaguard/bin/python -m pytest backend/tests/v2 -q
/home/lu/miniforge3/envs/dermfm-zero/bin/python -m pytest tests/test_pad_multimodal_mvp.py -q
```

后端测试覆盖登录与报告、自然语言字段抽取、多模态 Service、风险分级、RAG、Care Pathway、Agent、MCP、隐私出境和端到端业务链。多模态测试使用独立环境验证 Prompt、融合模型和推理接口。

## 目录说明

```text
backend/app/                 FastAPI 业务代码
  api/                       登录与业务接口
  chat/                      结构化抽取与报告 LLM Provider
  harness/                   LangGraph 任务图、确认门与上传资产
  database/                  本地持久化模型与仓储
  knowledge/rag_documents/  本地医学知识库
  mcp/                       MCP Server、Registry 与 Tools
  services/                  业务服务、RAG 与隐私治理
backend/tests/v2/            后端与 MCP 测试
frontend/                    React + Vite 医生工作台
pad_multimodal_mvp/          PAD 多模态训练及推理实现
scripts/                     启动、训练、推理与绘图入口
src/dermfm_provider.py       HAM7 实验视觉特征 Provider
artifacts/training/          训练日志、指标、图表与 checkpoint
data/                        本地数据库、病例和数据集
weights/                     本地模型实体权重
third_party/DermFM-Zero/     官方上游代码
```

`scripts/` 仅保留项目启动、MCP Server、PAD 数据准备/训练/推理和已完成实验绘图入口；不保留旧版 UI、已弃用 Agent 或规划性伪数据绘图代码。

`backend/app/knowledge/rag_documents/*.md` 是运行时知识库数据，`third_party/DermFM-Zero/**/README.md` 属于上游源码说明；它们不是重复的项目文档，不应删除。

## 开源协议

本项目自研代码以 [MIT License](./LICENSE) 发布。`third_party/DermFM-Zero/` 及其它第三方组件仍遵循各自原始许可证，不因本项目的 MIT License 而改变。
