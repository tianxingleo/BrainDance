# BrainDance 项目协作规范与开发协议 (v1.0)

**致团队成员：**

鉴于 BrainDance 项目涉及多技术栈（移动端 Flutter、后端 Go、AI 算法 Python）及端云协同架构，系统的复杂性要求我们必须遵循统一的工程规范。为确保多人协作时的代码稳定性、降低集成冲突并提升开发效率，即日起请严格执行以下协作协议。

------

## 一、 项目架构与目录权责 (Monorepo Structure)

本项目采用 **Monorepo（单体仓库）** 策略，所有代码托管于同一 GitHub 仓库。请严格遵守目录隔离原则，**严禁跨目录修改非本人负责的业务代码**。

### 1. 目录结构

Plaintext

```
BrainDance/
├── app/            # [移动端] Flutter 工程
├── server/         # [后端] Go 业务服务
├── ai_engine/      # [算法层] Python RAG/3DGS 脚本
├── docs/           # [文档] 接口定义(OpenAPI)、架构图、会议记录
├── deploy/         # [运维] Docker-compose、Nginx 配置
└── .gitignore      # 全局 Git 忽略配置
```

### 2. 分工权责

- **前端负责人**：全权负责 `/app` 目录。涉及 Flutter UI、ARCore 调用、Web 渲染器嵌入。
- **后端负责人**：全权负责 `/server` 与 `/deploy` 目录。涉及 API 接口设计、数据库 (MySQL/Redis) 维护、云端对象存储对接。
- **算法负责人**：全权负责 `/ai_engine` 目录。涉及 RAG 向量检索逻辑、Luma/Tripo API 对接脚本、Prompt 优化。

------

## 二、 Git 工作流规范 (Workflow)

我们采用 **Feature Branch Workflow**，**严禁**直接向 `main` 或 `dev` 分支推送代码。

### 1. 分支定义

- **`main` (生产分支)**：仅存放经过测试、随时可演示的稳定版本。**受到写保护**。
- **`dev` (开发主分支)**：日常开发的集成分支，所有新功能均合并至此。
- **`feat/xxx` (功能分支)**：日常开发的工作分支。

### 2. 开发流程标准动作

1. **同步代码**：开发前务必拉取最新 `dev` 分支：`git pull origin dev`。
2. **新建分支**：基于 `dev` 创建功能分支，命名规范：`feat/<姓名拼音>/<功能名>`。
   - *示例*：`feat/zhangsan/login-ui`
3. **提交代码 (Commit)**：在本地进行开发并提交。
4. **发起合并 (PR)**：在 GitHub 发起 **Pull Request**，目标分支选择 **`dev`**。
5. **代码评审 (Code Review)**：
   - PR 必须经过至少 1 名其他成员 Review。
   - 重点检查：逻辑漏洞、硬编码 Key、不规范命名。
   - Review 通过后方可合并。

------

## 三、 Commit 提交规范

Git Commit Message 必须清晰明了，杜绝 "update"、"fix" 等无意义描述。请遵循以下格式：

```
type(scope): subject
```

- **Type (类型)**：
  - `feat`: 新功能 (feature)
  - `fix`: 修补 bug
  - `docs`: 文档修改
  - `refactor`: 代码重构（无功能变动）
  - `chore`: 构建过程或辅助工具变动
- **Scope (范围)**：影响的模块（如 `app`, `server`, `rag`）。
- **Subject (描述)**：简短描述变更内容（中文）。

**示例**：

- `feat(app): 新增 AR 扫描页面的相机权限请求`
- `fix(server): 修复 Luma API 回调解析失败的问题`

------

## 四、 开发铁律 (Critical Rules)

### 1. API 契约先行 (Contract First)

- **原则**：在前后端联调前，必须先在 `docs/api_v1.yaml` 中定义好接口路径、参数和响应格式。
- **禁止**：后端未定义接口文档，口头告知前端字段；前端未确认文档，臆测字段名开发。

### 2. 敏感信息零容忍 (Security)

- **严禁上传**：API Key (OpenAI, Luma, Google Maps)、数据库密码、云服务 Secret。
- **规范**：所有敏感配置必须通过 **环境变量 (.env)** 或 **本地配置文件 (config.yaml)** 加载，并将该文件加入 `.gitignore`。

### 3. 大文件管理

- GitHub 单文件限制为 100MB。
- **禁止**：直接提交 `.ply` 模型文件、MP4 演示视频、大型预训练模型权重。
- **规范**：大文件请上传至网盘或对象存储，在代码中仅保留**下载链接 (URL)**。

### 4. 依赖管理

- 引入新的第三方库时，必须同步更新配置文件（`pubspec.yaml`, `go.mod`, `requirements.txt`），并在群内告知队友，防止他人拉取代码后运行报错。

------

## 五、 沟通与复盘

- **Issue 追踪**：所有待开发功能和已知 Bug 必须录入 GitHub Issues，并拖入 Project 看板管理状态（Todo -> In Progress -> Done）。
- **同步机制**：遇到技术阻塞（Block）超过 2 小时，必须立即在群内求助，禁止独自埋头苦干导致进度停滞。

请各位收到后回复“**收到**”，并严格按照上述规范执行。

BrainDance 开发组 2025年12月2日