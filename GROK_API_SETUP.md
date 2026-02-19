# Grok AI 启用指南

## 📋 前置条件

Grok AI 是 xAI 公司提供的大语言模型服务,需要 API Key 才能使用。

## 🔑 获取 xAI API Key

### 步骤1: 注册 xAI 账号

1. 访问 xAI 控制台: https://console.x.ai/
2. 使用 X (Twitter) 账号登录,或创建新账号
3. 完成账号验证

### 步骤2: 创建 API Key

1. 登录后进入 API Keys 页面
2. 点击 "Create API Key"
3. 给 Key 命名 (如 "AQuantTrader")
4. 复制生成的 API Key (格式: `xai-xxxxxxxxxxxxxxxxxxxxxxxx`)
5. **重要**: 妥善保存,离开页面后无法再次查看

### 步骤3: 充值账户 (可选)

- xAI 提供免费额度供测试
- 超出免费额度需要充值
- 当前价格 (2026年2月):
  - grok-3-mini: ~$0.15/1M tokens
  - grok-4-1: ~$2/1M tokens

## ⚙️ 配置系统

### 方式1: 环境变量 (推荐)

**临时设置** (仅当前终端有效):
```bash
export XAI_API_KEY="xai-xxxxxxxxxxxxxxxxxxxxxxxx"
```

**永久设置** (写入 shell 配置文件):
```bash
# macOS/Linux (bash)
echo 'export XAI_API_KEY="xai-xxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bash_profile
source ~/.bash_profile

# macOS/Linux (zsh)
echo 'export XAI_API_KEY="xai-xxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.zshrc
source ~/.zshrc
```

### 方式2: .env 文件 (备选)

在项目根目录创建 `.env` 文件:
```bash
cd /Users/niuyj/Downloads/workspace_Claude/stock/a-quant-trader
echo 'XAI_API_KEY=xai-xxxxxxxxxxxxxxxxxxxxxxxx' >> .env
```

### 验证配置

运行以下命令验证环境变量是否设置成功:
```bash
echo $XAI_API_KEY
```

应该输出您的 API Key (以 `xai-` 开头)。

## 🚀 重启应用

配置完成后,重启应用以生效:

```bash
cd /Users/niuyj/Downloads/workspace_Claude/stock/a-quant-trader

# 停止旧进程
pkill -f "streamlit run"

# 启动新进程
nohup python3 -m streamlit run src/web/app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &
```

## ✅ 验证功能

1. 打开浏览器访问 http://localhost:8501
2. 进入 **"📊 个股分析"** Tab
3. 输入股票代码 (如 `AAPL`)
4. 点击 **"🚀 开始分析"**
5. 切换到 **"🤖 AI分析"** Tab
6. 应该能看到 **"🤖 个股社交情绪"** 和 **"🌍 市场整体情绪"** 面板

如果看到 "💡 Grok AI分析未启用或无数据",说明:
- API Key 未设置或无效
- 环境变量未生效 (需要重启应用)
- API 调用失败 (检查网络或额度)

## 💰 成本控制

系统已内置成本控制机制:

### 默认预算
- 每日上限: $5 USD (可在 `config/settings.yaml` 调整)
- 超出后自动停止调用

### 缓存机制
- 个股情绪: 4小时缓存
- 市场状态: 1小时缓存
- 深度分析: 24小时缓存

### 典型消耗
- 单次个股情绪分析: ~$0.01-0.02
- 单次市场状态分析: ~$0.02-0.03
- 单次深度分析: ~$0.10-0.15
- 日常使用 (10只股票): ~$0.2-0.5/天

## 🔧 调整配置

编辑 `config/settings.yaml`:

```yaml
grok:
  enabled: true  # ✅ 已启用

  # 调整每日预算
  rate_limit:
    max_daily_cost_usd: 10.0  # 改为 $10/天

  # 调整缓存时间 (秒)
  cache_ttl:
    sentiment: 7200   # 改为 2小时
    market: 1800      # 改为 30分钟
```

修改后需重启应用。

## 🆓 免费替代方案

如果不想使用付费 API,系统仍能正常工作:

- ✅ 行研报告共识 (免费,yfinance/AKShare)
- ✅ HMM 市场状态识别 (免费)
- ✅ 行业轮动分析 (免费)
- ✅ 智能风控 (免费)
- ✅ DL 信号过滤 (免费,本地模型)
- ❌ Grok AI 情绪分析 (需付费 API)

缺少 Grok 只会影响社交情绪维度,不影响核心策略决策。

## ❓ 常见问题

### Q1: API Key 无效
**A**: 检查:
- Key 格式是否正确 (以 `xai-` 开头)
- 是否复制完整 (不包含空格/换行)
- xAI 账户状态是否正常

### Q2: 环境变量不生效
**A**: 确保:
- 使用 `source` 命令重新加载配置文件
- 或重新打开终端
- 或直接在启动命令前设置: `XAI_API_KEY=xxx python3 -m streamlit run ...`

### Q3: 超出免费额度
**A**:
- 检查 xAI 控制台的 Usage 页面
- 充值账户或等待下月重置
- 减少 `max_daily_cost_usd` 限制

### Q4: API 调用太慢
**A**:
- Grok API 响应时间通常 2-5秒
- 网络问题可能导致超时
- 增加缓存时间减少调用频率

## 📞 技术支持

- xAI 官方文档: https://docs.x.ai/
- xAI 状态页: https://status.x.ai/
- 系统日志: `tail -f streamlit.log`

---

**最后更新**: 2026-02-19
