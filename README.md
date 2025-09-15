## 项目

文档重复检测系统（DocumentDuplicationDetection）

## 作用

- 提供文档相似度检测与重复内容比对
- 支持单文件之间的差异比对
- 支持上传文件与存储库批量比对
- 支持关键词在存储库中的批量搜索

## 简介

本项目提供一套基于 Flask 的 Web 应用：
- 单文件比较：上传两个同类型文件（PDF/DOCX/DOC/XLSX/XLS），生成语义相似度与差异明细
- 存储库比较：上传一个文件，与指定目录下同类型文件批量比较，输出相似度排行与高频词、相似原因
- 关键词匹配：在存储库中按关键词进行全文搜索

底层支持多种文本抽取与比对策略，针对 Word/Excel/PDF 做了专用适配；支持多进程处理与资源清理，适合大批量文件场景。

## 说明

- 后端：Flask + concurrent.futures（多进程）
- 文本与相似度：NLTK、SentenceTransformer（可切换 TF-IDF 模式）
- 文件解析：python-docx、openpyxl、textract 等
- 结果页面：提供相似度标注（≥95% 红色、≥80% 黄色）、高频词展示与相似原因说明

## 使用方法

1) 安装依赖

```bash
pip install -r requirements.txt
```

2) 启动（推荐）

```bash
python start_app.py
```

3) 访问

- 本地访问：`http://localhost:5000`
- 局域网访问：`http://YOUR_IP:5000`

4) 基本操作

- 单文件比较：选择“单文件比较”，上传两个同格式文件，提交后跳转到进度页与结果页
- 存储库比较：选择“存储库比较”，上传文件，选择存储路径与文件类型，提交后显示比对结果
- 关键词匹配：选择“关键词匹配”，输入关键词并选择路径与类型，查看匹配结果

## 启动说明

以下内容合并自《启动说明.md》并做了精简：

### 1. 正常启动（推荐）

```bash
python start_app.py
```

- 自动下载和加载 SentenceTransformer 模型
- 提供较高精度的文档相似度计算（首次启动可能下载约 400MB 模型文件）

### 2. 快速启动（如遇启动卡住）

```bash
# Windows CMD
set FAST_START=1 && python start_app.py

# PowerShell  
$env:FAST_START=1; python start_app.py

# Linux/Mac
FAST_START=1 python start_app.py

# 或使用快速启动脚本
python quick_start.py
```

### 3. 强制 TF-IDF 模式（无需下载模型）

```bash
# Windows CMD
set FORCE_TFIDF=1 && python start_app.py

# PowerShell
$env:FORCE_TFIDF=1; python start_app.py

# Linux/Mac  
FORCE_TFIDF=1 python start_app.py
```

## 常见问题

### 启动时卡住或超时

现象：显示“应用导入超时！可能是模型加载过程卡住”。

解决：
1. 使用快速启动：`python quick_start.py`
2. 检查网络（首次需要访问 Hugging Face）
3. 使用 TF-IDF：`set FORCE_TFIDF=1 && python start_app.py`

### 网络连接问题（无法下载模型）

解决：
1. 检查防火墙
2. 配置代理：设置 `HTTP_PROXY` 和 `HTTPS_PROXY`
3. 手动下载模型到本地
4. 临时使用 TF-IDF 模式

### 内存不足

解决：
1. 使用 TF-IDF 模式
2. 关闭其他占用内存的程序
3. 选择更小的模型

## 目录结构（节选）

```
app.py
start_app.py
single_compare.py
storage_compare.py
keyword_search.py
templates/
static/
utils.py
```

## 许可证

本项目仅供内部或学习使用，如需商业使用请根据所依赖库的许可证合规处理。