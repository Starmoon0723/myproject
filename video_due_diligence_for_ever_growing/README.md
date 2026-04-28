# Video Due Diligence API

基于FastAPI构建的视频尽调和ASR（自动语音识别）处理API服务。

## 目录
- [项目结构](#项目结构)
- [功能特性](#功能特性)
- [安装依赖](#安装依赖)
- [配置文件](#配置文件)
- [运行服务](#运行服务)
- [API接口](#api接口)
- [开发说明](#开发说明)
- [测试状态](#测试状态)
- [注意事项](#注意事项)

## 项目结构

```
.
├── app.py                 # 应用入口文件
├── config.yaml            # 全局配置文件
├── requirements.txt       # 项目依赖
├── vibe.md               # 项目需求文档
├── video_due_diligence/  # 主要代码目录
│   ├── main.py           # FastAPI应用主文件
│   ├── core/             # 核心业务逻辑
│   │   ├── __init__.py
│   │   └── config.py     # 全局配置类
│   ├── api/              # API处理逻辑
│   │   ├── __init__.py
│   │   ├── asr.py        # ASR相关API
│   │   └── video_guidance.py  # 视频尽调相关API
│   ├── schemas/          # 数据模型定义
│   │   ├── __init__.py
│   │   ├── asr.py        # ASR数据模型
│   │   └── video_guidance.py  # 视频尽调数据模型
│   ├── utils/            # 工具类
│   │   ├── __init__.py
│   │   ├── header_utils.py   # Header处理和数据反序列化工具
│   │   └── result_cache.py   # 结果缓存类（带后台清理线程）
│   └── routes/           # 路由配置
│       ├── __init__.py
│       └── api.py        # API路由配置
└── tests/                # 测试目录
```

## 功能特性

1. **ASR音频分析** - 异步处理音频并返回文本结果
2. **视频尽调采集指引** - 分析视频内容并提供采集指引
3. **结果缓存机制** - 当回调失败时临时存储结果供拉取
4. **安全验证** - 完整的Header签名验证机制
5. **配置管理** - 统一的配置文件管理
6. **自动清理** - 后台线程定时清理过期缓存

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置文件

项目使用`config.yaml`文件进行配置管理：

```yaml
# 全局配置文件
api:
  secret: "your-api-secret-here"
  request_timeout: 300  # 请求超时时间（秒）

cache:
  default_ttl: 3600     # 默认缓存过期时间（秒）
  cleanup_interval: 60  # 清理间隔（秒）

server:
  host: "0.0.0.0"
  port: 8000
  reload: true
```

## 运行服务

```bash
python app.py
```

服务将根据配置文件中的设置运行

## API接口

### ASR音频分析

- `POST /api/v1/credit-av-audit/asr/submit` - 提交ASR处理请求
- `POST /api/v1/credit-av-audit/asr/pull` - 拉取ASR处理结果

### 视频尽调采集指引

- `POST /api/v1/credit-av-audit/video-guidance/submit` - 提交视频尽调请求
- `POST /api/v1/credit-av-audit/video-guidance/pull` - 拉取视频尽调结果

## 开发说明

1. 所有API接口都包含完整的Header验证机制
2. 音频和视频数据通过Base64编码传输
3. 异步处理结果通过回调URL返回，失败时存储在缓存中供拉取
4. 项目使用Pydantic进行数据验证和序列化
5. 结果缓存使用后台线程定时清理过期数据
6. 配置统一管理，便于部署和维护

## 注意事项

- 需要在`config.yaml`中配置正确的API密钥
- 结果缓存默认过期时间和清理间隔可在配置文件中调整
- 所有接口均遵循vibe.md中定义的接口规范