# 整体架构
## 通用约定

### Header
- Content-Type: application/json
- x-timestamp: Unix 时间戳（秒）例如 1738689600
- x-nonce: 随机串（建议 16~32 字节，base64url/hex 均可）
- x-signature: 签名值（base64url 编码）
- x-signature-method: 固定 HMAC-SHA256
- x-signature-version: 固定 v1
- x-request-id: 便于链路追踪

### StringToSign
```
api_secret = "......"
signature = sha256(api_secret + timestamp + nonce)
```

### 错误码

|code |http_status	|含义	|处理建议|
| ----  | ---- |  ----  | ---- |
|0|	200|	成功|	-
|1001|	400|	参数缺失/非法（JSON 结构错误、字段为空、base64 非法）|	修正入参
|1002|	400|	分段时长不符合预期（或 metadata 校验失败）|	调整切分
|1003|	413|	分段过大（超出服务限制）|	降码率/缩短分段
|1004|	415|	媒体格式不支持（非 wav/mp4 或编码不支持）|	转码
|1101|	401|	未鉴权/鉴权失败|	更新 token
|1103|	403|	无权限/租户不匹配|	检查权限
|1201|	429|	请求频率超限|	降频/重试
|1301|	500|	服务内部错误|	重试/联系支持
|1302|	503|	服务不可用/拥塞|	退避重试
|1404|	404|	request_id 不存在或已清理|	确认 request_id 或重新提交

### 回调接口
- Method：POST
- Application：json

## 接口设计

1. 音频分析 ASR（异步+回调）
- 接入点：POST /api/v1/credit-av-audit/asr/submit
- 入参结构
`audio_b64`：wav 分段的 base64

```json
{
  "request_id": "string",
  "session_id": "string",
  "segment_index": 1,
  "segment_ts_ms": 0,
  "audio_b64": "string",
  "audio_format": "wav",
  "callback_url": "https://example.com/callback/asr",
  "is_last": false,
}
```

- ACK
```json
{
  "code": 0,
  "message": "accepted",
  "request_id": "string",
  "session_id": "string",
  "accepted_at": "2026-02-03T16:20:00Z"
}
```

- Callback

status 枚举

- SUCCESS：识别成功
- PARTIAL：部分成功（例如某段噪声严重）
- FAILED：失败（同时返回 error）

```json
{
  "result_type": "asr",
  "request_id": "string",
  "session_id": "string",
  "segment_index": 3,
  "segment_ts_ms": 15000,
  "status": "SUCCESS",
  "result": {
    "language": "zh-CN",
    "utterances": [
      {
        "start_ms": 15120,
        "end_ms": 17200,
        "text": "请您确认一下本次贷款用途是什么？",
        "confidence": 0.93
      },
      {
        "start_ms": 17310,
        "end_ms": 19800,
        "text": "主要用于店铺周转和补充货款。",
        "confidence": 0.90
      }
    ]
  },
  "trace": {
    "model": "asr-model-v1",
    "latency_ms": 420
  }
}
```

- Failed

```json
{
  "result_type": "asr",
  "request_id": "req-xxx",
  "session_id": "sess-xxx",
  "segment_index": 3,
  "status": "FAILED",
  "error": {
    "code": 1004,
    "message": "unsupported audio encoding"
  }
}
```

2. 视频尽调采集指引（异步+回调）
- 接入点：POST /api/v1/credit-av-audit/video-guidance/submit
- 入参结构
```json
{
  "request_id": "string",
  "session_id": "string",
  "segment_index": 1,
  "segment_ts_ms": 0,
  "video_b64": "string",
  "video_format": "mp4",
  "task_list": [
    {
      "task_id": "T01",
      "name": "门头照",
      "desc": "需要清晰拍摄店铺门头与招牌文字",
    },
    {
      "task_id": "T02",
      "name": "营业执照",
      "desc": "需要完整展示营业执照且文字可读",
    }
  ],
  "callback_url": "https://example.com/callback/video-guidance",
  "risk_ruleset": "default-v1",
  "is_last": false,
  "ext": {
    "loan_id": "string",
    "customer_id": "string"
  }
}
```

- ACK
```json
{
  "code": 0,
  "message": "accepted",
  "request_id": "string",
  "session_id": "string",
  "accepted_at": "2026-02-03T16:20:00Z"
}
```

- Callback
```json
{
  "result_type": "video_guidance",
  "request_id": "string",
  "session_id": "string",
  "segment_index": 5,
  "segment_ts_ms": 42000,
  "frames": 100,
  "fps": 24,
  "status": "SUCCESS",
  "result": {
    "hit_tasks": [
      {
        "task_id": "T01",
        "video_quality": "清晰 / 模糊",
        "completation": "完整 / 不完整",
        "scene_desc": "检测到店铺门头与招牌区域，文字清晰度较高。具体内容为……",
        "risk": "",
      },
      {
        "task_id": "T02",
        "video_quality": "清晰 / 模糊",
        "completation": "完整 / 不完整",
        "scene_desc": "检测到营业执照",
        "risk": "画面抖动导致部分文字不可读，建议靠近门头并稳定拍摄 3 秒以上",
      }
    ],
    "guidance": {
      "next_action": "CAPTURE_T02",
      "suggestion": "请将镜头对准营业执照，确保完整入镜且停留 3 秒"
    }
  },
  "trace": {
    "model": "vlm-guidance-v2",
    "latency_ms": 880
  }
}
```

3. 结果推送（同步）

在ASR、视频尽调 callback失败时，框架临时缓存结果，等待请求方主动取回

- 接入点：
  - POST /api/v1/credit-av-audit/asr/pull
  - POST /api/v1/credit-av-audit/video-guidance/pull

- 入参结构：
```json
{
  "request_id": "string"
}
```

- ASR结果返回
```json
{
  "code": 0,
  "message": "ok",
  "request_id": "string",
  "purged": true,
  "result": {
    "result_type": "asr",
    "request_id": "string",
    "session_id": "string",
    "segment_index": 3,
    "segment_ts_ms": 15000,
    "status": "SUCCESS",
    "result": {
      "language": "zh-CN",
      "utterances": []
    },
    "trace": {
      "model": "asr-model-v1",
      "latency_ms": 420
    }
  },
  "trace": {
    "cached_at": "2026-02-03T16:20:10Z",
    "expired_at": "2026-02-04T16:20:10Z"
  }
}
```

- video结果返回
```json
{
  "code": 0,
  "message": "ok",
  "request_id": "string",
  "purged": true,
  "result": {
    "result_type": "video_guidance",
    "request_id": "string",
    "session_id": "string",
    "segment_index": 5,
    "segment_ts_ms": 42000,
    "frames": 100,
    "fps": 24,
    "status": "SUCCESS",
    "result": {
      "hit_tasks": [
        {
          "task_id": "T01",
          "match_score": 0.87,
          "state": "COMPLETED",
          "scene_desc": "检测到店铺门头与招牌区域，文字清晰度较高",
          "risk": ""
        }
      ],
      "guidance": {
        "next_action": "CAPTURE_T02",
        "suggestion": "请将镜头对准营业执照，确保完整入镜且停留 3 秒"
      }
    },
    "trace": {
      "model": "vlm-guidance-v2",
      "latency_ms": 880
    }
  },
  "trace": {
    "cached_at": "2026-02-03T16:20:10Z",
    "expired_at": "2026-02-04T16:20:10Z"
  }
}
```

## 项目实现说明

### 已实现功能

1. **完整的API接口**
   - ASR音频分析提交和结果拉取接口
   - 视频尽调采集指引提交和结果拉取接口
   - 健康检查接口

2. **安全验证机制**
   - 完整的Header签名验证，包括时间戳、随机数、签名等
   - 防重放攻击机制
   - HMAC-SHA256签名算法

3. **数据处理**
   - Base64编码的音频和视频数据解码
   - 数据模型验证（使用Pydantic）

4. **结果缓存**
   - 带TTL的缓存机制
   - 后台线程定时清理过期缓存
   - 结果拉取后自动删除

5. **配置管理**
   - 统一配置文件管理
   - 支持API密钥、缓存时间等配置项

### 项目结构

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

### 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 运行服务
python app.py

# 测试
python -m pytest tests/ -v

```

### 配置说明

在`config.yaml`中配置以下参数：

```yaml
api:
  secret: "your-api-secret-here"      # API密钥
  request_timeout: 300                # 请求超时时间（秒）

cache:
  default_ttl: 3600                   # 默认缓存过期时间（秒）
  cleanup_interval: 60                # 清理间隔（秒）

server:
  host: "0.0.0.0"                     # 服务监听地址
  port: 8000                          # 服务端口
  reload: true                        # 是否启用热重载
```