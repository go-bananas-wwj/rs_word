# rs-words · 用真实河流卫星影像拼出汉字

> 将中国河流（长江、黄河、珠江等）的卫星影像切片拼成 CJK 文字的艺术生成工具。

## 项目背景 / 灵感

本项目受到 NASA "Your Name in Landsat" 与 NODA "天地画布 · AI 生花" 的启发：把真实的地球遥感影像当作"像素"，拼出具有地理意义的汉字。河流的蜿蜒形态天然接近汉字的笔画，因此我们从 OpenStreetMap 提取主要流域矢量，从 Microsoft Planetary Computer 获取 Sentinel-2 / Landsat 卫星影像，再按笔画相似度拼接成最终作品。

## 快速开始

```bash
# 0. 激活项目 Conda 环境
source /opt/conda/etc/profile.d/conda.sh && conda activate rs_words

# 1. 安装（含开发依赖）
pip install -e ".[dev]"

# 2. 准备一款 CJK 字体（例如思源黑体）
mkdir -p /data/rs_word/fonts
cp /path/to/SourceHanSansCN-Regular.otf /data/rs_word/fonts/

# 3. 构建卫星影像切片库（需要流域 OSM 数据与 Planetary Computer 访问，耗时较长）
make build-bank

# 4. 生成河流汉字（默认使用网格模式拼贴遥感影像）
rs-words create "河" \
  --font /data/rs_word/fonts/SourceHanSansCN-Regular.otf \
  --output /data/rs_word/outputs/河.png \
  --meta /data/rs_word/outputs/河.json \
  --font-size 512 --tile-size 64

# 5. 启动网页 Demo
make run-web
```

网页服务默认运行在 <http://localhost:8000>，提交文字后可在浏览器中直接预览并下载结果。

## 水体 mask 与笔画素材筛选

高质量笔画模式需要先把真实河道从遥感切片中分割出来，再用河道 mask 匹配笔画形状。推荐流程：

```bash
# 1. 下载多区域 Sentinel-2 四波段 GeoTIFF，同时写 RGB 预览
python scripts/build_diverse_patch_bank.py --format geotiff4 --rgb-preview

# 2. 用公开水体分割模型生成 mask（需要先在 rs_words Conda 环境安装/配置 OmniWaterMask）
python scripts/build_water_masks.py --backend omniwatermask --patch-bank /data/rs_word/patch_bank

# 3. 如果模型环境暂时不可用，可先用 NDWI fallback 跑通链路
python scripts/build_water_masks.py --backend ndwi --patch-bank /data/rs_word/patch_bank

# 4. 用 mask 和河道几何指标重建笔画素材库
python scripts/build_stroke_library.py

# 5. 使用笔画拼贴模式生成
rs-words create "河" \
  --mode stroke \
  --font /data/rs_word/fonts/SourceHanSansCN-Regular.otf \
  --output /data/rs_word/outputs/河_stroke.png \
  --meta /data/rs_word/outputs/河_stroke.json
```

`build_water_masks.py` 会把 mask 写入 `/data/rs_word/water_masks/`，并把 `water_mask_path`、`mask_backend`、`river_metrics` 写回 patch bank metadata。`RiverMatcher` 会优先使用这些 mask；没有 mask 的旧数据仍会回退到 RGB 边缘匹配。

OmniWaterMask 可通过包内 API 或 `OMNIWATERMASK_COMMAND` 包装命令接入。`rivgraph` 当前作为可选增强依赖；未安装时，本项目使用 `skimage` skeleton 指标提供方向、连通域、骨架长度和分叉密度等基础几何过滤。

## 数据目录

所有大文件统一存放在 `/data/rs_word/`，与代码仓库分离，避免误提交：

| 目录 | 用途 |
|---|---|
| `/data/rs_word/osm/` | 从 OpenStreetMap 下载的河流流域矢量数据 |
| `/data/rs_word/satellite_chips/raw/` | 从 Planetary Computer 下载的原始卫星影像切片 |
| `/data/rs_word/patch_bank/` | 构建完成的影像切片库及其元数据 |
| `/data/rs_word/water_masks/` | 水体/河道二值 mask 与几何指标来源 |
| `/data/rs_word/outputs/` | 生成的汉字图片与 JSON 元数据 |
| `/data/rs_word/fonts/` | 用户自备的 CJK 字体文件 |

## 开发

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate rs_words
make test
# 等价于
pytest -v
ruff check src/rs_words tests scripts
```

## 命令行用法

```bash
rs-words create [OPTIONS] TEXT
```

主要选项：

| 选项 | 说明 | 默认值 |
|---|---|---|
| `TEXT` | 要渲染的中文文本（位置参数） | - |
| `-o, --output PATH` | 输出图片路径 | `/data/rs_word/outputs/out.png` |
| `--meta PATH` | 输出元数据 JSON 路径 | 无 |
| `--font PATH` | CJK 字体路径 | `/data/rs_word/fonts/` 下可用字体 |
| `--patch-bank PATH` | 切片库目录 | `/data/rs_word/patch_bank/` |
| `--font-size INT` | 渲染字号 | `256` |
| `--k INT` | 每个笔画候选匹配数量 | `5` |
| `--mode {grid,stroke}` | 合成模式：网格拼图 / 笔画拼图 | `grid` |
| `--tile-size INT` | grid 模式下每个瓦片的像素大小 | `128` |
| `--help` | 显示帮助信息 | - |

示例：

```bash
rs-words create "长江" \
  --font /data/rs_word/fonts/SourceHanSansCN-Regular.otf \
  --output /data/rs_word/outputs/长江.png \
  --meta /data/rs_word/outputs/长江.json \
  --font-size 512
```

## 网页 Demo

```bash
make run-web
```

启动 FastAPI 服务（默认 `0.0.0.0:8000`），访问首页即可在表单中输入文字并提交。后端通过 `/api/create` 端点调用与 CLI 相同的生成逻辑，返回 PNG 的 Base64 编码及对应元数据。

## 许可与来源

- **卫星影像**：通过 [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) 获取，使用公共数据资产（Sentinel-2、Landsat 等，需遵守各数据原始许可）。
- **河流矢量**：源自 [OpenStreetMap](https://www.openstreetmap.org/)，遵循 [ODbL](https://opendatacommons.org/licenses/odbl/) 开放数据库许可。
- **字体**：由用户自行提供，请确保拥有合法使用授权。
- **本项目代码**：MIT 许可。
