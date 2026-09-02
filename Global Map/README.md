# Global Map

`Global Map` 是“雷霆医疗队”全球健康转型项目的交互可视化端。页面将全球健康脆弱性、压力-响应失配、政策证据与中国省级映射放在同一个可点击的展示界面中。

## 功能

- 3D 地球仪展示不同国家的脆弱性类型，支持点击诊断。
- 中国地图展示大陆 31 个省级行政区的压力、响应和适配缺口。
- 诊断面板联动显示风险因子、资源短板、政策路径和中国参照。
- 模块 D 证据卡展示政策候选的设计、机制与迁移证据边界。

## 立即运行

需要 Node.js 18 或更高版本。

```bash
cd "Global Map"
npm ci
npm run data
npm run dev
```

浏览器打开 Vite 输出的本地地址。生产构建与完整校验：

```bash
npm run verify
npm run preview
```

## 数据构建

`npm run data` 会调用 `scripts/build-data.mjs`，然后生成：

- `public/data/global-health-map.json`
- `public/data/china-provinces.geojson`

默认使用 `data_snapshot/` 中的最小派生数据快照，因此前端可以在不下载数百 MB 原始数据的情况下复现。若需从完整分析产物重建，可设置：

```bash
export THUNDER_PROJECT_ROOT=/absolute/path/to/雷霆医疗队
export THUNDER_EXTERNAL_DATA_ROOT=/absolute/path/to/external_data
npm run data
```

也可单独指定 `THUNDER_ASSET_ROOT` 或 `THUNDER_CLEAN_ROOT`。路径只在运行时解析，生成的前端 JSON 不保存本机绝对路径。

## 目录

```text
Global Map/
├── data_snapshot/        前端复现所需的最小派生数据
├── public/data/         由 npm run data 生成的运行时数据
├── scripts/             数据转换脚本
├── src/components/      地球仪、中国地图和诊断面板
├── src/                 React 应用入口与样式
└── package.json         运行、构建与校验命令
```

## 边界

- 地图为研究和答辩可视化，不作为行政区划或导航依据。
- 台湾省、香港特别行政区和澳门特别行政区属展示记录，不参与大陆 31 省统计。
- 模块 D 的最强表述是“严格准因果强候选”，不声明随机实验级因果。
