import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  ArrowRight,
  Crosshair,
  Globe2,
  Layers3,
  MapPinned,
  MousePointerClick,
  Route,
  ShieldCheck,
} from 'lucide-react'
import { feature } from 'topojson-client'
import countriesTopo from 'world-atlas/countries-110m.json'
import { AnimatedNumber } from './components/AnimatedNumber.jsx'
import { ChinaMap } from './components/ChinaMap.jsx'
import { DiagnosticPanel } from './components/DiagnosticPanel.jsx'
import { EvidenceRail } from './components/EvidenceRail.jsx'
import { GlobalGlobe } from './components/GlobalGlobe.jsx'
import './App.css'

const TYPE_COLORS = {
  1: '#d94d61',
  2: '#f2b84b',
  3: '#2fa996',
  missing: '#d8dedc',
}

const VIEW_TABS = [
  { key: 'global', label: '全球画像', icon: Globe2 },
  { key: 'china', label: '中国映射', icon: MapPinned },
  { key: 'policy', label: '政策模拟', icon: Route },
]

const GLOBE_PRESETS = [
  { key: 'overview', label: '全球总览', view: 'global' },
  { key: 'highBurden', label: '高负担国家', view: 'global' },
  { key: 'lowBuffer', label: '低缓冲国家', view: 'global' },
  { key: 'china', label: '中国周边', view: 'china' },
  { key: 'policy', label: '政策样板', view: 'policy' },
]

const WORKFLOW_HINTS = [
  '点国家看脆弱性',
  '看中国相似省份',
  '调政策情景',
]

const SIMILARITY_FIELDS = [
  { country: 'pressureScore', province: 'pressureScore', weight: 0.24, label: '综合压力' },
  { country: 'burdenPressureScore', province: 'burdenPressureScore', weight: 0.18, label: '疾病负担' },
  { country: 'riskPressureScore', province: 'riskExposureScore', weight: 0.18, label: '风险暴露' },
  { country: 'responseScore', province: 'responseScore', weight: 0.24, label: '响应能力' },
  { country: 'adaptationGapScore', province: 'adaptationGapScore', weight: 0.16, label: '适配缺口' },
]

const isFiniteNumber = (value) => Number.isFinite(Number(value))

const clamp01 = (value) => Math.max(0, Math.min(1, value))

const pickRiskLabels = (item, fields) => {
  for (const field of fields) {
    if (Array.isArray(item?.[field]) && item[field].length) return item[field]
  }
  return []
}

const formatSimilarityReason = (country, province, matchedFields, sharedRisks) => {
  const riskText = sharedRisks.length ? `共同风险：${sharedRisks.join('、')}` : '风险结构接近'
  const fieldText = matchedFields.length
    ? `${matchedFields.slice(0, 2).join('、')}最接近`
    : '核心指标距离较近'
  return `${fieldText}；${riskText}`
}

const buildChinaSimilarityReferences = (country, provinces) => {
  if (!country || country.iso3 === 'CHN' || !Array.isArray(provinces)) return []

  const countryRisks = pickRiskLabels(country, ['dominantRiskTriplet', 'topRisks'])
  return provinces
    .filter((province) => !province.displayOnly)
    .map((province) => {
      const usableFields = SIMILARITY_FIELDS.filter(
        (field) => isFiniteNumber(country[field.country]) && isFiniteNumber(province[field.province]),
      )
      if (!usableFields.length) return null

      const weightTotal = usableFields.reduce((sum, field) => sum + field.weight, 0)
      const normalizedDistance = Math.sqrt(
        usableFields.reduce((sum, field) => {
          const diff = Number(country[field.country]) - Number(province[field.province])
          return sum + (field.weight / weightTotal) * diff * diff
        }, 0),
      )

      const provinceRisks = pickRiskLabels(province, ['topRisks', 'dominantRiskTriplet'])
      const sharedRisks = countryRisks.filter((risk) => provinceRisks.includes(risk))
      const riskOverlap = countryRisks.length ? sharedRisks.length / countryRisks.length : 0
      const score = clamp01((1 - normalizedDistance) * 0.88 + riskOverlap * 0.12)
      const matchedFields = usableFields
        .map((field) => ({
          label: field.label,
          distance: Math.abs(Number(country[field.country]) - Number(province[field.province])),
        }))
        .sort((a, b) => a.distance - b.distance)
        .map((field) => field.label)

      return {
        province: province.province,
        abcdType: province.abcdType || province.resourceType,
        score,
        sharedRisks,
        reason: formatSimilarityReason(country, province, matchedFields, sharedRisks),
        policyPriority: province.policyPriority,
      }
    })
    .filter(Boolean)
    .sort((a, b) => b.score - a.score)
    .slice(0, 4)
}

function App() {
  const [mapData, setMapData] = useState(null)
  const [chinaGeo, setChinaGeo] = useState(null)
  const [activeView, setActiveView] = useState('global')
  const [selectedCountryIso3, setSelectedCountryIso3] = useState('CHN')
  const [selectedProvinceName, setSelectedProvinceName] = useState('')
  const [globeReady, setGlobeReady] = useState(false)
  const [globeFocus, setGlobeFocus] = useState('overview')
  const [activeEvidenceStep, setActiveEvidenceStep] = useState('global')

  useEffect(() => {
    Promise.all([
      fetch('/data/global-health-map.json').then((res) => res.json()),
      fetch('/data/china-provinces.geojson').then((res) => res.json()),
    ]).then(([payload, geojson]) => {
      setMapData(payload)
      setChinaGeo(geojson)
      setSelectedProvinceName(payload.china.summary.topGapProvinces[0])
    })
  }, [])

  const countryByNumeric = useMemo(() => {
    if (!mapData) return new Map()
    return new Map(mapData.global.countries.map((country) => [country.isoNumeric, country]))
  }, [mapData])

  const countryByIso = useMemo(() => {
    if (!mapData) return new Map()
    return new Map(mapData.global.countries.map((country) => [country.iso3, country]))
  }, [mapData])

  const countryFeatures = useMemo(() => {
    const features = feature(countriesTopo, countriesTopo.objects.countries).features
    return features.map((item) => ({
      ...item,
      properties: {
        ...item.properties,
        appData: countryByNumeric.get(item.id),
      },
    }))
  }, [countryByNumeric])

  const selectedCountry = countryByIso.get(selectedCountryIso3) ?? countryByIso.get('CHN')

  const provinceByName = useMemo(() => {
    if (!mapData) return new Map()
    return new Map(mapData.china.provinces.map((province) => [province.province, province]))
  }, [mapData])

  const selectedProvince =
    provinceByName.get(selectedProvinceName) ??
    provinceByName.get(mapData?.china.summary.topGapProvinces?.[0])

  const similarChineseProvinces = useMemo(
    () => buildChinaSimilarityReferences(selectedCountry, mapData?.china.provinces),
    [selectedCountry, mapData],
  )

  const selectedCountryLabel = selectedCountry?.nameZh || selectedCountry?.name || selectedCountry?.iso3
  const selectedCountrySummary = selectedCountry?.iso3 === 'CHN'
    ? '当前进入中国省级映射，右侧显示省级适配缺口和治理建议。'
    : `已生成 ${similarChineseProvinces.length} 个中国省级相似参照，可继续切到政策模拟。`

  const handleCountrySelect = (country) => {
    if (!country) return
    setSelectedCountryIso3(country.iso3)
    setGlobeFocus('selected')
    if (country.iso3 === 'CHN') {
      setActiveView('china')
      setActiveEvidenceStep('china')
    } else {
      setActiveView('global')
      setActiveEvidenceStep('global')
    }
  }

  const handleProvinceSelect = (province) => {
    setSelectedProvinceName(province.province)
    setSelectedCountryIso3('CHN')
    setGlobeFocus('china')
    setActiveView('china')
    setActiveEvidenceStep('china')
  }

  const handleViewChange = (view) => {
    setActiveView(view)
    if (view === 'china') {
      setSelectedCountryIso3('CHN')
      setGlobeFocus('china')
      setActiveEvidenceStep('china')
      return
    }
    if (view === 'policy') {
      setGlobeFocus('policy')
      setActiveEvidenceStep('policy')
      return
    }
    setGlobeFocus('overview')
    setActiveEvidenceStep('global')
  }

  const handlePresetSelect = (preset) => {
    setGlobeFocus(preset.key)
    setActiveView(preset.view)
    setActiveEvidenceStep(
      preset.key === 'policy' ? 'policy' : preset.key === 'china' ? 'china' : preset.key === 'overview' ? 'global' : 'risk',
    )
    if (preset.key === 'china') setSelectedCountryIso3('CHN')
  }

  const handleEvidenceStep = (step) => {
    const stepFocus = {
      foundation: 'overview',
      global: 'overview',
      risk: 'highBurden',
      policy: 'policy',
      china: 'china',
    }
    const stepView = {
      policy: 'policy',
      china: 'china',
    }
    const nextFocus = stepFocus[step] ?? 'overview'
    setActiveEvidenceStep(step)
    setGlobeFocus(nextFocus)
    setActiveView(stepView[step] ?? 'global')
    if (step === 'china') setSelectedCountryIso3('CHN')
  }

  if (!mapData || !chinaGeo) {
    return (
      <main className="loading-screen">
        <div className="loading-mark">
          <Globe2 size={34} />
        </div>
        <h1>全球健康转型可视化正在载入</h1>
        <p>正在读取 193 个国家与 31 个省级诊断结果</p>
      </main>
    )
  }

  return (
    <main className={`app-shell view-${activeView}`}>
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark">
            <Activity size={22} />
          </div>
          <div>
            <h1>雷霆医疗队 Global Map</h1>
            <p>全球健康转型中的暴露-负担-响应失配诊断</p>
          </div>
        </div>

        <nav className="view-tabs" aria-label="可视化视图">
          {VIEW_TABS.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              type="button"
              className={activeView === key ? 'active' : ''}
              onClick={() => handleViewChange(key)}
              title={label}
            >
              <Icon size={17} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="status-pill">
          <ShieldCheck size={17} />
          <span>模块D证据锁定</span>
        </div>
      </header>

      <section className="demo-control-strip" aria-label="演示视角控制">
        <div>
          <span>演示路径</span>
          <strong>全球证据如何落到中国省级政策建议</strong>
        </div>
        <div className="workflow-hints" aria-label="演示链路">
          {WORKFLOW_HINTS.map((hint, index) => (
            <span key={hint}>
              <b>{index + 1}</b>
              {hint}
            </span>
          ))}
        </div>
        <div className="globe-preset-bar" aria-label="地球仪视角">
          {GLOBE_PRESETS.map((preset) => (
            <button
              key={preset.key}
              type="button"
              className={globeFocus === preset.key ? 'active' : ''}
              onClick={() => handlePresetSelect(preset)}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </section>

      <section className="dashboard-grid">
        <section className="globe-pane" aria-label="全球健康脆弱性地球仪">
          <div className="pane-header">
            <div>
              <span className="section-kicker">Global Evidence</span>
              <h2>全球脆弱性类型与政策路径</h2>
            </div>
            <div className="legend-row">
              <span><i style={{ background: TYPE_COLORS[1] }} />高负担转型承压型</span>
              <span><i style={{ background: TYPE_COLORS[2] }} />低缓冲低承载型</span>
              <span><i style={{ background: TYPE_COLORS[3] }} />相对稳健型</span>
            </div>
          </div>

          <GlobalGlobe
            features={countryFeatures}
            selectedIso3={selectedCountry?.iso3}
            focusMode={globeFocus}
            typeColors={TYPE_COLORS}
            onSelectCountry={handleCountrySelect}
            onReady={() => setGlobeReady(true)}
          />

          <div className="globe-insight-card">
            <div>
              <MousePointerClick size={16} />
              <span>当前选中</span>
            </div>
            <strong>{selectedCountryLabel}</strong>
            <p>{selectedCountrySummary}</p>
          </div>

          <div className="globe-overlay">
            <div className="metric-tile">
              <AnimatedNumber value={mapData.global.countries.length} />
              <span>国家样本</span>
            </div>
            <div className="metric-tile">
              <AnimatedNumber value={mapData.global.summaryByType.find((item) => item.code === 1)?.count ?? 0} />
              <span>高负担转型承压型</span>
            </div>
            <div className="metric-tile">
              <AnimatedNumber value={globeReady ? 5 : 0} />
              <span>演示视角</span>
            </div>
          </div>
        </section>

        <section className="china-pane" aria-label="中国省级映射">
          <div className="pane-header compact">
            <div>
              <span className="section-kicker">China Transfer</span>
              <h2>中国省级适配缺口映射</h2>
            </div>
            <button type="button" className="focus-china" onClick={() => handleViewChange('china')}>
              <MapPinned size={17} />
              省级模式
            </button>
          </div>

          <ChinaMap
            geojson={chinaGeo}
            provinces={mapData.china.provinces}
            selectedProvince={selectedProvince?.province}
            active={activeView === 'china'}
            onSelectProvince={handleProvinceSelect}
          />

          <div className="china-summary">
            <div>
              <AnimatedNumber value={mapData.china.summary.provinceCount} />
              <span>省级样本</span>
            </div>
            <div>
              <AnimatedNumber value={Math.round(mapData.china.summary.avgGap * 100)} />
              <span>平均缺口指数</span>
            </div>
            <div>
              <strong>{mapData.china.summary.topGapProvinces[0]}</strong>
              <span>首要关注省份</span>
            </div>
          </div>
        </section>

        <DiagnosticPanel
          activeView={activeView}
          country={selectedCountry}
          province={selectedProvince}
          similarProvinces={similarChineseProvinces}
          policyScorecard={mapData.global.topPolicyScorecard}
          countries={mapData.global.countries}
          provinces={mapData.china.provinces}
        />
      </section>

      <EvidenceRail activeStep={activeEvidenceStep} onSelectStep={handleEvidenceStep} />

      <section className="linkage-bar">
        <div>
          <Layers3 size={18} />
          <span>全球类型</span>
        </div>
        <ArrowRight size={17} />
        <div>
          <Crosshair size={18} />
          <span>风险归因</span>
        </div>
        <ArrowRight size={17} />
        <div>
          <Route size={18} />
          <span>政策路径</span>
        </div>
        <ArrowRight size={17} />
        <div>
          <MapPinned size={18} />
          <span>中国省级落地</span>
        </div>
      </section>
    </main>
  )
}

export default App
