import { useState } from 'react'
import { Calculator, Crosshair, Gauge, RotateCcw, TrendingDown, TrendingUp } from 'lucide-react'
import { AnimatedNumber } from './AnimatedNumber.jsx'

const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0))
const pct = (value) => Math.round(clamp01(value) * 100)

const getCountryName = (country) => country?.nameZh || country?.name || country?.iso3 || '未知国家'

const countryToScenario = (country) => ({
  label: getCountryName(country),
  burden: clamp01(country?.burdenPressureScore),
  risk: clamp01(country?.riskPressureScore),
  response: clamp01(country?.responseScore),
  policy: clamp01(country?.resourceResponseScore ?? country?.responseScore),
})

const provinceToScenario = (province) => ({
  label: province?.province || '未知省份',
  burden: clamp01(province?.burdenPressureScore),
  risk: clamp01(province?.riskExposureScore),
  response: clamp01(province?.responseScore),
  policy: clamp01(province?.policyScore ?? province?.responseScore),
})

const scenarioFromSource = (sourceKind, sourceId, countries, provinces, selectedCountry, selectedProvince) => {
  if (sourceKind === 'province') {
    const province = provinces.find((item) => item.province === sourceId) ?? selectedProvince ?? provinces[0]
    return provinceToScenario(province)
  }
  if (sourceKind === 'custom') {
    return { label: '自定义情景', burden: 0.62, risk: 0.58, response: 0.48, policy: 0.42 }
  }
  const country = countries.find((item) => item.iso3 === sourceId) ?? selectedCountry ?? countries[0]
  return countryToScenario(country)
}

const calculateScenario = (inputs) => {
  const burden = clamp01(inputs.burden)
  const risk = clamp01(inputs.risk)
  const response = clamp01(inputs.response)
  const policy = clamp01(inputs.policy)
  const pressure = clamp01(burden * 0.58 + risk * 0.42)
  const responsePower = clamp01(response * 0.72 + policy * 0.28)
  const gap = clamp01(Math.max(0, pressure - responsePower))

  let type = '相对均衡型'
  if (pressure >= 0.58 && responsePower < 0.5) type = '高压低响应型'
  else if (pressure >= 0.58 && responsePower >= 0.5) type = '高压高响应型'
  else if (pressure < 0.45 && responsePower < 0.5) type = '低压低响应型'

  const boostedGap = clamp01(Math.max(0, pressure - clamp01(responsePower + 0.1)))
  const riskDownGap = clamp01(Math.max(0, clamp01((burden * 0.58 + clamp01(risk - 0.1) * 0.42)) - responsePower))

  return {
    burden,
    risk,
    response,
    policy,
    pressure,
    responsePower,
    gap,
    type,
    responseBoostDrop: Math.max(0, gap - boostedGap),
    riskCutDrop: Math.max(0, gap - riskDownGap),
  }
}

const adviceForScenario = (result) => {
  if (result.type === '高压低响应型') {
    return '优先补基层服务、慢病筛查和护理承载能力，再叠加控烟、控盐、血压血糖管理等风险治理。'
  }
  if (result.type === '高压高响应型') {
    return '重点从“增量投入”转向“精准治理”，围绕高血压、吸烟、代谢风险做高强度干预。'
  }
  if (result.type === '低压低响应型') {
    return '当前压力不高但响应基础偏弱，适合先补资源底座、服务覆盖和政策执行能力。'
  }
  return '压力和响应整体接近，但仍要盯住局部风险短板，保持慢病管理和资源配置的动态校准。'
}

const recordVector = (record, kind) => {
  if (kind === 'province') {
    return {
      id: record.province,
      label: record.province,
      type: record.abcdType || record.resourceType,
      burden: clamp01(record.burdenPressureScore),
      risk: clamp01(record.riskExposureScore),
      response: clamp01(record.responseScore),
      policy: clamp01(record.policyScore ?? record.responseScore),
    }
  }
  return {
    id: record.iso3,
    label: getCountryName(record),
    type: record.vulnerabilityTypeLabel,
    burden: clamp01(record.burdenPressureScore),
    risk: clamp01(record.riskPressureScore),
    response: clamp01(record.responseScore),
    policy: clamp01(record.resourceResponseScore ?? record.responseScore),
  }
}

const similarityScore = (result, vector) => {
  const distance = Math.sqrt(
    (result.burden - vector.burden) ** 2 * 0.26 +
    (result.risk - vector.risk) ** 2 * 0.24 +
    (result.response - vector.response) ** 2 * 0.32 +
    (result.policy - vector.policy) ** 2 * 0.18,
  )
  return clamp01(1 - distance)
}

const nearestReferences = (result, records, kind, excludeId = '') =>
  records
    .filter((record) => !record.displayOnly)
    .map((record) => {
      const vector = recordVector(record, kind)
      return {
        id: vector.id,
        label: vector.label,
        type: vector.type,
        score: similarityScore(result, vector),
      }
    })
    .filter((item) => item.id !== excludeId)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)

function SliderRow({ label, value, onChange, tone }) {
  return (
    <label className="scenario-slider">
      <span>{label}</span>
      <input
        type="range"
        min="0"
        max="100"
        value={pct(value)}
        onChange={(event) => onChange(Number(event.target.value) / 100)}
      />
      <strong className={tone}>{pct(value)}</strong>
    </label>
  )
}

function ReferenceList({ title, items }) {
  return (
    <section className="scenario-reference">
      <h4>{title}</h4>
      <div>
        {items.map((item) => (
          <article key={`${title}-${item.label}`}>
            <span>{item.label}</span>
            <strong>{pct(item.score)}</strong>
            <em>{item.type}</em>
          </article>
        ))}
      </div>
    </section>
  )
}

function ScenarioQuadrant({ result }) {
  const x = pct(result.responsePower)
  const y = pct(result.pressure)
  return (
    <div className="scenario-quadrant" aria-label={`压力 ${y}，响应 ${x}`}>
      <span className="quadrant-label high-pressure">高压力</span>
      <span className="quadrant-label low-pressure">低压力</span>
      <span className="quadrant-label low-response">低响应</span>
      <span className="quadrant-label high-response">高响应</span>
      <i className="threshold pressure" />
      <i className="threshold response" />
      <b className="scenario-dot" style={{ left: `${x}%`, bottom: `${y}%` }} />
    </div>
  )
}

export function ScenarioSimulator({ countries = [], provinces = [], selectedCountry, selectedProvince }) {
  const realProvinces = provinces.filter((province) => !province.displayOnly)
  const defaultCountryId = selectedCountry?.iso3 ?? countries[0]?.iso3 ?? ''
  const defaultProvinceId = selectedProvince?.province ?? realProvinces[0]?.province ?? ''
  const [sourceKind, setSourceKind] = useState('country')
  const [sourceId, setSourceId] = useState(defaultCountryId)
  const [inputs, setInputs] = useState(() =>
    scenarioFromSource('country', defaultCountryId, countries, realProvinces, selectedCountry, selectedProvince),
  )

  const result = calculateScenario(inputs)
  const similarCountries = nearestReferences(result, countries, 'country', sourceKind === 'country' ? sourceId : '')
  const similarProvinces = nearestReferences(result, realProvinces, 'province', sourceKind === 'province' ? sourceId : '')

  const loadSource = (nextKind, nextId = '') => {
    const resolvedId =
      nextKind === 'province'
        ? nextId || defaultProvinceId
        : nextKind === 'country'
          ? nextId || defaultCountryId
          : ''
    setSourceKind(nextKind)
    setSourceId(resolvedId)
    setInputs(scenarioFromSource(nextKind, resolvedId, countries, realProvinces, selectedCountry, selectedProvince))
  }

  const updateInput = (field, value) => {
    setInputs((current) => ({ ...current, [field]: clamp01(value), label: sourceKind === 'custom' ? '自定义情景' : current.label }))
  }

  return (
    <section className="scenario-simulator" aria-label="政策适配计算器">
      <div className="scenario-head">
        <div>
          <span>Scenario Simulator</span>
          <h3>政策适配计算器</h3>
        </div>
        <Calculator size={20} />
      </div>

      <div className="scenario-source-tabs" aria-label="情景来源">
        <button type="button" className={sourceKind === 'country' ? 'active' : ''} onClick={() => loadSource('country')}>
          国家
        </button>
        <button type="button" className={sourceKind === 'province' ? 'active' : ''} onClick={() => loadSource('province')}>
          省份
        </button>
        <button type="button" className={sourceKind === 'custom' ? 'active' : ''} onClick={() => loadSource('custom')}>
          自定义
        </button>
      </div>

      {sourceKind !== 'custom' && (
        <select
          className="scenario-select"
          value={sourceId}
          onChange={(event) => loadSource(sourceKind, event.target.value)}
          aria-label={sourceKind === 'province' ? '选择中国省份' : '选择国家'}
        >
          {(sourceKind === 'province' ? realProvinces : countries).map((item) => {
            const value = sourceKind === 'province' ? item.province : item.iso3
            const label = sourceKind === 'province' ? item.province : getCountryName(item)
            return (
              <option key={value} value={value}>
                {label}
              </option>
            )
          })}
        </select>
      )}

      <div className="scenario-sliders">
        <SliderRow label="疾病负担" value={inputs.burden} tone="coral" onChange={(value) => updateInput('burden', value)} />
        <SliderRow label="风险暴露" value={inputs.risk} tone="amber" onChange={(value) => updateInput('risk', value)} />
        <SliderRow label="响应能力" value={inputs.response} tone="teal" onChange={(value) => updateInput('response', value)} />
        <SliderRow label="政策强度" value={inputs.policy} tone="ink" onChange={(value) => updateInput('policy', value)} />
      </div>

      <div className="scenario-result-card">
        <div className="scenario-result-title">
          <Gauge size={17} />
          <span>{inputs.label}</span>
        </div>
        <strong>{result.type}</strong>
        <div className="scenario-metrics">
          <span>压力 <AnimatedNumber value={pct(result.pressure)} /></span>
          <span>响应 <AnimatedNumber value={pct(result.responsePower)} /></span>
          <span>缺口 <AnimatedNumber value={pct(result.gap)} /></span>
        </div>
        <ScenarioQuadrant result={result} />
        <p>{adviceForScenario(result)}</p>
      </div>

      <div className="scenario-actions">
        <button type="button" onClick={() => updateInput('response', inputs.response + 0.1)}>
          <TrendingUp size={15} />
          响应+10
        </button>
        <button type="button" onClick={() => updateInput('risk', inputs.risk - 0.1)}>
          <TrendingDown size={15} />
          风险-10
        </button>
        <button type="button" onClick={() => loadSource(sourceKind, sourceId)}>
          <RotateCcw size={15} />
          恢复
        </button>
      </div>

      <div className="scenario-impact">
        <Crosshair size={15} />
        <span>
          响应提升10分，缺口约下降 <AnimatedNumber value={pct(result.responseBoostDrop)} /> 分；风险下降10分，缺口约下降 <AnimatedNumber value={pct(result.riskCutDrop)} /> 分。
        </span>
      </div>

      <div className="scenario-reference-grid">
        <ReferenceList title="相似国家" items={similarCountries} />
        <ReferenceList title="相似省份" items={similarProvinces} />
      </div>
    </section>
  )
}
