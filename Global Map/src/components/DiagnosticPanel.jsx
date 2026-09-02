import { BarChart3, ClipboardCheck, Gauge, ListChecks, MapPinned, MoveUpRight } from 'lucide-react'
import { AnimatedNumber } from './AnimatedNumber.jsx'
import { ScenarioSimulator } from './ScenarioSimulator.jsx'

const percentNumber = (value) => {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? Math.max(0, Math.min(100, Math.round(parsed * 100))) : null
}
const pct = (value) => percentNumber(value) ?? 'NA'
const fixed = (value, digits = 1) => (value == null ? 'NA' : Number(value).toFixed(digits))

function MetricBar({ label, value, tone = 'teal' }) {
  const percent = percentNumber(value)
  return (
    <div className="metric-bar">
      <div>
        <span>{label}</span>
        <strong>{percent ?? 'NA'}</strong>
      </div>
      <i>
        <b className={tone} style={{ width: `${percent ?? 0}%` }} />
      </i>
    </div>
  )
}

function MiniBars3D({ bars }) {
  return (
    <div className="mini-3d-chart" aria-label="立体指标柱状图">
      {bars.map((bar) => (
        <div className="bar-slot" key={bar.label}>
          <div className="bar-space">
            <span
              className={`bar-prism ${bar.tone}`}
              style={{ height: `${Math.max(8, percentNumber(bar.value) ?? 0)}%` }}
            />
          </div>
          <small>{bar.label}</small>
        </div>
      ))}
    </div>
  )
}

function PillList({ items }) {
  if (!items?.length) return <span className="empty-note">暂无匹配</span>
  return (
    <div className="pill-list">
      {items.map((item) => (
        <span key={item}>{item}</span>
      ))}
    </div>
  )
}

function ChinaReferenceBlock({ country, similarProvinces }) {
  const isChina = country?.iso3 === 'CHN'
  return (
    <section className="diagnostic-section similarity-section">
      <h3><MapPinned size={16} />中国相似参照</h3>
      {isChina ? (
        <div className="reference-guide">
          <strong>点击任意非中国国家</strong>
          <p>这里会按压力、负担、风险、响应和缺口结构，显示最相似的中国省份。</p>
        </div>
      ) : similarProvinces?.length ? (
        <div className="similarity-list">
          {similarProvinces.map((item) => (
            <article className="similarity-card" key={item.province}>
              <div>
                <strong>{item.province}</strong>
                <span>{item.abcdType}</span>
              </div>
              <AnimatedNumber value={pct(item.score)} className="similarity-score" />
              <i className="similarity-meter" aria-hidden="true">
                <b style={{ width: `${pct(item.score)}%` }} />
              </i>
              <p>{item.reason}</p>
              <em>{item.policyPriority}</em>
            </article>
          ))}
        </div>
      ) : (
        <p>暂无可比省级样本。</p>
      )}
    </section>
  )
}

function CountryBlock({ country, similarProvinces }) {
  if (!country) return null
  return (
    <>
      <div className="diagnostic-title">
        <div>
          <span>Selected Country</span>
          <h2>{country.nameZh || country.name}</h2>
        </div>
        <strong>{country.iso3}</strong>
      </div>

      <div className="type-banner type-global">
        <MoveUpRight size={18} />
        <span>{country.vulnerabilityTypeLabel}</span>
      </div>

      <MiniBars3D
        bars={[
          { label: '负担', value: country.burdenPressureScore, tone: 'coral' },
          { label: '风险', value: country.riskPressureScore, tone: 'amber' },
          { label: '响应', value: country.responseScore, tone: 'teal' },
          { label: '缺口', value: country.adaptationGapScore, tone: 'ink' },
        ]}
      />

      <div className="metric-stack">
        <MetricBar label="综合压力" value={country.pressureScore} tone="coral" />
        <MetricBar label="响应能力" value={country.responseScore} tone="teal" />
        <MetricBar label="适配缺口" value={country.adaptationGapScore} tone="amber" />
      </div>

      <section className="diagnostic-section">
        <h3><Gauge size={16} />响应状态</h3>
        <p>{country.responseDiagnosisType}</p>
      </section>

      {country.iso3 !== 'CHN' && (
        <section className="diagnostic-section transfer-bridge">
          <h3><MapPinned size={16} />迁移解释</h3>
          <p>系统会把该国的压力、风险、响应和缺口向量，与中国 31 个省级样本做结构距离匹配。</p>
        </section>
      )}

      <section className="diagnostic-section">
        <h3><BarChart3 size={16} />主要风险</h3>
        <PillList
          items={
            country.dominantRiskTriplet?.length
              ? country.dominantRiskTriplet
              : country.typeTopRisks?.map((risk) => risk.label)
          }
        />
      </section>

      <ChinaReferenceBlock country={country} similarProvinces={similarProvinces} />

      <section className="diagnostic-section">
        <h3><ClipboardCheck size={16} />政策路径</h3>
        <p>{country.policyPathway.package}</p>
        <em>{country.policyPathway.chinaTranslation}</em>
      </section>
    </>
  )
}

function ProvinceBlock({ province }) {
  if (!province) return null
  return (
    <>
      <div className="diagnostic-title">
        <div>
          <span>Selected Province</span>
          <h2>{province.province}</h2>
        </div>
        <strong>2024</strong>
      </div>

      <div className="type-banner type-china">
        <MapPinned size={18} />
        <span>{province.abcdType || province.resourceType}</span>
      </div>

      {province.displayOnly && (
        <section className="diagnostic-section display-caveat">
          <h3><MapPinned size={16} />展示口径</h3>
          <p>{province.cardText}</p>
          <em>{province.dataCaveat}</em>
        </section>
      )}

      <MiniBars3D
        bars={[
          { label: '压力', value: province.pressureScore, tone: 'coral' },
          { label: '响应', value: province.responseScore, tone: 'teal' },
          { label: '缺口', value: province.adaptationGapScore, tone: 'amber' },
          { label: '政策', value: province.policyScore, tone: 'ink' },
        ]}
      />

      <div className="metric-stack">
        <MetricBar label="健康压力" value={province.pressureScore} tone="coral" />
        <MetricBar label="响应能力" value={province.responseScore} tone="teal" />
        <MetricBar label="适配缺口" value={province.adaptationGapScore} tone="amber" />
      </div>

      <section className="diagnostic-section">
        <h3><ListChecks size={16} />优先任务</h3>
        <p>{province.policyPriority}</p>
      </section>

      <section className="diagnostic-section">
        <h3><BarChart3 size={16} />省级风险</h3>
        <PillList items={province.topRisks} />
        <em>PM2.5 {fixed(province.pm25)} μg/m3，65岁及以上占比 {fixed(province.age65)}%</em>
      </section>

      <section className="diagnostic-section">
        <h3><ClipboardCheck size={16} />建议政策包</h3>
        <p>{province.recommendedPackage}</p>
      </section>
    </>
  )
}

function PolicyBlock({ policyScorecard, countries, provinces, selectedCountry, selectedProvince }) {
  return (
    <>
      <div className="diagnostic-title">
        <div>
          <span>Module D</span>
          <h2>政策证据锁定</h2>
        </div>
        <AnimatedNumber value={policyScorecard.length} />
      </div>

      <ScenarioSimulator
        countries={countries}
        provinces={provinces}
        selectedCountry={selectedCountry}
        selectedProvince={selectedProvince}
      />

      <section className="diagnostic-section">
        <h3><ClipboardCheck size={16} />模块D证据库</h3>
        <p>下列政策证据为计算器建议口径提供支撑，不把单一政策硬说成随机实验因果。</p>
      </section>

      <div className="policy-list">
        {policyScorecard.slice(0, 5).map((policy) => (
          <article key={policy.policyId}>
            <div>
              <h3>{policy.policyLabel}</h3>
              <span>{policy.domain} / {policy.grade}</span>
            </div>
            <AnimatedNumber value={pct(policy.totalScore)} />
            <p>{policy.answerLine}</p>
          </article>
        ))}
      </div>
    </>
  )
}

export function DiagnosticPanel({
  activeView,
  country,
  province,
  similarProvinces,
  policyScorecard,
  countries,
  provinces,
}) {
  return (
    <aside className="diagnostic-panel" aria-label="诊断信息面板">
      {activeView === 'policy' ? (
        <PolicyBlock
          policyScorecard={policyScorecard}
          countries={countries}
          provinces={provinces}
          selectedCountry={country}
          selectedProvince={province}
        />
      ) : activeView === 'china' ? (
        <ProvinceBlock province={province} />
      ) : (
        <CountryBlock country={country} similarProvinces={similarProvinces} />
      )}
    </aside>
  )
}
