import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { parse } from 'csv-parse/sync'
import countries from 'i18n-iso-countries'

const APP_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const PROJECT_ROOT = path.resolve(process.env.THUNDER_PROJECT_ROOT || path.join(APP_ROOT, '..', '雷霆医疗队'))
const ASSET_ROOT = path.resolve(process.env.THUNDER_ASSET_ROOT || path.join(PROJECT_ROOT, '06_report_assets'))
const CLEAN_ROOT = path.resolve(process.env.THUNDER_CLEAN_ROOT || path.join(PROJECT_ROOT, '09_data_clean'))
const EXTERNAL_ROOT = path.resolve(process.env.THUNDER_EXTERNAL_DATA_ROOT || path.join(APP_ROOT, '..', '..', 'external_data'))
const SNAPSHOT_ROOT = path.join(APP_ROOT, 'data_snapshot')
const DATA_OUT = path.join(APP_ROOT, 'public', 'data')

const firstExisting = (candidates, label) => {
  const match = candidates.find((candidate) => fs.existsSync(candidate))
  if (match) return match
  throw new Error(`Missing ${label}. Checked:\n${candidates.map((candidate) => `- ${candidate}`).join('\n')}`)
}

const sourceFile = (primary, snapshotName = path.basename(primary)) =>
  firstExisting([primary, path.join(SNAPSHOT_ROOT, snapshotName)], snapshotName)

const files = {
  countryNames: sourceFile(path.join(CLEAN_ROOT, 'external_country.csv')),
  countryZh: sourceFile(path.join(CLEAN_ROOT, 'external_country_alias_zh.csv')),
  vulnerability: sourceFile(path.join(ASSET_ROOT, 'b_global_abcd', 'vulnerability_country_labels_latest.csv')),
  response: sourceFile(path.join(ASSET_ROOT, 'b_global_abcd', 'country_response_diagnosis_latest.csv')),
  riskByType: sourceFile(path.join(ASSET_ROOT, 'b_global_abcd', 'risk_attribution_type_risk_summary.csv')),
  pathways: sourceFile(path.join(ASSET_ROOT, 'c_module_d_policy', 'policy_response_pathways.csv')),
  policyScorecard: sourceFile(path.join(ASSET_ROOT, 'c_module_d_policy', 'policy_d6_extreme_quasi_causal_scorecard.csv')),
  lockStatus: sourceFile(path.join(ASSET_ROOT, 'c_module_d_policy', 'module_d_lock_status.json')),
  chinaCards: firstExisting(
    [
      path.join(ASSET_ROOT, 'd_china_mapping', 'china_provincial_policy_cards.csv'),
      path.join(ASSET_ROOT, 'd_china_mapping', 'china_planb_provincial_policy_cards.csv'),
      path.join(SNAPSHOT_ROOT, 'china_provincial_policy_cards.csv'),
    ],
    'china_provincial_policy_cards.csv',
  ),
  chinaMapping: sourceFile(path.join(ASSET_ROOT, 'd_china_mapping', 'china_abcd_provincial_mapping_v2.csv')),
  chinaPolicy: sourceFile(path.join(ASSET_ROOT, 'd_china_mapping', 'china_provincial_policy_response_score.csv')),
  chinaGeo: firstExisting(
    [
      path.join(EXTERNAL_ROOT, '10_China_GeoJSON', 'china_geojson', 'china_provinces_100000_full.geojson'),
      path.join(SNAPSHOT_ROOT, 'china-provinces.geojson'),
    ],
    'china-provinces.geojson',
  ),
}

const readCsv = (file) => {
  const text = fs.readFileSync(file, 'utf8')
  return parse(text, { columns: true, skip_empty_lines: true, bom: true })
}

const readJson = (file, fallback = {}) => {
  if (!fs.existsSync(file)) return fallback
  return JSON.parse(fs.readFileSync(file, 'utf8'))
}

const num = (value, fallback = null) => {
  if (value === undefined || value === null || value === '') return fallback
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

const text = (value, fallback = '') => {
  if (value === undefined || value === null) return fallback
  const trimmed = String(value).trim()
  return trimmed || fallback
}

const clamp01 = (value) => {
  const parsed = num(value, 0)
  return Math.max(0, Math.min(1, parsed))
}

const splitList = (value) =>
  text(value)
    .split(/[/、,，;；]+/)
    .map((item) => item.trim())
    .filter(Boolean)

const topN = (rows, count, scoreField) =>
  [...rows]
    .sort((a, b) => num(b[scoreField], 0) - num(a[scoreField], 0))
    .slice(0, count)

fs.mkdirSync(DATA_OUT, { recursive: true })

const countryNames = new Map()
for (const row of readCsv(files.countryNames)) {
  countryNames.set(row.code, row.title)
}

const countryZh = new Map()
for (const row of readCsv(files.countryZh)) {
  if (!countryZh.has(row.iso3)) countryZh.set(row.iso3, row.country_name_zh)
}

const vulnerabilityRows = readCsv(files.vulnerability)
const responseRows = readCsv(files.response)
const pathwayRows = readCsv(files.pathways)
const riskRows = readCsv(files.riskByType)
const scorecardRows = readCsv(files.policyScorecard)
const lockStatusRaw = readJson(files.lockStatus, {})
const { project_root: _projectRoot, output_files: outputFiles = {}, ...lockStatusBody } = lockStatusRaw
const lockStatus = {
  ...lockStatusBody,
  output_files: Object.fromEntries(
    Object.entries(outputFiles).map(([key, value]) => [key, path.basename(String(value))]),
  ),
}

const vulnerabilityByIso = new Map(vulnerabilityRows.map((row) => [row.iso3, row]))
const responseByIso = new Map(responseRows.map((row) => [row.iso3, row]))

const pathwayByTypeAndResponse = new Map()
const pathwayByType = new Map()
for (const row of topN(pathwayRows, pathwayRows.length, 'priority_score')) {
  const type = text(row.vulnerability_type_label)
  const responseType = text(row.response_diagnosis_type)
  const exactKey = `${type}__${responseType}`
  if (!pathwayByTypeAndResponse.has(exactKey)) pathwayByTypeAndResponse.set(exactKey, row)
  if (!pathwayByType.has(type)) pathwayByType.set(type, row)
}

const riskByType = new Map()
for (const row of riskRows) {
  const type = text(row.vulnerability_type_label)
  if (!riskByType.has(type)) riskByType.set(type, [])
  riskByType.get(type).push({
    label: text(row.risk_label),
    share: clamp01(row.contribution_share),
    rank: num(row.rank_within_type, 99),
  })
}
for (const [type, rows] of riskByType) {
  rows.sort((a, b) => a.rank - b.rank)
  riskByType.set(type, rows.slice(0, 4))
}

const countryRecords = []
for (const [iso3, vulnerability] of vulnerabilityByIso) {
  const response = responseByIso.get(iso3) ?? {}
  const isoNumericRaw = countries.alpha3ToNumeric(iso3)
  if (!isoNumericRaw) continue
  const isoNumeric = String(isoNumericRaw).padStart(3, '0')

  const typeLabel = text(vulnerability.vulnerability_type_label, text(response.vulnerability_type_label))
  const responseType = text(response.response_diagnosis_type, '未识别')
  const pathway =
    pathwayByTypeAndResponse.get(`${typeLabel}__${responseType}`) ??
    pathwayByType.get(typeLabel) ??
    {}

  countryRecords.push({
    iso3,
    isoNumeric,
    name: countryNames.get(iso3) ?? countries.getName(iso3, 'en') ?? iso3,
    nameZh: countryZh.get(iso3) ?? '',
    year: num(vulnerability.year, num(response.year, null)),
    vulnerabilityTypeCode: num(vulnerability.vulnerability_type_code, num(response.vulnerability_type_code, null)),
    vulnerabilityTypeLabel: typeLabel,
    vulnerabilityScore: num(vulnerability.overall_vulnerability_score, null),
    pressureScore: clamp01(response.combined_pressure_score),
    burdenPressureScore: clamp01(response.burden_pressure_score),
    riskPressureScore: clamp01(response.risk_pressure_score),
    responseScore: clamp01(response.adjusted_response_score ?? response.resource_response_score),
    resourceResponseScore: clamp01(response.resource_response_score),
    adaptationGapScore: clamp01(response.adaptation_gap_score),
    responseDiagnosisType: responseType,
    dominantRiskTriplet: splitList(response.dominant_risk_triplet).slice(0, 3),
    typeTopRisks: riskByType.get(typeLabel) ?? [],
    weakestResourceComponents: splitList(response.weakest_resource_components).slice(0, 3),
    policyPathway: {
      domain: text(pathway.policy_domain, '政策路径待匹配'),
      package: text(pathway.policy_package, '根据类型画像补齐响应能力'),
      target: text(pathway.target_pressure_or_gap, '压力-响应适配缺口'),
      evidence: text(pathway.evidence_basis, '全球类型与响应失配综合证据'),
      strength: text(pathway.evidence_strength, '综合证据'),
      priorityScore: num(pathway.priority_score, null),
      chinaTranslation: text(pathway.china_policy_translation, '转化为中国省级分层治理建议'),
    },
  })
}

const summaryByType = Object.values(
  countryRecords.reduce((acc, row) => {
    const key = row.vulnerabilityTypeLabel || '未识别'
    acc[key] ??= {
      label: key,
      code: row.vulnerabilityTypeCode,
      count: 0,
      avgPressure: 0,
      avgResponse: 0,
      avgGap: 0,
    }
    acc[key].count += 1
    acc[key].avgPressure += row.pressureScore
    acc[key].avgResponse += row.responseScore
    acc[key].avgGap += row.adaptationGapScore
    return acc
  }, {}),
).map((row) => ({
  ...row,
  avgPressure: row.avgPressure / row.count,
  avgResponse: row.avgResponse / row.count,
  avgGap: row.avgGap / row.count,
}))

const scorecard = topN(scorecardRows, 8, 'd6_total_score').map((row) => ({
  policyId: text(row.policy_id),
  policyLabel: text(row.policy_label),
  domain: text(row.policy_domain),
  outcome: text(row.outcome_label),
  grade: text(row.d6_grade),
  totalScore: num(row.d6_total_score, null),
  designScore: num(row.design_integrity_score, null),
  mechanismScore: num(row.mechanism_score, null),
  transferScore: num(row.china_transfer_readiness_score, null),
  answerLine: text(row.answer_line),
  boundaryNote: text(row.boundary_note),
}))

const chinaCards = readCsv(files.chinaCards)
const chinaPolicy = new Map(readCsv(files.chinaPolicy).map((row) => [row.province, row]))
const chinaMapping = new Map(readCsv(files.chinaMapping).map((row) => [row.province, row]))

const provinceRecords = chinaCards.map((row) => {
  const policy = chinaPolicy.get(row.province) ?? {}
  const mapping = chinaMapping.get(row.province) ?? {}
  return {
    province: text(row.province),
    year: num(row.year, 2024),
    abcdType: text(row.province_abcd_type, text(row.province_resource_response_scaffold_type)),
    resourceType: text(row.province_resource_response_scaffold_type, text(row.resource_scaffold_type)),
    riskType: text(row.province_risk_proxy_type),
    globalChinaProfile: text(row.global_china_profile),
    pressureScore: clamp01(row.province_combined_pressure_score),
    burdenPressureScore: clamp01(row.province_burden_pressure_score),
    riskExposureScore: clamp01(row.province_risk_exposure_score),
    responseScore: clamp01(row.province_response_score),
    adaptationGapScore: clamp01(row.province_adaptation_gap_score),
    adaptationGapPercentile: clamp01(row.province_adaptation_gap_percentile),
    policyScore: clamp01(row.province_d_policy_score),
    paymentPolicyScore: clamp01(policy.payment_reform_policy_score ?? row.province_d_payment_policy_score),
    chronicPolicyScore: clamp01(row.chronic_policy_execution_score),
    pm25: num(row.pm25_population_weighted_ug_m3, null),
    age65: num(row.census_2020_age65_plus_pct, null),
    medicalStaffPer10k: num(row.medical_staff_per_10k_population, null),
    institutionsPerMillion: num(row.medical_institutions_per_million_population, null),
    topRisks: splitList(row.national_top_risks_placeholder).slice(0, 3),
    policyPriority: text(row.province_policy_priority),
    recommendedPackage: text(row.recommended_policy_package),
    cardText: text(row.card_text),
    dataCaveat: text(row.data_caveat),
    responseDiagnosisType: text(mapping.response_diagnosis_type, text(row.province_d_policy_type)),
    policyExposureNote: text(policy.policy_exposure_note),
  }
})

const chinaDisplayAverage = {
  pressureScore: provinceRecords.reduce((sum, row) => sum + row.pressureScore, 0) / provinceRecords.length,
  responseScore: provinceRecords.reduce((sum, row) => sum + row.responseScore, 0) / provinceRecords.length,
  adaptationGapScore: provinceRecords.reduce((sum, row) => sum + row.adaptationGapScore, 0) / provinceRecords.length,
  policyScore: provinceRecords.reduce((sum, row) => sum + row.policyScore, 0) / provinceRecords.length,
}

const displayOnlyRegions = [
  {
    province: '台湾省',
    year: 2024,
    abcdType: '中国整体展示口径',
    resourceType: '纳入中国版图展示，未单独估计省级指标',
    riskType: '展示口径',
    globalChinaProfile: '与中国大陆合并进入国家层面展示',
    pressureScore: chinaDisplayAverage.pressureScore,
    burdenPressureScore: chinaDisplayAverage.pressureScore,
    riskExposureScore: chinaDisplayAverage.pressureScore,
    responseScore: chinaDisplayAverage.responseScore,
    adaptationGapScore: chinaDisplayAverage.adaptationGapScore,
    adaptationGapPercentile: chinaDisplayAverage.adaptationGapScore,
    policyScore: chinaDisplayAverage.policyScore,
    paymentPolicyScore: chinaDisplayAverage.policyScore,
    chronicPolicyScore: chinaDisplayAverage.policyScore,
    pm25: null,
    age65: null,
    medicalStaffPer10k: null,
    institutionsPerMillion: null,
    topRisks: ['纳入中国整体风险口径'],
    policyPriority: '不单独输出省级排序，作为中国整体版图展示。',
    recommendedPackage: '采用中国整体政策框架展示，不伪造台湾单独省级数据。',
    cardText: '台湾省在地图中纳入中国整体展示；当前省级指标表未单独估计台湾。',
    dataCaveat: '展示用补充记录，不参与31个省级样本统计。',
    responseDiagnosisType: '中国整体展示',
    policyExposureNote: '展示记录',
    displayOnly: true,
  },
  {
    province: '香港特别行政区',
    year: 2024,
    abcdType: '未单独估计',
    resourceType: '地图展示区域',
    pressureScore: null,
    responseScore: null,
    adaptationGapScore: null,
    policyScore: null,
    topRisks: [],
    policyPriority: '未纳入31个省级样本。',
    recommendedPackage: '不输出单独政策排序。',
    cardText: '香港特别行政区仅作为中国地图展示区域。',
    dataCaveat: '展示记录，不参与省级统计。',
    displayOnly: true,
  },
  {
    province: '澳门特别行政区',
    year: 2024,
    abcdType: '未单独估计',
    resourceType: '地图展示区域',
    pressureScore: null,
    responseScore: null,
    adaptationGapScore: null,
    policyScore: null,
    topRisks: [],
    policyPriority: '未纳入31个省级样本。',
    recommendedPackage: '不输出单独政策排序。',
    cardText: '澳门特别行政区仅作为中国地图展示区域。',
    dataCaveat: '展示记录，不参与省级统计。',
    displayOnly: true,
  },
]

const chinaSummary = {
  provinceCount: provinceRecords.length,
  avgPressure: provinceRecords.reduce((sum, row) => sum + row.pressureScore, 0) / provinceRecords.length,
  avgResponse: provinceRecords.reduce((sum, row) => sum + row.responseScore, 0) / provinceRecords.length,
  avgGap: provinceRecords.reduce((sum, row) => sum + row.adaptationGapScore, 0) / provinceRecords.length,
  topGapProvinces: topN(provinceRecords, 6, 'adaptationGapScore').map((row) => row.province),
}

const payload = {
  generatedAt: new Date().toISOString(),
  source: {
    mode: Object.values(files).some((file) => file.startsWith(SNAPSHOT_ROOT)) ? 'bundled-snapshot' : 'full-analysis-output',
    project: '雷霆医疗队',
    note: 'Global Map reads finalized analysis outputs and does not modify the Python project.',
  },
  global: {
    countries: countryRecords,
    summaryByType,
    topPolicyScorecard: scorecard,
    moduleDLockStatus: lockStatus,
  },
  china: {
    provinces: [...provinceRecords, ...displayOnlyRegions],
    summary: chinaSummary,
  },
}

fs.writeFileSync(path.join(DATA_OUT, 'global-health-map.json'), JSON.stringify(payload, null, 2))
fs.copyFileSync(files.chinaGeo, path.join(DATA_OUT, 'china-provinces.geojson'))

console.log(`wrote ${path.join(DATA_OUT, 'global-health-map.json')}`)
console.log(`copied ${path.join(DATA_OUT, 'china-provinces.geojson')}`)
console.log(`countries=${countryRecords.length}, provinces=${provinceRecords.length}`)
