import { Database, GitBranch, Map, PieChart, ShieldCheck } from 'lucide-react'

const steps = [
  {
    key: 'foundation',
    icon: Database,
    title: '数据底座',
    body: '统一国家、省级、风险、资源与政策字段',
  },
  {
    key: 'global',
    icon: Map,
    title: '全球分型',
    body: '识别三类健康脆弱性画像',
  },
  {
    key: 'risk',
    icon: PieChart,
    title: '风险归因',
    body: '定位主要暴露和疾病负担来源',
  },
  {
    key: 'policy',
    icon: ShieldCheck,
    title: '政策路径',
    body: '用准因果证据筛选可迁移政策包',
  },
  {
    key: 'china',
    icon: GitBranch,
    title: '中国落地',
    body: '映射到省级适配缺口和治理建议',
  },
]

export function EvidenceRail({ activeStep, onSelectStep }) {
  return (
    <section className="evidence-rail" aria-label="ABCDE分析闭环">
      {steps.map(({ key, icon: Icon, title, body }) => (
        <button
          type="button"
          key={key}
          className={activeStep === key ? 'active' : ''}
          onClick={() => onSelectStep(key)}
          title={title}
        >
          <Icon size={18} />
          <div>
            <h3>{title}</h3>
            <p>{body}</p>
          </div>
        </button>
      ))}
    </section>
  )
}
