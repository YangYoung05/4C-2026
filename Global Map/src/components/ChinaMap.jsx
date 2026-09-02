import { useMemo, useRef } from 'react'
import { geoPath, geoTransform } from 'd3-geo'
import { useElementSize } from './useElementSize.js'

const gapColor = (value) => {
  if (value == null) return '#d7dedb'
  if (value >= 0.66) return '#d94d61'
  if (value >= 0.46) return '#f2b84b'
  return '#2fa996'
}

const displayRegionFill = '#edf4f1'

const DISPLAY_REGION_LABELS = {
  台湾省: { label: '台湾', dx: 22, dy: -12, anchor: 'start' },
  香港特别行政区: { label: '香港', dx: 24, dy: -18, anchor: 'start' },
  澳门特别行政区: { label: '澳门', dx: -24, dy: 20, anchor: 'end' },
}

const coordinateBounds = (geojson) => {
  const bounds = {
    minX: Infinity,
    maxX: -Infinity,
    minY: Infinity,
    maxY: -Infinity,
  }

  const walk = (coordinates) => {
    if (typeof coordinates?.[0] === 'number') {
      bounds.minX = Math.min(bounds.minX, coordinates[0])
      bounds.maxX = Math.max(bounds.maxX, coordinates[0])
      bounds.minY = Math.min(bounds.minY, coordinates[1])
      bounds.maxY = Math.max(bounds.maxY, coordinates[1])
      return
    }
    coordinates?.forEach(walk)
  }

  geojson.features.forEach((feature) => walk(feature.geometry.coordinates))
  return bounds
}

const createLinearLonLatProjection = (geojson, width, height) => {
  const { minX, maxX, minY, maxY } = coordinateBounds(geojson)
  const drawWidth = width * 0.9
  const drawHeight = height * 0.75
  const scale = Math.min(drawWidth / (maxX - minX), drawHeight / (maxY - minY))
  const offsetX = (width - (maxX - minX) * scale) / 2
  const offsetY = height * 0.08 + (drawHeight - (maxY - minY) * scale) / 2

  const projectPoint = ([lon, lat]) => [
    offsetX + (lon - minX) * scale,
    offsetY + (maxY - lat) * scale,
  ]

  const projection = geoTransform({
    point(lon, lat) {
      const [x, y] = projectPoint([lon, lat])
      this.stream.point(x, y)
    },
  })

  return { projection, projectPoint }
}

const titleText = (item) => {
  if (!item.province) return `${item.name || '未命名区域'}: 暂无省级诊断数据`
  if (item.province.displayOnly) {
    return `${item.name}: ${item.province.cardText || item.province.dataCaveat || '展示区域，未单独估计省级指标'}`
  }
  const score = Number(item.province.adaptationGapScore)
  return Number.isFinite(score) ? `${item.name}: 适配缺口 ${Math.round(score * 100)}` : `${item.name}: 暂无适配缺口指标`
}

export function ChinaMap({ geojson, provinces, selectedProvince, active, onSelectProvince }) {
  const wrapRef = useRef(null)
  const { width, height } = useElementSize(wrapRef)
  const provinceByName = useMemo(
    () => new Map(provinces.map((province) => [province.province, province])),
    [provinces],
  )

  const paths = useMemo(() => {
    if (!geojson || width <= 1 || height <= 1) return []
    const { projection, projectPoint } = createLinearLonLatProjection(geojson, width, height)
    const path = geoPath(projection)
    return geojson.features.map((feature) => {
      const province = provinceByName.get(feature.properties.name)
      const markerConfig = DISPLAY_REGION_LABELS[feature.properties.name]
      const markerPoint = feature.properties.centroid || feature.properties.center
      return {
        key: feature.properties.adcode,
        name: feature.properties.name,
        d: path(feature),
        province,
        marker: markerConfig && markerPoint
          ? {
              ...markerConfig,
              point: projectPoint(markerPoint),
            }
          : null,
      }
    })
  }, [geojson, height, provinceByName, width])

  return (
    <div ref={wrapRef} className={`china-map-stage ${active ? 'is-active' : ''}`}>
      <svg viewBox={`0 0 ${width || 1} ${height || 1}`} role="img" aria-label="中国省级适配缺口地图">
        <defs>
          <filter id="provinceGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor="#2fa996" floodOpacity="0.45" />
          </filter>
        </defs>
        {paths.map((item) => {
          const isSelected = item.name === selectedProvince
          const isDisplayRegion = item.province?.displayOnly
          const className = [
            'province',
            isSelected ? 'selected' : '',
            isDisplayRegion ? 'display-region' : '',
          ].filter(Boolean).join(' ')
          return (
            <path
              key={item.key}
              d={item.d}
              className={className}
              fill={isDisplayRegion ? displayRegionFill : gapColor(item.province?.adaptationGapScore)}
              stroke={isSelected ? '#10231f' : isDisplayRegion ? '#0c6f61' : '#ffffff'}
              strokeWidth={isSelected ? 1.8 : isDisplayRegion ? 1.15 : 0.75}
              filter={isSelected ? 'url(#provinceGlow)' : undefined}
              onClick={() => item.province && onSelectProvince(item.province)}
            >
              <title>{titleText(item)}</title>
            </path>
          )
        })}

        {paths.filter((item) => item.marker).map((item) => {
          const [x, y] = item.marker.point
          const labelX = x + item.marker.dx
          const labelY = y + item.marker.dy
          return (
            <g
              key={`${item.key}-marker`}
              className="display-region-marker"
              role="button"
              tabIndex="0"
              onClick={() => item.province && onSelectProvince(item.province)}
              onKeyDown={(event) => {
                if ((event.key === 'Enter' || event.key === ' ') && item.province) {
                  event.preventDefault()
                  onSelectProvince(item.province)
                }
              }}
            >
              <line x1={x} y1={y} x2={labelX} y2={labelY - 4} />
              <circle cx={x} cy={y} r={4} />
              <text x={labelX} y={labelY} textAnchor={item.marker.anchor}>
                {item.marker.label}
              </text>
              <title>{titleText(item)}</title>
            </g>
          )
        })}
      </svg>

      <div className="map-scale" aria-hidden="true">
        <span><i className="low" />低缺口</span>
        <span><i className="mid" />中缺口</span>
        <span><i className="high" />高缺口</span>
        <span><i className="display" />展示/未估计</span>
        <span><i className="empty" />未匹配</span>
      </div>
    </div>
  )
}
