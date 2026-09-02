import { useEffect, useMemo, useRef } from 'react'
import { geoCentroid } from 'd3-geo'
import Globe from 'react-globe.gl'
import * as THREE from 'three'
import { useElementSize } from './useElementSize.js'

const htmlEscape = (value) =>
  String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')

const CHINA_ANCHOR = { lat: 39.9, lng: 116.4 }

const focusViews = {
  overview: { lat: 18, lng: 45, altitude: 1.55 },
  highBurden: { lat: 34, lng: 83, altitude: 1.25 },
  lowBuffer: { lat: 8, lng: 25, altitude: 1.32 },
  china: { lat: 30, lng: 105, altitude: 1.25 },
  policy: { lat: 26, lng: 70, altitude: 1.38 },
}

const featureCentroid = (feature) => {
  const [lng, lat] = geoCentroid(feature)
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null
  return { lat, lng }
}

const arcLabel = (arc) => `<div class="map-tooltip">
  <strong>${htmlEscape(arc.label)}</strong>
  <span>${htmlEscape(arc.typeLabel)}</span>
  <em>${htmlEscape(arc.policyPackage)}</em>
</div>`

export function GlobalGlobe({ features, selectedIso3, focusMode, typeColors, onSelectCountry, onReady }) {
  const wrapRef = useRef(null)
  const globeRef = useRef(null)
  const readyRef = useRef(false)
  const { width, height } = useElementSize(wrapRef)
  const globeMaterial = useMemo(
    () =>
      new THREE.MeshPhongMaterial({
        color: '#0b332d',
        emissive: '#05231f',
        shininess: 14,
        specular: '#2fa996',
      }),
    [],
  )

  const featureByIso = useMemo(() => {
    const entries = features
      .filter((item) => item.properties.appData)
      .map((item) => [item.properties.appData.iso3, item])
    return new Map(entries)
  }, [features])

  const policyArcs = useMemo(() => {
    const candidates = features
      .map((item) => ({ feature: item, country: item.properties.appData, centroid: featureCentroid(item) }))
      .filter(({ country, centroid }) => (
        country &&
        centroid &&
        country.iso3 !== 'CHN' &&
        country.responseScore >= 0.58 &&
        country.adaptationGapScore <= 0.18
      ))
      .sort((a, b) => {
        const scoreA = a.country.responseScore - a.country.adaptationGapScore + (a.country.vulnerabilityTypeCode === 1 ? 0.08 : 0)
        const scoreB = b.country.responseScore - b.country.adaptationGapScore + (b.country.vulnerabilityTypeCode === 1 ? 0.08 : 0)
        return scoreB - scoreA
      })

    const selected = []
    const seenTypes = new Set()
    for (const candidate of candidates) {
      const type = candidate.country.vulnerabilityTypeCode
      if (seenTypes.has(type) && selected.length < 5) continue
      selected.push(candidate)
      seenTypes.add(type)
      if (selected.length >= 8) break
    }

    return selected.map(({ country, centroid }) => ({
      ...centroid,
      startLat: CHINA_ANCHOR.lat,
      startLng: CHINA_ANCHOR.lng,
      endLat: centroid.lat,
      endLng: centroid.lng,
      iso3: country.iso3,
      label: country.nameZh || country.name,
      typeCode: country.vulnerabilityTypeCode,
      typeLabel: country.vulnerabilityTypeLabel,
      policyPackage: country.policyPathway.package,
      color: typeColors[country.vulnerabilityTypeCode] ?? typeColors.missing,
      country,
    }))
  }, [features, typeColors])

  const selectedArc = useMemo(() => {
    if (!selectedIso3 || selectedIso3 === 'CHN') return null
    const feature = featureByIso.get(selectedIso3)
    const country = feature?.properties.appData
    const centroid = feature ? featureCentroid(feature) : null
    if (!country || !centroid) return null
    return {
      ...centroid,
      startLat: CHINA_ANCHOR.lat,
      startLng: CHINA_ANCHOR.lng,
      endLat: centroid.lat,
      endLng: centroid.lng,
      iso3: country.iso3,
      label: country.nameZh || country.name,
      typeCode: country.vulnerabilityTypeCode,
      typeLabel: country.vulnerabilityTypeLabel,
      policyPackage: country.policyPathway.package,
      color: typeColors[country.vulnerabilityTypeCode] ?? typeColors.missing,
      country,
    }
  }, [featureByIso, selectedIso3, typeColors])

  const displayedArcs = useMemo(() => {
    if (focusMode === 'policy') return policyArcs
    if (selectedArc) {
      const rest = policyArcs.filter((arc) => arc.iso3 !== selectedArc.iso3).slice(0, 4)
      return [selectedArc, ...rest]
    }
    return policyArcs.slice(0, 5)
  }, [focusMode, policyArcs, selectedArc])

  const focusPoints = useMemo(() => {
    const chinaFeature = featureByIso.get('CHN')
    const chinaCountry = chinaFeature?.properties.appData
    const chinaPoint = chinaCountry
      ? {
          ...CHINA_ANCHOR,
          iso3: 'CHN',
          label: '中国',
          color: typeColors[chinaCountry.vulnerabilityTypeCode],
          country: chinaCountry,
          altitude: 0.07,
          radius: 0.5,
        }
      : null

    return [
      ...(chinaPoint ? [chinaPoint] : []),
      ...policyArcs.slice(0, focusMode === 'policy' ? 8 : 5).map((arc) => ({
        lat: arc.lat,
        lng: arc.lng,
        iso3: arc.iso3,
        label: arc.label,
        color: arc.color,
        country: arc.country,
        altitude: 0.035 + arc.country.responseScore * 0.05,
        radius: 0.22 + arc.country.responseScore * 0.22,
      })),
    ]
  }, [featureByIso, focusMode, policyArcs, typeColors])

  const ringsData = useMemo(() => {
    const selectedFeature = featureByIso.get(selectedIso3)
    const selectedCountry = selectedFeature?.properties.appData
    const selectedCentroid = selectedFeature ? featureCentroid(selectedFeature) : null
    return [
      { ...CHINA_ANCHOR, color: '#7de1d1' },
      ...(selectedCountry && selectedCentroid
        ? [{ ...selectedCentroid, color: typeColors[selectedCountry.vulnerabilityTypeCode] ?? '#ffffff' }]
        : []),
    ]
  }, [featureByIso, selectedIso3, typeColors])

  useEffect(() => {
    if (!globeRef.current || width <= 0 || height <= 0 || readyRef.current) return
    globeRef.current.controls().autoRotate = true
    globeRef.current.controls().autoRotateSpeed = 0.5
    globeRef.current.pointOfView(focusViews.overview, 1200)
    readyRef.current = true
    onReady?.()
  }, [height, onReady, width])

  useEffect(() => {
    if (!readyRef.current || !globeRef.current) return
    const selectedFeature = featureByIso.get(selectedIso3)
    const selectedCentroid = selectedFeature ? featureCentroid(selectedFeature) : null

    if (focusMode === 'selected' && selectedCentroid) {
      globeRef.current.controls().autoRotate = false
      globeRef.current.pointOfView(
        { lat: selectedCentroid.lat, lng: selectedCentroid.lng, altitude: selectedIso3 === 'CHN' ? 1.2 : 1.35 },
        1200,
      )
      return
    }

    const view = focusViews[focusMode] ?? focusViews.overview
    globeRef.current.controls().autoRotate = focusMode === 'overview'
    globeRef.current.pointOfView(view, 1200)
  }, [featureByIso, focusMode, selectedIso3])

  return (
    <div ref={wrapRef} className="globe-stage">
      {width > 0 && height > 0 && (
        <Globe
          ref={globeRef}
          width={width}
          height={height}
          backgroundColor="rgba(0,0,0,0)"
          globeMaterial={globeMaterial}
          polygonsData={features}
          polygonAltitude={(polygon) => (polygon.properties.appData?.iso3 === selectedIso3 ? 0.055 : 0.018)}
          polygonCapColor={(polygon) => {
            const record = polygon.properties.appData
            if (!record) return 'rgba(216, 222, 220, 0.16)'
            return typeColors[record.vulnerabilityTypeCode] ?? typeColors.missing
          }}
          polygonSideColor={(polygon) =>
            polygon.properties.appData?.iso3 === selectedIso3
              ? 'rgba(255,255,255,0.34)'
              : 'rgba(21, 68, 61, 0.45)'
          }
          polygonStrokeColor={(polygon) =>
            polygon.properties.appData?.iso3 === selectedIso3 ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.28)'
          }
          polygonLabel={(polygon) => {
            const record = polygon.properties.appData
            if (!record) {
              return `<div class="map-tooltip"><strong>${htmlEscape(polygon.properties.name)}</strong><span>未进入分析样本</span></div>`
            }
            return `<div class="map-tooltip">
              <strong>${htmlEscape(record.nameZh || record.name)}</strong>
              <span>${htmlEscape(record.vulnerabilityTypeLabel)}</span>
              <em>压力 ${Math.round(record.pressureScore * 100)} / 响应 ${Math.round(record.responseScore * 100)} / 缺口 ${Math.round(record.adaptationGapScore * 100)}</em>
            </div>`
          }}
          onPolygonClick={(polygon) => onSelectCountry(polygon.properties.appData)}
          arcsData={displayedArcs}
          arcStartLat="startLat"
          arcStartLng="startLng"
          arcEndLat="endLat"
          arcEndLng="endLng"
          arcColor="color"
          arcLabel={arcLabel}
          arcDashLength={0.55}
          arcDashGap={0.18}
          arcDashAnimateTime={2600}
          arcStroke={(arc) => (arc.iso3 === selectedIso3 ? 1.05 : 0.62)}
          pointsData={focusPoints}
          pointLat="lat"
          pointLng="lng"
          pointColor="color"
          pointAltitude="altitude"
          pointRadius="radius"
          pointResolution={24}
          pointLabel={(point) => `<div class="map-tooltip">
            <strong>${htmlEscape(point.label)}</strong>
            <span>${htmlEscape(point.country.vulnerabilityTypeLabel)}</span>
            <em>响应 ${Math.round(point.country.responseScore * 100)} / 缺口 ${Math.round(point.country.adaptationGapScore * 100)}</em>
          </div>`}
          onPointClick={(point) => onSelectCountry(point.country)}
          ringsData={ringsData}
          ringLat="lat"
          ringLng="lng"
          ringColor={(ring) => [ring.color, 'rgba(255,255,255,0)']}
          ringAltitude={0.01}
          ringMaxRadius={3.2}
          ringPropagationSpeed={1.2}
          ringRepeatPeriod={1600}
          atmosphereColor="#3fd2be"
          atmosphereAltitude={0.18}
        />
      )}
    </div>
  )
}
