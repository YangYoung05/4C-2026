import { useEffect, useRef } from 'react'
import { animate } from 'animejs'

const formatValue = (value, decimals, prefix, suffix) => {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return `${value ?? 'NA'}`
  return `${prefix}${numeric.toFixed(decimals)}${suffix}`
}

export function AnimatedNumber({
  value,
  decimals = 0,
  duration = 760,
  prefix = '',
  suffix = '',
  className,
}) {
  const ref = useRef(null)

  useEffect(() => {
    const node = ref.current
    const target = Number(value)
    if (!node || !Number.isFinite(target)) return

    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    if (reduceMotion) {
      node.textContent = formatValue(target, decimals, prefix, suffix)
      return
    }

    const counter = { value: Math.max(0, target * 0.72) }

    const animation = animate(counter, {
      value: target,
      duration,
      ease: 'outExpo',
      onUpdate: () => {
        node.textContent = formatValue(counter.value, decimals, prefix, suffix)
      },
    })

    return () => animation.revert()
  }, [decimals, duration, prefix, suffix, value])

  return (
    <strong ref={ref} className={className}>
      {formatValue(value, decimals, prefix, suffix)}
    </strong>
  )
}
