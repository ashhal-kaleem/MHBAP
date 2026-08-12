import { useEffect, useRef } from 'react'
import { Activity, Camera, CameraOff, Mic, MicOff } from 'lucide-react'
import type { Prediction } from '@/types'

interface CapturePreviewProps {
  active: boolean
  /** Number of prediction frames received from the backend WS stream. */
  frameCount: number
  /**
   * Latest prediction — drives the visualisation colours and node pulses.
   * frameSrc is accepted but intentionally ignored: we never render the
   * real camera feed for privacy.
   */
  prediction?: Prediction | null
  /** Accepted for API compatibility; the raw frame is never rendered. */
  frameSrc?: string | null
  onPermissionDenied?: (err: string) => void
}

// ── Canonical 68-point face landmark positions (normalised 0-1) ──────────
// Derived from dlib/MediaPipe face mesh representative subset.
// Groups: jaw, brow_l, brow_r, nose, eye_l, eye_r, mouth, iris_l, iris_r
const LANDMARKS: { x: number; y: number; group: string }[] = [
  // jaw outline (17 pts)
  { x: 0.18, y: 0.82, group: 'jaw' }, { x: 0.22, y: 0.90, group: 'jaw' },
  { x: 0.28, y: 0.95, group: 'jaw' }, { x: 0.35, y: 0.98, group: 'jaw' },
  { x: 0.43, y: 0.99, group: 'jaw' }, { x: 0.50, y: 0.99, group: 'jaw' },
  { x: 0.57, y: 0.99, group: 'jaw' }, { x: 0.65, y: 0.98, group: 'jaw' },
  { x: 0.72, y: 0.95, group: 'jaw' }, { x: 0.78, y: 0.90, group: 'jaw' },
  { x: 0.82, y: 0.82, group: 'jaw' },
  // left brow (5 pts)
  { x: 0.22, y: 0.42, group: 'brow' }, { x: 0.29, y: 0.37, group: 'brow' },
  { x: 0.36, y: 0.35, group: 'brow' }, { x: 0.43, y: 0.36, group: 'brow' },
  { x: 0.48, y: 0.38, group: 'brow' },
  // right brow (5 pts)
  { x: 0.52, y: 0.38, group: 'brow' }, { x: 0.57, y: 0.36, group: 'brow' },
  { x: 0.64, y: 0.35, group: 'brow' }, { x: 0.71, y: 0.37, group: 'brow' },
  { x: 0.78, y: 0.42, group: 'brow' },
  // nose bridge + tip (9 pts)
  { x: 0.50, y: 0.44, group: 'nose' }, { x: 0.50, y: 0.50, group: 'nose' },
  { x: 0.50, y: 0.56, group: 'nose' }, { x: 0.50, y: 0.62, group: 'nose' },
  { x: 0.42, y: 0.67, group: 'nose' }, { x: 0.46, y: 0.68, group: 'nose' },
  { x: 0.50, y: 0.69, group: 'nose' }, { x: 0.54, y: 0.68, group: 'nose' },
  { x: 0.58, y: 0.67, group: 'nose' },
  // left eye (6 pts)
  { x: 0.27, y: 0.46, group: 'eye' }, { x: 0.32, y: 0.43, group: 'eye' },
  { x: 0.38, y: 0.43, group: 'eye' }, { x: 0.43, y: 0.46, group: 'eye' },
  { x: 0.38, y: 0.49, group: 'eye' }, { x: 0.32, y: 0.49, group: 'eye' },
  // right eye (6 pts)
  { x: 0.57, y: 0.46, group: 'eye' }, { x: 0.62, y: 0.43, group: 'eye' },
  { x: 0.68, y: 0.43, group: 'eye' }, { x: 0.73, y: 0.46, group: 'eye' },
  { x: 0.68, y: 0.49, group: 'eye' }, { x: 0.62, y: 0.49, group: 'eye' },
  // mouth outer (12 pts)
  { x: 0.35, y: 0.76, group: 'mouth' }, { x: 0.41, y: 0.73, group: 'mouth' },
  { x: 0.50, y: 0.72, group: 'mouth' }, { x: 0.59, y: 0.73, group: 'mouth' },
  { x: 0.65, y: 0.76, group: 'mouth' }, { x: 0.59, y: 0.80, group: 'mouth' },
  { x: 0.50, y: 0.82, group: 'mouth' }, { x: 0.41, y: 0.80, group: 'mouth' },
  // irises (2 pts)
  { x: 0.35, y: 0.46, group: 'iris' }, { x: 0.65, y: 0.46, group: 'iris' },
]

// ── Connection lines between landmark groups ──────────────────────────────
const CONNECTIONS: [number, number][] = [
  // jaw
  [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],
  // brows
  [11,12],[12,13],[13,14],[14,15],  [16,17],[17,18],[18,19],[19,20],
  // nose
  [21,22],[22,23],[23,24],[24,25],[25,26],[26,27],[27,28],[28,29],[29,25],
  // left eye
  [30,31],[31,32],[32,33],[33,34],[34,35],[35,30],
  // right eye
  [36,37],[37,38],[38,39],[39,40],[40,41],[41,36],
  // mouth outer
  [42,43],[43,44],[44,45],[45,46],[46,47],[47,48],[48,49],[49,42],
]

const GROUP_BASE_COLOR: Record<string, string> = {
  jaw:   'rgba(99,179,237,0.55)',   // sky-blue
  brow:  'rgba(167,139,250,0.70)',  // violet
  nose:  'rgba(99,179,237,0.55)',
  eye:   'rgba(192,132,252,0.90)',  // bright violet — most expressive
  mouth: 'rgba(129,230,217,0.75)',  // teal
  iris:  'rgba(255,255,255,0.95)',  // white — always prominent
}

function lerp(a: number, b: number, t: number) { return a + (b - a) * t }

/**
 * Draws one frame of the face-mesh visualisation onto the canvas.
 * Uses real ML values (stress, engagement) to modulate colours and pulse.
 */
function drawFaceMesh(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  t: number,           // seconds since session start
  stress: number,      // [0,1]
  engagement: number,  // [0,1]
) {
  ctx.clearRect(0, 0, w, h)

  // ── Background ──────────────────────────────────────────────────────
  ctx.fillStyle = '#0a0a14'
  ctx.fillRect(0, 0, w, h)

  // ── Scan grid ───────────────────────────────────────────────────────
  const gridAlpha = 0.06 + 0.04 * Math.sin(t * 0.4)
  ctx.strokeStyle = `rgba(99,179,237,${gridAlpha})`
  ctx.lineWidth = 0.5
  const step = 20
  for (let x = 0; x < w; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke()
  }
  for (let y = 0; y < h; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke()
  }

  // ── Horizontal scan line ─────────────────────────────────────────────
  const scanY = ((t * 0.15) % 1) * h
  const scanGrad = ctx.createLinearGradient(0, scanY - 18, 0, scanY + 4)
  scanGrad.addColorStop(0, 'rgba(99,179,237,0)')
  scanGrad.addColorStop(1, `rgba(99,179,237,${0.18 + 0.10 * engagement})`)
  ctx.fillStyle = scanGrad
  ctx.fillRect(0, scanY - 18, w, 22)

  // ── Face bounding ellipse ────────────────────────────────────────────
  const cx = w * 0.50, cy = h * 0.60
  const rx = w * 0.30, ry = h * 0.36
  const stressHue = lerp(200, 0, stress)   // blue → red
  ctx.beginPath()
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2)
  ctx.strokeStyle = `hsla(${stressHue},80%,65%,${0.35 + 0.15 * Math.sin(t * 1.2)})`
  ctx.lineWidth = 1.5
  ctx.setLineDash([6, 4])
  ctx.stroke()
  ctx.setLineDash([])

  // ── Connection lines ──────────────────────────────────────────────────
  const lineAlpha = 0.28 + 0.12 * engagement
  CONNECTIONS.forEach(([a, b]) => {
    const p = LANDMARKS[a], q = LANDMARKS[b]
    ctx.beginPath()
    ctx.moveTo(p.x * w, p.y * h)
    ctx.lineTo(q.x * w, q.y * h)
    ctx.strokeStyle = `rgba(99,179,237,${lineAlpha})`
    ctx.lineWidth = 0.6
    ctx.stroke()
  })

  // ── Landmark dots ─────────────────────────────────────────────────────
  LANDMARKS.forEach((lm, i) => {
    const px = lm.x * w, py = lm.y * h
    const pulse = 0.5 + 0.5 * Math.sin(t * 2.5 + i * 0.4)
    const base = GROUP_BASE_COLOR[lm.group] ?? 'rgba(150,150,150,0.6)'

    // radius: irises largest, others scale with engagement pulse
    let r = lm.group === 'iris' ? 3.5 : (lm.group === 'eye' ? 2.2 : 1.4)
    r *= (0.85 + 0.3 * pulse * engagement)

    // glow
    const glow = ctx.createRadialGradient(px, py, 0, px, py, r * 3.5)
    glow.addColorStop(0, base.replace(/[\d.]+\)$/, `${0.4 + 0.3 * pulse})`))
    glow.addColorStop(1, 'rgba(0,0,0,0)')
    ctx.beginPath()
    ctx.arc(px, py, r * 3.5, 0, Math.PI * 2)
    ctx.fillStyle = glow
    ctx.fill()

    // solid dot
    ctx.beginPath()
    ctx.arc(px, py, r, 0, Math.PI * 2)
    ctx.fillStyle = base
    ctx.fill()
  })

  // ── Corner brackets (HUD feel) ────────────────────────────────────────
  const bx0 = cx - rx - 10, by0 = cy - ry - 10
  const bx1 = cx + rx + 10, by1 = cy + ry + 10
  const brack = 12
  const brackAlpha = 0.55 + 0.25 * Math.sin(t * 0.8)
  ctx.strokeStyle = `rgba(167,139,250,${brackAlpha})`
  ctx.lineWidth = 1.5
  // top-left
  ctx.beginPath(); ctx.moveTo(bx0, by0 + brack); ctx.lineTo(bx0, by0); ctx.lineTo(bx0 + brack, by0); ctx.stroke()
  // top-right
  ctx.beginPath(); ctx.moveTo(bx1 - brack, by0); ctx.lineTo(bx1, by0); ctx.lineTo(bx1, by0 + brack); ctx.stroke()
  // bottom-left
  ctx.beginPath(); ctx.moveTo(bx0, by1 - brack); ctx.lineTo(bx0, by1); ctx.lineTo(bx0 + brack, by1); ctx.stroke()
  // bottom-right
  ctx.beginPath(); ctx.moveTo(bx1 - brack, by1); ctx.lineTo(bx1, by1); ctx.lineTo(bx1, by1 - brack); ctx.stroke()

  // ── Stress indicator arc (top-right corner of bounding box) ──────────
  if (stress > 0) {
    const arc_cx = bx1 + 6, arc_cy = by0 - 6
    const arc_r = 10
    ctx.beginPath()
    ctx.arc(arc_cx, arc_cy, arc_r, -Math.PI / 2, -Math.PI / 2 + stress * Math.PI * 2)
    ctx.strokeStyle = `hsla(${lerp(120, 0, stress)},90%,55%,0.85)`
    ctx.lineWidth = 2.5
    ctx.stroke()
  }
}

// ── Component ─────────────────────────────────────────────────────────────

export function CapturePreview({
  active,
  frameCount,
  prediction,
}: CapturePreviewProps) {
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const rafRef     = useRef<number | null>(null)
  const startRef   = useRef<number>(0)
  const predRef    = useRef<Prediction | null>(null)

  // Keep latest prediction in a ref so the rAF loop always uses current data
  // without needing to be re-created each render
  useEffect(() => {
    predRef.current = prediction ?? null
  }, [prediction])

  useEffect(() => {
    if (!active) {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
      return
    }

    startRef.current = performance.now()

    function draw(now: number) {
      const canvas = canvasRef.current
      if (!canvas) { rafRef.current = requestAnimationFrame(draw); return }
      const ctx = canvas.getContext('2d')
      if (!ctx) { rafRef.current = requestAnimationFrame(draw); return }

      const t = (now - startRef.current) / 1000
      const p = predRef.current
      drawFaceMesh(ctx, canvas.width, canvas.height, t, p?.stress ?? 0, p?.engagement ?? 0.5)
      rafRef.current = requestAnimationFrame(draw)
    }

    rafRef.current = requestAnimationFrame(draw)
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [active])

  return (
    <div className="rounded-2xl bg-white border border-gray-100 shadow-sm overflow-hidden flex flex-col h-full min-h-[200px]">
      {/* ── Viewport ──────────────────────────────────────────────────── */}
      <div className="relative bg-[#0a0a14] flex-1 min-h-[200px] overflow-hidden">

        {/* Idle placeholder */}
        {!active && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-gray-600">
            <Camera className="h-8 w-8 opacity-25" />
            <span className="text-xs opacity-40 font-medium tracking-wide">No active session</span>
          </div>
        )}

        {/* Live animation canvas */}
        {active && (
          <canvas
            ref={canvasRef}
            width={320}
            height={240}
            className="absolute inset-0 w-full h-full"
            style={{ imageRendering: 'pixelated' }}
            aria-label="Privacy-safe face detection visualisation"
            aria-hidden="true"
          />
        )}

        {/* LIVE badge */}
        {active && (
          <span className="absolute top-2.5 left-2.5 z-20 inline-flex items-center gap-1 rounded-full bg-red-600/90 px-2 py-0.5 text-[10px] font-bold text-white shadow-sm select-none">
            <span className="h-1.5 w-1.5 rounded-full bg-white animate-pulse" />
            LIVE
          </span>
        )}

        {/* Frame counter */}
        {active && (
          <span className="absolute top-2.5 right-2.5 z-20 inline-flex items-center gap-1 rounded-full bg-black/40 px-2 py-0.5 text-[10px] font-mono text-white/70 select-none">
            <Activity className="h-2.5 w-2.5" />
            {frameCount}
          </span>
        )}

        {/* Stress readout overlay (bottom-left) */}
        {active && prediction && (
          <div className="absolute bottom-2.5 left-2.5 z-20 flex flex-col gap-0.5">
            <span className="text-[9px] font-mono text-cyan-400/80 select-none">
              STR {(prediction.stress * 100).toFixed(0)}%
            </span>
            <span className="text-[9px] font-mono text-violet-400/80 select-none">
              ENG {(prediction.engagement * 100).toFixed(0)}%
            </span>
          </div>
        )}

        {/* Privacy notice */}
        {active && (
          <span className="absolute bottom-2.5 right-2.5 z-20 text-[9px] text-white/20 font-medium select-none">
            no video stored
          </span>
        )}
      </div>

      {/* ── Status row ────────────────────────────────────────────────── */}
      <div className="flex items-center gap-4 px-4 py-2.5 bg-white border-t border-gray-50">
        <div className="flex items-center gap-1.5 text-xs">
          {active
            ? <Camera className="h-3.5 w-3.5 text-blue-500" />
            : <CameraOff className="h-3.5 w-3.5 text-gray-300" />}
          <span className={active ? 'text-blue-600 font-medium' : 'text-gray-300'}>
            {active ? 'Video' : 'Idle'}
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-xs">
          {active
            ? <Mic className="h-3.5 w-3.5 text-blue-500" />
            : <MicOff className="h-3.5 w-3.5 text-gray-300" />}
          <span className={active ? 'text-blue-600 font-medium' : 'text-gray-300'}>
            {active ? 'Audio' : 'Idle'}
          </span>
        </div>
        {active && (
          <span className="ml-auto text-[10px] text-gray-300">Backend capture</span>
        )}
      </div>
    </div>
  )
}
