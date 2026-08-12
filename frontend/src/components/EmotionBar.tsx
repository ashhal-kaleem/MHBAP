const EMOTION_COLORS: Record<string, { bar: string; badge: string }> = {
  neutral:   { bar: 'bg-blue-400',    badge: 'bg-blue-50 text-blue-700 border-blue-100' },
  happy:     { bar: 'bg-green-400',   badge: 'bg-green-50 text-green-700 border-green-100' },
  sad:       { bar: 'bg-indigo-400',  badge: 'bg-indigo-50 text-indigo-700 border-indigo-100' },
  angry:     { bar: 'bg-red-400',     badge: 'bg-red-50 text-red-700 border-red-100' },
  fearful:   { bar: 'bg-purple-400',  badge: 'bg-purple-50 text-purple-700 border-purple-100' },
  disgusted: { bar: 'bg-yellow-500',  badge: 'bg-yellow-50 text-yellow-700 border-yellow-100' },
  surprised: { bar: 'bg-pink-400',    badge: 'bg-pink-50 text-pink-700 border-pink-100' },
  focused:   { bar: 'bg-teal-400',    badge: 'bg-teal-50 text-teal-700 border-teal-100' },
  confused:  { bar: 'bg-orange-400',  badge: 'bg-orange-50 text-orange-700 border-orange-100' },
}

const FALLBACK = { bar: 'bg-gray-400', badge: 'bg-gray-50 text-gray-600 border-gray-100' }

interface Props {
  label: string
  scores: Record<string, number>
}

export function EmotionBar({ label, scores }: Props) {
  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1])
  const isWaiting = !label || label === '—'
  const colorSet = EMOTION_COLORS[label?.toLowerCase()] ?? FALLBACK

  return (
    <div className="rounded-2xl bg-white border border-gray-100 shadow-sm p-5 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-gray-900">Emotion</span>
        {isWaiting ? (
          <div className="h-5 w-20 rounded-full bg-gray-100 animate-pulse" />
        ) : (
          <span className={`rounded-full px-2.5 py-0.5 text-xs font-bold uppercase border ${colorSet.badge}`}>
            {label}
          </span>
        )}
      </div>

      {isWaiting ? (
        <div className="space-y-2.5 pt-1">
          {[80, 55, 35, 20].map((w) => (
            <div key={w} className="space-y-1">
              <div className="flex justify-between">
                <div className="h-3 w-16 rounded bg-gray-100 animate-pulse" />
                <div className="h-3 w-8 rounded bg-gray-100 animate-pulse" />
              </div>
              <div className="h-2 w-full rounded-full bg-gray-100 overflow-hidden">
                <div className="h-2 rounded-full bg-gray-200 animate-pulse" style={{ width: `${w}%` }} />
              </div>
            </div>
          ))}
        </div>
      ) : entries.length === 0 ? (
        <p className="text-xs text-gray-400 text-center py-4">No emotion data</p>
      ) : (
        <div className="space-y-2">
          {entries.map(([name, val]) => {
            const ec = EMOTION_COLORS[name.toLowerCase()] ?? FALLBACK
            return (
              <div key={name} className="space-y-0.5">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-600 capitalize font-medium">{name}</span>
                  <span className="text-gray-500 tabular-nums">{Math.round(val * 100)}%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-gray-100 overflow-hidden">
                  <div
                    className={`h-2 rounded-full ${ec.bar}`}
                    style={{ width: `${val * 100}%`, transition: 'width 0.45s cubic-bezier(0.4,0,0.2,1)' }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
