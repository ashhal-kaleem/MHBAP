interface Props {
  label: string
  scores: Record<string, number>
}

const EMOTION_COLORS: Record<string, string> = {
  neutral:  'bg-blue-400',
  happy:    'bg-green-400',
  sad:      'bg-indigo-400',
  angry:    'bg-red-400',
  fearful:  'bg-purple-400',
  disgusted:'bg-yellow-500',
  surprised:'bg-pink-400',
  focused:  'bg-teal-400',
  confused: 'bg-orange-400',
}

export function EmotionBar({ label, scores }: Props) {
  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1])
  return (
    <div className="rounded-2xl bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm p-5 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-gray-900">Emotion</span>
        <span className="rounded-full bg-blue-50 px-2 py-0.5 text-xs font-bold text-blue-700 uppercase border border-blue-100">
          {label}
        </span>
      </div>
      {entries.map(([name, val]) => (
        <div key={name} className="space-y-0.5">
          <div className="flex justify-between text-xs text-gray-500">
            <span className="capitalize">{name}</span>
            <span>{Math.round(val * 100)}%</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-gray-100 overflow-hidden">
            <div
              className={`h-1.5 rounded-full ${EMOTION_COLORS[name] ?? 'bg-gray-400'}`}
              style={{ width: `${val * 100}%`, transition: 'width 0.4s ease' }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
