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
    <div className="rounded-xl bg-gray-800 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-gray-200">Emotion</span>
        <span className="rounded-full bg-gray-700 px-2 py-0.5 text-xs font-bold text-white uppercase">
          {label}
        </span>
      </div>
      {entries.map(([name, val]) => (
        <div key={name} className="space-y-0.5">
          <div className="flex justify-between text-xs text-gray-400">
            <span className="capitalize">{name}</span>
            <span>{Math.round(val * 100)}%</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-gray-700">
            <div
              className={`h-1.5 rounded-full ${EMOTION_COLORS[name] ?? 'bg-gray-500'}`}
              style={{ width: `${val * 100}%`, transition: 'width 0.4s ease' }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
