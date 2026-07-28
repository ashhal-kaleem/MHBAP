interface Props {
  weights: Record<string, number> | null
  explanation: string | null
}

const MODAL_COLORS: Record<string, string> = {
  face:  'bg-blue-500',
  voice: 'bg-green-500',
  gaze:  'bg-purple-500',
  pose:  'bg-yellow-500',
  hci:   'bg-pink-500',
}

export function XAIPanel({ weights, explanation }: Props) {
  if (!weights) {
    return (
      <div className="rounded-xl bg-gray-800 p-4 text-sm text-gray-500">
        XAI data will appear once the inference pipeline is running.
      </div>
    )
  }

  const entries = Object.entries(weights).sort((a, b) => b[1] - a[1])

  return (
    <div className="rounded-xl bg-gray-800 p-4 space-y-3">
      <p className="text-sm font-semibold text-gray-200">Modality Contributions (SHAP)</p>
      {entries.map(([mod, w]) => (
        <div key={mod} className="space-y-0.5">
          <div className="flex justify-between text-xs text-gray-400">
            <span className="capitalize">{mod}</span>
            <span>{Math.round(w * 100)}%</span>
          </div>
          <div className="h-2 w-full rounded-full bg-gray-700">
            <div
              className={`h-2 rounded-full ${MODAL_COLORS[mod] ?? 'bg-gray-500'}`}
              style={{ width: `${w * 100}%`, transition: 'width 0.4s ease' }}
            />
          </div>
        </div>
      ))}
      {explanation && (
        <p className="mt-2 rounded-lg bg-gray-700 px-3 py-2 text-xs text-gray-300 leading-relaxed">
          {explanation}
        </p>
      )}
    </div>
  )
}
