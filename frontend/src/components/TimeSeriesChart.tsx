import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import type { MetricSeries } from '@/types'

interface Props { data: MetricSeries[] }

export function TimeSeriesChart({ data }: Props) {
  return (
    <div className="rounded-xl bg-gray-800 p-4">
      <p className="mb-3 text-sm font-semibold text-gray-200">Signal History</p>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="time" tick={{ fill: '#9ca3af', fontSize: 10 }}
            interval="preserveStartEnd" />
          <YAxis domain={[0, 1]} tick={{ fill: '#9ca3af', fontSize: 10 }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151' }}
            labelStyle={{ color: '#d1d5db' }}
            itemStyle={{ color: '#d1d5db' }}
            formatter={(v: number) => `${Math.round(v * 100)}%`}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: '#9ca3af' }} />
          <Line type="monotone" dataKey="stress"     stroke="#f87171" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="engagement" stroke="#4ade80" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="attention"  stroke="#60a5fa" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="fatigue"    stroke="#fb923c" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
