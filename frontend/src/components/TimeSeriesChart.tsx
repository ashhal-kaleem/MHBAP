import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import type { MetricSeries } from '@/types'

interface Props {
  data: MetricSeries[]
  title?: string
}

const LINES = [
  { key: 'stress',     label: 'Stress',     color: '#ef4444' },
  { key: 'engagement', label: 'Engagement', color: '#22c55e' },
  { key: 'attention',  label: 'Attention',  color: '#3b82f6' },
  { key: 'fatigue',    label: 'Fatigue',    color: '#f97316' },
] as const

export function TimeSeriesChart({ data, title = 'Signal History' }: Props) {
  const isEmpty = data.length === 0

  return (
    <div className="rounded-2xl bg-white border border-gray-100 shadow-sm p-5">
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm font-semibold text-gray-900">{title}</p>
        {data.length > 0 && (
          <span className="text-xs text-gray-400 tabular-nums">{data.length} frames</span>
        )}
      </div>

      {isEmpty ? (
        <div className="h-52 flex items-center justify-center text-gray-300 text-sm">
          No data yet
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data} margin={{ top: 4, right: 12, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
            <XAxis
              dataKey="time"
              tick={{ fill: '#9ca3af', fontSize: 10 }}
              interval="preserveStartEnd"
              tickLine={false}
              axisLine={{ stroke: '#e5e7eb' }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fill: '#9ca3af', fontSize: 10 }}
              tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              tickLine={false}
              axisLine={false}
              width={40}
            />
            <ReferenceLine y={0.5} stroke="#e5e7eb" strokeDasharray="4 4" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                border: '1px solid #f3f4f6',
                borderRadius: '0.75rem',
                boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.08)',
                padding: '8px 12px',
              }}
              labelStyle={{ color: '#374151', fontWeight: 600, fontSize: 11, marginBottom: 4 }}
              itemStyle={{ color: '#6b7280', fontSize: 12 }}
              formatter={(v: number, name: string) => {
                const line = LINES.find(l => l.key === name)
                return [`${Math.round(v * 100)}%`, line?.label ?? name]
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, color: '#6b7280', paddingTop: 12 }}
              formatter={(value: string) => LINES.find(l => l.key === value)?.label ?? value}
            />
            {LINES.map(({ key, color }) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={color}
                dot={false}
                strokeWidth={2}
                activeDot={{ r: 4, strokeWidth: 0 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
