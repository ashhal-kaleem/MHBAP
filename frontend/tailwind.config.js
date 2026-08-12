/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ivory: '#FAF9F6',
        plum: {
          light:   '#7c3aed',
          DEFAULT: '#4a148c',
          dark:    '#2d0c5a',
        },
        sage: {
          light:   '#A3C6A3',
          DEFAULT: '#8fbc8f',
          dark:    '#699969',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
      },
      boxShadow: {
        'glass': '0 4px 24px -4px rgba(0,0,0,0.08)',
      },
      backgroundOpacity: {
        '3':  '0.03',
        '8':  '0.08',
      },
      animation: {
        'spin-slow': 'spin 3s linear infinite',
      },
    },
  },
  plugins: [],
  // Safelist classes used dynamically (emotion colors, metric accents)
  safelist: [
    'bg-plum/3', 'bg-plum/8', 'bg-plum/5', 'bg-plum/10', 'bg-plum/20',
    'text-plum', 'text-plum-dark', 'text-plum-light',
    'border-plum', 'border-plum/20', 'border-plum/25', 'border-plum/30',
    'ring-plum/10',
    { pattern: /^(bg|text|border)-(red|green|blue|orange|purple|pink|teal|yellow|amber|indigo)-(50|100|200|400|500|600|700)$/ },
  ],
}
