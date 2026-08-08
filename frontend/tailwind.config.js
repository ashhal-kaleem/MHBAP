/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  '#f0f4ff',
          500: '#4f6ef7',
          700: '#3451d1',
          900: '#1e2e8a',
        },
        ivory: '#FAF9F6',
        plum: {
          light: '#6E3A9C',
          DEFAULT: '#4a148c',
          dark: '#2d0c5a',
        },
        sage: {
          light: '#A3C6A3',
          DEFAULT: '#8fbc8f',
          dark: '#699969',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
