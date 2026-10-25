/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        tactical: {
          dark: '#080c10',
          panel: '#0e141b',
          border: '#1f2937',
          cyan: '#00f0ff',
          amber: '#ffaa00',
          red: '#ff3344',
          emerald: '#10b981',
          purple: '#a855f7'
        }
      },
      fontFamily: {
        mono: ['Fira Code', 'Courier New', 'monospace'],
        tactical: ['Chakra Petch', 'system-ui', 'sans-serif']
      }
    },
  },
  plugins: [],
}

