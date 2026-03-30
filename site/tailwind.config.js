/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        bg: { DEFAULT: '#0d1117', 2: '#161b22', 3: '#1c2128' },
        accent: { DEFAULT: '#58a6ff', green: '#7ee787', purple: '#d2a8ff', orange: '#ffa657' },
        txt: { DEFAULT: '#e6edf3', 2: '#8b949e' },
        border: '#30363d',
      },
      fontFamily: {
        sans: ['Segoe UI', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['Cascadia Code', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
};
