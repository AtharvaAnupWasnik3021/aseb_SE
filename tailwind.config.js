/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        cyan: {
          400: '#22d3ee',
          500: '#06b6d4',
        },
        neon: {
          cyan: '#00f5ff',
          blue: '#0080ff',
          violet: '#8b5cf6',
          emerald: '#10b981',
          pink: '#f0abfc',
        },
        bio: {
          dark: '#030712',
          darker: '#010208',
          surface: '#0a0f1e',
          border: '#1a2744',
          card: '#0d1530',
        },
      },
      fontFamily: {
        display: ['"Orbitron"', 'monospace'],
        body: ['"Rajdhani"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 20s linear infinite',
        'float': 'float 6s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'grid-flow': 'grid-flow 20s linear infinite',
        'particle': 'particle 8s linear infinite',
        'scan': 'scan 3s ease-in-out infinite',
        'data-stream': 'data-stream 2s linear infinite',
        'orbit': 'orbit 10s linear infinite',
        'morph': 'morph 8s ease-in-out infinite',
        'counter': 'counter 2s ease-out forwards',
        'slide-up': 'slide-up 0.8s ease-out forwards',
        'slide-in-left': 'slide-in-left 0.8s ease-out forwards',
        'slide-in-right': 'slide-in-right 0.8s ease-out forwards',
        'fade-in': 'fade-in 1s ease-out forwards',
        'scale-in': 'scale-in 0.6s ease-out forwards',
        'bounce-slow': 'bounce 3s infinite',
        'ping-slow': 'ping 3s cubic-bezier(0, 0, 0.2, 1) infinite',
        'wave': 'wave 2.5s ease-in-out infinite',
        'typewriter': 'typewriter 3s steps(40) forwards',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 5px #00f5ff, 0 0 10px #00f5ff, 0 0 20px #00f5ff' },
          '100%': { boxShadow: '0 0 10px #00f5ff, 0 0 30px #00f5ff, 0 0 60px #00f5ff' },
        },
        'grid-flow': {
          '0%': { backgroundPosition: '0 0' },
          '100%': { backgroundPosition: '50px 50px' },
        },
        scan: {
          '0%, 100%': { top: '0%', opacity: 1 },
          '50%': { top: '100%', opacity: 0.5 },
        },
        'data-stream': {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100%)' },
        },
        orbit: {
          '0%': { transform: 'rotate(0deg) translateX(80px) rotate(0deg)' },
          '100%': { transform: 'rotate(360deg) translateX(80px) rotate(-360deg)' },
        },
        morph: {
          '0%, 100%': { borderRadius: '60% 40% 30% 70% / 60% 30% 70% 40%' },
          '25%': { borderRadius: '30% 60% 70% 40% / 50% 60% 30% 60%' },
          '50%': { borderRadius: '50% 60% 30% 60% / 40% 70% 60% 30%' },
          '75%': { borderRadius: '40% 70% 60% 30% / 70% 40% 50% 60%' },
        },
        'slide-up': {
          '0%': { transform: 'translateY(60px)', opacity: 0 },
          '100%': { transform: 'translateY(0)', opacity: 1 },
        },
        'slide-in-left': {
          '0%': { transform: 'translateX(-60px)', opacity: 0 },
          '100%': { transform: 'translateX(0)', opacity: 1 },
        },
        'slide-in-right': {
          '0%': { transform: 'translateX(60px)', opacity: 0 },
          '100%': { transform: 'translateX(0)', opacity: 1 },
        },
        'fade-in': {
          '0%': { opacity: 0 },
          '100%': { opacity: 1 },
        },
        'scale-in': {
          '0%': { transform: 'scale(0.8)', opacity: 0 },
          '100%': { transform: 'scale(1)', opacity: 1 },
        },
        wave: {
          '0%, 100%': { transform: 'scaleY(1)' },
          '50%': { transform: 'scaleY(2)' },
        },
        typewriter: {
          '0%': { width: '0' },
          '100%': { width: '100%' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
      backgroundImage: {
        'grid-pattern': 'linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px)',
        'radial-glow': 'radial-gradient(ellipse at center, rgba(0,245,255,0.15) 0%, transparent 70%)',
        'hero-gradient': 'radial-gradient(ellipse at 50% 50%, rgba(0, 128, 255, 0.08) 0%, rgba(139, 92, 246, 0.05) 50%, transparent 70%)',
      },
      boxShadow: {
        'neon-cyan': '0 0 20px rgba(0, 245, 255, 0.5), 0 0 40px rgba(0, 245, 255, 0.2)',
        'neon-blue': '0 0 20px rgba(0, 128, 255, 0.5), 0 0 40px rgba(0, 128, 255, 0.2)',
        'neon-violet': '0 0 20px rgba(139, 92, 246, 0.5), 0 0 40px rgba(139, 92, 246, 0.2)',
        'glass': '0 8px 32px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255,255,255,0.05)',
        'card-hover': '0 0 30px rgba(0, 245, 255, 0.15), 0 20px 60px rgba(0, 0, 0, 0.5)',
      },
    },
  },
  plugins: [],
}
