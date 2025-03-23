module.exports = {
  theme: {
    extend: {
      animation: {
        fadeIn: 'fadeIn 0.5s ease-in-out',
        fadeOut: 'fadeIn 0.5s ease-in-out, fadeOut 0.5s ease-in-out 2.5s forwards',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeOut: {
          '0%': { opacity: '1' },
          '100%': { opacity: '0' },
        },
      },
    },
  },
  plugins: [
    function({ addUtilities }) {
      const newUtilities = {
        '.loading-bar-animation': {
          background: 'linear-gradient(90deg, rgba(59, 130, 246, 0.5) 0%, rgba(59, 130, 246, 1) 50%, rgba(59, 130, 246, 0.5) 100%)',
          backgroundSize: '200% 100%',
          animation: 'loading-bar 1.5s infinite linear'
        },
      }
      
      addUtilities(newUtilities)
    },
    function({ addKeyframes }) {
      addKeyframes({
        'loading-bar': {
          '0%': { backgroundPosition: '0% 0' },
          '100%': { backgroundPosition: '200% 0' }
        }
      })
    }
  ]
} 