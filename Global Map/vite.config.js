import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 2200,
    rollupOptions: {
      output: {
        manualChunks: {
          globe: ['react-globe.gl', 'three'],
          maps: ['d3-geo', 'topojson-client'],
          react: ['react', 'react-dom'],
        },
      },
    },
  },
})
