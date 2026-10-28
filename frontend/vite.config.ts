import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api/images': {
        target: process.env.IMAGE_API_PROXY ?? 'http://127.0.0.1:8100',
        // Preserve the browser origin for the DRES same-origin check.
        changeOrigin: false,
        headers: process.env.IMAGE_API_KEY ? { Authorization: `Bearer ${process.env.IMAGE_API_KEY}` } : undefined,
      },
    },
    watch: { ignored: ['**/test-results/**', '**/playwright-report/**'] },
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    css: true,
    maxWorkers: 2,
    exclude: ['tests/**', 'node_modules/**', 'dist/**'],
  },
})
