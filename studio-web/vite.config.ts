import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

const STUDIO_BACKEND = process.env.STUDIO_BACKEND_ORIGIN || 'http://127.0.0.1:8000';
const STUDIO_WS_BACKEND = STUDIO_BACKEND.replace(/^http/, 'ws');

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: STUDIO_BACKEND,
        changeOrigin: true,
      },
      '/health': {
        target: STUDIO_BACKEND,
        changeOrigin: true,
      },
      '/ws': {
        target: STUDIO_WS_BACKEND,
        ws: true,
        changeOrigin: true,
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
  },
});
