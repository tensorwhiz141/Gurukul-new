import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    historyApiFallback: true,
    proxy: {
      // Proxy for AnimateDiff API on separate device to handle CORS
      '/api/vision': {
        target: 'http://192.168.0.121:8501',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/vision/, '/generate-video'),
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log('AnimateDiff API proxy error:', err);
          });
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log('AnimateDiff API proxy request to 192.168.0.121:8501:', req.method, req.url);
            // Add the API key header to proxied requests
            proxyReq.setHeader('x-api-key', 'shashank_ka_vision786');
          });
          proxy.on('proxyRes', (proxyRes, req, res) => {
            console.log('AnimateDiff API proxy response:', proxyRes.statusCode, req.url);
          });
        }
      },
      // Additional proxy for test endpoint on separate device
      '/api/test-vision': {
        target: 'http://192.168.0.121:8501',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/test-vision/, '/test-generate-video'),
        configure: (proxy, options) => {
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log('AnimateDiff test API proxy request to 192.168.0.121:8501:', req.method, req.url);
            // No API key needed for test endpoint
          });
        }
      }
    }
  }
})
