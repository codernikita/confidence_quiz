import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The study-guidance API (server/index.mjs) holds the Gemini key, so the
// browser reaches it through this proxy and never sees the key.
const api = {
  "/api": `http://localhost:${process.env.GUIDANCE_PORT || 8787}`,
};

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    sourcemap: false,
    target: "es2019",          // Safari 13+/older Edge still parse this
    rollupOptions: {
      output: {
        manualChunks: {
          firebase: ["firebase/app", "firebase/auth", "firebase/firestore"],
          react: ["react", "react-dom"],
        },
      },
    },
  },
  server: { port: 5173, open: true, proxy: api },
  preview: { proxy: api },
});
