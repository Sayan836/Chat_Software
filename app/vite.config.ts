import { defineConfig } from "vite";

export default defineConfig({
  clearScreen: false,
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    watch: {
      ignored: [
        "**/.venv/**",
        "**/.chat_software_state/**",
        "**/backend/**/*.egg-info/**",
        "**/backend/**/__pycache__/**",
        "**/src-tauri/target/**"
      ]
    }
  }
});
