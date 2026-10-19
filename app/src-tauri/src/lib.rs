#[tauri::command]
fn backend_url() -> String {
    std::env::var("CHAT_SOFTWARE_BACKEND_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8765".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![backend_url])
        .run(tauri::generate_context!())
        .expect("failed to run Chat Software shell");
}
