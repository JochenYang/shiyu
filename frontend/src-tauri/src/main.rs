#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Command, Stdio};
use std::sync::Mutex;
use tauri::Manager;

struct AppState {
    backend_process: Mutex<Option<std::process::Child>>,
}

fn find_backend_dir(app: &tauri::App) -> Option<std::path::PathBuf> {
    // 1. Try resolving via Tauri's official path resolver (production mode)
    if let Some(resource_path) = app.path_resolver().resolve_resource("resources/backend") {
        if resource_path.join("run_server.py").exists() {
            return Some(resource_path);
        }
    }

    // 2. Fallback: scan parent directories (local development mode)
    let cwd = std::env::current_dir().ok()?;
    let mut dir = cwd.as_path();
    for _ in 0..5 {
        if dir.join("backend").join("run_server.py").exists() {
            return Some(dir.join("backend"));
        }
        dir = dir.parent()?;
    }
    None
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let child = if let Some(backend_dir) = find_backend_dir(app) {
                let main_py = backend_dir.join("run_server.py");
                let python_exe = if cfg!(windows) {
                    backend_dir.join("venv").join("Scripts").join("python.exe")
                } else {
                    backend_dir.join("venv").join("bin").join("python")
                };

                let (exe, using_venv) = if python_exe.exists() {
                    (python_exe.clone(), true)
                } else {
                    // Fallback: try conda/miniconda python, then system python
                    let conda = std::path::PathBuf::from("D:/tools/miniconda3/python.exe");
                    if conda.exists() {
                        (conda, false)
                    } else {
                        (std::path::PathBuf::from("python"), false)
                    }
                };

                println!("[shiyu] backend_dir: {}", backend_dir.display());
                println!("[shiyu] python: {} (venv={})", exe.display(), using_venv);
                println!("[shiyu] main_py: {}", main_py.display());

                let mut cmd = Command::new(&exe);
                cmd.arg(&main_py)
                    .current_dir(&backend_dir);

                if cfg!(debug_assertions) {
                    cmd.stdout(Stdio::inherit())
                       .stderr(Stdio::inherit());
                } else {
                    cmd.stdout(Stdio::null())
                       .stderr(Stdio::null());
                }

                #[cfg(windows)]
                {
                    use std::os::windows::process::CommandExt;
                    cmd.creation_flags(0x08000000);
                }

                let result = cmd.spawn();

                match result {
                    Ok(c) => Some(c),
                    Err(e) => {
                        eprintln!("[shiyu] failed to start backend: {}", e);
                        None
                    }
                }
            } else {
                eprintln!("[shiyu] backend directory not found");
                None
            };

            app.manage(AppState {
                backend_process: Mutex::new(child),
            });

            if let Some(main_window) = app.get_window("main") {
                let icon_bytes = include_bytes!("../../assets/logo.png");
                let icon = tauri::Icon::Raw(icon_bytes.to_vec());
                let _ = main_window.set_icon(icon); // Logo padding cropped & refreshed
            }

            Ok(())
        })
        .on_window_event(|event| {
            if let tauri::WindowEvent::Destroyed = event.event() {
                if let Some(state) = event.window().try_state::<AppState>() {
                    if let Ok(mut guard) = state.backend_process.lock() {
                        if let Some(ref mut c) = *guard {
                            let _ = c.kill();
                            println!("[shiyu] backend process killed");
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
