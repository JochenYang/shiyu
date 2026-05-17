#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Command, Stdio};
use std::sync::Mutex;
use tauri::Manager;

struct AppState {
    backend_process: Mutex<Option<std::process::Child>>,
}

fn find_backend_dir() -> Option<std::path::PathBuf> {
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
            let child = if let Some(backend_dir) = find_backend_dir() {
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

                let result = Command::new(&exe)
                    .arg(&main_py)
                    .current_dir(&backend_dir)
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .spawn();

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
