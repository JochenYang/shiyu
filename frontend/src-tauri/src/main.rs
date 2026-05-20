#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Command, Stdio};
use std::sync::Mutex;
use tauri::Manager;

#[cfg_attr(windows, allow(dead_code))]
struct AppState {
    backend_process: Mutex<Option<std::process::Child>>,
}

/// Resolve the backend directory, trying:
///   1. Tauri resources (production mode) — checks for shiyu-backend.exe
///   2. PyInstaller dist/ (local build, dev mode) — checks for shiyu-backend.exe
///   3. backend/ source (local venv, dev mode) — checks for run_server.py
fn find_backend_dir(app: &tauri::App) -> Option<std::path::PathBuf> {
    let exe_name = if cfg!(windows) { "shiyu-backend.exe" } else { "shiyu-backend" };

    // 1. Try resolving via Tauri's official path resolver (production mode)
    if let Some(resource_path) = app.path_resolver().resolve_resource("resources/backend") {
        if resource_path.join(exe_name).exists() {
            return Some(resource_path);
        }
    }

    // 2. Fallback: scan parent directories for local builds and dev environments
    let cwd = std::env::current_dir().ok()?;
    let mut dir = cwd.as_path();
    for _ in 0..5 {
        // 2a. PyInstaller local build (backend/dist/)
        if dir.join("backend").join("dist").join(exe_name).exists() {
            return Some(dir.join("backend").join("dist"));
        }
        // 2b. Original venv dev mode (backend/run_server.py)
        if dir.join("backend").join("run_server.py").exists() {
            return Some(dir.join("backend"));
        }
        dir = dir.parent()?;
    }
    None
}

/// Start the backend process.
///
/// Production mode: run the PyInstaller-built `shiyu-backend` executable directly.
/// Dev mode: run `python run_server.py` via venv or system Python.
fn start_backend(backend_dir: &std::path::Path) -> Option<std::process::Child> {
    let (exe, args, mode_label) = {
        // Check for PyInstaller executable first (production only)
        let exe_name = if cfg!(windows) { "shiyu-backend.exe" } else { "shiyu-backend" };
        let exe_path = backend_dir.join(exe_name);
        // In debug mode, prefer Python source for hot-reload compatibility
        if exe_path.exists() && !cfg!(debug_assertions) {
            println!("[shiyu] Found PyInstaller executable: {}", exe_path.display());
            (exe_path, vec!["--port=11235".to_string()], "pyinstaller")
        } else {
            // Dev mode: use venv or system Python
            let main_py = backend_dir.join("run_server.py");
            let python_exe = if cfg!(windows) {
                backend_dir.join("venv").join("Scripts").join("python.exe")
            } else {
                backend_dir.join("venv").join("bin").join("python")
            };

            let (python_bin, using_venv) = if python_exe.exists() {
                (python_exe, true)
            } else {
                // Fallback: try conda/miniconda python, then system python
                let conda = std::path::PathBuf::from("D:/tools/miniconda3/python.exe");
                if conda.exists() {
                    (conda, false)
                } else {
                    (std::path::PathBuf::from("python"), false)
                }
            };

            println!("[shiyu] Using Python: {} (venv={})", python_bin.display(), using_venv);
            println!("[shiyu] Main script: {}", main_py.display());
            (python_bin, vec![main_py.to_string_lossy().to_string()], "venv")
        }
    };

    let mut cmd = Command::new(&exe);
    cmd.args(&args)
        .current_dir(backend_dir);

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
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    println!("[shiyu] Starting backend (mode={}): {} {}", mode_label, exe.display(), args.join(" "));

    match cmd.spawn() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("[shiyu] failed to start backend: {}", e);
            None
        }
    }
}

fn kill_backend(_app_handle: &tauri::AppHandle) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // Kill by image name to reliably catch all subprocesses (PyInstaller onefile
        // creates a launcher parent + Python child — PID-based kill can miss orphans).
        let _ = std::process::Command::new("taskkill")
            .args(&["/F", "/IM", "shiyu-backend.exe"])
            .creation_flags(0x08000000) // CREATE_NO_WINDOW
            .output();
    }

    #[cfg(not(windows))]
    {
        // On Unix, Child::kill() uses SIGKILL, which is sufficient.
        if let Some(state) = _app_handle.try_state::<AppState>() {
            if let Ok(mut guard) = state.backend_process.lock() {
                if let Some(ref mut c) = *guard {
                    let _ = c.kill();
                }
            }
        }
    }
}

fn main() {
    let app = tauri::Builder::default()
        .setup(|app| {
            let child = find_backend_dir(app)
                .and_then(|dir| start_backend(&dir));

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
            // Kill backend as early as possible (window close), not just on app exit
            match event.event() {
                tauri::WindowEvent::CloseRequested { .. } | tauri::WindowEvent::Destroyed => {
                    kill_backend(&event.window().app_handle());
                }
                _ => {}
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    // Use RunEvent::Exit as a guaranteed fallback — fires regardless of how the app exits
    app.run(|app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            kill_backend(app_handle);
        }
    });
}
