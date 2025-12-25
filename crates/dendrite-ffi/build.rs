//! Build script for dendrite-ffi.
//!
//! This script:
//! 1. Generates Rust bindings for FlashInfer C++ headers
//! 2. Compiles any custom CUDA kernels
//! 3. Links against FlashInfer and CUDA libraries

fn main() {
    // Re-run if build script changes
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc/");

    // Check for CUDA
    let cuda_available = std::env::var("CUDA_HOME").is_ok()
        || std::env::var("CUDA_PATH").is_ok()
        || std::path::Path::new("/usr/local/cuda").exists();

    if !cuda_available {
        println!("cargo:warning=CUDA not found, building without GPU support");
        return;
    }

    // Get CUDA path
    let cuda_path = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");

    // TODO: Build FlashInfer integration
    // This would involve:
    // 1. Using bindgen to generate bindings from FlashInfer headers
    // 2. Compiling wrapper functions if needed
    // 3. Linking against FlashInfer library

    // For now, we just set up the link paths
    if let Ok(flashinfer_path) = std::env::var("FLASHINFER_PATH") {
        println!("cargo:rustc-link-search=native={}/lib", flashinfer_path);
        println!("cargo:rustc-link-lib=dylib=flashinfer");

        // Generate bindings if header exists
        let header = format!("{}/include/flashinfer.h", flashinfer_path);
        if std::path::Path::new(&header).exists() {
            generate_bindings(&header);
        }
    }
}

#[allow(dead_code)]
fn generate_bindings(header_path: &str) {
    let bindings = bindgen::Builder::default()
        .header(header_path)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("flashinfer_.*")
        .allowlist_type("flashinfer_.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("flashinfer_bindings.rs"))
        .expect("Couldn't write bindings!");
}
