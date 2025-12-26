//! GPU availability check example.
//!
//! Run with:
//! ```bash
//! cargo run -p dendrite-ffi --features cuda --example gpu_check
//! ```

fn main() {
    println!("Checking CUDA availability...");
    println!("CUDA available: {}", dendrite_ffi::cuda_available());

    #[cfg(feature = "cuda")]
    {
        use dendrite_ffi::cuda;

        match cuda::device_count() {
            Ok(count) => println!("GPU count: {}", count),
            Err(e) => println!("Error getting device count: {}", e),
        }

        if cuda::is_available() {
            match cuda::Device::new(0) {
                Ok(device) => {
                    println!("Created device with ordinal: {}", device.ordinal());
                    match device.synchronize() {
                        Ok(_) => println!("Device synchronized successfully!"),
                        Err(e) => println!("Sync error: {}", e),
                    }
                }
                Err(e) => println!("Error creating device: {}", e),
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled. Build with --features cuda");
    }
}
