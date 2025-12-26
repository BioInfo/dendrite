//! GPU availability check example.

fn main() {
    println!("Checking CUDA availability...");
    println!("CUDA available: {}", dendrite_ffi::cuda::is_available());
    
    match dendrite_ffi::cuda::device_count() {
        Ok(count) => println!("GPU count: {}", count),
        Err(e) => println!("Error getting device count: {}", e),
    }
    
    #[cfg(feature = "cuda")]
    {
        match dendrite_ffi::cuda::Device::new(0) {
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
