//! CUDA utilities.

use crate::error::{FfiError, Result};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaStream};

/// Check if CUDA is available.
#[cfg(feature = "cuda")]
pub fn is_available() -> bool {
    CudaDevice::new(0).is_ok()
}

#[cfg(not(feature = "cuda"))]
pub fn is_available() -> bool {
    false
}

/// Get number of available CUDA devices.
#[cfg(feature = "cuda")]
pub fn device_count() -> Result<usize> {
    cudarc::driver::result::device::get_count()
        .map(|c| c as usize)
        .map_err(|e| FfiError::CudaError(e.to_string()))
}

#[cfg(not(feature = "cuda"))]
pub fn device_count() -> Result<usize> {
    Ok(0)
}

/// CUDA device wrapper.
#[cfg(feature = "cuda")]
pub struct Device {
    device: std::sync::Arc<CudaDevice>,
}

#[cfg(feature = "cuda")]
impl Device {
    /// Create a new CUDA device handle.
    pub fn new(ordinal: usize) -> Result<Self> {
        let device = CudaDevice::new(ordinal).map_err(|e| FfiError::CudaError(e.to_string()))?;
        Ok(Self { device })
    }

    /// Get device ordinal.
    pub fn ordinal(&self) -> usize {
        self.device.ordinal()
    }

    /// Synchronize device.
    pub fn synchronize(&self) -> Result<()> {
        self.device
            .synchronize()
            .map_err(|e| FfiError::CudaError(e.to_string()))
    }

    /// Get free and total memory.
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        // Note: cudarc doesn't expose mem_get_info directly
        // This is a placeholder
        Ok((0, 0))
    }
}

/// CUDA stream wrapper.
#[cfg(feature = "cuda")]
pub struct Stream {
    stream: CudaStream,
}

#[cfg(feature = "cuda")]
impl Stream {
    /// Create a new CUDA stream.
    pub fn new(device: &Device) -> Result<Self> {
        let stream = device
            .device
            .fork_default_stream()
            .map_err(|e| FfiError::CudaError(e.to_string()))?;
        Ok(Self { stream })
    }

    /// Synchronize stream.
    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| FfiError::CudaError(e.to_string()))
    }
}
