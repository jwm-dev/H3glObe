use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use h3o::{Resolution, CellIndex};
use wgpu::{Device, Queue};
use crate::elevation::ElevationData;

use std::collections::HashSet;
use wgpu::{self, ShaderModule};

pub struct ElevationPreloader {
    thread_handle: Option<thread::JoinHandle<()>>,
    is_running: Arc<AtomicBool>,
    is_complete: Arc<AtomicBool>,
    progress: Arc<Mutex<f32>>,
    current_resolution: Arc<Mutex<Resolution>>,
    target_resolution: Arc<Mutex<Resolution>>,
    elevations: Arc<ElevationData>,
}

impl ElevationPreloader {
    pub fn new(
        initial_resolution: Resolution,
        elevations: Arc<ElevationData>,
        device: Arc<Device>,
        queue: Arc<Queue>
    ) -> Self {
        let is_running = Arc::new(AtomicBool::new(true));
        let is_complete = Arc::new(AtomicBool::new(false));
        let progress = Arc::new(Mutex::new(0.0));
        let current_resolution = Arc::new(Mutex::new(initial_resolution));
        let target_resolution = Arc::new(Mutex::new(initial_resolution));
        
        // Clone references for thread
        let thread_is_running = is_running.clone();
        let thread_is_complete = is_complete.clone();
        let thread_progress = progress.clone();
        let thread_current = current_resolution.clone();
        let thread_target = target_resolution.clone();
        let thread_elevations = elevations.clone();
        let thread_device = device;
        let thread_queue = queue;
        
        // Create background thread for processing
        let thread_handle = thread::spawn(move || {
            println!("Elevation preloader thread started");
            
            // Sleep briefly to let the main thread continue initializing
            thread::sleep(Duration::from_millis(500));
            
            // Background processing loop
            while thread_is_running.load(Ordering::SeqCst) {
                // Check if we need to process a new resolution
                let current = *thread_current.lock().unwrap();
                let target = *thread_target.lock().unwrap();
                
                if current != target {
                    println!("Resolution change requested: {} -> {}", 
                        u8::from(current), u8::from(target));
                    
                    // Reset progress
                    *thread_progress.lock().unwrap() = 0.0;
                    thread_is_complete.store(false, Ordering::SeqCst);
                    
                    // Check if this resolution is already in the cache
                    if thread_elevations.is_resolution_loaded(target) {
                        println!("Resolution {} already loaded from cache", u8::from(target));
                        *thread_progress.lock().unwrap() = 1.0;
                        thread_is_complete.store(true, Ordering::SeqCst);
                        *thread_current.lock().unwrap() = target;
                        continue;
                    }
                    
                    println!("Processing resolution {} using GPU acceleration", u8::from(target));
                    
                    // Start measuring time
                    let start_time = Instant::now();
                    
                    // Process data using GPU compute shader
                    let result = process_resolution_gpu(
                        target,
                        thread_elevations.clone(),
                        thread_device.clone(),
                        thread_queue.clone(),
                        |p| {
                            // Update progress
                            *thread_progress.lock().unwrap() = p;
                        }
                    );
                    
                    match result {
                        Ok(_) => {
                            let elapsed = start_time.elapsed();
                            println!("Resolution {} processed in {:?}", u8::from(target), elapsed);
                            
                            // Set progress to complete
                            *thread_progress.lock().unwrap() = 1.0;
                            thread_is_complete.store(true, Ordering::SeqCst);
                            
                            // Update current resolution
                            *thread_current.lock().unwrap() = target;
                        },
                        Err(e) => {
                            eprintln!("Error processing resolution {}: {}", u8::from(target), e);
                        }
                    }
                }
                
                // Sleep a bit to avoid excessive CPU usage
                thread::sleep(Duration::from_millis(100));
            }
            
            println!("Elevation preloader thread stopped");
        });
        
        Self {
            thread_handle: Some(thread_handle),
            is_running,
            is_complete,
            progress,
            current_resolution,
            target_resolution,
            elevations,
        }
    }
    
    pub fn set_target_resolution(&mut self, resolution: Resolution) {
        // Only change target if different to avoid unnecessary processing
        let current = *self.current_resolution.lock().unwrap();
        if current != resolution {
            println!("Setting target resolution to {}", u8::from(resolution));
            *self.target_resolution.lock().unwrap() = resolution;
            self.is_complete.store(false, Ordering::SeqCst);
        }
    }
    
    pub fn progress(&self) -> f32 {
        *self.progress.lock().unwrap()
    }
    
    pub fn is_complete(&self) -> bool {
        self.is_complete.load(Ordering::SeqCst)
    }
    
    pub fn current_resolution(&self) -> Resolution {
        *self.current_resolution.lock().unwrap()
    }
    
    pub fn target_resolution(&self) -> Resolution {
        *self.target_resolution.lock().unwrap()
    }
    
    pub fn get_status_text(&self) -> String {
        let current = u8::from(*self.current_resolution.lock().unwrap());
        let target = u8::from(*self.target_resolution.lock().unwrap());
        
        if current == target {
            if self.is_complete.load(Ordering::SeqCst) {
                format!("Resolution: {}", current)
            } else {
                format!("Processing resolution: {} ({:.1}%)", 
                    current, self.progress() * 100.0)
            }
        } else {
            format!("Switching resolution: {} → {} ({:.1}%)", 
                current, target, self.progress() * 100.0)
        }
    }
}

impl Drop for ElevationPreloader {
    fn drop(&mut self) {
        // Signal the thread to stop
        self.is_running.store(false, Ordering::SeqCst);
        
        // Wait for thread to finish
        if let Some(handle) = self.thread_handle.take() {
            handle.join().ok();
        }
    }
}

// Process a resolution using GPU compute shader
fn process_resolution_gpu(
    resolution: Resolution,
    elevations: Arc<ElevationData>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    progress_callback: impl Fn(f32) + Send + 'static,
) -> Result<(), String> {
    // Call into the ElevationData struct to process this resolution with GPU acceleration
    elevations.process_cells_for_resolution_gpu(resolution, Some(progress_callback))?;
    
    Ok(())
}

pub struct Preloader {
    /// The GPU device
    device: Arc<Device>,
    /// The GPU queue
    queue: Arc<Queue>,
    /// The elevation compute shader
    compute_shader: ShaderModule,
    /// List of resolutions currently being processed
    processing: Mutex<HashSet<Resolution>>,
    /// List of resolutions that have been loaded
    loaded_resolutions: Mutex<HashSet<Resolution>>,
    /// Reference to the elevation data
    elevation_data: Arc<ElevationData>,
}

impl Preloader {
    /// Create a new preloader instance
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, elevation_data: Arc<ElevationData>) -> Self {
        // Load the compute shader
        let shader_source = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Elevation Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/elevation_compute.wgsl").into()),
        });
        
        Self {
            device,
            queue,
            compute_shader: shader_source,
            processing: Mutex::new(HashSet::new()),
            loaded_resolutions: Mutex::new(HashSet::new()),
            elevation_data,
        }
    }
    
    /// Check if a resolution is currently being processed
    pub fn is_processing(&self, resolution: Resolution) -> bool {
        let processing = self.processing.lock().unwrap();
        processing.contains(&resolution)
    }
    
    /// Check if a resolution has been loaded
    pub fn is_resolution_loaded(&self, resolution: Resolution) -> bool {
        // First check our internal cache
        {
            let loaded = self.loaded_resolutions.lock().unwrap();
            if loaded.contains(&resolution) {
                return true;
            }
        }
        
        // Then check with the elevation data provider
        self.elevation_data.is_resolution_loaded(resolution)
    }
    
    /// Start background loading of a specific resolution
    pub fn preload_resolution(&self, resolution: Resolution) -> bool {
        // Check if already loaded or processing
        if self.is_resolution_loaded(resolution) {
            return true; // Already loaded
        }
        
        {
            let mut processing = self.processing.lock().unwrap();
            if processing.contains(&resolution) {
                return false; // Already processing
            }
            
            // Mark as processing
            processing.insert(resolution);
        }
        
        // Clone necessary references for the thread
        let device = self.device.clone();
        let queue = self.queue.clone();
        let elevation_data = self.elevation_data.clone();
        
        // Get a reference back to self for updating the processing state
        let processing = Arc::new(self.processing.clone());
        let loaded_resolutions = Arc::new(self.loaded_resolutions.clone());
        
        // Start a background thread
        thread::spawn(move || {
            println!("Starting background processing for resolution {}", u8::from(resolution));
            
            // Here we could use the GPU to process data, but for now we'll use the CPU implementation
            // as a fallback to ensure compatibility
            
            // Try GPU first, fall back to CPU if needed
            let result = elevation_data.process_cells_for_resolution_gpu(
                resolution,
                Some(|progress| {
                    println!("Processing resolution {}: {:.1}%", u8::from(resolution), progress * 100.0);
                }),
            );
            
            if result.is_err() {
                // Fall back to CPU-based implementation
                println!("GPU processing failed, falling back to CPU for resolution {}", u8::from(resolution));
                elevation_data.preload_elevation_data(resolution);
            }
            
            // Mark resolution as loaded
            {
                let mut loaded = loaded_resolutions.lock().unwrap();
                loaded.insert(resolution);
            }
            
            // Mark as no longer processing
            {
                let mut processing_guard = processing.lock().unwrap();
                processing_guard.remove(&resolution);
            }
            
            println!("Completed processing for resolution {}", u8::from(resolution));
        });
        
        true
    }
    
    /// Preload a range of resolutions in the background
    pub fn preload_resolution_range(&self, min_res: u8, max_res: u8) {
        for res in min_res..=max_res {
            if let Ok(resolution) = Resolution::try_from(res) {
                if !self.is_resolution_loaded(resolution) {
                    self.preload_resolution(resolution);
                    
                    // Sleep briefly between starting each resolution to avoid overwhelming the system
                    thread::sleep(std::time::Duration::from_millis(100));
                }
            }
        }
    }
}