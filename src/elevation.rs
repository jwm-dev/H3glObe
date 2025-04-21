use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use gdal::{Dataset, GeoTransform};
use h3o::{CellIndex, LatLng, Resolution};
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use bytemuck::{Pod, Zeroable};
use wgpu::{self, util::DeviceExt};

// Constants
const SEA_LEVEL: f32 = 0.0; // Sea level elevation in meters

// Color definitions
const OCEAN_COLOR: [f32; 3] = [0.0, 0.3, 0.7]; // Deep blue for oceans
const LAND_COLOR: [f32; 3] = [1.0, 1.0, 1.0];  // White for land (changed from green)

// Maximum resolution we'll use for direct sampling of GeoTIFF files
// For higher resolutions, we'll use cached data
const MAX_DIRECT_SAMPLING_RESOLUTION: u8 = 9; // Adjust based on performance needs

// Static elevation data cache
static ELEVATION_CACHE: OnceCell<Arc<ElevationData>> = OnceCell::new();

// Params struct for compute shader
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ComputeParams {
    cell_count: u32,
    resolution: u32,
    sea_level: f32,
    padding: f32, // To ensure 16-byte alignment
}

// Cache for opened GDAL datasets to avoid repeatedly loading the same files
#[derive(Debug)]
struct DatasetCache {
    datasets: HashMap<PathBuf, Dataset>,
}

// Elevation data for a specific cell
#[derive(Clone, Debug)]
pub struct CellElevation {
    pub elevation: f32,
    pub color: [f32; 3],
}

// Main struct to hold and process elevation data
#[derive(Debug)]
pub struct ElevationData {
    // Map from H3 cell index to elevation data
    cell_elevations: Mutex<HashMap<u64, CellElevation>>,
    // Directory containing the GeoTIFF files
    etopo_dir: PathBuf,
    // Available GeoTIFF files
    etopo_files: Vec<PathBuf>,
    // Cached datasets
    dataset_cache: Mutex<DatasetCache>,
    // Downsampled elevation data for lower resolutions
    downsampled_data: Mutex<HashMap<u8, HashMap<u64, CellElevation>>>,
    // Flag to track if GPU compute is available
    gpu_compute_available: Mutex<bool>,
    // Cache of processed resolutions
    processed_resolutions: Mutex<Vec<Resolution>>,
    // Cached cells for each resolution
    cells_res0: Option<Vec<CellIndex>>,
    cells_res1: Option<Vec<CellIndex>>,
    cells_res2: Option<Vec<CellIndex>>,
    cells_res3: Option<Vec<CellIndex>>,
    cells_res4: Option<Vec<CellIndex>>,
    cells_res5: Option<Vec<CellIndex>>,
    cells_res6: Option<Vec<CellIndex>>,
    cells_res7: Option<Vec<CellIndex>>,
    cells_res8: Option<Vec<CellIndex>>,
    cells_res9: Option<Vec<CellIndex>>,
    cells_res10: Option<Vec<CellIndex>>,
    cells_res11: Option<Vec<CellIndex>>,
    cells_res12: Option<Vec<CellIndex>>,
    cells_res13: Option<Vec<CellIndex>>,
    cells_res14: Option<Vec<CellIndex>>,
    cells_res15: Option<Vec<CellIndex>>,
}

impl ElevationData {
    // Initialize the elevation data singleton
    pub fn initialize(etopo_dir: &Path) -> Arc<Self> {
        ELEVATION_CACHE.get_or_init(|| {
            Arc::new(Self::new(etopo_dir))
        }).clone()
    }
    
    // Get the elevation data singleton
    pub fn get_instance() -> Option<Arc<Self>> {
        ELEVATION_CACHE.get().cloned()
    }
    
    // Create a new elevation data instance
    fn new(etopo_dir: &Path) -> Self {
        // Scan for available GeoTIFF files
        let etopo_files = scan_etopo_files(etopo_dir);
        
        println!("Found {} ETOPO GeoTIFF files", etopo_files.len());
        
        Self {
            cell_elevations: Mutex::new(HashMap::new()),
            etopo_dir: etopo_dir.to_path_buf(),
            etopo_files,
            dataset_cache: Mutex::new(DatasetCache { datasets: HashMap::new() }),
            downsampled_data: Mutex::new(HashMap::new()),
            gpu_compute_available: Mutex::new(false),
            processed_resolutions: Mutex::new(Vec::new()),
            cells_res0: None,
            cells_res1: None,
            cells_res2: None,
            cells_res3: None,
            cells_res4: None,
            cells_res5: None,
            cells_res6: None,
            cells_res7: None,
            cells_res8: None,
            cells_res9: None,
            cells_res10: None,
            cells_res11: None,
            cells_res12: None,
            cells_res13: None,
            cells_res14: None,
            cells_res15: None,
        }
    }
    
    // Get elevation and color for a cell, loading data if needed
    pub fn get_cell_elevation(&self, cell: CellIndex) -> CellElevation {
        let cell_int = u64::from(cell);
        let cell_res = u8::from(cell.resolution());
        
        // Check if we already have this cell's elevation in the primary cache
        {
            let elevations = self.cell_elevations.lock().unwrap();
            if let Some(elevation) = elevations.get(&cell_int) {
                return elevation.clone();
            }
        }
        
        // For lower resolutions, check if we have downsampled data
        if cell_res < MAX_DIRECT_SAMPLING_RESOLUTION {
            let result = {
                let downsampled = self.downsampled_data.lock().unwrap();
                if let Some(res_map) = downsampled.get(&cell_res) {
                    res_map.get(&cell_int).cloned()
                } else {
                    None
                }
            };
            
            if let Some(elevation) = result {
                return elevation;
            }
        }
        
        // If this is a high-resolution cell, compute from children or use direct sampling
        if cell_res > MAX_DIRECT_SAMPLING_RESOLUTION {
            return self.compute_aggregated_elevation(cell);
        }
        
        // For low-resolution cells, sample the center point directly
        // Get the lat/lng of the cell center
        let latlng = LatLng::from(cell);
        let lat = latlng.lat_radians().to_degrees();
        let lng = latlng.lng_radians().to_degrees();
        
        // Find and load the appropriate GeoTIFF file
        if let Some(elevation) = self.get_elevation_from_geotiff(lat, lng) {
            let color = elevation_to_color(elevation);
            let cell_elevation = CellElevation { elevation, color };
            
            // Cache this elevation in the appropriate resolution cache
            if cell_res < MAX_DIRECT_SAMPLING_RESOLUTION {
                let mut downsampled = self.downsampled_data.lock().unwrap();
                let res_map = downsampled.entry(cell_res).or_insert_with(HashMap::new);
                res_map.insert(cell_int, cell_elevation.clone());
            } else {
                let mut elevations = self.cell_elevations.lock().unwrap();
                elevations.insert(cell_int, cell_elevation.clone());
            }
            
            return cell_elevation;
        }
        
        // Default elevation if we couldn't find data
        let default = CellElevation {
            elevation: 0.0,
            color: OCEAN_COLOR,
        };
        
        {
            // Cache the default in the appropriate resolution cache
            if cell_res < MAX_DIRECT_SAMPLING_RESOLUTION {
                let mut downsampled = self.downsampled_data.lock().unwrap();
                let res_map = downsampled.entry(cell_res).or_insert_with(HashMap::new);
                res_map.insert(cell_int, default.clone());
            } else {
                let mut elevations = self.cell_elevations.lock().unwrap();
                elevations.insert(cell_int, default.clone());
            }
        }
        
        default
    }
    
    // Check if a resolution has already been processed and cached
    pub fn is_resolution_loaded(&self, resolution: Resolution) -> bool {
        let processed = self.processed_resolutions.lock().unwrap();
        processed.contains(&resolution)
    }
    
    // Process cells for a specific resolution using GPU acceleration
    pub fn process_cells_for_resolution_gpu(
        &self,
        resolution: Resolution,
        progress_callback: Option<impl Fn(f32) + Send + 'static>,
    ) -> Result<(), String> {
        let res_value = u8::from(resolution);
        
        // Check if this resolution is already processed
        {
            let processed = self.processed_resolutions.lock().unwrap();
            if processed.contains(&resolution) {
                return Ok(());
            }
        }
        
        println!("GPU-accelerated processing for resolution {}", res_value);
        
        // Step 1: Get the device and queue for GPU operations
        let device_option = get_wgpu_device_and_queue();
        if device_option.is_none() {
            eprintln!("No GPU device available, falling back to CPU");
            // Fall back to CPU implementation
            self.preload_elevation_data(resolution);
            return Ok(());
        }
        
        let (device, queue) = device_option.unwrap();
        
        // Step 2: Generate the list of cells we need to process for this resolution
        let cells = self.get_cells_for_resolution(resolution);
        let num_cells = cells.len();
        
        // If no cells to process, we're done
        if num_cells == 0 {
            return Ok(());
        }
        
        // Report initial progress
        if let Some(ref callback) = progress_callback {
            callback(0.1); // Initial progress
        }
        
        // Step 3: Prepare cell data for GPU processing
        // Create a buffer of H3Cell structs
        let mut cell_data = Vec::with_capacity(num_cells);
        for cell in &cells {
            let latlng = LatLng::from(*cell);
            let lat = latlng.lat_radians().to_degrees() as f32;
            let lng = latlng.lng_radians().to_degrees() as f32;
            
            // Pack cell data for the GPU
            cell_data.push(H3CellGpu {
                lat,
                lng,
                cell_index: u64::from(*cell),
                resolution: res_value as u32,
                _padding: 0, // Add padding for 16-byte alignment
            });
        }
        
        // Report progress
        if let Some(ref callback) = progress_callback {
            callback(0.2); // Cell data prepared
        }
        
        // Step 4: Create GPU buffers for input and output
        // Cell input buffer
        let cell_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("H3 Cell Buffer"),
            contents: bytemuck::cast_slice(&cell_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        // Result output buffer
        let result_buffer_size = std::mem::size_of::<CellElevationResultGpu>() * num_cells;
        let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Elevation Result Buffer"),
            size: result_buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create a staging buffer for reading results back to CPU
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: result_buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Step 5: Prepare elevation data for the GPU
        // We'll need to extract the data we need from our GeoTIFF files
        let mut elevation_array = Vec::new();
        let mut min_elevation = f32::MAX;
        let mut max_elevation = f32::MIN;
        let mut width: u32 = 0;
        let mut height: u32 = 0;
        
        // For simplicity in this implementation, we'll create a global elevation grid
        // In a full implementation, we'd load tiles as needed, but this gives us a proof of concept
        let sample_elevation_map = self.create_global_elevation_map(&mut elevation_array, &mut min_elevation, &mut max_elevation, &mut width, &mut height);
        
        if !sample_elevation_map {
            eprintln!("Failed to create elevation map for GPU");
            // Fall back to CPU
            self.preload_elevation_data(resolution);
            return Ok(());
        }
        
        // Create buffer for elevation data
        let elevation_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Elevation Data Buffer"),
            contents: bytemuck::cast_slice(&elevation_array),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        // Report progress
        if let Some(ref callback) = progress_callback {
            callback(0.4); // Elevation data prepared
        }
        
        // Step 6: Create uniform buffer for configuration
        let elevation_map_config = ElevationMapGpu {
            width,
            height,
            min_elevation,
            max_elevation,
        };
        
        let color_config = ColorRampGpu {
            min_elevation,
            max_elevation,
            sea_level: SEA_LEVEL,
            padding: 0.0, // Padding for alignment
        };
        
        let elevation_config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Elevation Map Config"),
            contents: bytemuck::bytes_of(&elevation_map_config),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let color_config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Color Ramp Config"),
            contents: bytemuck::bytes_of(&color_config),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Step 7: Load and compile the compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Elevation Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/elevation_compute.wgsl").into()),
        });
        
        // Step 8: Create bind group layout and pipeline
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Elevation Compute Bind Group Layout"),
            entries: &[
                // Elevation map config
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Color config
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // H3 cells input
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Elevation data
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Results output
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Elevation Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: elevation_config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: color_config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: elevation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: result_buffer.as_entire_binding(),
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Elevation Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Elevation Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Report progress
        if let Some(ref callback) = progress_callback {
            callback(0.6); // Pipeline setup complete
        }
        
        // Step 9: Execute the compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Elevation Compute Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Elevation Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Calculate workgroups based on number of cells
            // Our shader uses workgroup_size(64)
            let workgroup_count = (num_cells as f32 / 64.0).ceil() as u32;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        
        // Copy result to staging buffer for reading
        encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, result_buffer_size as u64);
        
        // Submit the work
        queue.submit(std::iter::once(encoder.finish()));
        
        // Report progress
        if let Some(ref callback) = progress_callback {
            callback(0.8); // Compute submitted
        }
        
        // Step 10: Read back the results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        
        // Wait for GPU work to complete
        device.poll(wgpu::MaintainBase::Wait);
        
        // Process results
        if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
            let data = buffer_slice.get_mapped_range();
            let results: &[CellElevationResultGpu] = bytemuck::cast_slice(&data);
            
            // Create a new resolution cache map
            let mut cell_data = HashMap::with_capacity(num_cells);
            
            // Process the results
            for result in results {
                let color = [result.color[0], result.color[1], result.color[2]];
                
                let cell_elevation = CellElevation {
                    elevation: result.elevation,
                    color,
                };
                
                cell_data.insert(result.cell_index, cell_elevation);
            }
            
            // Store the processed data in our downsampled cache
            {
                let mut downsampled = self.downsampled_data.lock().unwrap();
                downsampled.insert(res_value, cell_data);
                
                // Add to processed resolutions
                let mut processed = self.processed_resolutions.lock().unwrap();
                if !processed.contains(&resolution) {
                    processed.push(resolution);
                }
            }
            
            // Drop the mapped data
            drop(data);
            staging_buffer.unmap();
            
            // Report completion
            if let Some(ref callback) = progress_callback {
                callback(1.0);
            }
            
            println!("GPU processing complete for resolution {}", res_value);
            
            Ok(())
        } else {
            Err("Failed to read back GPU results".to_string())
        }
    }
    
    // Create a global elevation map from our GeoTIFF data for GPU processing
    fn create_global_elevation_map(&self, 
                                  elevation_array: &mut Vec<f32>, 
                                  min_elevation: &mut f32, 
                                  max_elevation: &mut f32,
                                  width: &mut u32,
                                  height: &mut u32) -> bool {
        // For this implementation, we'll create a simplified global elevation map
        // In a production system, you might want to use more sophisticated tile management
        
        // Create a low-resolution global grid (1 degree resolution is sufficient for preprocessing)
        *width = 360;  // 1 degree per cell
        *height = 180; // 1 degree per cell
        elevation_array.resize((*width * *height) as usize, 0.0);
        
        // Fill in the array with elevation data
        for lat_idx in 0..*height {
            let lat = 90.0 - (lat_idx as f64 * 180.0 / *height as f64);
            
            for lng_idx in 0..*width {
                let lng = -180.0 + (lng_idx as f64 * 360.0 / *width as f64);
                
                // Get elevation from our GeoTIFF files
                if let Some(elevation) = self.get_elevation_from_geotiff(lat, lng) {
                    let idx = (lat_idx * *width + lng_idx) as usize;
                    elevation_array[idx] = elevation;
                    
                    // Update min/max
                    *min_elevation = min_elevation.min(elevation);
                    *max_elevation = max_elevation.max(elevation);
                }
            }
        }
        
        // If min/max are still at their initial values, something went wrong
        if *min_elevation == f32::MAX || *max_elevation == f32::MIN {
            return false;
        }
        
        true
    }
    
    // Compute elevation for a cell by averaging its children
    fn compute_aggregated_elevation(&self, cell: CellIndex) -> CellElevation {
        let cell_int = u64::from(cell);
        let cell_res = u8::from(cell.resolution());
        
        // For medium resolution cells, sample directly rather than aggregating children
        if cell_res <= MAX_DIRECT_SAMPLING_RESOLUTION {
            return self.direct_compute_elevation(cell);
        }
        
        // For high-resolution cells, use a limited set of children to improve performance
        // Instead of going all the way to res 15, limit the depth
        let target_res = if cell_res >= 12 {
            cell_res + 1 // Just one level deeper for very high resolutions
        } else {
            std::cmp::min(cell_res + 2, MAX_DIRECT_SAMPLING_RESOLUTION) // Limited depth for medium resolutions
        };
        
        let target_res_enum = Resolution::try_from(target_res).unwrap();
        let children = get_limited_children(cell, target_res_enum);
        
        if children.is_empty() {
            // Fall back to direct computation
            return self.direct_compute_elevation(cell);
        }
        
        // Process children (no parallel processing for small sets to reduce overhead)
        let elevations: Vec<CellElevation> = children.iter()
            .map(|child| self.get_cell_elevation(*child))
            .collect();
        
        // Count types
        let mut total_elevation = 0.0;
        let mut land_count = 0;
        let mut ocean_count = 0;
        
        for elev in &elevations {
            total_elevation += elev.elevation;
            if elev.elevation > SEA_LEVEL {
                land_count += 1;
            } else {
                ocean_count += 1;
            }
        }
        
        // Compute average elevation
        let avg_elevation = if elevations.is_empty() {
            0.0
        } else {
            total_elevation / elevations.len() as f32
        };
        
        // Determine dominant type - simplified to just land vs ocean
        let color = if land_count > ocean_count {
            LAND_COLOR
        } else {
            OCEAN_COLOR
        };
        
        let result = CellElevation {
            elevation: avg_elevation,
            color,
        };
        
        // Cache the result
        {
            let mut elevations_map = self.cell_elevations.lock().unwrap();
            elevations_map.insert(cell_int, result.clone());
        }
        
        result
    }
    
    // Direct elevation computation for cells
    fn direct_compute_elevation(&self, cell: CellIndex) -> CellElevation {
        let latlng = LatLng::from(cell);
        let lat = latlng.lat_radians().to_degrees();
        let lng = latlng.lng_radians().to_degrees();
        
        if let Some(elevation) = self.get_elevation_from_geotiff(lat, lng) {
            let color = elevation_to_color(elevation);
            CellElevation { elevation, color }
        } else {
            // Default for missing data
            CellElevation {
                elevation: 0.0,
                color: OCEAN_COLOR,
            }
        }
    }
    
    // Extract elevation from GeoTIFF at the given coordinates
    fn get_elevation_from_geotiff(&self, lat: f64, lng: f64) -> Option<f32> {
        // Find the right ETOPO file for this coordinate
        let file_path = self.find_etopo_file_for_coords(lat, lng)?;
        
        // Get or open the dataset
        let dataset_and_transform = self.get_or_open_dataset(&file_path)?;
        let dataset = &dataset_and_transform.0;
        let geotransform = dataset_and_transform.1;
        
        // Convert lat/lng to pixel coordinates
        let (pixel_x, pixel_y) = self.latlon_to_pixel(lat, lng, &geotransform);
        
        // Make sure the pixel is within bounds and convert to usize
        let size = dataset.raster_size();
        if pixel_x < 0 || pixel_y < 0 || pixel_x >= size.0 as isize || pixel_y >= size.1 as isize {
            return None;
        }
        
        let pixel_x_usize = pixel_x as usize;
        let pixel_y_usize = pixel_y as usize;
        
        // Get band (assuming band 1 for elevation data)
        let band = match dataset.rasterband(1) {
            Ok(b) => b,
            Err(_) => { return None; }
        };
        
        // Try different approaches to read the pixel value
        
        // Approach 1: Direct pixel reading
        match band.read_as::<f32>((pixel_x_usize.try_into().unwrap(), pixel_y_usize.try_into().unwrap()), (1, 1), (1, 1), None) {
            Ok(data) => {
                if !data.data.is_empty() {
                    let elevation = data.data[0];
                    Some(elevation)
                } else {
                    // If the buffer is empty, try another approach
                    self.try_alternate_reading_method(band, pixel_x_usize, pixel_y_usize)
                }
            },
            Err(_) => {
                self.try_alternate_reading_method(band, pixel_x_usize, pixel_y_usize)
            }
        }
    }
    
    // Alternative method to read pixel data
    fn try_alternate_reading_method(&self, band: gdal::raster::RasterBand, x: usize, y: usize) -> Option<f32> {
        // Try to read a small window around the target pixel (3x3)
        // This sometimes helps with edge cases or when direct pixel access fails
        let window_size = 3;
        let x_start = if x >= window_size / 2 { x - window_size / 2 } else { 0 };
        let y_start = if y >= window_size / 2 { y - window_size / 2 } else { 0 };
        
        match band.read_as::<f32>((x_start.try_into().unwrap(), y_start.try_into().unwrap()), (window_size, window_size), (window_size, window_size), None) {
            Ok(data) => {
                if !data.data.is_empty() {
                    // Calculate the index of our target pixel in the window
                    let x_offset = x - x_start;
                    let y_offset = y - y_start;
                    let index = y_offset * window_size + x_offset;
                    
                    if index < data.data.len() {
                        let elevation = data.data[index];
                        Some(elevation)
                    } else {
                        // If we can't get the exact pixel, use the average of available data
                        let avg_elevation = data.data.iter().sum::<f32>() / data.data.len() as f32;
                        Some(avg_elevation)
                    }
                } else {
                    // Last resort: try reading the entire band and sampling from it
                    match band.read_as::<f32>((0, 0), band.size(), band.size(), None) {
                        Ok(full_data) => {
                            if !full_data.data.is_empty() {
                                let width = band.size().0;
                                let index = y * width + x;
                                if index < full_data.data.len() {
                                    Some(full_data.data[index])
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        },
                        Err(_) => {
                            None
                        }
                    }
                }
            },
            Err(_) => {
                None
            }
        }
    }
    
    // Get or open a dataset from cache, returning owned dataset and geotransform
    // Note: Instead of cloning Dataset which isn't supported, open a new Dataset instance
    fn get_or_open_dataset(&self, file_path: &Path) -> Option<(Dataset, GeoTransform)> {
        // First, check if we have this dataset in cache
        {
            let cache = self.dataset_cache.lock().unwrap();
            if let Some(dataset) = cache.datasets.get(file_path) {
                if let Ok(gt) = dataset.geo_transform() {
                    // Since we can't clone Dataset, open a new instance
                    if let Ok(new_dataset) = Dataset::open(file_path) {
                        return Some((new_dataset, gt));
                    }
                }
            }
        }
        
        // If not in cache or couldn't get transform, open a new dataset
        match Dataset::open(file_path) {
            Ok(dataset) => {
                match dataset.geo_transform() {
                    Ok(gt) => {
                        // Store in cache
                        let mut cache = self.dataset_cache.lock().unwrap();
                        // Make a new Dataset for the cache
                        if let Ok(cache_dataset) = Dataset::open(file_path) {
                            cache.datasets.insert(file_path.to_path_buf(), cache_dataset);
                        }
                        // Return the original dataset and transform
                        Some((dataset, gt))
                    },
                    Err(_) => {
                        None
                    }
                }
            },
            Err(_) => {
                None
            }
        }
    }
    
    // Convert latitude and longitude to pixel coordinates
    fn latlon_to_pixel(&self, lat: f64, lng: f64, geotransform: &GeoTransform) -> (isize, isize) {
        // Note: GDAL GeoTransform format is:
        // [0] = top left x (longitude of the top-left corner)
        // [1] = w-e pixel resolution
        // [2] = rotation, 0 if image is "north up"
        // [3] = top left y (latitude of the top-left corner)
        // [4] = rotation, 0 if image is "north up"
        // [5] = n-s pixel resolution (negative for northern hemisphere)
        
        // Calculate pixel coordinates correctly:
        // For longitude (x): how many pixels from the left edge
        let pixel_x = ((lng - geotransform[0]) / geotransform[1]) as isize;
        
        // For latitude (y): how many pixels from the top edge
        // Handle potential sign issues based on hemisphere
        let pixel_y = if geotransform[5] < 0.0 {
            // Northern hemisphere typically has negative [5]
            ((lat - geotransform[3]) / geotransform[5]) as isize
        } else {
            // Southern hemisphere might have positive [5]
            ((geotransform[3] - lat) / geotransform[5].abs()) as isize
        };
        
        (pixel_x, pixel_y)
    }
    
    // Find the appropriate ETOPO file for given coordinates
    fn find_etopo_file_for_coords(&self, lat: f64, lng: f64) -> Option<PathBuf> {
        // Ensure longitude is in -180 to 180 range
        let lng = if lng > 180.0 { lng - 360.0 } else if lng < -180.0 { lng + 360.0 } else { lng };
        
        // Find the right file based on the ETOPO naming convention
        for file_path in &self.etopo_files {
            let file_name = file_path.file_name()?.to_str()?;
            
            // Parse the file name to extract coordinates
            if let Some((file_lat, file_lng, is_north, is_east)) = parse_etopo_filename(file_name) {
                // Convert file coordinates to adjust for northern/southern and eastern/western hemispheres
                let file_lat_numeric = if is_north { file_lat as f64 } else { -(file_lat as f64) };
                let file_lng_numeric = if is_east { file_lng as f64 } else { -(file_lng as f64) };
                
                // Calculate the bottom and right edges (15 arc-seconds tiles are 15 degrees)
                let lat_bottom = file_lat_numeric - 15.0;
                let lng_right = file_lng_numeric + 15.0;

                // Different comparison logic for northern vs southern hemisphere
                let lat_in_range = if is_north {
                    // For northern hemisphere: lat should be <= top edge and > bottom edge
                    lat <= file_lat_numeric && lat > lat_bottom
                } else {
                    // For southern hemisphere: lat should be >= bottom edge (which is more negative) and < top edge
                    lat >= lat_bottom && lat < file_lat_numeric
                };
                
                // Check if our coordinate falls within this tile
                if lat_in_range && lng >= file_lng_numeric && lng < lng_right {
                    return Some(file_path.clone());
                }
            }
        }
        
        None
    }
    
    // Get color for a cell based on its index
    pub fn get_color_for_cell(&self, cell: CellIndex) -> [f32; 3] {
        let elevation = self.get_cell_elevation(cell);
        elevation.color
    }
    
    // Preload elevation data for lower resolution cells to improve performance
    pub fn preload_elevation_data(&self, resolution: Resolution) {
        let res_value = u8::from(resolution);
        
        // Only preload for resolutions below our cutoff
        if res_value >= MAX_DIRECT_SAMPLING_RESOLUTION {
            println!("Skipping preload for resolution {}, above cutoff", res_value);
            return;
        }
        
        println!("Preloading elevation data for resolution {}", res_value);
        
        // Check if this resolution is already processed
        {
            let processed = self.processed_resolutions.lock().unwrap();
            if processed.contains(&resolution) {
                println!("Resolution {} already processed, skipping", res_value);
                return;
            }
        }
        
        // Process cells using our collection logic
        let cells = self.get_cells_for_resolution(resolution);
        println!("Processing {} cells for resolution {}", cells.len(), res_value);
        
        // Create maps for elevation data
        let mut cell_data = HashMap::with_capacity(cells.len());
        
        // Process in parallel for better performance
        let elevations: Vec<(u64, CellElevation)> = cells.par_iter()
            .map(|&cell| {
                let cell_int = u64::from(cell);
                let elevation = self.direct_compute_elevation(cell);
                (cell_int, elevation)
            })
            .collect();
        
        // Store in the resolution-specific cache
        for (cell_int, elevation) in elevations {
            cell_data.insert(cell_int, elevation);
        }
        
        // Update caches
        {
            let mut downsampled = self.downsampled_data.lock().unwrap();
            downsampled.insert(res_value, cell_data);
            
            // Mark as processed
            let mut processed = self.processed_resolutions.lock().unwrap();
            if !processed.contains(&resolution) {
                processed.push(resolution);
            }
        }
        
        println!("Preloaded data for resolution {}", res_value);
    }

    // Add get_cells_for_resolution method implementation
    pub fn get_cells_for_resolution(&self, resolution: Resolution) -> Vec<CellIndex> {
        // Implementation that returns cells for the given resolution
        // This can be based on the existing process_cells_for_resolution_gpu method
        let cells = match resolution {
            Resolution::Zero => self.cells_res0.clone(),
            Resolution::One => self.cells_res1.clone(),
            Resolution::Two => self.cells_res2.clone(),
            Resolution::Three => self.cells_res3.clone(),
            Resolution::Four => self.cells_res4.clone(),
            Resolution::Five => self.cells_res5.clone(),
            Resolution::Six => self.cells_res6.clone(),
            Resolution::Seven => self.cells_res7.clone(),
            Resolution::Eight => self.cells_res8.clone(),
            Resolution::Nine => self.cells_res9.clone(),
            Resolution::Ten => self.cells_res10.clone(),
            Resolution::Eleven => self.cells_res11.clone(),
            Resolution::Twelve => self.cells_res12.clone(),
            Resolution::Thirteen => self.cells_res13.clone(),
            Resolution::Fourteen => self.cells_res14.clone(),
            Resolution::Fifteen => self.cells_res15.clone(),
        };
        
        cells.unwrap_or_default()
    }
}

// Convert elevation value to a color
fn elevation_to_color(elevation: f32) -> [f32; 3] {
    if elevation > SEA_LEVEL {
        LAND_COLOR // White for any land (elevation > 0)
    } else {
        OCEAN_COLOR // Blue for sea (elevation <= 0)
    }
}

// Get a limited set of children cells for performance
fn get_limited_children(cell: CellIndex, target_res: Resolution) -> Vec<CellIndex> {
    let cell_res = cell.resolution();
    
    // If we're already at the target resolution, return this cell
    if cell_res == target_res {
        return vec![cell];
    }
    
    // If we're beyond the target, return empty
    if u8::from(cell_res) > u8::from(target_res) {
        return vec![];
    }
    
    // If just one level away, use direct children
    if u8::from(cell_res) + 1 == u8::from(target_res) {
        let child_res = Resolution::try_from(u8::from(cell_res) + 1).unwrap();
        return cell.children(child_res).collect();
    }
    
    // For deeper levels, use sampling strategy instead of complete enumeration
    // For example, rather than getting ALL children several levels down,
    // just sample a few representative children
    let next_res = Resolution::try_from(u8::from(cell_res) + 1).unwrap();
    let direct_children: Vec<CellIndex> = cell.children(next_res).collect();
    
    // For performance, if there are many direct children, just sample a subset
    let sample_children = if direct_children.len() > 7 {
        // Sample every other child for better performance
        direct_children.into_iter()
            .step_by(2)
            .collect()
    } else {
        direct_children
    };
    
    // Recursively get children from the sampled subset
    let mut result = Vec::new();
    for child in sample_children {
        result.extend(get_limited_children(child, target_res));
    }
    
    result
}

// Get all children cells at a target resolution - original implementation
// This is kept but not used directly for performance reasons
fn get_cell_children_at_resolution(cell: CellIndex, target_res: Resolution) -> Vec<CellIndex> {
    let cell_res = cell.resolution();
    
    // If we're already at the target resolution, return this cell
    if cell_res == target_res {
        return vec![cell];
    }
    
    // If we're beyond the target, return empty
    if u8::from(cell_res) > u8::from(target_res) {
        return vec![];
    }
    
    // If just one level away, use direct children
    if u8::from(cell_res) + 1 == u8::from(target_res) {
        let child_res = Resolution::try_from(u8::from(cell_res) + 1).unwrap();
        return cell.children(child_res).collect();
    }
    
    // Otherwise, recursively get children
    let mut result = Vec::new();
    let next_res = Resolution::try_from(u8::from(cell_res) + 1).unwrap();
    let direct_children: Vec<CellIndex> = cell.children(next_res).collect();
    
    for child in direct_children {
        result.extend(get_cell_children_at_resolution(child, target_res));
    }
    
    result
}

// Scan for ETOPO files in the specified directory
fn scan_etopo_files(directory: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    
    // Make sure the directory exists
    if !directory.exists() || !directory.is_dir() {
        println!("Error: ETOPO directory does not exist or is not a directory: {:?}", directory);
        return files;
    }
    
    // Try to read directory contents
    match fs::read_dir(directory) {
        Ok(entries) => {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    
                    // Check if it's a surface elevation file
                    if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                        if file_name.ends_with("_surface.tif") && file_name.starts_with("ETOPO_2022") {
                            files.push(path);
                        }
                    }
                }
            }
        }
        Err(e) => {
            println!("Error reading ETOPO directory: {}", e);
        }
    }
    
    files
}

// Parse an ETOPO filename to extract coordinates
// Example: ETOPO_2022_v1_15s_N00E000_surface.tif
fn parse_etopo_filename(filename: &str) -> Option<(u8, u16, bool, bool)> {
    // Split by underscores
    let parts: Vec<&str> = filename.split('_').collect();
    
    // Need at least 5 parts: ETOPO, 2022, v1, 15s, N00E000
    if parts.len() < 5 {
        return None;
    }
    
    // Extract the coordinate part (e.g., "N00E000")
    let coords = parts[4];
    if coords.len() < 7 {
        return None;
    }
    
    // Extract information
    let is_north = coords.starts_with('N');
    let is_east = coords.contains('E');
    
    // Extract the latitude (2 digits after N/S)
    let lat_str = &coords[1..3];
    let lat = match lat_str.parse::<u8>() {
        Ok(val) => val,
        Err(_) => return None,
    };
    
    // Extract the longitude (3 digits after E/W)
    let lng_pos = if is_east { coords.find('E') } else { coords.find('W') };
    if let Some(pos) = lng_pos {
        if pos + 3 >= coords.len() {
            return None;
        }
        
        let lng_str = &coords[pos+1..pos+4];
        if let Ok(lng) = lng_str.parse::<u16>() {
            return Some((lat, lng, is_north, is_east));
        }
    }
    
    None
}

// Add GPU-related struct definitions at the end of the file
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct H3CellGpu {
    lat: f32,
    lng: f32,
    cell_index: u64,
    resolution: u32,
    _padding: u32, // Add padding for 16-byte alignment
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CellElevationResultGpu {
    cell_index: u64,
    elevation: f32,
    color: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ElevationMapGpu {
    width: u32,
    height: u32,
    min_elevation: f32,
    max_elevation: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ColorRampGpu {
    min_elevation: f32,
    max_elevation: f32,
    sea_level: f32,
    padding: f32, // For 16-byte alignment
}

// Helper function to get wgpu device and queue
fn get_wgpu_device_and_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
    // Create a new instance and request an adapter
    pollster::block_on(async {
        // Create a WGPU instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let adapter = match instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await {
            Ok(adapter) => adapter,
            Err(_) => return None,
        };
        
        // Request the device and queue
        match adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::default(),
            },
        ).await {
            Ok((device, queue)) => Some((device, queue)),
            Err(_) => None,
        }
    })
}