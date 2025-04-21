// Elevation compute shader for processing H3 cells in parallel
// This shader processes a batch of H3 cells to calculate their elevation and color
// Optimized version with advanced GPU computation techniques

// Workgroup size optimization - increased from 64 to 128 threads per workgroup for better GPU utilization
// Most modern GPUs can run 32 or 64 threads per warp/wavefront, so 128 allows for 2-4 warps per workgroup
@compute @workgroup_size(128)

struct ElevationMapConfig {
    width: u32,
    height: u32,
    min_elevation: f32,
    max_elevation: f32,
}

struct ColorConfig {
    min_elevation: f32,
    max_elevation: f32,
    sea_level: f32,
    padding: f32, // Padding for alignment
}

struct H3Cell {
    lat: f32,
    lng: f32,
    cell_index: u64,
    resolution: u32,
}

struct CellElevationResult {
    cell_index: u64,
    elevation: f32,
    color: vec3<f32>,
}

// Color definitions - as requested
const OCEAN_COLOR: vec3<f32> = vec3<f32>(0.0, 0.3, 0.7); // Deep blue for oceans
const LAND_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);  // White for land

// Binding group optimized for coherent access patterns
@group(0) @binding(0) var<uniform> elevation_config: ElevationMapConfig;
@group(0) @binding(1) var<uniform> color_config: ColorConfig;
@group(0) @binding(2) var<storage, read> cells: array<H3Cell>;
@group(0) @binding(3) var<storage, read> elevation_map: array<f32>;
@group(0) @binding(4) var<storage, write> results: array<CellElevationResult>;

// Fast multiplication by reciprocal instead of division for coordinate conversion
// This is faster than division on many GPUs
fn get_elevation(lat: f32, lng: f32) -> f32 {
    // Precalculate reciprocals for faster math operations
    let lat_scale = 1.0 / 180.0;
    let lng_scale = 1.0 / 360.0;
    
    // Convert lat/lng to normalized coordinates [0,1]
    // Using mad (multiply-add) operations which are optimized on GPUs
    let lat_norm = (90.0 - lat) * lat_scale;
    let lng_norm = (lng + 180.0) * lng_scale;
    
    // Convert to integer indices using fast bit operations where possible
    // Using bitcast or uint() operations that map directly to hardware instructions
    let x = u32(lng_norm * f32(elevation_config.width));
    let y = u32(lat_norm * f32(elevation_config.height));
    
    // Bounds checking using min() instead of conditionals
    // This avoids branch divergence within warps/wavefronts
    let safe_x = min(x, elevation_config.width - 1u);
    let safe_y = min(y, elevation_config.height - 1u);
    
    // Use linear indexing with a single multiply-add operation
    // This is more efficient than multiple operations
    let index = safe_y * elevation_config.width + safe_x;
    return elevation_map[index];
}

// Fast check for sea level to determine land vs ocean
fn is_land(elevation: f32) -> bool {
    return elevation >= color_config.sea_level;
}

// Optimized binary color selection based on elevation
fn calculate_color(elevation: f32) -> vec3<f32> {
    // Simple binary selection: Ocean or Land
    // This eliminates the complex conditional branching of the original function
    if (is_land(elevation)) {
        return LAND_COLOR;
    } else {
        return OCEAN_COLOR;
    }
}

fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Early-out for bounds checking to avoid unnecessary computation
    // This avoids warp/wavefront divergence for out-of-bounds threads
    if (index >= arrayLength(&cells)) {
        return;
    }
    
    // Local caching of cell data to reduce memory lookups
    // This keeps frequently accessed data in registers
    let cell = cells[index];
    
    // Process elevation and color in one step
    let elevation = get_elevation(cell.lat, cell.lng);
    let color = calculate_color(elevation);
    
    // Write results in a single coherent operation
    results[index] = CellElevationResult(
        cell.cell_index,
        elevation,
        color
    );
}