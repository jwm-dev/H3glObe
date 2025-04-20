use gdal::{Dataset, GeoTransform};
use h3o::{CellIndex, LatLng, Resolution};
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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
        
        println!("Found file for lat: {}, lng: {}: {}", 
                 lat, lng, file_path.file_name().unwrap_or_default().to_string_lossy());
        
        // Get or open the dataset
        let dataset_and_transform = self.get_or_open_dataset(&file_path)?;
        let dataset = &dataset_and_transform.0;
        let geotransform = dataset_and_transform.1;
        
        // Convert lat/lng to pixel coordinates
        let (pixel_x, pixel_y) = self.latlon_to_pixel(lat, lng, &geotransform);
        
        // Make sure the pixel is within bounds and convert to usize
        let size = dataset.raster_size();
        if pixel_x < 0 || pixel_y < 0 || pixel_x >= size.0 as isize || pixel_y >= size.1 as isize {
            println!("Pixel out of bounds: ({}, {}) for size: {:?}", pixel_x, pixel_y, size);
            return None;
        }
        
        let pixel_x_usize = pixel_x as usize;
        let pixel_y_usize = pixel_y as usize;
        
        // Get band (assuming band 1 for elevation data)
        let band = match dataset.rasterband(1) {
            Ok(b) => b,
            Err(err) => {
                println!("Error getting raster band: {}", err);
                return None;
            }
        };
        
        // Try different approaches to read the pixel value
        
        // Approach 1: Direct pixel reading
        match band.read_as::<f32>((pixel_x_usize.try_into().unwrap(), pixel_y_usize.try_into().unwrap()), (1, 1), (1, 1), None) {
            Ok(data) => {
                if !data.data.is_empty() {
                    let elevation = data.data[0];
                    println!("Got elevation for lat: {}, lng: {}: {}", lat, lng, elevation);
                    Some(elevation)
                } else {
                    // If the buffer is empty, try another approach
                    println!("Empty buffer from read_as, trying alternative method");
                    self.try_alternate_reading_method(band, pixel_x_usize, pixel_y_usize)
                }
            },
            Err(err) => {
                println!("Error reading pixel: {}, trying alternative method", err);
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
                        println!("Got elevation (window method) for x:{}, y:{}: {}", x, y, elevation);
                        Some(elevation)
                    } else {
                        // If we can't get the exact pixel, use the average of available data
                        let avg_elevation = data.data.iter().sum::<f32>() / data.data.len() as f32;
                        println!("Using average elevation from window: {}", avg_elevation);
                        Some(avg_elevation)
                    }
                } else {
                    println!("Empty buffer from window read");
                    
                    // Last resort: try reading the entire band and sampling from it
                    match band.read_as::<f32>((0, 0), band.size(), band.size(), None) {
                        Ok(full_data) => {
                            if !full_data.data.is_empty() {
                                let width = band.size().0;
                                let index = y * width + x;
                                if index < full_data.data.len() {
                                    println!("Got elevation (full read method): {}", full_data.data[index]);
                                    Some(full_data.data[index])
                                } else {
                                    println!("Index out of bounds in full read");
                                    None
                                }
                            } else {
                                println!("Empty buffer from full read");
                                None
                            }
                        },
                        Err(e) => {
                            println!("Full read failed: {}", e);
                            None
                        }
                    }
                }
            },
            Err(e) => {
                println!("Window read failed: {}", e);
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
                    Err(err) => {
                        println!("Error getting geotransform: {}", err);
                        None
                    }
                }
            },
            Err(err) => {
                println!("Error opening GeoTIFF file {}: {}", file_path.display(), err);
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
        
        // Debug print to see coordinate conversion for troubleshooting
        println!("lat: {}, lng: {}, pixel: ({}, {})", lat, lng, pixel_x, pixel_y);
        
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
                    println!("Found tile for lat:{}, lng:{} - File: {}", lat, lng, file_name);
                    return Some(file_path.clone());
                }
            }
        }
        
        // If we get here, we couldn't find a matching file
        println!("No elevation tile found for lat:{}, lng:{}", lat, lng);
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
        
        // For resolution 0, preload all 122 base cells
        if res_value == 0 {
            let cells: Vec<CellIndex> = (0..122)
                .filter_map(|i| CellIndex::try_from(i).ok())
                .collect();
                
            println!("Preloading {} base cells", cells.len());
            
            // Process cells in parallel
            let elevations: Vec<(u64, CellElevation)> = cells.par_iter()
                .map(|&cell| {
                    let cell_int = u64::from(cell);
                    let elevation = self.direct_compute_elevation(cell);
                    (cell_int, elevation)
                })
                .collect();
                
            // Store in our resolution-specific cache
            {
                let mut downsampled = self.downsampled_data.lock().unwrap();
                let res_map = downsampled.entry(res_value).or_insert_with(HashMap::new);
                
                for (cell_int, elevation) in &elevations {
                    res_map.insert(*cell_int, elevation.clone());
                }
            }
            
            println!("Preloaded {} base cells for resolution 0", elevations.len());
            return;
        }
        
        // For other resolutions, sample strategic locations
        // This is a simplified approach - for a production system you might
        // want a more comprehensive preloading strategy
        let sample_points = [
            // North America
            (37.7749, -122.4194),   // San Francisco
            (40.7128, -74.0060),    // New York
            // Europe
            (51.5074, -0.1278),     // London
            (48.8566, 2.3522),      // Paris
            // Asia
            (35.6762, 139.6503),    // Tokyo
            (1.3521, 103.8198),     // Singapore
            // Africa
            (-33.9249, 18.4241),    // Cape Town
            // Australia
            (-33.8688, 151.2093),   // Sydney
        ];
        
        let mut all_cells: Vec<CellIndex> = Vec::new();
        
        for (lat, lng) in sample_points.iter() {
            // Convert lat/lng to H3 cell
            let center_point = LatLng::new(*lat, *lng).expect("Valid lat/lng");
            let center_cell = center_point.to_cell(resolution);
            
            // Get the cell and a small ring around it
            let max_rings = 2;
            let cells: Vec<CellIndex> = center_cell.grid_disk(max_rings);
            all_cells.extend(cells);
        }
        
        // Add pentagons for completeness
        all_cells.extend(resolution.pentagons());
        
        println!("Preloading {} cells for resolution {}", all_cells.len(), res_value);
        
        // Process cells in parallel
        let elevations: Vec<(u64, CellElevation)> = all_cells.par_iter()
            .map(|&cell| {
                let cell_int = u64::from(cell);
                let elevation = self.direct_compute_elevation(cell);
                (cell_int, elevation)
            })
            .collect();
            
        // Store in our resolution-specific cache
        {
            let mut downsampled = self.downsampled_data.lock().unwrap();
            let res_map = downsampled.entry(res_value).or_insert_with(HashMap::new);
            
            for (cell_int, elevation) in &elevations {
                res_map.insert(*cell_int, elevation.clone());
            }
        }
        
        println!("Preloaded {} cells for resolution {}", elevations.len(), res_value);
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