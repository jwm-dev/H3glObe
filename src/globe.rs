use crate::camera::Camera;
use crate::elevation::ElevationData;
use wgpu::util::DeviceExt;
use h3o::{CellIndex, LatLng, Resolution};
use glam::Mat4;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const EARTH_RADIUS: f32 = 1.0; // Normalized globe radius

// Define a hexagon vertex with position and color
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// Define a transform uniform for instances
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct GlobeUniform {
    model: [[f32; 4]; 4],
}

// H3 Resolution wrapper
#[derive(Clone)]
pub struct H3Resolution {
    resolution: u8,
}

impl H3Resolution {
    pub fn new(resolution: u8) -> Self {
        // Ensure resolution is within valid range (0-15)
        let clamped = resolution.clamp(0, 15);
        Self { resolution: clamped }
    }

    pub fn increase(&mut self) {
        if self.resolution < 15 {
            self.resolution += 1;
        }
    }

    pub fn decrease(&mut self) {
        if self.resolution > 0 {
            self.resolution -= 1;
        }
    }

    pub fn value(&self) -> u8 {
        self.resolution
    }
}

pub struct Globe {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    triangle_index_buffer: wgpu::Buffer,
    num_indices: u32,
    num_triangle_indices: u32,
    globe_uniform: GlobeUniform,
    globe_uniform_buffer: wgpu::Buffer,
    globe_bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,
    h3_resolution: H3Resolution,
    device: wgpu::Device,
    base_cell_colors: Vec<[f32; 3]>,
    render_solid: bool,
    elevation_data: Option<Arc<ElevationData>>,
    
    // Fields for hot-switching support
    pending_resolution: Option<H3Resolution>,
    is_loading_resolution: bool,
    background_loader_handle: Option<std::thread::JoinHandle<()>>,
}

impl Globe {
    pub fn new(device: &wgpu::Device, resolution: H3Resolution, elevation_data: Option<Arc<ElevationData>>) -> Self {
        // Generate a color palette for base cells (used as fallback if no elevation data)
        let base_cell_colors = generate_base_cell_colors();

        // Create initial H3 cells
        // If elevation data is provided, it will be used for coloring
        let (vertices, indices, triangle_indices) = generate_h3_cells_with_elevation(&resolution, elevation_data.clone());
        
        // Create vertex and index buffers
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        let triangle_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Triangle Index Buffer"),
            contents: bytemuck::cast_slice(&triangle_indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create a globe uniform for world transform
        let globe_uniform = GlobeUniform {
            model: Mat4::IDENTITY.to_cols_array_2d(),
        };
        
        // Create uniform buffer
        let globe_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Globe Uniform Buffer"),
            contents: bytemuck::cast_slice(&[globe_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create bind group layout for the globe
        let globe_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Globe Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        // Create bind group
        let globe_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Globe Bind Group"),
            layout: &globe_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globe_uniform_buffer.as_entire_binding(),
            }],
        });
        
        // Load shaders
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Globe Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shader.wgsl").into()),
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Globe Pipeline Layout"),
            bind_group_layouts: &[
                &globe_bind_group_layout,
                &create_camera_bind_group_layout(device),
            ],
            push_constant_ranges: &[],
        });
        
        // Create depth texture
        let (depth_texture, depth_texture_view) = create_depth_texture(
            device,
            wgpu::Extent3d {
                width: 1280,
                height: 720,
                depth_or_array_layers: 1,
            },
        );
        
        // Create render pipeline with updated fields for wgpu 0.25.0
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Globe Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb, // Use appropriate format for your swapchain
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
        
        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            triangle_index_buffer,
            num_indices: indices.len() as u32,
            num_triangle_indices: triangle_indices.len() as u32,
            globe_uniform,
            globe_uniform_buffer,
            globe_bind_group,
            depth_texture,
            depth_texture_view,
            h3_resolution: resolution,
            device: device.clone(),
            base_cell_colors,
            render_solid: false,
            elevation_data,
            pending_resolution: None,
            is_loading_resolution: false,
            background_loader_handle: None,
        }
    }
    
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, camera: &'a Camera) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        
        // Use the correct index buffer based on render mode
        if self.render_solid {
            render_pass.set_index_buffer(self.triangle_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.set_bind_group(0, &self.globe_bind_group, &[]);
            render_pass.set_bind_group(1, camera.bind_group(), &[]);
            render_pass.draw_indexed(0..self.num_triangle_indices, 0, 0..1);
        } else {
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.set_bind_group(0, &self.globe_bind_group, &[]);
            render_pass.set_bind_group(1, camera.bind_group(), &[]);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }
    }
    
    pub fn depth_texture_view(&self) -> &wgpu::TextureView {
        &self.depth_texture_view
    }
    
    // Update depth texture size
    pub fn update_depth_texture(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (new_depth_texture, new_depth_view) = create_depth_texture(
            device,
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            }
        );
        
        self.depth_texture = new_depth_texture;
        self.depth_texture_view = new_depth_view;
    }

    // Update the increase_resolution method to use hot-switching
    pub fn increase_resolution(&mut self) {
        // Only allow resolution change if not already loading
        if self.is_loading_resolution {
            println!("Resolution change already in progress, please wait...");
            return;
        }
        
        // Create a new resolution object
        let mut new_res = H3Resolution::new(self.h3_resolution.value());
        new_res.increase();
        
        // If resolution didn't change (already at max), do nothing
        if new_res.value() == self.h3_resolution.value() {
            return;
        }
        
        // Set the pending resolution
        self.pending_resolution = Some(new_res);
        
        // Start hot-switching process
        self.start_resolution_hot_switch();
    }
    
    // Update the decrease_resolution method to use hot-switching
    pub fn decrease_resolution(&mut self) {
        // Only allow resolution change if not already loading
        if self.is_loading_resolution {
            println!("Resolution change already in progress, please wait...");
            return;
        }
        
        // Create a new resolution object
        let mut new_res = H3Resolution::new(self.h3_resolution.value());
        new_res.decrease();
        
        // If resolution didn't change (already at min), do nothing
        if new_res.value() == self.h3_resolution.value() {
            return;
        }
        
        // Set the pending resolution
        self.pending_resolution = Some(new_res);
        
        // Start hot-switching process
        self.start_resolution_hot_switch();
    }
    
    // New method to start the resolution hot-switching process
    fn start_resolution_hot_switch(&mut self) {
        if self.pending_resolution.is_none() || self.is_loading_resolution {
            return;
        }
        
        // Mark that we're loading a new resolution
        self.is_loading_resolution = true;
        
        // Get the new resolution
        let new_res = self.pending_resolution.clone().unwrap();
        println!("Starting hot-switch to resolution {}", new_res.value());
        
        // Clone the elevation data reference for the background thread
        let elevation_data_clone = self.elevation_data.clone();
        let device_clone = self.device.clone();
        let new_res_for_thread = new_res.clone();
        
        // Start a background thread to process the new resolution
        let handle = std::thread::spawn(move || {
            // Check if we have elevation data
            if let Some(elev_data) = elevation_data_clone {
                // Use GPU acceleration to process cells for the new resolution
                let res_obj = Resolution::try_from(new_res_for_thread.value()).unwrap();
                
                // Set up a progress callback
                let new_res_for_callback = new_res_for_thread.clone();
                let progress_callback = move |progress: f32| {
                    println!("Processing resolution {}: {:.1}%", new_res_for_callback.value(), progress * 100.0);
                };
                
                // Process cells using GPU acceleration
                match elev_data.process_cells_for_resolution_gpu(res_obj, Some(progress_callback)) {
                    Ok(_) => println!("Successfully processed cells for resolution {}", new_res_for_thread.value()),
                    Err(e) => eprintln!("Failed to process cells for resolution {}: {}", new_res_for_thread.value(), e),
                }
                
                // Pre-generate geometry for faster switching
                let (_vertices, _indices, _triangle_indices) = 
                    generate_h3_cells_with_elevation(&new_res_for_thread, Some(elev_data));
                println!("Pre-generated geometry for resolution {}", new_res_for_thread.value());
            } else {
                // No elevation data, just pre-generate the geometry
                let (_vertices, _indices, _triangle_indices) = generate_h3_cells(&new_res_for_thread);
                println!("Pre-generated geometry for resolution {} (no elevation data)", new_res_for_thread.value());
            }
        });
        
        self.background_loader_handle = Some(handle);
    }
    
    // Method to check if resolution loading is complete
    pub fn check_resolution_loading(&mut self) -> bool {
        if let Some(elev_data) = &self.elevation_data {
            if let Some(loading_res) = self.pending_resolution.clone() {
                // If we're waiting for a resolution to load, check its status
                // Convert loading_res to Resolution type
                let res_obj = Resolution::try_from(loading_res.value()).unwrap_or(Resolution::Zero);
                let is_loaded = elev_data.is_resolution_loaded(res_obj);
                
                if is_loaded {
                    // If loaded, update the current resolution and clear loading flag
                    // Update resolution only if it's different from what we already have
                    if self.h3_resolution.value() != loading_res.value() {
                        self.h3_resolution = loading_res;
                        // Regenerate the geometry to update the visible mesh - this was missing!
                        self.regenerate_geometry();
                    }
                    self.pending_resolution = None;
                    self.is_loading_resolution = false;
                    return true; // Resolution change completed
                }
            }
        } else if let Some(handle) = self.background_loader_handle.take() {
            if handle.is_finished() {
                // For non-elevation mode, just check if the background thread is done
                // This is a simpler case as we just wait for the thread to complete
                if let Some(new_res) = self.pending_resolution.take() {
                    // Join the thread to ensure it's completed properly
                    if let Err(e) = handle.join() {
                        eprintln!("Error joining background thread: {:?}", e);
                    }
                    
                    // Update the resolution and regenerate geometry
                    self.h3_resolution = new_res;
                    // Regenerate the geometry to update the visible mesh - this was missing!
                    self.regenerate_geometry();
                    self.is_loading_resolution = false;
                    return true; // Resolution change completed
                }
            } else {
                // Put the handle back if thread is still running
                self.background_loader_handle = Some(handle);
            }
        }
        
        false // No resolution change completed
    }
    
    // Method to get current loading status
    pub fn is_loading(&self) -> bool {
        self.is_loading_resolution
    }
    
    // Method to get loading progress (stub - would need to be enhanced with actual progress tracking)
    pub fn loading_progress(&self) -> f32 {
        if self.is_loading_resolution {
            // In a real implementation, we'd track actual progress
            0.5 // Placeholder
        } else {
            0.0
        }
    }
    
    fn regenerate_geometry(&mut self) {
        // Generate new geometry for the updated resolution, using elevation data if available
        let (vertices, indices, triangle_indices) = generate_h3_cells_with_elevation(&self.h3_resolution, self.elevation_data.clone());
        
        // Create new buffers - ensure we have at least some valid indices
        let index_data = if indices.is_empty() {
            vec![0u32, 0, 0] // Fallback triangle to prevent empty buffer errors
        } else {
            indices
        };

        let triangle_index_data = if triangle_indices.is_empty() {
            vec![0u32, 0, 0] // Fallback triangle to prevent empty buffer errors
        } else {
            triangle_indices
        };
        
        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        
        let index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        let triangle_index_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Triangle Index Buffer"),
            contents: bytemuck::cast_slice(&triangle_index_data),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });
        
        // Update buffer references
        self.vertex_buffer = vertex_buffer;
        self.index_buffer = index_buffer;
        self.triangle_index_buffer = triangle_index_buffer;
        self.num_indices = index_data.len() as u32;
        self.num_triangle_indices = triangle_index_data.len() as u32;
        
        println!("Regenerated geometry with {} indices and {} triangle indices", self.num_indices, self.num_triangle_indices);
    }

    pub fn toggle_render_mode(&mut self) {
        self.render_solid = !self.render_solid;
        self.update_pipeline();
        println!("Render mode: {}", if self.render_solid { "Solid" } else { "Wireframe" });
    }
    
    pub fn set_render_solid(&mut self, solid: bool) {
        if self.render_solid != solid {
            self.render_solid = solid;
            self.update_pipeline();
        }
    }
    
    pub fn is_render_solid(&self) -> bool {
        self.render_solid
    }
    
    fn update_pipeline(&mut self) {
        // Load shaders
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Globe Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shader.wgsl").into()),
        });
        
        // Create the globe bind group layout
        let globe_bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Globe Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        // Create pipeline layout
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Globe Pipeline Layout"),
            bind_group_layouts: &[
                &globe_bind_group_layout,
                &create_camera_bind_group_layout(&self.device),
            ],
            push_constant_ranges: &[],
        });
        
        // Create render pipeline with updated topology based on render mode
        self.pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Globe Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(if self.render_solid { "fs_main_solid" } else { "fs_main" }),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: if self.render_solid { 
                    wgpu::PrimitiveTopology::TriangleList 
                } else { 
                    wgpu::PrimitiveTopology::LineList 
                },
                strip_index_format: None,
                // Fix for backwards culling - changing from CW to CcW for front face winding
                front_face: if self.render_solid {
                    wgpu::FrontFace::Ccw
                } else {
                    wgpu::FrontFace::Ccw
                },
                cull_mode: if self.render_solid { Some(wgpu::Face::Back) } else { None },
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
    }

    // Add the get_resolution method to retrieve current resolution
    pub fn get_resolution(&self) -> Resolution {
        Resolution::try_from(self.h3_resolution.value()).unwrap_or(Resolution::Zero)
    }
}

// Helper function to create camera bind group layout
fn create_camera_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Camera Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

// Helper function to create depth texture
fn create_depth_texture(
    device: &wgpu::Device,
    size: wgpu::Extent3d,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    
    (texture, view)
}

// Generate colors for H3 base cells
fn generate_base_cell_colors() -> Vec<[f32; 3]> {
    // Generate unique colors for all 122 base cells
    let mut colors = Vec::with_capacity(122);
    for i in 0..122 {
        // Generate a unique HSV color and convert to RGB
        let h = (i as f32 * 137.5) % 360.0; // Golden angle to get good distribution
        let s = 0.5 + 0.5 * ((i as f32 * 37.5).sin() * 0.5 + 0.5);
        let v = 0.5 + 0.5 * ((i as f32 * 29.3).cos() * 0.5 + 0.5);
        
        // Convert HSV to RGB
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;
        
        let (r, g, b) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };
        
        colors.push([r + m, g + m, b + m]);
    }
    
    colors
}

// Convert lat/lng to 3D point on a sphere (CORRECTED^2 VERSION)
fn lat_lng_to_point(lat: f64, lng: f64) -> [f32; 3] {
    let lat_rad = lat.to_radians();
    let lng_rad = lng.to_radians();
    
    // Standard spherical to Cartesian conversion
    let x = ((-lng_rad).cos() * lat_rad.cos()) as f32 * EARTH_RADIUS; 
    let z = ((-lng_rad).sin() * lat_rad.cos()) as f32 * EARTH_RADIUS;
    let y = (lat_rad.sin()) as f32 * EARTH_RADIUS;
    
    [x, y, z]
}

// Calculate surface normal at a point on a sphere
fn calculate_normal(vertex: [f32; 3]) -> [f32; 3] {
    // For a sphere, the normal is simply the normalized position vector
    let length = (vertex[0] * vertex[0] + vertex[1] * vertex[1] + vertex[2] * vertex[2]).sqrt();
    [
        vertex[0] / length,
        vertex[1] / length,
        vertex[2] / length,
    ]
}

fn generate_h3_cells(resolution: &H3Resolution) -> (Vec<Vertex>, Vec<u32>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut line_indices = Vec::new();
    let mut triangle_indices = Vec::new();
    
    // Get actual resolution value (0-15)
    let res_value = resolution.value();
    
    println!("Generating H3 cells at resolution {}...", res_value);
    
    // At resolution 0, we'll display the base icosahedron for context
    if res_value == 0 {
        // Add the base icosahedron
        create_icosahedron_base(&mut vertices, &mut line_indices);
        
        // Get all base cells (resolution 0) using h3o
        let h3_res = Resolution::try_from(0).unwrap(); // Resolution 0
        
        // Generate and add all resolution 0 cells
        generate_h3_base_cells(h3_res, &mut vertices, &mut line_indices, &mut triangle_indices);
    } else {
        // For higher resolutions, we'll generate cells based on a coverage area
        let h3_res = Resolution::try_from(res_value as u8).unwrap();
        generate_h3_cells_by_resolution(h3_res, &mut vertices, &mut line_indices, &mut triangle_indices);
    }
    
    println!("Created {} vertices, {} line indices, and {} triangle indices", 
             vertices.len(), line_indices.len(), triangle_indices.len());
    
    (vertices, line_indices, triangle_indices)
}

// Generate all base cells (resolution 0)
fn generate_h3_base_cells(_res: Resolution, vertices: &mut Vec<Vertex>, line_indices: &mut Vec<u32>, triangle_indices: &mut Vec<u32>) {
    // Iterate through all base cells (122 at resolution 0)
    for base_index in 0..122u64 {
        // Try to create a cell index for this base cell
        if let Ok(cell_index) = CellIndex::try_from(base_index) {
            // Get the boundary of this cell
            let boundary = cell_index.boundary();
            add_h3_cell_boundary(cell_index, &boundary, vertices, line_indices, triangle_indices);
        }
    }
}

// Generate H3 cells at a specific resolution within a view area
fn generate_h3_cells_by_resolution(res: Resolution, vertices: &mut Vec<Vertex>, line_indices: &mut Vec<u32>, triangle_indices: &mut Vec<u32>) {
    // For higher resolutions, we can't render the whole globe at once
    // So we'll focus on a section with reasonable density
    
    // Sample area covering a section of the globe with visible cells
    // For demonstration, we'll sample cells around the equator
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
    
    let res_val = u8::from(res);
    
    // For resolutions 0-4, we'll render all cells (no limit)
    // For higher resolutions, we'll still limit to prevent overwhelming the GPU
    let max_rings = match res_val {
        0 => 100, // Effectively unlimited for res 0
        1 => 100, // Effectively unlimited for res 1
        2 => 100, // Effectively unlimited for res 2
        3 => 100, // Effectively unlimited for res 3
        4 => 100, // Effectively unlimited for res 4
        5 => 4,
        6 => 3,
        7 => 2,
        _ => 1  // For very high resolutions, still limit to avoid overwhelming the GPU
    };
    
    println!("Using {} rings at resolution {}", max_rings, res_val);
    
    for (lat, lng) in sample_points.iter() {
        // Convert lat/lng to H3 cell
        let center_point = LatLng::new(*lat, *lng).expect("Valid lat/lng");
        let center_cell = center_point.to_cell(res);
        
        // Get the cell and its rings
        let cells = if max_rings > 0 { 
            center_cell.grid_disk(max_rings)
        } else {
            // For very high resolutions, just get the center cell
            vec![center_cell]
        };
        
        println!("Generated {} cells around point {}, {}", cells.len(), lat, lng);
        
        // Add all cells in the collection
        for cell in cells {
            let boundary = cell.boundary();
            add_h3_cell_boundary(cell, &boundary, vertices, line_indices, triangle_indices);
        }
    }
    
    // Always include the 12 pentagon cells at the current resolution
    add_pentagon_cells(res, vertices, line_indices, triangle_indices);
}

// Add the 12 pentagon cells at the specified resolution
fn add_pentagon_cells(res: Resolution, vertices: &mut Vec<Vertex>, line_indices: &mut Vec<u32>, triangle_indices: &mut Vec<u32>) {
    // Start with 12 base icosahedron pentagons and get their children at the desired resolution
    for base_index in 0..122u64 {
        if let Ok(cell_index) = CellIndex::try_from(base_index) {
            // Check if this is a pentagon
            if cell_index.is_pentagon() {
                // For resolution 0, we already added this pentagon
                if u8::from(res) == 0 {
                    continue;
                }
                
                // For higher resolutions, get the descendant pentagon
                let pentagon_at_res = get_pentagon_descendant(cell_index, res);
                let boundary = pentagon_at_res.boundary();
                
                // Add with special coloring for pentagons
                add_h3_cell_boundary(pentagon_at_res, &boundary, vertices, line_indices, triangle_indices);
            }
        }
    }
}

// Get the pentagon descendant at a specific resolution
fn get_pentagon_descendant(base_cell: CellIndex, target_res: Resolution) -> CellIndex {
    let base_res = base_cell.resolution();
    if base_res == target_res {
        return base_cell;
    }
    
    // Get the center point of the cell - using From trait implementation for LatLng from CellIndex
    let center = LatLng::from(base_cell);
    
    // Convert to the target resolution
    let cell_at_res = center.to_cell(target_res);
    
    // Find the pentagon at this resolution near the point
    // We know each pentagon at resolution 0 has exactly one pentagon descendant at each resolution
    for ring in 0..3 {
        let neighbors: Vec<CellIndex> = cell_at_res.grid_disk(ring);
        for neighbor in neighbors {
            if neighbor.is_pentagon() {
                return neighbor;
            }
        }
    }
    
    // Fallback to the original cell converted to target resolution
    cell_at_res
}

// Add an H3 cell boundary to our vertex and index buffers
fn add_h3_cell_boundary(cell: CellIndex, boundary: &[LatLng], vertices: &mut Vec<Vertex>, line_indices: &mut Vec<u32>, triangle_indices: &mut Vec<u32>) {
    let vertex_start_index = vertices.len() as u32;
    let is_pentagon = cell.is_pentagon();
    
    // Color based on cell properties
    let color = if is_pentagon {
        [0.9, 0.1, 0.1] // Red for pentagons
    } else {
        // Create a quasi-unique color based on the cell index
        let h3_index = u64::from(cell);
        let hue = ((h3_index % 360) as f32) / 360.0;
        
        // Simple HSV to RGB conversion for varying colors
        hsv_to_rgb(hue, 0.7, 0.8)
    };
    
    // Add vertices for the boundary points
    for point in boundary {
        let pos = lat_lng_to_point(point.lat_radians().to_degrees(), point.lng_radians().to_degrees());
        let normal = calculate_normal(pos);
        
        vertices.push(Vertex {
            position: pos,
            normal,
            color,
        });
    }
    
    // Connect boundary vertices to form edges (for LineList topology)
    let num_vertices = boundary.len();
    for i in 0..num_vertices {
        let next = (i + 1) % num_vertices;
        line_indices.push(vertex_start_index + i as u32);
        line_indices.push(vertex_start_index + next as u32);
    }
    
    // Generate triangles for solid face rendering
    add_cell_face_triangles(boundary, vertex_start_index, vertices, triangle_indices, color);
}

// Generate triangles for solid face rendering of a cell
fn add_cell_face_triangles(boundary: &[LatLng], vertex_start_index: u32, vertices: &mut Vec<Vertex>, triangle_indices: &mut Vec<u32>, color: [f32; 3]) {
    if boundary.len() < 3 {
        return; // Need at least 3 vertices to form a triangle
    }
    
    // Better center calculation - use 3D centroid instead of lat/lng averaging
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_z = 0.0;
    
    // Calculate centroid in 3D Cartesian space (proper for spherical geometry)
    for point in boundary {
        let pos = lat_lng_to_point(point.lat_radians().to_degrees(), point.lng_radians().to_degrees());
        sum_x += pos[0] as f64;
        sum_y += pos[1] as f64;
        sum_z += pos[2] as f64;
    }
    
    let count = boundary.len() as f64;
    let center_x = sum_x / count;
    let center_y = sum_y / count;
    let center_z = sum_z / count;
    
    // Project the centroid back to the sphere surface
    let center_length = (center_x * center_x + center_y * center_y + center_z * center_z).sqrt();
    let scale = EARTH_RADIUS as f64 / center_length;
    
    let center_pos = [
        (center_x * scale) as f32,
        (center_y * scale) as f32,
        (center_z * scale) as f32
    ];
    
    let center_normal = calculate_normal(center_pos);
    
    // Add the center vertex
    let center_index = vertices.len() as u32;
    vertices.push(Vertex {
        position: center_pos,
        normal: center_normal,
        color: [color[0] * 0.95, color[1] * 0.95, color[2] * 0.95], // Slightly darker for center
    });
    
    // Create triangles using fan triangulation from center
    let num_vertices = boundary.len();
    for i in 0..num_vertices {
        let next = (i + 1) % num_vertices;
        
        // Add triangle: center -> current -> next
        // These are added AFTER the line indices and don't interfere with them
        triangle_indices.push(center_index);
        triangle_indices.push(vertex_start_index + i as u32);
        triangle_indices.push(vertex_start_index + next as u32);
    }
}

// Simple HSV to RGB conversion for cell coloring
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = v - c;
    
    let (r1, g1, b1) = if h < 1.0/6.0 {
        (c, x, 0.0)
    } else if h < 2.0/6.0 {
        (x, c, 0.0)
    } else if h < 3.0/6.0 {
        (0.0, c, x)
    } else if h < 4.0/6.0 {
        (0.0, x, c)
    } else if h < 5.0/6.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    
    [r1 + m, g1 + m, b1 + m]
}

// Add this function to create points along a great arc
fn create_great_arc(start: [f32; 3], end: [f32; 3], segments: usize) -> Vec<[f32; 3]> {
    let mut points = Vec::with_capacity(segments + 1);
    points.push(start);
    
    // Convert to unit vectors for spherical interpolation
    let start_norm = normalize(start);
    let end_norm = normalize(end);
    
    // Calculate the angle between the vectors
    let dot = start_norm[0] * end_norm[0] + start_norm[1] * end_norm[1] + start_norm[2] * end_norm[2];
    let angle = dot.clamp(-1.0, 1.0).acos();
    
    // Create intermediate points
    for i in 1..segments {
        let t = i as f32 / segments as f32;
        let sin_angle = angle.sin();
        
        // Use spherical linear interpolation (SLERP)
        let a = (angle * (1.0 - t)).sin() / sin_angle;
        let b = (angle * t).sin() / sin_angle;
        
        let x = start_norm[0] * a + end_norm[0] * b;
        let y = start_norm[1] * a + end_norm[1] * b;
        let z = start_norm[2] * a + end_norm[2] * b;
        
        // Project back to sphere surface
        let point = normalize([x, y, z]);
        points.push([
            point[0] * EARTH_RADIUS, 
            point[1] * EARTH_RADIUS, 
            point[2] * EARTH_RADIUS
        ]);
    }
    
    points.push(end);
    points
}

fn create_icosahedron_base(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>) {
    // Use exact Cartesian coordinates known to form a proper icosahedron
    // These coordinates are precisely defined to create an icosahedron
    let radius = EARTH_RADIUS;
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0; // Golden ratio

    // The 12 vertices of a regular icosahedron
    let ico_vertices = [
        normalize([-1.0, t, 0.0]),
        normalize([1.0, t, 0.0]),
        normalize([-1.0, -t, 0.0]),
        normalize([1.0, -t, 0.0]),
        normalize([0.0, -1.0, t]),
        normalize([0.0, 1.0, t]),
        normalize([0.0, -1.0, -t]),
        normalize([0.0, 1.0, -t]),
        normalize([t, 0.0, -1.0]),
        normalize([t, 0.0, 1.0]),
        normalize([-t, 0.0, -1.0]),
        normalize([-t, 0.0, 1.0])
    ];
    
    // The 30 edges, triple-checked to ensure they form a proper icosahedron
    let edges = [
        // The 5 edges emanating from vertex 0
        [0, 1], [0, 5], [0, 7], [0, 10], [0, 11],
        // The 5 edges emanating from vertex 1
        [1, 5], [1, 7], [1, 8], [1, 9],
        // The 5 edges emanating from vertex 2
        [2, 3], [2, 4], [2, 6], [2, 10], [2, 11],
        // The 5 edges emanating from vertex 3
        [3, 4], [3, 6], [3, 8], [3, 9],
        // Remaining edges (avoiding duplicates)
        [4, 5], [4, 9], [4, 11],
        [5, 9], [5, 11],
        [6, 7], [6, 8], [6, 10],
        [7, 8], [7, 10],
        [8, 9],
        [10, 11]
    ];

    // Create the faces for verification (20 faces of the icosahedron)
    let _faces = [
        [0, 1, 5], [0, 5, 11], [0, 11, 10], [0, 10, 7], [0, 7, 1],
        [1, 7, 8], [1, 8, 9], [1, 9, 5], [5, 9, 4], [5, 4, 11],
        [11, 4, 2], [11, 2, 10], [10, 2, 6], [10, 6, 7], [7, 6, 8],
        [8, 6, 3], [8, 3, 9], [9, 3, 4], [4, 3, 2], [2, 3, 6]
    ];
    
    // Create actual vertices scaled to proper radius
    let base_idx = vertices.len() as u32;
    
    // Add icosahedron vertices
    for v in &ico_vertices {
        let position = [v[0] * radius, v[1] * radius, v[2] * radius];
        
        vertices.push(Vertex {
            position,
            normal: [v[0], v[1], v[2]],
            color: [0.9, 0.2, 0.2], // Bright red for vertices
        });
    }
    
    // Render edges using explicit LineList topology
    for edge in &edges {
        let v1_idx = base_idx + edge[0] as u32;
        let v2_idx = base_idx + edge[1] as u32;
        
        indices.push(v1_idx);
        indices.push(v2_idx);
    }
    
    // Render edges with great arcs for a truly spherical icosahedron
    let segments = 12; // Enough segments for smooth arcs
    
    for edge in &edges {
        let start_vertex = base_idx + edge[0] as u32;
        let end_vertex = base_idx + edge[1] as u32;
        
        let start_pos = vertices[start_vertex as usize].position;
        let end_pos = vertices[end_vertex as usize].position;
        
        // Create great arc points between vertices
        let arc_points = create_great_arc(start_pos, end_pos, segments);
        
        // Add arc segment vertices and connect them
        let first_new_vertex = vertices.len() as u32;
        
        // Add interior vertices (skip first and last which are existing vertices)
        for i in 1..arc_points.len()-1 {
            let pos = arc_points[i];
            let normal = normalize(pos);
            
            vertices.push(Vertex {
                position: pos,
                normal,
                color: [0.7, 0.7, 0.7], // Gray for arc segments
            });
        }
        
        // Connect segments with LineList topology
        // Connect start vertex to first segment
        if segments > 2 {
            indices.push(start_vertex);
            indices.push(first_new_vertex);
            
            // Connect intermediate segments
            for i in 0..segments-3 {
                indices.push(first_new_vertex + i as u32);
                indices.push(first_new_vertex + i as u32 + 1);
            }
            
            // Connect last segment to end vertex
            indices.push(first_new_vertex + segments as u32 - 3);
            indices.push(end_vertex);
        } else {
            // Direct connection for low segment count
            indices.push(start_vertex);
            indices.push(end_vertex);
        }
    }
    
    // Optional: Add vertices and edges for face centers to show the H3 hexagons
    // This can be uncommented if you want to see the face centers
    /*
    // Add face centers
    for face in &faces {
        let v0 = ico_vertices[face[0]];
        let v1 = ico_vertices[face[1]];
        let v2 = ico_vertices[face[2]];
        
        // Calculate face center by averaging the vertices and normalizing
        let center = normalize([
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0
        ]);
        
        let center_pos = [center[0] * radius, center[1] * radius, center[2] * radius];
        let center_idx = vertices.len() as u32;
        
        // Add the face center vertex
        vertices.push(Vertex {
            position: center_pos,
            normal: center,
            color: [0.2, 0.5, 0.8], // Blue for face centers
        });
        
        // Connect face center to face vertices
        for &vertex_idx in face {
            indices.push(center_idx);
            indices.push(base_idx + vertex_idx as u32);
        }
    }
    */
}

// Helper function to normalize a vector
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / length, v[1] / length, v[2] / length]
}

// Add great arc edges between boundary points for smoother cell rendering
fn add_great_arc_edges(boundary: &[LatLng], base_index: u32, vertices: &mut Vec<Vertex>, line_indices: &mut Vec<u32>, color: [f32; 3]) {
    let segments = 8; // Number of segments per edge
    let num_points = boundary.len();
    
    for i in 0..num_points {
        let next = (i + 1) % num_points;
        
        // Get 3D positions of the two endpoints
        let p1 = lat_lng_to_point(boundary[i].lat_radians().to_degrees(), boundary[i].lng_radians().to_degrees());
        let p2 = lat_lng_to_point(boundary[next].lat_radians().to_degrees(), boundary[next].lng_radians().to_degrees());
        
        // Create great arc points
        let arc_points = create_great_arc(p1, p2, segments);
        
        // Skip first and last points as they're the original vertices
        let first_new_vertex = vertices.len() as u32;
        for j in 1..arc_points.len() - 1 {
            let pos = arc_points[j];
            let normal = calculate_normal(pos);
            
            vertices.push(Vertex {
                position: pos,
                normal,
                color,
            });
        }
        
        // Connect segments
        if segments > 2 {
            // Connect start to first internal point
            line_indices.push(base_index + i as u32);
            line_indices.push(first_new_vertex);
            
            // Connect internal points
            let internal_points = (segments - 2) as u32;
            for j in 0..internal_points - 1 {
                line_indices.push(first_new_vertex + j);
                line_indices.push(first_new_vertex + j + 1);
            }
            
            // Connect last internal point to end
            if internal_points > 0 {  // Ensure we have at least one internal point
                line_indices.push(first_new_vertex + internal_points - 1);
                line_indices.push(base_index + next as u32);
            }
        }
    }
}

// Generate H3 cells with elevation-based coloring
fn generate_h3_cells_with_elevation(resolution: &H3Resolution, elevation_data: Option<Arc<ElevationData>>) -> (Vec<Vertex>, Vec<u32>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut line_indices = Vec::new();
    let mut triangle_indices = Vec::new();
    
    // Get actual resolution value (0-15)
    let res_value = resolution.value();
    
    println!("Generating H3 cells at resolution {} with elevation data...", res_value);
    
    // At resolution 0, we'll display the base icosahedron for context
    if res_value == 0 {
        // Add the base icosahedron
        create_icosahedron_base(&mut vertices, &mut line_indices);
        
        // Get all base cells (resolution 0) using h3o
        let h3_res = Resolution::try_from(0).unwrap(); // Resolution 0
        
        // Generate and add all resolution 0 cells
        generate_h3_base_cells_with_elevation(h3_res, &mut vertices, &mut line_indices, &mut triangle_indices, elevation_data.clone());
    } else {
        // For higher resolutions, we'll generate cells based on a coverage area
        let h3_res = Resolution::try_from(res_value as u8).unwrap();
        generate_h3_cells_by_resolution_with_elevation(h3_res, &mut vertices, &mut line_indices, &mut triangle_indices, elevation_data.clone());
    }
    
    println!("Created {} vertices, {} line indices, and {} triangle indices", 
             vertices.len(), line_indices.len(), triangle_indices.len());
    
    (vertices, line_indices, triangle_indices)
}

// Generate all base cells (resolution 0) with elevation coloring
fn generate_h3_base_cells_with_elevation(
    _res: Resolution, 
    vertices: &mut Vec<Vertex>, 
    line_indices: &mut Vec<u32>, 
    triangle_indices: &mut Vec<u32>,
    elevation_data: Option<Arc<ElevationData>>
) {
    // Iterate through all base cells (122 at resolution 0)
    for base_index in 0..122u64 {
        // Try to create a cell index for this base cell
        if let Ok(cell_index) = CellIndex::try_from(base_index) {
            // Get the boundary of this cell
            let boundary = cell_index.boundary();
            add_h3_cell_boundary_with_elevation(cell_index, &boundary, vertices, line_indices, triangle_indices, elevation_data.clone());
        }
    }
}

// Generate H3 cells at a specific resolution with elevation-based coloring
fn generate_h3_cells_by_resolution_with_elevation(
    res: Resolution, 
    vertices: &mut Vec<Vertex>, 
    line_indices: &mut Vec<u32>, 
    triangle_indices: &mut Vec<u32>,
    elevation_data: Option<Arc<ElevationData>>
) {
    // For higher resolutions, we can't render the whole globe at once
    // So we'll focus on a section with reasonable density
    
    // Sample area covering a section of the globe with visible cells
    // These are the same sample points as before
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
    
    let res_val = u8::from(res);
    
    // For resolutions 0-4, we'll render all cells (no limit)
    // For higher resolutions, limit to prevent overwhelming the GPU
    let max_rings = match res_val {
        0 => 100, // Effectively unlimited for res 0
        1 => 100, // Effectively unlimited for res 1
        2 => 100, // Effectively unlimited for res 2
        3 => 100, // Effectively unlimited for res 3
        4 => 100, // Effectively unlimited for res 4
        5 => 4,
        6 => 3,
        7 => 2,
        _ => 1  // For very high resolutions, limit to 1 ring
    };
    
    println!("Using {} rings at resolution {}", max_rings, res_val);
    
    for (lat, lng) in sample_points.iter() {
        // Convert lat/lng to H3 cell
        let center_point = LatLng::new(*lat, *lng).expect("Valid lat/lng");
        let center_cell = center_point.to_cell(res);
        
        // Get the cell and its rings
        let cells = if max_rings > 0 { 
            center_cell.grid_disk(max_rings)
        } else {
            // For very high resolutions, just get the center cell
            vec![center_cell]
        };
        
        println!("Generated {} cells around point {}, {}", cells.len(), lat, lng);
        
        // Add all cells in the collection
        for cell in cells {
            let boundary = cell.boundary();
            add_h3_cell_boundary_with_elevation(cell, &boundary, vertices, line_indices, triangle_indices, elevation_data.clone());
        }
    }
    
    // Always include the 12 pentagon cells at the current resolution
    add_pentagon_cells_with_elevation(res, vertices, line_indices, triangle_indices, elevation_data);
}

// Add pentagon cells with elevation data
fn add_pentagon_cells_with_elevation(
    res: Resolution, 
    vertices: &mut Vec<Vertex>, 
    line_indices: &mut Vec<u32>, 
    triangle_indices: &mut Vec<u32>,
    elevation_data: Option<Arc<ElevationData>>
) {
    // Start with 12 base icosahedron pentagons and get their children at the desired resolution
    for base_index in 0..122u64 {
        if let Ok(cell_index) = CellIndex::try_from(base_index) {
            // Check if this is a pentagon
            if cell_index.is_pentagon() {
                // For resolution 0, we already added this pentagon
                if u8::from(res) == 0 {
                    continue;
                }
                
                // For higher resolutions, get the descendant pentagon
                let pentagon_at_res = get_pentagon_descendant(cell_index, res);
                let boundary = pentagon_at_res.boundary();
                
                // Add with special coloring for pentagons - elevation data is secondary for these special cells
                add_h3_cell_boundary_with_elevation(pentagon_at_res, &boundary, vertices, line_indices, triangle_indices, elevation_data.clone());
            }
        }
    }
}

// Add an H3 cell boundary with elevation-based coloring
fn add_h3_cell_boundary_with_elevation(
    cell: CellIndex, 
    boundary: &[LatLng], 
    vertices: &mut Vec<Vertex>, 
    line_indices: &mut Vec<u32>, 
    triangle_indices: &mut Vec<u32>,
    elevation_data: Option<Arc<ElevationData>>
) {
    let vertex_start_index = vertices.len() as u32;
    let is_pentagon = cell.is_pentagon();
    
    // Get color based on elevation if available, otherwise use old coloring scheme
    let color = if let Some(elev_data) = &elevation_data {
        // For pentagons, always use a special color to highlight them
        if is_pentagon {
            [0.9, 0.1, 0.1] // Red for pentagons
        } else {
            // Get the color based on elevation
            elev_data.get_color_for_cell(cell)
        }
    } else {
        // Fall back to old coloring scheme
        if is_pentagon {
            [0.9, 0.1, 0.1] // Red for pentagons
        } else {
            // Create a quasi-unique color based on the cell index
            let h3_index = u64::from(cell);
            let hue = ((h3_index % 360) as f32) / 360.0;
            
            // Simple HSV to RGB conversion for varying colors
            hsv_to_rgb(hue, 0.7, 0.8)
        }
    };
    
    // Add vertices for the boundary points
    for point in boundary {
        let pos = lat_lng_to_point(point.lat_radians().to_degrees(), point.lng_radians().to_degrees());
        let normal = calculate_normal(pos);
        
        vertices.push(Vertex {
            position: pos,
            normal,
            color,
        });
    }
    
    // Connect boundary vertices to form edges (for LineList topology)
    let num_vertices = boundary.len();
    for i in 0..num_vertices {
        let next = (i + 1) % num_vertices;
        line_indices.push(vertex_start_index + i as u32);
        line_indices.push(vertex_start_index + next as u32);
    }
    
    // Generate triangles for solid face rendering
    add_cell_face_triangles(boundary, vertex_start_index, vertices, triangle_indices, color);
}