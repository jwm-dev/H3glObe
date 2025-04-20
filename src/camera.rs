use glam::{Mat4, Quat, Vec3};
use std::f32::consts::PI;
use wgpu::util::DeviceExt;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};

// Configuration constants for camera behavior
const CAMERA_SPEED: f32 = 0.5;          // Slower base speed
const CAMERA_SENSITIVITY: f32 = 0.002;  // Much lower mouse sensitivity
const SCROLL_SENSITIVITY: f32 = 0.15;   // Controlled scroll zoom
const MIN_ZOOM: f32 = 1.5;              // Minimum distance from globe center
const MAX_ZOOM: f32 = 20.0;             // Maximum distance - increased for better overview
const ORBIT_SMOOTHING: f32 = 0.85;      // Smooth camera movement (0-1, higher = more smoothing)

// Define Camera Uniform struct for passing to shaders
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    view_position: [f32; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            view_position: [0.0, 0.0, 3.0, 1.0],
        }
    }
}

pub struct Camera {
    // Camera positioning
    position: Vec3,
    target: Vec3,
    up: Vec3,
    
    // Projection parameters
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
    
    // Matrices
    projection_matrix: Mat4,
    view_matrix: Mat4,
    
    // GPU resources
    uniform: CameraUniform,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    
    // Orbital control state
    is_orbiting: bool,
    yaw: f32,
    pitch: f32,
    distance: f32,
    
    // Mouse tracking
    last_mouse_position: Option<(f32, f32)>,
    
    // Keyboard movement state
    moving_up: bool,
    moving_down: bool,
    moving_left: bool,
    moving_right: bool,
    moving_forward: bool,
    moving_backward: bool,
    
    // Smooth movement
    target_yaw: f32,
    target_pitch: f32,
    target_distance: f32,
}

impl Camera {
    pub fn new(device: &wgpu::Device, aspect: f32, fovy: f32, znear: f32, zfar: f32) -> Self {
        let position = Vec3::new(0.0, 0.0, 5.0);
        let target = Vec3::ZERO;
        let up = Vec3::Y;
        
        let projection_matrix = Mat4::perspective_rh(fovy.to_radians(), aspect, znear, zfar);
        let view_matrix = Mat4::look_at_rh(position, target, up);
        
        let mut uniform = CameraUniform::new();
        
        // Create the bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        });
        
        // Create the uniform buffer
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create the bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        
        // Set initial distance and angles
        let initial_distance = 5.0;
        let initial_yaw = 0.0;
        let initial_pitch = 0.0;
        
        let mut camera = Self {
            position,
            target,
            up,
            aspect,
            fovy,
            znear,
            zfar,
            projection_matrix,
            view_matrix,
            uniform,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            is_orbiting: false,
            yaw: initial_yaw,
            pitch: initial_pitch,
            distance: initial_distance,
            last_mouse_position: None,
            moving_up: false,
            moving_down: false,
            moving_left: false,
            moving_right: false,
            moving_forward: false,
            moving_backward: false,
            target_yaw: initial_yaw,
            target_pitch: initial_pitch,
            target_distance: initial_distance,
        };
        
        // Initialize uniform with camera data
        camera.update_view_matrix();
        
        camera
    }
    
    pub fn update_aspect_ratio(&mut self, aspect: f32) {
        self.aspect = aspect;
        self.projection_matrix = Mat4::perspective_rh(self.fovy.to_radians(), aspect, self.znear, self.zfar);
        self.update_view_matrix();
    }
    
    pub fn process_input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput { 
                event: KeyEvent { 
                    state, logical_key: key, ..
                }, ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                let mut processed = true;

                if let winit::keyboard::Key::Character(k) = key {
                    match k.as_str() {
                        "w" => self.moving_forward = is_pressed,
                        "s" => self.moving_backward = is_pressed,
                        "a" => self.moving_left = is_pressed,
                        "d" => self.moving_right = is_pressed,
                        "q" => self.moving_up = is_pressed,
                        "e" => self.moving_down = is_pressed,
                        "r" => {
                            // Reset camera position
                            if is_pressed {
                                self.target = Vec3::ZERO;
                                self.yaw = 0.0;
                                self.pitch = 0.0;
                                self.distance = 5.0;
                                self.target_yaw = 0.0;
                                self.target_pitch = 0.0;
                                self.target_distance = 5.0;
                                self.update_view_matrix();
                            }
                        }
                        _ => processed = false,
                    }
                } else {
                    processed = false;
                }

                processed
            }
            WindowEvent::MouseInput { 
                state, button, ..
            } => {
                if *button == MouseButton::Left {
                    self.is_orbiting = *state == ElementState::Pressed;
                    
                    // Reset last mouse position when starting to orbit
                    if self.is_orbiting {
                        self.last_mouse_position = None;
                    }
                    true
                } else if *button == MouseButton::Right && *state == ElementState::Pressed {
                    // Reset view on right click
                    self.target = Vec3::ZERO;
                    self.yaw = 0.0;
                    self.pitch = 0.0;
                    self.distance = 5.0;
                    self.target_yaw = 0.0;
                    self.target_pitch = 0.0;
                    self.target_distance = 5.0;
                    self.update_view_matrix();
                    true
                } else {
                    false
                }
            }
            WindowEvent::MouseWheel { 
                delta, ..
            } => {
                match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        // Adjust zoom level with scroll - smoother
                        self.target_distance -= y * SCROLL_SENSITIVITY;
                        self.target_distance = self.target_distance.clamp(MIN_ZOOM, MAX_ZOOM);
                        true
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                        self.target_distance -= (pos.y as f32) * 0.002;
                        self.target_distance = self.target_distance.clamp(MIN_ZOOM, MAX_ZOOM);
                        true
                    }
                }
            }
            WindowEvent::CursorMoved { 
                position, ..
            } => {
                if self.is_orbiting {
                    let new_pos = (position.x as f32, position.y as f32);
                    
                    if let Some((last_x, last_y)) = self.last_mouse_position {
                        // Calculate delta from last position
                        let delta_x = new_pos.0 - last_x;
                        let delta_y = new_pos.1 - last_y;
                        
                        // Apply smoothed movement with reduced sensitivity
                        self.target_yaw -= delta_x * CAMERA_SENSITIVITY;
                        self.target_pitch = (self.target_pitch - delta_y * CAMERA_SENSITIVITY)
                            .clamp(-PI / 2.0 + 0.1, PI / 2.0 - 0.1);
                    }
                    
                    // Update last position
                    self.last_mouse_position = Some(new_pos);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }
    
    pub fn update(&mut self, dt: f32) {
        let mut needs_update = false;
        
        // Apply smoothing to orbital movement
        if (self.yaw - self.target_yaw).abs() > 0.001 {
            self.yaw = self.yaw * ORBIT_SMOOTHING + self.target_yaw * (1.0 - ORBIT_SMOOTHING);
            needs_update = true;
        }
        
        if (self.pitch - self.target_pitch).abs() > 0.001 {
            self.pitch = self.pitch * ORBIT_SMOOTHING + self.target_pitch * (1.0 - ORBIT_SMOOTHING);
            needs_update = true;
        }
        
        if (self.distance - self.target_distance).abs() > 0.01 {
            self.distance = self.distance * ORBIT_SMOOTHING + self.target_distance * (1.0 - ORBIT_SMOOTHING);
            needs_update = true;
        }
        
        // Handle keyboard movement with proper orbit update
        let mut movement = Vec3::ZERO;
        
        // Calculate camera's local coordinate system
        let rotation = Quat::from_euler(
            glam::EulerRot::YXZ,
            self.yaw,
            self.pitch,
            0.0,
        );
        
        let forward = rotation * -Vec3::Z;
        let right = rotation * Vec3::X;
        let local_up = Vec3::Y;  // Use world up for consistent vertical movement
        
        // Apply movement based on pressed keys
        if self.moving_forward {
            movement += forward;
        }
        if self.moving_backward {
            movement -= forward;
        }
        if self.moving_right {
            movement += right;
        }
        if self.moving_left {
            movement -= right;
        }
        if self.moving_up {
            movement += local_up;
        }
        if self.moving_down {
            movement -= local_up;
        }
        
        // Normalize and scale movement, apply to target
        if movement != Vec3::ZERO {
            movement = movement.normalize() * CAMERA_SPEED * dt * self.distance * 0.2;
            self.target += movement;
            needs_update = true;
        }
        
        if needs_update {
            self.update_view_matrix();
        }
    }
    
    fn update_view_matrix(&mut self) {
        // Calculate position based on orbital parameters
        let rotation = Quat::from_euler(
            glam::EulerRot::YXZ,
            self.yaw,
            self.pitch,
            0.0,
        );
        
        // Calculate position from spherical coordinates to cartesian
        let offset = rotation * Vec3::new(0.0, 0.0, self.distance);
        self.position = self.target + offset;
        
        // Update view matrix
        self.view_matrix = Mat4::look_at_rh(self.position, self.target, Vec3::Y);
        
        // Update uniform with current values
        self.update_uniform();
    }
    
    // Update the camera uniform with current camera data
    fn update_uniform(&mut self) {
        // Calculate view-projection matrix
        let view_proj = self.projection_matrix * self.view_matrix;
        
        // Update the uniform directly
        self.uniform.view_proj = view_proj.to_cols_array_2d();
        self.uniform.view_position = [self.position.x, self.position.y, self.position.z, 1.0];
    }

    pub fn update_buffer(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniform]));
    }
    
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
    
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
    
    // New helper methods for common operations
    
    // Get the current camera position
    pub fn position(&self) -> Vec3 {
        self.position
    }
    
    // Get the current view direction
    pub fn view_direction(&self) -> Vec3 {
        (self.target - self.position).normalize()
    }
    
    // Get the current target point
    pub fn target(&self) -> Vec3 {
        self.target
    }
    
    // Set the camera to look at a specific point
    pub fn look_at(&mut self, point: Vec3) {
        self.target = point;
        self.update_view_matrix();
    }
}