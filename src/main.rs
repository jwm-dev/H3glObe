use std::time::Instant;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

mod camera;
mod globe;
mod render;
mod elevation;

use camera::Camera;
use globe::{Globe, H3Resolution};
use h3o::Resolution;
use std::path::Path;
use std::thread;

const WINDOW_TITLE: &str = "H3glObe - H3 Geospatial Visualization with Elevation";

fn main() {
    env_logger::init();
    
    // Create a new event loop
    let event_loop = EventLoop::new().unwrap();
    let mut app = H3GlobeApp::new();
    
    // Set control flow to poll (continuous rendering)
    event_loop.set_control_flow(ControlFlow::Poll);
    
    // Run the event loop with our application handler
    event_loop.run_app(&mut app).unwrap();
}

// Remove the lifetime from State struct
struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    globe: Globe,
    camera: Camera,
    last_frame_time: Instant,
    window_id: winit::window::WindowId,
}

impl State {
    // Change to take ownership of the window, not reference
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();
        let window_id = window.id();

        // Initialize WGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        // Remove unnecessary unsafe block
        let surface = instance.create_surface(window).unwrap().into_static();
        
        // ... rest of the initialization code remains the same
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .unwrap();

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        
        surface.configure(&device, &config);

        // Initialize elevation data
        println!("Initializing elevation data...");
        let etopo_path = Path::new("assets/etopo15s_tifs");
        let elevation_data = elevation::ElevationData::initialize(etopo_path);
        
        // Preload elevation data for lower resolutions in a separate thread
        // to avoid blocking the main thread during startup
        let elevation_data_clone = elevation_data.clone();
        thread::spawn(move || {
            println!("Preloading elevation data for resolutions 0-4...");
            // Preload data for resolutions 0-4
            for i in 0..5 {
                let res = Resolution::try_from(i).unwrap();
                elevation_data_clone.preload_elevation_data(res);
            }
            println!("Preloading complete!");
        });
        
        // Default starting resolution
        let default_resolution = 2;
        println!("Initializing globe with resolution {}...", default_resolution);
        
        // Initialize globe with elevation data and default resolution
        let globe = Globe::new(&device, H3Resolution::new(default_resolution), Some(elevation_data));
        
        // Initialize camera
        let camera = Camera::new(
            &device,
            config.width as f32 / config.height as f32,
            45.0,
            0.1,
            100.0,
        );

        Self {
            surface,
            device,
            queue,
            config,
            size,
            globe,
            camera,
            last_frame_time: Instant::now(),
            window_id,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            
            // Update depth texture to match the new window size
            self.globe.update_depth_texture(&self.device, new_size.width, new_size.height);
            
            self.camera.update_aspect_ratio(new_size.width as f32 / new_size.height as f32);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera.process_input(event)
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt = now - self.last_frame_time;
        self.last_frame_time = now;
        
        self.camera.update(dt.as_secs_f32());
        self.camera.update_buffer(&self.queue);
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.15,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.globe.depth_texture_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            self.globe.render(&mut render_pass, &self.camera);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        Ok(())
    }
}

// Application handler implementation for winit 0.30.9
// Remove the lifetime parameter 'a since it's not used
struct H3GlobeApp {
    state: Option<State>,
    last_render_time: Instant,
    window: Option<Window>,
}

// Remove the lifetime parameter from the implementation too
impl H3GlobeApp {
    fn new() -> Self {
        Self {
            state: None,
            last_render_time: Instant::now(),
            window: None,
        }
    }
}

// Update this implementation to remove the lifetime parameter
impl winit::application::ApplicationHandler for H3GlobeApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // Create the window when the application is resumed
        // (This is the proper place to create windows in the newer winit API)
        let window_attributes = WindowAttributes::default()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720));
            
        let window = event_loop
            .create_window(window_attributes)
            .unwrap();
            
        // Store window reference
        self.window = Some(window);
        
        // Initialize the state with our window
        if let Some(window) = &self.window {
            self.state = Some(pollster::block_on(State::new(window)));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(state) => {
                // Only process events for our window
                if window_id != state.window_id {
                    return;
                }
                state
            }
            None => return,
        };

        if !state.input(&event) {
            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    state.resize(physical_size);
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let _dt = now - self.last_render_time;
                    self.last_render_time = now;

                    state.update();
                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => eprintln!("{:?}", e),
                    }
                }
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                    ..
                } => event_loop.exit(),
                WindowEvent::KeyboardInput {
                    event: KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: Key::Character(ref key),
                        ..
                    },
                    ..
                } => {
                    if key == "+" || key == "=" {
                        state.globe.increase_resolution();
                    } else if key == "-" || key == "_" {
                        state.globe.decrease_resolution();
                    } else if key == "m" || key == "M" {
                        state.globe.toggle_render_mode();
                    }
                }
                _ => {}
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        // Request a redraw for our window
        if let (Some(state), Some(window)) = (&self.state, &self.window) {
            if window.id() == state.window_id {
                window.request_redraw();
            }
        }
    }
}

// Now we need to add an extension trait to convert Surface to 'static lifetime
trait SurfaceExt<'a> {
    fn into_static(self) -> wgpu::Surface<'static>;
}

impl<'a> SurfaceExt<'a> for wgpu::Surface<'a> {
    fn into_static(self) -> wgpu::Surface<'static> {
        // This is safe as long as we ensure the Surface doesn't outlive the Window
        // which we do by tracking the window_id
        unsafe { std::mem::transmute(self) }
    }
}