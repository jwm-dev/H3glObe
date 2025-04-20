// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
}

struct CameraUniform {
    view_proj: mat4x4<f32>,
    view_position: vec4<f32>,
}

struct GlobeUniform {
    model: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> globe: GlobeUniform;

@group(1) @binding(0)
var<uniform> camera: CameraUniform;

@vertex
fn vs_main(
    vertex: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Apply world transformation
    let world_position = globe.model * vec4<f32>(vertex.position, 1.0);
    
    // Project to clip space
    out.clip_position = camera.view_proj * world_position;
    
    // Pass other attributes to fragment shader
    out.position = world_position.xyz;
    out.normal = normalize((globe.model * vec4<f32>(vertex.normal, 0.0)).xyz);
    out.color = vertex.color;
    
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple lighting model
    let light_position = vec3<f32>(5.0, 5.0, 5.0);
    let light_color = vec3<f32>(1.0, 1.0, 1.0);
    let ambient_strength = 0.2;
    let diffuse_strength = 0.6;
    let specular_strength = 0.3;
    let shininess = 32.0;
    
    // Calculate ambient light
    let ambient = ambient_strength * light_color;
    
    // Calculate diffuse light
    let light_dir = normalize(light_position - in.position);
    let diff = max(dot(in.normal, light_dir), 0.0);
    let diffuse = diffuse_strength * diff * light_color;
    
    // Calculate specular light
    let view_dir = normalize(camera.view_position.xyz - in.position);
    let reflect_dir = reflect(-light_dir, in.normal);
    let spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
    let specular = specular_strength * spec * light_color;
    
    // Calculate lighting on cell color
    let result = (ambient + diffuse + specular) * in.color;
    
    return vec4<f32>(result, 0.9);
}

// Solid face fragment shader with enhanced lighting
@fragment
fn fs_main_solid(in: VertexOutput) -> @location(0) vec4<f32> {
    // Enhanced lighting for solid faces
    let light_position = vec3<f32>(5.0, 5.0, 5.0);
    let light_color = vec3<f32>(1.0, 1.0, 1.0);
    let ambient_strength = 0.3;
    let diffuse_strength = 0.7;
    
    // Additional light from camera direction for better visibility
    let camera_light = 0.2;
    
    // Calculate ambient light
    let ambient = ambient_strength * light_color;
    
    // Calculate diffuse light from main light source
    let light_dir = normalize(light_position - in.position);
    let diff = max(dot(in.normal, light_dir), 0.0);
    let diffuse = diffuse_strength * diff * light_color;
    
    // Add camera-based lighting for better visibility of facing surfaces
    let view_dir = normalize(camera.view_position.xyz - in.position);
    let camera_diff = max(dot(in.normal, view_dir), 0.0);
    let camera_diffuse = camera_light * camera_diff * light_color;
    
    // Combine lighting with cell color
    let result = (ambient + diffuse + camera_diffuse) * in.color;
    
    // Slightly higher opacity for solid faces
    return vec4<f32>(result, 0.95);
}