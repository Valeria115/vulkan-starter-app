#version 450

// Входные данные одной вершины
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 tint;
} pc;

// Данные, которые уйдут во фрагментный шейдер
layout(location = 0) out vec3 frag_color;

void main() {
    gl_Position = pc.mvp * vec4(in_position, 1.0);

    frag_color = in_color * pc.tint.rgb;
}
