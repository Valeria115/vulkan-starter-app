#version 450

layout(location = 0) in vec3 frag_world_position;
layout(location = 1) in vec3 frag_normal;

layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 view_proj;
    vec4 camera_position;
    vec4 ambient;
    vec4 dir_direction;
    vec4 dir_color;
    ivec4 counts;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelData {
    mat4 model;
    mat4 normal_matrix;
    vec4 albedo;
    vec4 specular;
} object;

// Один точечный источник
struct PointLight {
    vec4 position;
    vec4 color;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight data[];
} point_lights;

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color;
    vec4 params;
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight data[];
} spot_lights;

layout(location = 0) out vec4 out_color;

vec3 shade(vec3 N, vec3 V, vec3 L, vec3 radiance) {
    float n_dot_l = max(dot(N, L), 0.0);
    vec3 diffuse = radiance * object.albedo.rgb * n_dot_l;

    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), object.specular.w);
    vec3 specular = radiance * object.specular.rgb * (n_dot_l > 0.0 ? spec : 0.0);

    return diffuse + specular;
}

void main() {
    vec3 N = normalize(frag_normal);
    vec3 V = normalize(scene.camera_position.xyz - frag_world_position);

    // Рассеянный свет
    vec3 color = scene.ambient.rgb * scene.ambient.w * object.albedo.rgb;

    // Направленный источник
    vec3 L = normalize(-scene.dir_direction.xyz);
    color += shade(N, V, L, scene.dir_color.rgb * scene.dir_color.w);

    // Точечные источники
    for (int i = 0; i < scene.counts.x; ++i) {
        vec3 to_light = point_lights.data[i].position.xyz - frag_world_position;
        float dist2 = max(dot(to_light, to_light), 0.0001);
        vec3 Lp = to_light * inversesqrt(dist2);
        float attenuation = 1.0 / dist2;

        vec3 radiance = point_lights.data[i].color.rgb * point_lights.data[i].color.w * attenuation;
        color += shade(N, V, Lp, radiance);
    }

    // Прожекторы
    for (int i = 0; i < scene.counts.y; ++i) {
        vec3 to_light = spot_lights.data[i].position.xyz - frag_world_position;
        float dist2 = max(dot(to_light, to_light), 0.0001);
        vec3 Ls = to_light * inversesqrt(dist2);
        float attenuation = 1.0 / dist2;

        float cos_theta = dot(-Ls, spot_lights.data[i].direction.xyz);
        float cos_inner = spot_lights.data[i].params.x;
        float cos_outer = spot_lights.data[i].params.y;

        // 1 внутри внутреннего конуса, 0 снаружи внешнего
        float t = clamp((cos_theta - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);
        float cone = smoothstep(0.0, 1.0, t);

        vec3 radiance = spot_lights.data[i].color.rgb * spot_lights.data[i].color.w * attenuation * cone;
        color += shade(N, V, Ls, radiance);
    }

    out_color = vec4(color, 1.0);
}