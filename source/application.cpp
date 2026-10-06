#include "application.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


namespace application {

using graphics::internal::context;

namespace {
	// 1. Типы данных
// одна вершина фигуры
struct Vertex {
	glm::vec3 position;
	glm::vec3 color;
};

static_assert(sizeof(Vertex) == 6 * sizeof(float), "Vertex must be tightly packed");

struct PushConstants {
	glm::mat4 mvp;
	glm::vec4 tint;
};
static_assert(sizeof(PushConstants) == 80, "PushConstants layout must match the shader");

// буфер Vulkan вместе с его памятью
struct Buffer {
	VkBuffer handle = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
};

// Все, что пользователь меняет в интерфейсе
struct Scene {
	int projection = 0;
	float fov_degrees = 60.0f;
	float ortho_half_height = 2.3f;
	float camera_distance = 4.0f;

	glm::vec3 position{0.0f};
	glm::vec3 rotation_degrees{20.0f, 30.0f, 0.0f};
	glm::vec3 scale{1.0f};
	glm::vec3 tint{1.0f};

    // параметры анимации
    bool playing = true;
    float speed = 1.0f;
    float path_radius = 1.2f;
    float path_height = 0.6f;
    float spin_speed = 60.0f;
};

// 2. Геометрия: усеченный правильный тетраэдр
constexpr int corner_count = 12;
constexpr int corners[corner_count][3] = {
	{ 3,  1,  1},
    {-3, -1,  1},
    {-3,  1, -1},
    { 3, -1, -1},
    { 1,  3,  1},
    {-1, -3,  1},
    { 1, -3, -1},
    {-1,  3, -1},
    { 1,  1,  3},
    {-1, -1,  3},
    { 1, -1, -3},
    {-1,  1, -3},
};

constexpr uint16_t indices[] = {
	// 4 правильных треугольника
	0, 4, 8,
	1, 5, 9,
	2, 7, 11,
	3, 6, 10,

	// шестиугольники
	1, 5, 6,  1, 6, 10,  1, 10, 11,  1, 11, 2,

	0, 3, 10,  0, 10, 11,  0, 11, 7,  0, 7, 4,

	0, 3, 6,  0, 6, 5,  0, 5, 9,  0, 9, 8,

	1, 2, 7, 1, 7, 4,  1, 4, 8,  1, 8, 9,
};

constexpr uint32_t indices_total = sizeof(indices) / sizeof(indices[0]);
static_assert(indices_total == 60, "Truncated tetrahedron must have 20 triangles");

// 3. Глобальное состояние приложения
Scene scene;
glm::mat4 mvp(1.0f);

double animation_time = 0.0;
double last_time = -1.0;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

Buffer vertex_buffer;
Buffer index_buffer;

// 4. Вспомогательные функции
// Читает скомпилированный шейдер из файла
std::vector<uint32_t> readSpirv(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file " << path << std::endl;
		return {};
	}
	const std::streamsize size = file.tellg();
	if (size <= 0 || size % 4 != 0) {
		std::cerr << "Shader file " << path << " has invalid size" << std::endl;
		return {};
	}

	std::vector<uint32_t> code(size_t(size) / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), size);
	return code;
}

// Создает шейдерный модуль из .spv файла
VkShaderModule createShaderModule(const char* path) {
	const std::vector<uint32_t> code = readSpirv(path);
	if (code.empty()) {
		return VK_NULL_HANDLE;
	}

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size() * sizeof(uint32_t),
		.pCode = code.data(),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module from " << path << std::endl;
		return VK_NULL_HANDLE;
	}

	return module;

}

//Создает буффер нужного типа и копирует в него данные
bool createBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info, &out.handle, &out.allocation, nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan buffer\n";
        return false;
    }

    if (vmaCopyMemoryToAllocation(context.allocator, data, out.allocation, 0, size) != VK_SUCCESS) {
        std::cerr << "Failed to copy data into Vulkan buffer\n";
        return false;
    }

    return true;
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.handle != VK_NULL_HANDLE) {
        vmaDestroyBuffer(context.allocator, buffer.handle, buffer.allocation);
        buffer.handle = VK_NULL_HANDLE;
        buffer.allocation = nullptr;
    }
}

// Создает графический конвейер
bool createPipeline() {
	VkShaderModule vert = createShaderModule("shaders/tetra.vert.spv");
    VkShaderModule frag = createShaderModule("shaders/tetra.frag.spv");
	const auto cleanup = [&](bool result) {
        vkDestroyShaderModule(context.device, vert, nullptr);
        vkDestroyShaderModule(context.device, frag, nullptr);
        return result;
    };

	if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
        return cleanup(false);
    }

	const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
	const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return cleanup(false);
    }

	// Какие шейдеры используются
	const VkPipelineShaderStageCreateInfo stages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag,
            .pName = "main",
        },
    };
	const VkVertexInputBindingDescription binding = {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

	const VkVertexInputAttributeDescription attributes[] = {
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, position),
        },
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, color),
        },
    };
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 2,
        .pVertexAttributeDescriptions = attributes,
    };

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

	const VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

	const VkPipelineRasterizationStateCreateInfo rasterization = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f,
    };

	const VkPipelineMultisampleStateCreateInfo multisample = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };
	const VkPipelineColorBlendAttachmentState blend_attachment = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

	 const VkPipelineColorBlendStateCreateInfo color_blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blend_attachment,
    };

    const VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
	const VkPipelineDynamicStateCreateInfo dynamic_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamic_states,
    };

	const VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic_state,
        .layout = pipeline_layout,
        .renderPass = context.render_pass,
        .subpass = 0,
    };

	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline\n";
        return cleanup(false);
    }

    return cleanup(true);
}


} // namespace


// 5. Функции приложения, которые вызывает main
// Создает все объекты Vulkan
bool initialize() {
	std::vector<Vertex> vertices;
	vertices.reserve(corner_count);

	for (const auto& c : corners) {
		const glm::vec3 position = glm::vec3(float(c[0]), float(c[1]), float(c[2])) / 3.0f;
		const glm::vec3 color = position * 0.5f + glm::vec3(0.5f);
		vertices.push_back({position, color});
	}

	if (!createBuffer(vertices.data(), vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer)) {

		return false;
	}

	if (!createBuffer(indices, sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer)) {
		return false;
	}
	return createPipeline();
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);
	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	
	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);
}

void update([[maybe_unused]] double time) {
    if (last_time < 0.0) {
        last_time = time;
    }

    const double dt = std::fmin(time - last_time, 0.1);
    last_time = time;
    if (scene.playing) {
        animation_time += dt * double(scene.speed);
    }

	ImGui::Begin("Truncated tetrahedron");

	ImGui::SeparatorText("Projection");
	ImGui::RadioButton("Perspective", &scene.projection, 0);
	ImGui::SameLine();
	ImGui::RadioButton("Orthographic", &scene.projection, 1);
	if (scene.projection == 0) {
		ImGui::SliderFloat("FOV", &scene.fov_degrees, 20.0f, 120.0f);
	} else {
		ImGui::SliderFloat("Half height", &scene.ortho_half_height, 0.5f, 5.0f);
	}

	ImGui::SeparatorText("Transform");
	ImGui::DragFloat3("Position", &scene.position.x, 0.01f);
	ImGui::SliderFloat3("Rotation", &scene.rotation_degrees.x, -180.0f, 180.0f);
	ImGui::DragFloat3("Scale", &scene.scale.x, 0.01f, 0.1f, 5.0f);

	ImGui::SeparatorText("Color");
	ImGui::ColorEdit3("Tint", &scene.tint.x);

    ImGui::SeparatorText("Animation");
    if (ImGui::Button(scene.playing ? "Pause" : "Play")) {
        scene.playing = !scene.playing;
    }
    ImGui::SameLine();
    if (ImGui::Button("Restart")) {
        animation_time = 0.0;
    }
    ImGui::SliderFloat("Speed", &scene.speed, 0.0f, 5.0f);
    ImGui::SliderFloat("Path radius", &scene.path_radius, 0.0f, 3.0f);
    ImGui::SliderFloat("Path height", &scene.path_height, 0.0f, 2.0f);
    ImGui::SliderFloat("Spin (deg/s)", &scene.spin_speed, 0.0f, 360.0f);

    ImGui::Separator();
	if (ImGui::Button("Reset all")) {
		scene = Scene{};
        animation_time = 0.0;
	}

	ImGui::End();

	// Матрицы
	const VkExtent2D extent = context.swapchain_extent;
    const float aspect = float(extent.width) / float(extent.height);

	glm::mat4 projection(1.0f);
    if (scene.projection == 0) {
        projection = glm::perspective(glm::radians(scene.fov_degrees), aspect, 0.1f, 100.0f);
    } else {
        const float h = scene.ortho_half_height;
        const float w = h * aspect;
        projection = glm::ortho(-w, w, -h, h, 0.1f, 100.0f);
    }

	const glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -scene.camera_distance));

    const float t = float(animation_time);
    const glm::vec3 path_offset(
        scene.path_radius * std::sin(t),
        scene.path_height * std::sin(2.0f * t),
        0.5f * scene.path_radius * std::sin(3.0f * t)
    );

    const float spin = scene.spin_speed * t;
    
	// Сначала рястажение, потом повороты, сдвиг
	glm::mat4 model = glm::translate(glm::mat4(1.0f), scene.position + path_offset);
    model = glm::rotate(model, glm::radians(scene.rotation_degrees.x + 0.5f * spin), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(scene.rotation_degrees.y + spin), glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(scene.rotation_degrees.z), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::scale(model, scene.scale);

	mvp = projection * view * model;

}

void render(const graphics::internal::FrameData& fd) {
	const VkCommandBuffer cmd = fd.command_buffer;
    const VkExtent2D extent = context.swapchain_extent;
	vkResetCommandBuffer(cmd, 0);

    const VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin_info);

	VkClearValue clear_values[2] = {};
    clear_values[0].color = {{0.08f, 0.08f, 0.10f, 1.0f}};
    clear_values[1].depthStencil = {1.0f, 0};

    const VkRenderPassBeginInfo pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = { .offset = {0, 0}, .extent = extent },
        .clearValueCount = 2,
        .pClearValues = clear_values,
    };
    vkCmdBeginRenderPass(cmd, &pass_info, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	const VkViewport viewport = {
        .x = 0.0f,
        .y = float(extent.height),
        .width = float(extent.width),
        .height = -float(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    const VkRect2D scissor = { .offset = {0, 0}, .extent = extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

	const PushConstants constants = {
        .mvp = mvp,
        .tint = glm::vec4(scene.tint, 1.0f),
    };
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &constants);

	const VkDeviceSize vertex_offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.handle, &vertex_offset);
    vkCmdBindIndexBuffer(cmd, index_buffer.handle, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, indices_total, 1, 0, 0, 0);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);

}

} // namespace application