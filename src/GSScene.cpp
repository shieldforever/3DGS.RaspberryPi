//
// Created by steven on 11/30/23.
//

#include <fstream>
#include "GSScene.h"

#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <random>
#include "shaders.h"

#include "vulkan/Utils.h"
#include "vulkan/DescriptorSet.h"
#include "vulkan/pipelines/ComputePipeline.h"
#include "spdlog/spdlog.h"
#include "vulkan/Shader.h"

namespace {
size_t plyTypeSize(const std::string& type) {
    if (type == "char" || type == "int8" || type == "uchar" || type == "uint8") return 1;
    if (type == "short" || type == "int16" || type == "ushort" || type == "uint16") return 2;
    if (type == "int" || type == "int32" || type == "uint" || type == "uint32" || type == "float" || type == "float32") return 4;
    if (type == "double" || type == "float64" || type == "int64" || type == "uint64") return 8;
    throw std::runtime_error("Unsupported PLY property type: " + type);
}

float readPlyScalarAsFloat(const char* data, const std::string& type) {
    if (type == "float" || type == "float32") {
        float v = 0.0f;
        std::memcpy(&v, data, sizeof(float));
        return v;
    }
    if (type == "double" || type == "float64") {
        double v = 0.0;
        std::memcpy(&v, data, sizeof(double));
        return static_cast<float>(v);
    }
    if (type == "char" || type == "int8") {
        int8_t v = 0;
        std::memcpy(&v, data, sizeof(int8_t));
        return static_cast<float>(v);
    }
    if (type == "uchar" || type == "uint8") {
        uint8_t v = 0;
        std::memcpy(&v, data, sizeof(uint8_t));
        return static_cast<float>(v);
    }
    if (type == "short" || type == "int16") {
        int16_t v = 0;
        std::memcpy(&v, data, sizeof(int16_t));
        return static_cast<float>(v);
    }
    if (type == "ushort" || type == "uint16") {
        uint16_t v = 0;
        std::memcpy(&v, data, sizeof(uint16_t));
        return static_cast<float>(v);
    }
    if (type == "int" || type == "int32") {
        int32_t v = 0;
        std::memcpy(&v, data, sizeof(int32_t));
        return static_cast<float>(v);
    }
    if (type == "uint" || type == "uint32") {
        uint32_t v = 0;
        std::memcpy(&v, data, sizeof(uint32_t));
        return static_cast<float>(v);
    }
    if (type == "int64") {
        int64_t v = 0;
        std::memcpy(&v, data, sizeof(int64_t));
        return static_cast<float>(v);
    }
    if (type == "uint64") {
        uint64_t v = 0;
        std::memcpy(&v, data, sizeof(uint64_t));
        return static_cast<float>(v);
    }

    throw std::runtime_error("Unsupported PLY property type: " + type);
}
}

void GSScene::load(const std::shared_ptr<VulkanContext>&context) {
    auto startTime = std::chrono::high_resolution_clock::now();

    std::ifstream plyFile(filename, std::ios::binary);
    loadPlyHeader(plyFile);
    if (header.format != "binary_little_endian") {
        throw std::runtime_error("Only binary_little_endian PLY is supported. Got: " + header.format);
    }

    std::unordered_map<std::string, size_t> propertyIndex;
    std::vector<size_t> propertyOffsets(header.vertexProperties.size(), 0);
    size_t vertexStride = 0;
    for (size_t i = 0; i < header.vertexProperties.size(); ++i) {
        propertyIndex[header.vertexProperties[i].name] = i;
        propertyOffsets[i] = vertexStride;
        vertexStride += plyTypeSize(header.vertexProperties[i].type);
    }
    std::vector<char> vertexRaw(vertexStride);

    vertexBuffer = createBuffer(context, header.numVertices * sizeof(Vertex));
    auto vertexStagingBuffer = Buffer::staging(context, header.numVertices * sizeof(Vertex));
    auto* verteces = static_cast<Vertex *>(vertexStagingBuffer->allocation_info.pMappedData);

    auto readProperty = [&](const std::vector<char>& raw, const std::string& name, float fallback = 0.0f) -> float {
        auto it = propertyIndex.find(name);
        if (it == propertyIndex.end()) {
            return fallback;
        }
        const size_t idx = it->second;
        const char* ptr = raw.data() + propertyOffsets[idx];
        return readPlyScalarAsFloat(ptr, header.vertexProperties[idx].type);
    };

    for (auto i = 0; i < header.numVertices; i++) {
        assert(plyFile.is_open());
        assert(!plyFile.eof());
        plyFile.read(vertexRaw.data(), static_cast<std::streamsize>(vertexStride));
        if (plyFile.gcount() != static_cast<std::streamsize>(vertexStride)) {
            throw std::runtime_error("Unexpected end of PLY vertex data while reading: " + filename);
        }

        const glm::vec3 position(
            readProperty(vertexRaw, "x", 0.0f),
            readProperty(vertexRaw, "y", 0.0f),
            readProperty(vertexRaw, "z", 0.0f)
        );
        const glm::vec3 normal(
            readProperty(vertexRaw, "nx", 0.0f),
            readProperty(vertexRaw, "ny", 0.0f),
            readProperty(vertexRaw, "nz", 0.0f)
        );
        const glm::vec3 scale(
            readProperty(vertexRaw, "scale_0", 0.0f),
            readProperty(vertexRaw, "scale_1", 0.0f),
            readProperty(vertexRaw, "scale_2", 0.0f)
        );
        const float opacity = readProperty(vertexRaw, "opacity", 0.0f);
        const glm::vec4 rotation(
            readProperty(vertexRaw, "rot_0", 0.0f),
            readProperty(vertexRaw, "rot_1", 0.0f),
            readProperty(vertexRaw, "rot_2", 0.0f),
            readProperty(vertexRaw, "rot_3", 1.0f)
        );

        float shs[48] = {};
        shs[0] = readProperty(vertexRaw, "f_dc_0", 0.0f);
        shs[1] = readProperty(vertexRaw, "f_dc_1", 0.0f);
        shs[2] = readProperty(vertexRaw, "f_dc_2", 0.0f);
        for (int k = 0; k < 45; ++k) {
            shs[3 + k] = readProperty(vertexRaw, "f_rest_" + std::to_string(k), 0.0f);
        }

        // Parse MOE extension fields so models containing abc/d can be loaded safely.
        // They are currently not consumed by the renderer.
        const glm::vec3 abc(
            readProperty(vertexRaw, "abc_0", 1.0f),
            readProperty(vertexRaw, "abc_1", 1.0f),
            readProperty(vertexRaw, "abc_2", 1.0f)
        );
        const float d = propertyIndex.count("d") ? readProperty(vertexRaw, "d", 0.0f)
                                                 : readProperty(vertexRaw, "d_0", 0.0f);

        verteces[i].position = glm::vec4(position, 1.0f);
        verteces[i].scale_opacity = glm::vec4(glm::exp(scale), 1.0f / (1.0f + std::exp(-opacity)));
        verteces[i].rotation = normalize(rotation);
        verteces[i].shs[0] = shs[0];
        verteces[i].shs[1] = shs[1];
        verteces[i].shs[2] = shs[2];
        auto SH_N = 16;
        for (auto j = 1; j < SH_N; j++) {
            verteces[i].shs[j * 3 + 0] = shs[(j - 1) + 3];
            verteces[i].shs[j * 3 + 1] = shs[(j - 1) + SH_N + 2];
            verteces[i].shs[j * 3 + 2] = shs[(j - 1) + SH_N * 2 + 1];
        }
        verteces[i].abc_d = glm::vec4(abc, d);
        // Some datasets store non-zero normals; renderer does not consume them.
        (void)normal;
    }

    vertexBuffer->uploadFrom(vertexStagingBuffer);

    auto endTime = std::chrono::high_resolution_clock::now();
    spdlog::info("Loaded {} in {}ms", filename,
                 std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());

    precomputeCov3D(context);
}

void GSScene::loadTestScene(const std::shared_ptr<VulkanContext>&context) {
    int testObects = 1;
    header.numVertices = testObects;
    vertexBuffer = createBuffer(context, testObects * sizeof(Vertex));
    auto vertexStagingBuffer = Buffer::staging(context, testObects * sizeof(Vertex));
    auto* verteces = static_cast<Vertex *>(vertexStagingBuffer->allocation_info.pMappedData);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> posgen(-3.0, 3.0);
    std::uniform_real_distribution<> scalegen(100.0, 5000.0);
    std::uniform_real_distribution<> shsgen(-1.0, 1.0);


    for (auto i = 0; i < testObects; i++) {
        verteces[i].position = glm::vec4(posgen(gen), posgen(gen), posgen(gen), 1.0f);
        // verteces[i].normal = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        verteces[i].scale_opacity = glm::vec4(scalegen(gen), scalegen(gen), scalegen(gen), 0.5f);
        verteces[i].rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        for (auto j = 0; j < 48; j++) {
            verteces[i].shs[j] = shsgen(gen);
        }
        verteces[i].abc_d = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
    }

    vertexBuffer->uploadFrom(vertexStagingBuffer);

    precomputeCov3D(context);
}

void GSScene::loadPlyHeader(std::ifstream&plyFile) {
    if (!plyFile.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    header = {};
    std::string line;
    bool headerEnd = false;
    std::string currentElement;

    while (std::getline(plyFile, line)) {
        std::istringstream iss(line);
        std::string token;

        iss >> token;

        if (token == "ply") {
            // PLY format indicator
        }
        else if (token == "format") {
            iss >> header.format;
        }
        else if (token == "element") {
            iss >> currentElement;

            if (currentElement == "vertex") {
                iss >> header.numVertices;
            }
            else if (currentElement == "face") {
                iss >> header.numFaces;
            }
        }
        else if (token == "property") {
            std::string nextToken;
            iss >> nextToken;

            // Ignore list properties for now (usually face indices)
            if (nextToken == "list") {
                continue;
            }

            PlyProperty property;
            property.type = nextToken;
            iss >> property.name;

            if (currentElement == "vertex") {
                header.vertexProperties.push_back(property);
            }
            else if (currentElement == "face") {
                header.faceProperties.push_back(property);
            }
        }
        else if (token == "end_header") {
            headerEnd = true;
            break;
        }
    }

    if (!headerEnd) {
        throw std::runtime_error("Could not find end of header");
    }
}

std::shared_ptr<Buffer> GSScene::createBuffer(const std::shared_ptr<VulkanContext>&context, size_t i) {
    return std::make_shared<Buffer>(
        context, i, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_GPU_ONLY, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, false);
}

void GSScene::precomputeCov3D(const std::shared_ptr<VulkanContext>&context) {
    cov3DBuffer = createBuffer(context, header.numVertices * sizeof(float) * 6);

    auto pipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "precomp_cov3d", SPV_PRECOMP_COV3D, SPV_PRECOMP_COV3D_len));

    auto descriptorSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    descriptorSet->bindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             vertexBuffer);
    descriptorSet->bindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             cov3DBuffer);
    descriptorSet->build();

    pipeline->addDescriptorSet(0, descriptorSet);
    pipeline->addPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(float));
    pipeline->build();

    auto commandBuffer = context->beginOneTimeCommandBuffer();
    pipeline->bind(commandBuffer, 0, 0);
    float scaleFactor = 1.0f;
    commandBuffer->pushConstants(pipeline->pipelineLayout.get(), vk::ShaderStageFlagBits::eCompute, 0,
                                 sizeof(float), &scaleFactor);
    int numGroups = (header.numVertices + 255) / 256;
    commandBuffer->dispatch(numGroups, 1, 1);
    context->endOneTimeCommandBuffer(std::move(commandBuffer), VulkanContext::Queue::COMPUTE);

    spdlog::info("Precomputed Cov3D");
}
