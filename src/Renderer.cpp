#include "Renderer.h"

#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <filesystem>
#include <spdlog/spdlog.h> // Ensure spdlog is included

#include "vulkan/Swapchain.h"

#include <memory>
#include "shaders.h"
#include <utility>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "vulkan/Utils.h"

#include <spdlog/spdlog.h>
#include <array>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>

namespace {
bool envEnabled(const char* name) {
    if (const char* v = std::getenv(name)) {
        return std::string(v) != "0";
    }
    return false;
}
}

void Renderer::initialize() {
    initializeVulkan();
    createGui();
    loadSceneToGPU();
    initializeCpuPreprocessResources();
    createPerFrameRenderQueryPools();
    createRenderPipeline();
    createCommandPool();
    startWritebackThread();
}

void Renderer::handleInput() {
    auto translation = window->getCursorTranslation();
    auto keys = window->getKeys(); // W, A, S, D

    if ((!configuration.enableGui || (!guiManager.wantCaptureMouse() && !guiManager.mouseCapture)) && window->
        getMouseButton()[0]) {
        window->mouseCapture(true);
        guiManager.mouseCapture = true;
    }

    // rotate camera
    if (!configuration.enableGui || guiManager.mouseCapture) {
        if (translation[0] != 0.0 || translation[1] != 0.0) {
            camera.rotation = glm::rotate(camera.rotation, static_cast<float>(translation[0]) * 0.005f,
                                          glm::vec3(0.0f, -1.0f, 0.0f));
            camera.rotation = glm::rotate(camera.rotation, static_cast<float>(translation[1]) * 0.005f,
                                          glm::vec3(-1.0f, 0.0f, 0.0f));
        }
    }

    // move camera
    if (!configuration.enableGui || !guiManager.wantCaptureKeyboard()) {
        glm::vec3 direction = glm::vec3(0.0f, 0.0f, 0.0f);
        if (keys[0]) {
            direction += glm::vec3(0.0f, 0.0f, -1.0f);
        }
        if (keys[1]) {
            direction += glm::vec3(-1.0f, 0.0f, 0.0f);
        }
        if (keys[2]) {
            direction += glm::vec3(0.0f, 0.0f, 1.0f);
        }
        if (keys[3]) {
            direction += glm::vec3(1.0f, 0.0f, 0.0f);
        }
        if (keys[4]) {
            direction += glm::vec3(0.0f, 1.0f, 0.0f);
        }
        if (keys[5]) {
            direction += glm::vec3(0.0f, -1.0f, 0.0f);
        }
        if (keys[6]) {
            window->mouseCapture(false);
            guiManager.mouseCapture = false;
        }
        if (direction != glm::vec3(0.0f, 0.0f, 0.0f)) {
            direction = glm::normalize(direction);
            camera.position += (glm::mat4_cast(camera.rotation) * glm::vec4(direction, 1.0f)).xyz() * 0.3f;
        }
    }
}

std::array<double, 6> Renderer::retrieveTimestamps(bool logTiming) {
    const bool trace = envEnabled("VKGS_TRACE_FRAME");
    if (trace) {
        spdlog::info("[TRACE] retrieveTimestamps begin: nextId={}", queryManager->nextId);
    }
    std::vector<uint64_t> timestamps(queryManager->nextId);
    auto res = context->device->getQueryPoolResults(context->queryPool.get(), 0, queryManager->nextId,
                                                    timestamps.size() * sizeof(uint64_t),
                                                    timestamps.data(), sizeof(uint64_t),
                                                    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (res != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to retrieve timestamps");
    }
    if (trace) {
        spdlog::info("[TRACE] retrieveTimestamps end");
    }

    auto metrics = queryManager->parseResults(timestamps);
    for (auto& metric: metrics) {
        if (configuration.enableGui)
            guiManager.pushMetric(metric.first, metric.second / 1000000.0);
    }
    auto getMs = [&](const char* key) -> double {
        auto it = metrics.find(key);
        if (it == metrics.end()) {
            return -1.0;
        }
        return static_cast<double>(it->second) / 1000000.0;
    };

    std::array<double, 6> stageMs = {
        getMs("preprocess"),
        getMs("prefix_sum"),
        getMs("preprocess_sort"),
        getMs("sort"),
        getMs("tile_boundary"),
        getMs("render")
    };
    if (logTiming) {
        spdlog::info(
            "[TIMING][ms] preprocess={:.3f}, prefix_sum={:.3f}, preprocess_sort={:.3f}, sort={:.3f}, tile_boundary={:.3f}, render={:.3f}",
            cpuStageTiming.preprocessMs,
            cpuStageTiming.prefixSumMs,
            cpuStageTiming.preprocessSortMs,
            cpuStageTiming.sortMs,
            cpuStageTiming.tileBoundaryMs,
            stageMs[5]
        );
    }

    return stageMs;
}

void Renderer::createPerFrameRenderQueryPools() {
    frameRenderQueryPools.resize(FRAMES_IN_FLIGHT);
    framePendingTiming.assign(FRAMES_IN_FLIGHT, {});
    framePendingReadback.assign(FRAMES_IN_FLIGHT, {});
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        vk::QueryPoolCreateInfo info{};
        info.queryType = vk::QueryType::eTimestamp;
        info.queryCount = 2;
        frameRenderQueryPools[i] = context->device->createQueryPoolUnique(info);
    }
}

void Renderer::flushCompletedTimingForSlot(uint32_t slot, bool logTiming) {
    if (slot >= framePendingTiming.size() || !framePendingTiming[slot].valid) {
        return;
    }
    uint64_t ts[2] = {0, 0};
    auto res = context->device->getQueryPoolResults(frameRenderQueryPools[slot].get(), 0, 2,
                                                    sizeof(ts), ts, sizeof(uint64_t),
                                                    vk::QueryResultFlagBits::e64);
    if (res != vk::Result::eSuccess) {
        return;
    }
    const double renderMs = static_cast<double>(ts[1] - ts[0]) / 1000000.0;
    const auto cpu = framePendingTiming[slot].cpu;
    const double cpuTotal = cpu.preprocessMs + cpu.prefixSumMs +
                            cpu.preprocessSortMs + cpu.sortMs + cpu.tileBoundaryMs;
    timingCpuTotalMsAccum += cpuTotal;
    timingRenderMsAccum += renderMs;
    timingSampleCount++;
    framePendingTiming[slot].valid = false;

    if (logTiming) {
        spdlog::info(
            "[TIMING][ms] preprocess={:.3f}, prefix_sum={:.3f}, preprocess_sort={:.3f}, sort={:.3f}, tile_boundary={:.3f}, render={:.3f}",
            cpu.preprocessMs, cpu.prefixSumMs, cpu.preprocessSortMs, cpu.sortMs, cpu.tileBoundaryMs, renderMs);
    }
}

void Renderer::flushCompletedReadbackForSlot(uint32_t slot) {
    if (slot >= framePendingReadback.size() || !framePendingReadback[slot].valid) {
        return;
    }
    enqueueAsyncReadback(framePendingReadback[slot].imageIndex, framePendingReadback[slot].outputPath);
    framePendingReadback[slot].valid = false;
}

std::string Renderer::addFrameSuffixToPath(const std::string& basePath, uint64_t frameId) {
    if (basePath.empty()) {
        return basePath;
    }
    std::filesystem::path p(basePath);
    const std::string stem = p.stem().string();
    const std::string ext = p.extension().string();
    const std::string suffixed = fmt::format("{}_{:06d}{}", stem, static_cast<int>(frameId), ext);
    return (p.parent_path() / suffixed).string();
}

void Renderer::enqueueAsyncReadback(uint32_t imageIndex, const std::string& outputPath) {
    if (outputPath.empty()) {
        return;
    }
    auto [width, height] = swapchain->swapchainExtent;
    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4;
    auto stagingBuffer = std::make_shared<Buffer>(context, imageSize,
                              vk::BufferUsageFlagBits::eTransferDst,
                              VMA_MEMORY_USAGE_CPU_ONLY, VMA_ALLOCATION_CREATE_MAPPED_BIT, false);

    vk::CommandBufferAllocateInfo allocInfo{commandPool.get(), vk::CommandBufferLevel::ePrimary, 1};
    auto copyCmdBuffers = context->device->allocateCommandBuffersUnique(allocInfo);
    auto cmd = std::move(copyCmdBuffers[0]);

    cmd->begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = vk::ImageLayout::ePresentSrcKHR;
    barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eMemoryRead;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
    barrier.image = swapchain->swapchainImages[imageIndex]->image;
    barrier.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                         vk::DependencyFlagBits::eByRegion, nullptr, nullptr, barrier);

    vk::BufferImageCopy region{};
    region.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    region.imageExtent = vk::Extent3D(width, height, 1);
    cmd->copyImageToBuffer(swapchain->swapchainImages[imageIndex]->image,
                           vk::ImageLayout::eTransferSrcOptimal,
                           stagingBuffer->buffer, 1, &region);
    cmd->end();

    auto copyFence = context->device->createFenceUnique(vk::FenceCreateInfo{});
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmd.get());
    context->queues[VulkanContext::Queue::COMPUTE].queue.submit(submitInfo, copyFence.get());

    inflightReadbacks.push_back(InflightReadback{
        .stagingBuffer = std::move(stagingBuffer),
        .fence = std::move(copyFence),
        .commandBuffer = std::move(cmd),
        .imageSize = imageSize,
        .width = width,
        .height = height,
        .outputPath = outputPath
    });
}

void Renderer::pollAsyncReadbacks() {
    for (size_t i = 0; i < inflightReadbacks.size();) {
        auto status = context->device->getFenceStatus(inflightReadbacks[i].fence.get());
        if (status == vk::Result::eNotReady) {
            ++i;
            continue;
        }
        if (status != vk::Result::eSuccess) {
            spdlog::warn("Async readback fence failed with status {}", vk::to_string(status));
            inflightReadbacks.erase(inflightReadbacks.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(writebackMutex);
            writebackQueue.push_back(WritebackTask{
                .stagingBuffer = std::move(inflightReadbacks[i].stagingBuffer),
                .imageSize = inflightReadbacks[i].imageSize,
                .width = inflightReadbacks[i].width,
                .height = inflightReadbacks[i].height,
                .outputPath = std::move(inflightReadbacks[i].outputPath)
            });
            pendingWritebackCount.fetch_add(1, std::memory_order_relaxed);
        }
        writebackCv.notify_one();
        inflightReadbacks.erase(inflightReadbacks.begin() + static_cast<std::ptrdiff_t>(i));
    }
}

void Renderer::startWritebackThread() {
    if (writebackThreadRunning.exchange(true)) {
        return;
    }
    writebackThread = std::thread(&Renderer::writebackThreadMain, this);
}

void Renderer::stopWritebackThread() {
    if (!writebackThreadRunning.exchange(false)) {
        if (writebackThread.joinable()) {
            writebackThread.join();
        }
        return;
    }
    writebackCv.notify_all();
    if (writebackThread.joinable()) {
        writebackThread.join();
    }
}

void Renderer::writebackThreadMain() {
    while (true) {
        WritebackTask task {};
        {
            std::unique_lock<std::mutex> lock(writebackMutex);
            writebackCv.wait(lock, [&] {
                return !writebackThreadRunning.load(std::memory_order_relaxed) || !writebackQueue.empty();
            });
            if (writebackQueue.empty()) {
                if (!writebackThreadRunning.load(std::memory_order_relaxed)) {
                    return;
                }
                continue;
            }
            task = std::move(writebackQueue.front());
            writebackQueue.pop_front();
        }

        const std::filesystem::path outputPath = task.outputPath;
        if (!outputPath.parent_path().empty()) {
            std::error_code ec;
            std::filesystem::create_directories(outputPath.parent_path(), ec);
            if (ec) {
                spdlog::warn("Failed to create output directory {}: {}", outputPath.parent_path().string(), ec.message());
            }
        }
        std::ofstream ofs(outputPath, std::ios::binary);
        if (ofs.is_open()) {
            uint8_t* mapped = static_cast<uint8_t*>(task.stagingBuffer->allocation_info.pMappedData);
            if (mapped) {
                ofs.write(reinterpret_cast<char*>(mapped), static_cast<std::streamsize>(task.imageSize));
            }
            ofs.close();
        } else {
            spdlog::warn("Failed to open output file {}", outputPath.string());
        }
        pendingWritebackCount.fetch_sub(1, std::memory_order_relaxed);
    }
}

void Renderer::recreateSwapchain() {
    auto oldExtent = swapchain->swapchainExtent;
    spdlog::debug("Recreating swapchain");
    swapchain->recreate();
    if (swapchain->swapchainExtent == oldExtent) {
        return;
    }

    auto [width, height] = swapchain->swapchainExtent;
    auto tileX = (width + 16 - 1) / 16;
    auto tileY = (height + 16 - 1) / 16;
    const uint32_t boundaryValues = tileX * tileY * 2;
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        frameTileBoundaryBuffers[i]->realloc(boundaryValues * sizeof(uint32_t));
        frameTileBoundariesHost[i].assign(boundaryValues, 0u);
    }
    createRenderPipeline();
}

void Renderer::initializeVulkan() {
    spdlog::debug("Initializing Vulkan");
    window = configuration.window;
    context = std::make_shared<VulkanContext>(window->getRequiredInstanceExtensions(), std::vector<std::string>{},
                                              configuration.enableVulkanValidationLayers);

    context->createInstance();
    auto surface = static_cast<vk::SurfaceKHR>(window->createSurface(context));
    context->selectPhysicalDevice(configuration.physicalDeviceId, surface);

    vk::PhysicalDeviceFeatures pdf{};
    vk::PhysicalDeviceVulkan11Features pdf11{};
    vk::PhysicalDeviceVulkan12Features pdf12{};
    pdf.shaderStorageImageWriteWithoutFormat = true;
    pdf.shaderInt64 = true;
    // pdf.robustBufferAccess = true;
    // pdf12.shaderFloat16 = true;]
#ifndef __APPLE__
    pdf12.shaderBufferInt64Atomics = true;
    pdf12.shaderSharedInt64Atomics = true;
#endif

    context->createLogicalDevice(pdf, pdf11, pdf12);
    context->createDescriptorPool(1);

    swapchain = std::make_shared<Swapchain>(context, window, configuration.immediateSwapchain);

    for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
        inflightFences.emplace_back(
            context->device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled)));
    }

    renderFinishedSemaphores.resize(FRAMES_IN_FLIGHT);
    for (int i = 0; i < FRAMES_IN_FLIGHT; i++) {
        renderFinishedSemaphores[i] = context->device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    }
}

void Renderer::loadSceneToGPU() {
    spdlog::debug("Loading scene to GPU");
    scene = std::make_shared<GSScene>(configuration.scene);
    scene->load(context);

    // reset descriptor pool
    context->device->resetDescriptorPool(context->descriptorPool.get());
}

void Renderer::initializeCpuPreprocessResources() {
    const uint32_t vertexCount = static_cast<uint32_t>(scene->getNumVertices());
    auto verticesRaw = scene->vertexBuffer->download();
    auto cov3dRaw = scene->cov3DBuffer->download();

    hostVertices.resize(vertexCount);
    std::memcpy(hostVertices.data(), verticesRaw.data(), vertexCount * sizeof(GSScene::Vertex));

    hostCov3D.resize(static_cast<size_t>(vertexCount) * 6u);
    std::memcpy(hostCov3D.data(), cov3dRaw.data(), hostCov3D.size() * sizeof(float));

    frameVertexAttributesHost.clear();
    frameVertexAttributesHost.resize(FRAMES_IN_FLIGHT);
    for (auto& attrs : frameVertexAttributesHost) {
        attrs.resize(vertexCount);
    }
    frameSortedVerticesHost.assign(FRAMES_IN_FLIGHT, {});
    frameTileBoundariesHost.assign(FRAMES_IN_FLIGHT, {});
    frameNumInstances.assign(FRAMES_IN_FLIGHT, 0u);

    frameVertexAttributeBuffers.resize(FRAMES_IN_FLIGHT);
    frameSortVBuffers.resize(FRAMES_IN_FLIGHT);
    frameTileBoundaryBuffers.resize(FRAMES_IN_FLIGHT);
    const uint32_t sortCapacity = std::max<uint64_t>(1, scene->getNumVertices() * sortBufferSizeMultiplier);
    const auto [width, height] = swapchain->swapchainExtent;
    const uint32_t tileX = (width + 16 - 1) / 16;
    const uint32_t tileY = (height + 16 - 1) / 16;
    const uint32_t boundaryValues = tileX * tileY * 2;
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        frameVertexAttributeBuffers[i] = Buffer::storage(context, vertexCount * sizeof(VertexAttributeBuffer), false,
                                                         0, "cpuFrameVertexAttributeBuffer");
        frameSortVBuffers[i] = Buffer::storage(context, sortCapacity * sizeof(uint32_t), false, 0, "cpuFrameSortVBuffer");
        frameTileBoundaryBuffers[i] = Buffer::storage(context, boundaryValues * sizeof(uint32_t), false, 0,
                                                      "cpuFrameTileBoundaryBuffer");
        frameTileBoundariesHost[i].assign(boundaryValues, 0u);
    }
}

void Renderer::createPreprocessPipeline() {
    spdlog::debug("Creating preprocess pipeline");
    const char* debugMaskStatsEnv = std::getenv("VKGS_DEBUG_MASK_STATS");
    debugMaskStats = debugMaskStatsEnv != nullptr && std::string(debugMaskStatsEnv) != "0";
    debugMaskStatsLogged = false;

    uniformBuffer = Buffer::uniform(context, sizeof(UniformBuffer));
    gaussianMaskBuffer = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t), false, 0, "gaussianMaskBuffer");
    // Keep intermediate compute attribute buffer in device-preferred memory.
    // On some mobile drivers, host-visible heaps can cause inconsistent SSBO behavior under heavy compute.
    vertexAttributeBuffer = Buffer::storage(context, scene->getNumVertices() * sizeof(VertexAttributeBuffer), false);
    tileOverlapBuffer = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t), false);
    // tileOverlapBuffer = std::make_shared<Buffer>(context, scene->getNumVertices() * sizeof(uint32_t), 
    //                               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
    //                               VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT, false);
    preprocessPipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "preprocess", SPV_PREPROCESS, SPV_PREPROCESS_len));
    inputSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    inputSet->bindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                        scene->vertexBuffer);
    inputSet->bindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                        scene->cov3DBuffer);
    inputSet->build();
    preprocessPipeline->addDescriptorSet(0, inputSet);

    auto uniformOutputSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    uniformOutputSet->bindBufferToDescriptorSet(0, vk::DescriptorType::eUniformBuffer,
                                                vk::ShaderStageFlagBits::eCompute,
                                                uniformBuffer);
    uniformOutputSet->bindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer,
                                                vk::ShaderStageFlagBits::eCompute,
                                                gaussianMaskBuffer);
    uniformOutputSet->bindBufferToDescriptorSet(2, vk::DescriptorType::eStorageBuffer,
                                                vk::ShaderStageFlagBits::eCompute,
                                                vertexAttributeBuffer);
    uniformOutputSet->bindBufferToDescriptorSet(3, vk::DescriptorType::eStorageBuffer,
                                                vk::ShaderStageFlagBits::eCompute,
                                                tileOverlapBuffer);
    uniformOutputSet->build();

    preprocessPipeline->addDescriptorSet(1, uniformOutputSet);
    preprocessPipeline->addPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t));
    preprocessPipeline->build();
}

Renderer::Renderer(VulkanSplatting::RendererConfiguration configuration) : configuration(std::move(configuration)) {
}

void Renderer::createGui() {
    if (!configuration.enableGui) {
        return;
    }

    spdlog::debug("Creating GUI");

    imguiManager = std::make_shared<ImguiManager>(context, swapchain, window);
    imguiManager->init();
    guiManager.init();
}

void Renderer::createPrefixSumPipeline() {
    spdlog::debug("Creating prefix sum pipeline");
    prefixSumPingBuffer = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t), false);
    prefixSumPongBuffer = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t), false);

    // 修改为 Host Visible 且 Mapped 内存，以便 CPU 直接回读
    // prefixSumPingBuffer = std::make_shared<Buffer>(context, scene->getNumVertices() * sizeof(uint32_t), 
    //                               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
    //                               VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT, false);
                                  
    // prefixSumPongBuffer = std::make_shared<Buffer>(context, scene->getNumVertices() * sizeof(uint32_t), 
    //                               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
    //                               VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT, false);
    
    totalSumBufferHost = Buffer::staging(context, sizeof(uint32_t));

    prefixSumPipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "prefix_sum", SPV_PREFIX_SUM, SPV_PREFIX_SUM_len));
    auto descriptorSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    descriptorSet->bindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             prefixSumPingBuffer);
    descriptorSet->bindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             prefixSumPongBuffer);
    descriptorSet->build();

    prefixSumPipeline->addDescriptorSet(0, descriptorSet);
    prefixSumPipeline->addPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t));
    prefixSumPipeline->build();
}

void Renderer::createRadixSortPipeline() {
    spdlog::debug("Creating radix sort pipeline");
    
    // 修改点：将 uint64_t 拆分为两个 uint32_t 缓冲区进行排序
    // 树莓派不支持 uint64_t，所以我们将 key 拆分为 [low, high] 存储
    // sortKBufferEvenLow = std::make_shared<Buffer>(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier, 
    //                               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
    //                               VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT, false);
    // sortKBufferEvenHigh = std::make_shared<Buffer>(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier, 
    //                               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
    //                               VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT, false);
    // sortKBufferOddLow = std::make_shared<Buffer>(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier, 
    //                               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
    //                               VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT, false);
    // sortKBufferOddHigh = std::make_shared<Buffer>(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier, 
    //                               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
    //                               VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT, false);
    // sortVBufferEven = std::make_shared<Buffer>(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier, 
    //                               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
    //                               VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT, false);
    // sortVBufferOdd = std::make_shared<Buffer>(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier, 
    //                               vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
    //                               VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT, false);
    sortKBufferEvenLow = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier,
                                      false, 0, "sortKBufferEvenLow");
    sortKBufferEvenHigh = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier,
                                      false, 0, "sortKBufferEvenHigh");
    sortKBufferOddLow = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier,
                                     false, 0, "sortKBufferOddLow");
    sortKBufferOddHigh = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier,
                                     false, 0, "sortKBufferOddHigh");
    sortVBufferEven = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier,
                                      false, 0, "sortVBufferEven");
    sortVBufferOdd = Buffer::storage(context, scene->getNumVertices() * sizeof(uint32_t) * sortBufferSizeMultiplier,
                                     false, 0, "sortVBufferOdd");

    const char* radixBitsEnv = std::getenv("VKGS_RADIX_BITS");
    uint32_t requestedRadixBits = 8;
    if (radixBitsEnv != nullptr) {
        requestedRadixBits = static_cast<uint32_t>(std::atoi(radixBitsEnv));
    }
    if (requestedRadixBits != 4 && requestedRadixBits != 8) {
        spdlog::warn("Unsupported VKGS_RADIX_BITS={}, fallback to 8", requestedRadixBits);
        requestedRadixBits = 8;
    }
    radixBits = requestedRadixBits;
    radixBins = 1u << radixBits;
    radixPasses = (64u + radixBits - 1u) / radixBits;
    spdlog::info("Radix sort config: bits={}, bins={}, passes={}", radixBits, radixBins, radixPasses);

    uint32_t globalInvocationSize = scene->getNumVertices() * sortBufferSizeMultiplier / numRadixSortBlocksPerWorkgroup;
    uint32_t remainder = scene->getNumVertices() * sortBufferSizeMultiplier % numRadixSortBlocksPerWorkgroup;
    globalInvocationSize += remainder > 0 ? 1 : 0;

    auto numWorkgroups = (globalInvocationSize + 256 - 1) / 256;

    sortHistBuffer = Buffer::storage(context, numWorkgroups * radixBins * sizeof(uint32_t), false);
    sortOffsetBuffer = Buffer::storage(context, numWorkgroups * radixBins * sizeof(uint32_t), false);

    const bool use4BitRadix = radixBits == 4;

    sortHistPipeline = std::make_shared<ComputePipeline>(
        context, use4BitRadix
                     ? std::make_shared<Shader>(context, "hist4", SPV_HIST4, SPV_HIST4_len)
                     : std::make_shared<Shader>(context, "hist", SPV_HIST, SPV_HIST_len));
    spdlog::debug("Created hist pipeline");

    sortScanPipeline = std::make_shared<ComputePipeline>(
        context, use4BitRadix
                     ? std::make_shared<Shader>(context, "scan4", SPV_SCAN4, SPV_SCAN4_len)
                     : std::make_shared<Shader>(context, "scan", SPV_SCAN, SPV_SCAN_len));
    spdlog::debug("Created scan pipeline");

    sortScatterPipeline = std::make_shared<ComputePipeline>(
        context, use4BitRadix
                     ? std::make_shared<Shader>(context, "scatter4", SPV_SCATTER4, SPV_SCATTER4_len)
                     : std::make_shared<Shader>(context, "scatter", SPV_SCATTER, SPV_SCATTER_len));
    spdlog::debug("Created scatter pipeline");

    auto descriptorSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    descriptorSet->bindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferEvenLow);
    descriptorSet->bindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferOddLow);
    descriptorSet->bindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortHistBuffer);
    // 绑定高位 Key 缓冲区，匹配 hist.comp 中的 binding 5
    descriptorSet->bindBufferToDescriptorSet(5, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferEvenHigh);
    descriptorSet->bindBufferToDescriptorSet(5, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferOddHigh);
    // 移除 binding 6 的绑定，因为 hist.comp 不需要输出高位
    descriptorSet->build();
    sortHistPipeline->addDescriptorSet(0, descriptorSet);
    sortHistPipeline->addPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(RadixSortPushConstants));
    spdlog::debug("Building hist pipeline");
    sortHistPipeline->build();
    spdlog::debug("Built hist pipeline");

    descriptorSet = std::make_shared<DescriptorSet>(context, 1);
    descriptorSet->bindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortHistBuffer);
    descriptorSet->bindBufferToDescriptorSet(4, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortOffsetBuffer);
    descriptorSet->build();
    sortScanPipeline->addDescriptorSet(0, descriptorSet);
    sortScanPipeline->addPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(RadixSortPushConstants));
    spdlog::debug("Building scan pipeline");
    sortScanPipeline->build();
    spdlog::debug("Built scan pipeline");

    descriptorSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    descriptorSet->bindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferEvenLow);
    descriptorSet->bindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferOddLow);
    descriptorSet->bindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferOddLow);
    descriptorSet->bindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferEvenLow);
    descriptorSet->bindBufferToDescriptorSet(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortVBufferEven);
    descriptorSet->bindBufferToDescriptorSet(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortVBufferOdd);
    descriptorSet->bindBufferToDescriptorSet(3, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortVBufferOdd);
    descriptorSet->bindBufferToDescriptorSet(3, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortVBufferEven);
    descriptorSet->bindBufferToDescriptorSet(4, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortOffsetBuffer);
    descriptorSet->bindBufferToDescriptorSet(5, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferEvenHigh);
    descriptorSet->bindBufferToDescriptorSet(5, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferOddHigh);
    descriptorSet->bindBufferToDescriptorSet(6, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferOddHigh);
    descriptorSet->bindBufferToDescriptorSet(6, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferEvenHigh);
    descriptorSet->build();
    sortScatterPipeline->addDescriptorSet(0, descriptorSet);
    sortScatterPipeline->addPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(RadixSortPushConstants));
    spdlog::debug("Building scatter pipeline");
    sortScatterPipeline->build();
    spdlog::debug("Built scatter pipeline");
}

void Renderer::createPreprocessSortPipeline() {
    spdlog::debug("Creating preprocess sort pipeline");
    auto shader = std::make_shared<Shader>(context, "preprocess_sort", SPV_PREPROCESS_SORT, SPV_PREPROCESS_SORT_len);
    spdlog::debug("Shader modules for preprocess_sort created");
    preprocessSortPipeline = std::make_shared<ComputePipeline>(context, shader);
    spdlog::debug("ComputePipeline object for preprocess_sort created");
    
    auto descriptorSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    descriptorSet->bindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             vertexAttributeBuffer);
    descriptorSet->bindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             prefixSumPingBuffer);
    descriptorSet->bindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             prefixSumPongBuffer);
    // 修改：使用 Low/High 缓冲区替换旧的 Even 缓冲区
    descriptorSet->bindBufferToDescriptorSet(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferEvenLow);
    descriptorSet->bindBufferToDescriptorSet(4, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferEvenHigh);
    descriptorSet->bindBufferToDescriptorSet(3, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortVBufferEven);
    // descriptorSet->bindBufferToDescriptorSet(3, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
    //                                          sortVBufferOdd);
    descriptorSet->build();

    preprocessSortPipeline->addDescriptorSet(0, descriptorSet);
    preprocessSortPipeline->addPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t) * 2);
    preprocessSortPipeline->build();
    spdlog::debug("Built preprocess sort pipeline");
}

void Renderer::createTileBoundaryPipeline() {
    spdlog::debug("Creating tile boundary pipeline");
    auto [width, height] = swapchain->swapchainExtent;
    auto tileX = (width + 16 - 1) / 16;
    auto tileY = (height + 16 - 1) / 16;
    tileBoundaryBuffer = Buffer::storage(context, tileX * tileY * sizeof(uint32_t) * 2, false);
    tileBoundaryPipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "tile_boundary", SPV_TILE_BOUNDARY, SPV_TILE_BOUNDARY_len));
    auto descriptorSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    // 修改：使用缓冲区，Tile Boundary 只需要读取排序后的 Key 中的 Tile ID 部分
    descriptorSet->bindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             sortKBufferEvenHigh);
    descriptorSet->bindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             tileBoundaryBuffer);
    descriptorSet->build();

    tileBoundaryPipeline->addDescriptorSet(0, descriptorSet);
    tileBoundaryPipeline->addPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t) * 2);
    tileBoundaryPipeline->build();
    spdlog::debug("Built tile boundary pipeline");
}

void Renderer::createClearTileBoundaryPipeline() {
    clearTileBoundaryPipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "clear_boundaries", SPV_CLEAR_BOUNDARIES, SPV_CLEAR_BOUNDARIES_len));

    auto descriptorSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    descriptorSet->bindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                             tileBoundaryBuffer);
    descriptorSet->build();

    clearTileBoundaryPipeline->addDescriptorSet(0, descriptorSet);
    clearTileBoundaryPipeline->addPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t));
    clearTileBoundaryPipeline->build();
}

void Renderer::createRenderPipeline() {
    spdlog::debug("Creating render pipeline");
    renderPipeline = std::make_shared<ComputePipeline>(
        context, std::make_shared<Shader>(context, "render", SPV_RENDER, SPV_RENDER_len));
    auto inputSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        inputSet->bindBufferToDescriptorSet(0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                            frameVertexAttributeBuffers[i]);
        inputSet->bindBufferToDescriptorSet(1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                            frameTileBoundaryBuffers[i]);
        inputSet->bindBufferToDescriptorSet(2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute,
                                            frameSortVBuffers[i]);
    }
    inputSet->build();

    // Keep output descriptor set frame-indexable as well; Pipeline::bind uses the same currentFrame
    // for every set, so set=1 must also have FRAMES_IN_FLIGHT entries to avoid frame>0 OOB access.
    auto outputSet = std::make_shared<DescriptorSet>(context, FRAMES_IN_FLIGHT);
    for (uint32_t f = 0; f < FRAMES_IN_FLIGHT; ++f) {
        for (auto& image: swapchain->swapchainImages) {
            outputSet->bindImageToDescriptorSet(0, vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eCompute,
                                                image);
        }
    }
    outputSet->build();
    renderPipeline->addDescriptorSet(0, inputSet);
    renderPipeline->addDescriptorSet(1, outputSet);
    renderPipeline->addPushConstant(vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t) * 2);
    renderPipeline->build();
    spdlog::debug("Built render pipeline");
}

void Renderer::draw() {
    const bool trace = envEnabled("VKGS_TRACE_FRAME");
    const uint64_t frameId = frameSequenceId;
    const uint32_t slot = frameSlot;
    if (trace) {
        spdlog::info("[TRACE] frame={} slot={} draw begin", frameId, slot);
    }

    if (trace) {
        spdlog::info("[TRACE] frame={} waitForFences begin", frameId);
    }
    auto ret = context->device->waitForFences(inflightFences[slot].get(), VK_TRUE, UINT64_MAX);
    if (ret != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to wait for fence");
    }
    flushCompletedTimingForSlot(slot, true);
    flushCompletedReadbackForSlot(slot);
    context->device->resetFences(inflightFences[slot].get());
    if (trace) {
        spdlog::info("[TRACE] frame={} waitForFences end", frameId);
    }

    if (trace) {
        spdlog::info("[TRACE] frame={} acquireNextImage begin", frameId);
    }
    auto res = context->device->acquireNextImageKHR(swapchain->swapchain.get(), UINT64_MAX,
                                                    swapchain->imageAvailableSemaphores[slot].get(),
                                                    nullptr, &currentImageIndex);
    if (res == vk::Result::eErrorOutOfDateKHR) {
        recreateSwapchain();
        return;
    } else if (res != vk::Result::eSuccess && res != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("Failed to acquire swapchain image");
    }
    if (trace) {
        spdlog::info("[TRACE] frame={} acquireNextImage end image={}", frameId, currentImageIndex);
    }
    auto uniformData = buildUniformData();
    updateUniforms(uniformData, slot);
    if (trace) {
        spdlog::info("[TRACE] frame={} cpuPreprocess begin", frameId);
    }
    runCpuPreprocess(slot, uniformData);
    if (trace) {
        spdlog::info("[TRACE] frame={} cpuPreprocess end", frameId);
    }

    if (!recordRenderCommandBuffer(slot, slot)) {
        throw std::runtime_error("Failed to record render command buffer");
    }
    vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eComputeShader;
    auto submitInfo = vk::SubmitInfo{}.setWaitSemaphores(swapchain->imageAvailableSemaphores[slot].get())
            .setCommandBuffers(renderCommandBuffers[slot].get())
            .setSignalSemaphores(renderFinishedSemaphores[slot].get())
            .setWaitDstStageMask(waitStage);
    if (trace) {
        spdlog::info("[TRACE] frame={} queue submit begin", frameId);
    }
    context->queues[VulkanContext::Queue::COMPUTE].queue.submit(submitInfo, inflightFences[slot].get());
    framePendingTiming[slot].cpu = cpuStageTiming;
    framePendingTiming[slot].valid = true;
    if (trace) {
        spdlog::info("[TRACE] frame={} queue submit end", frameId);
    }

    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphores[slot].get();
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain->swapchain.get();
    presentInfo.pImageIndices = &currentImageIndex;

    try {
        if (trace) {
            spdlog::info("[TRACE] frame={} present begin", frameId);
        }
        ret = context->queues[VulkanContext::Queue::PRESENT].queue.presentKHR(presentInfo);
        if (trace) {
            spdlog::info("[TRACE] frame={} present end", frameId);
        }
    } catch (vk::OutOfDateKHRError& e) {
        recreateSwapchain();
        return;
    }

    if (ret == vk::Result::eErrorOutOfDateKHR || ret == vk::Result::eSuboptimalKHR) {
        recreateSwapchain();
    } else if (ret != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to present swapchain image");
    }
    if (!activeOutputRawPath.empty()) {
        framePendingReadback[slot].imageIndex = currentImageIndex;
        framePendingReadback[slot].outputPath = activeOutputRawPath;
        framePendingReadback[slot].valid = true;
    }

    frameSlot = (frameSlot + 1) % FRAMES_IN_FLIGHT;
    frameSequenceId++;
    if (trace) {
        spdlog::info("[TRACE] frame={} draw end -> next_slot={}", frameId, frameSlot);
    }
    
}

// void Renderer::run() {
//     while (running) {
//         if (!window->tick()) {
//             break;
//         }

//         draw();

//         auto now = std::chrono::high_resolution_clock::now();
//         auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFpsTime).count();
//         if (diff > 1000) {
//             spdlog::debug("FPS: {}", fpsCounter);
//             fpsCounter = 0;
//             lastFpsTime = now;
//         } else {
//             fpsCounter++;
//         }

//         retrieveTimestamps();
//     }

//     context->device->waitIdle();
// }


void Renderer::run() {
    using Clock = std::chrono::high_resolution_clock;
    const bool saveEveryFrame = envEnabled("VKGS_SAVE_EVERY_FRAME");
    const bool hasJobs = !configuration.renderJobs.empty();
    const uint64_t totalJobs = hasJobs ? static_cast<uint64_t>(configuration.renderJobs.size()) : 1u;
    uint64_t frameCounter = 0;
    const auto runStart = Clock::now();
    spdlog::info("Run loop start: jobs={}", totalJobs);

    for (uint64_t jobIdx = 0; jobIdx < totalJobs && running; ++jobIdx) {
        if (!window->tick()) {
            spdlog::info("Run loop break: window requested close");
            break;
        }

        if (hasJobs) {
            const auto& job = configuration.renderJobs[jobIdx];
            activeCameraOverride = job.camera;
            activeOutputRawPath = job.outputRawPath;
            if (job.camera.width != swapchain->swapchainExtent.width ||
                job.camera.height != swapchain->swapchainExtent.height) {
                spdlog::warn(
                    "Job {} camera size {}x{} differs from swapchain {}x{}, rendering with swapchain size",
                    jobIdx, job.camera.width, job.camera.height,
                    swapchain->swapchainExtent.width, swapchain->swapchainExtent.height);
            }
            spdlog::info("Rendering job {}/{} -> {}", jobIdx + 1, totalJobs, activeOutputRawPath);
        } else {
            activeCameraOverride = configuration.cameraOverride;
            activeOutputRawPath = saveEveryFrame ? addFrameSuffixToPath(configuration.outputRawPath, frameCounter)
                                                 : std::string{};
        }

        const auto frameStart = Clock::now();
        draw();
        pollAsyncReadbacks();
        const auto frameEnd = Clock::now();
        const double submitMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();
        frameE2EMsSamples.push_back(submitMs);
        spdlog::info("[LOOP][submit_ms] job={}/{}, output={}, latency={:.3f}",
                     jobIdx + 1, totalJobs, activeOutputRawPath.empty() ? "(disabled)" : activeOutputRawPath, submitMs);
        ++frameCounter;
    }

    // Flush pending timing snapshots from last submitted frames.
    for (uint32_t slot = 0; slot < FRAMES_IN_FLIGHT; ++slot) {
        if (slot < inflightFences.size()) {
            auto fenceRes = context->device->waitForFences(inflightFences[slot].get(), VK_TRUE, UINT64_MAX);
            if (fenceRes != vk::Result::eSuccess) {
                throw std::runtime_error("Failed to wait for submitted frame fence during shutdown flush");
            }
        }
        flushCompletedTimingForSlot(slot, true);
        flushCompletedReadbackForSlot(slot);
    }
    context->device->waitIdle();
    pollAsyncReadbacks();
    while (!inflightReadbacks.empty() || pendingWritebackCount.load(std::memory_order_relaxed) > 0) {
        pollAsyncReadbacks();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    spdlog::info("Run loop end: rendered_frames={}, timing_samples={}", frameCounter, timingSampleCount);
    if (timingSampleCount > 0) {
        const double cpuAvg = timingCpuTotalMsAccum / static_cast<double>(timingSampleCount);
        const double renderAvg = timingRenderMsAccum / static_cast<double>(timingSampleCount);
        const double serialEst = cpuAvg + renderAvg;
        const double pipelineEst = std::max(cpuAvg, renderAvg);
        const double overlapGain = serialEst > 0.0 ? (serialEst - pipelineEst) / serialEst * 100.0 : 0.0;
        spdlog::info(
            "[PIPELINE][avg_ms] samples={}, cpu_pre={:.3f}, render={:.3f}, serial_est={:.3f}, pipeline_est={:.3f}, overlap_gain={:.2f}%",
            timingSampleCount, cpuAvg, renderAvg, serialEst, pipelineEst, overlapGain);
    }
    if (!frameE2EMsSamples.empty()) {
        auto samples = frameE2EMsSamples;
        std::sort(samples.begin(), samples.end());
        const size_t n = samples.size();
        const auto pct = [&](double p) -> double {
            if (n == 1) return samples[0];
            const double idx = (p / 100.0) * static_cast<double>(n - 1);
            const size_t lo = static_cast<size_t>(std::floor(idx));
            const size_t hi = static_cast<size_t>(std::ceil(idx));
            const double t = idx - static_cast<double>(lo);
            return samples[lo] * (1.0 - t) + samples[hi] * t;
        };
        double sum = 0.0;
        for (double v : samples) sum += v;
        const double avg = sum / static_cast<double>(n);
        const double minv = samples.front();
        const double maxv = samples.back();
        const double p50 = pct(50.0);
        const double p95 = pct(95.0);
        const double fps = avg > 0.0 ? 1000.0 / avg : 0.0;
        spdlog::info(
            "[E2E][ms] frames={}, avg={:.3f}, min={:.3f}, p50={:.3f}, p95={:.3f}, max={:.3f}, fps={:.3f}",
            n, avg, minv, p50, p95, maxv, fps);
    }
    const auto runEnd = Clock::now();
    const double totalMs = std::chrono::duration<double, std::milli>(runEnd - runStart).count();
    spdlog::info("[E2E][total_ms] jobs={}, total={:.3f}, avg_per_job={:.3f}",
                 frameCounter, totalMs, frameCounter > 0 ? totalMs / static_cast<double>(frameCounter) : 0.0);
}

void Renderer::saveFrameRawByImageIndex(uint32_t imageIndex, const std::string& outputPathValue) {
    auto [width, height] = swapchain->swapchainExtent;
    spdlog::info("Capturing frame: {}x{}", width, height);
    
    VkDeviceSize imageSize = width * height * 4; // RGBA8
    auto stagingBuffer = std::make_shared<Buffer>(context, imageSize, 
                              vk::BufferUsageFlagBits::eTransferDst,
                              VMA_MEMORY_USAGE_CPU_ONLY, VMA_ALLOCATION_CREATE_MAPPED_BIT, false);

    vk::CommandBufferAllocateInfo allocInfo{commandPool.get(), vk::CommandBufferLevel::ePrimary, 1};
    auto copyCmdBuffers = context->device->allocateCommandBuffersUnique(allocInfo);
    auto cmd = copyCmdBuffers[0].get();

    cmd.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = vk::ImageLayout::ePresentSrcKHR;
    barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eMemoryRead;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
    barrier.image = swapchain->swapchainImages[imageIndex]->image;
    barrier.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                        vk::DependencyFlagBits::eByRegion, nullptr, nullptr, barrier);

    vk::BufferImageCopy region{};
    region.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    region.imageExtent = vk::Extent3D(width, height, 1);
    cmd.copyImageToBuffer(swapchain->swapchainImages[imageIndex]->image,
                         vk::ImageLayout::eTransferSrcOptimal, 
                         stagingBuffer->buffer, 1, &region);

    cmd.end();

    auto copyFence = context->device->createFenceUnique(vk::FenceCreateInfo{});
    vk::SubmitInfo submitInfo{};
    submitInfo.setCommandBuffers(cmd);
    context->queues[VulkanContext::Queue::COMPUTE].queue.submit(submitInfo, copyFence.get());
    auto copyRes = context->device->waitForFences(copyFence.get(), VK_TRUE, UINT64_MAX);
    if (copyRes != vk::Result::eSuccess) {
        throw std::runtime_error("Failed to wait for frame copy fence");
    }

    const std::filesystem::path outputPath = outputPathValue;
    if (!outputPath.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(outputPath.parent_path(), ec);
        if (ec) {
            spdlog::warn("Failed to create output directory {}: {}", outputPath.parent_path().string(), ec.message());
        }
    }

    std::ofstream ofs(outputPath, std::ios::binary);
    if (ofs.is_open()) {
        uint8_t* mapped = static_cast<uint8_t*>(stagingBuffer->allocation_info.pMappedData);
        if (mapped) {
            ofs.write(reinterpret_cast<char*>(mapped), imageSize);
            
            uint64_t totalR = 0, totalG = 0, totalB = 0;
            int nonZeroPixels = 0;
            for (uint32_t i = 0; i < width * height; ++i) {
                uint8_t r = mapped[i * 4 + 0];
                uint8_t g = mapped[i * 4 + 1];
                uint8_t b = mapped[i * 4 + 2];
                if (r > 0 || g > 0 || b > 0) {
                    nonZeroPixels++;
                    totalR += r; totalG += g; totalB += b;
                }
            }
            
            if (nonZeroPixels > 0) {
                spdlog::info("[DEBUG] Render successful! Non-zero pixels: {}/{}", nonZeroPixels, width * height);
                spdlog::info("[DEBUG] Average color of non-zero pixels: R={}, G={}, B={}", 
                             totalR / nonZeroPixels, totalG / nonZeroPixels, totalB / nonZeroPixels);
            } else {
                spdlog::warn("[DEBUG] WARNING: Render output is FULL BLACK (all pixels are 0,0,0)!");
            }
        }
        ofs.close();
        spdlog::info("Rendered image saved to {} ({}x{})", outputPath.string(), width, height);
    }
}

void Renderer::saveCurrentFrameRaw(const std::string& outputPathValue) {
    saveFrameRawByImageIndex(currentImageIndex, outputPathValue);
}

void Renderer::stop() {
    // wait till device is idle
    running = false;

    context->device->waitIdle();
    pollAsyncReadbacks();
    while (!inflightReadbacks.empty() || pendingWritebackCount.load(std::memory_order_relaxed) > 0) {
        pollAsyncReadbacks();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    stopWritebackThread();
}

void Renderer::createCommandPool() {
    spdlog::debug("Creating command pool");
    vk::CommandPoolCreateInfo poolInfo = {};
    poolInfo.queueFamilyIndex = context->queues[VulkanContext::Queue::COMPUTE].queueFamily;
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

    commandPool = context->device->createCommandPoolUnique(poolInfo, nullptr);
}

void Renderer::recordPreprocessCommandBuffer() {
    spdlog::debug("Recording preprocess command buffer");
    if (!preprocessCommandBuffer) {
        vk::CommandBufferAllocateInfo allocateInfo = {commandPool.get(), vk::CommandBufferLevel::ePrimary, 1};
        auto buffers = context->device->allocateCommandBuffersUnique(allocateInfo);
        preprocessCommandBuffer = std::move(buffers[0]);
    }
    preprocessCommandBuffer->reset();

    auto numGroups = (scene->getNumVertices() + 255) / 256;

    preprocessCommandBuffer->begin(vk::CommandBufferBeginInfo{});

    preprocessCommandBuffer->resetQueryPool(context->queryPool.get(), 0, 20);

    preprocessPipeline->bind(preprocessCommandBuffer, 0, 0);
    const uint32_t preprocessUseOriginIntersection = 0u;
    preprocessCommandBuffer->pushConstants(preprocessPipeline->pipelineLayout.get(),
                                           vk::ShaderStageFlagBits::eCompute, 0,
                                           sizeof(uint32_t), &preprocessUseOriginIntersection);
    preprocessCommandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, context->queryPool.get(),
                                            queryManager->registerQuery("preprocess_start"));
    preprocessCommandBuffer->dispatch(numGroups, 1, 1);

    tileOverlapBuffer->computeWriteReadBarrier(preprocessCommandBuffer.get());
    
    // 1. 强制执行屏障：确保 Preprocess Shader 所有的写操作都彻底落盘，并对 Transfer (CopyBuffer) 可见
    Utils::BarrierBuilder().addBufferBarrier(tileOverlapBuffer, 
                                            vk::AccessFlagBits::eShaderWrite, 
                                            vk::AccessFlagBits::eTransferRead)
                          .build(preprocessCommandBuffer.get(), 
                                 vk::PipelineStageFlagBits::eComputeShader, 
                                 vk::PipelineStageFlagBits::eTransfer);

    vk::BufferCopy copyRegion = {0, 0, tileOverlapBuffer->size};
    preprocessCommandBuffer->copyBuffer(tileOverlapBuffer->buffer, prefixSumPingBuffer->buffer, 1, &copyRegion);

    prefixSumPingBuffer->computeWriteReadBarrier(preprocessCommandBuffer.get());

    // 2. 强制执行屏障：确保 CopyBuffer 完成，数据对后续 Prefix Sum Shader 可见
    Utils::BarrierBuilder().addBufferBarrier(prefixSumPingBuffer, 
                                            vk::AccessFlagBits::eTransferWrite, 
                                            vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite)
                          .build(preprocessCommandBuffer.get(), 
                                 vk::PipelineStageFlagBits::eTransfer, 
                                 vk::PipelineStageFlagBits::eComputeShader);

    preprocessCommandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, context->queryPool.get(),
                                            queryManager->registerQuery("preprocess_end"));

    prefixSumPipeline->bind(preprocessCommandBuffer, 0, 0);
    preprocessCommandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, context->queryPool.get(),
                                            queryManager->registerQuery("prefix_sum_start"));
    const auto iters = static_cast<uint32_t>(std::ceil(std::log2(static_cast<float>(scene->getNumVertices()))));
    for (uint32_t timestep = 0; timestep <= iters; timestep++) {
        preprocessCommandBuffer->pushConstants(prefixSumPipeline->pipelineLayout.get(),
                                               vk::ShaderStageFlagBits::eCompute, 0,
                                               sizeof(uint32_t), &timestep);
        preprocessCommandBuffer->dispatch(numGroups, 1, 1);

        if (timestep % 2 == 0) {
            prefixSumPongBuffer->computeWriteReadBarrier(preprocessCommandBuffer.get());
            prefixSumPingBuffer->computeReadWriteBarrier(preprocessCommandBuffer.get());
        } else {
            prefixSumPingBuffer->computeWriteReadBarrier(preprocessCommandBuffer.get());
            prefixSumPongBuffer->computeReadWriteBarrier(preprocessCommandBuffer.get());
        }
    }

    // Ensure Prefix Sum is finished before copying to host
    Utils::BarrierBuilder().addBufferBarrier(iters % 2 == 0 ? prefixSumPingBuffer : prefixSumPongBuffer, 
                                            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead)
                          .build(preprocessCommandBuffer.get(), vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer);

    auto totalSumRegion = vk::BufferCopy{(scene->getNumVertices() - 1) * sizeof(uint32_t), 0, sizeof(uint32_t)};
    if (iters % 2 == 0) {
        preprocessCommandBuffer->copyBuffer(prefixSumPingBuffer->buffer, totalSumBufferHost->buffer, 1,
                                            &totalSumRegion);
    } else {
        preprocessCommandBuffer->copyBuffer(prefixSumPongBuffer->buffer, totalSumBufferHost->buffer, 1,
                                            &totalSumRegion);
    }

    preprocessCommandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, context->queryPool.get(),
                                        queryManager->registerQuery("prefix_sum_end"));

    preprocessCommandBuffer->end();
}

void Renderer::runCpuPreprocess(uint32_t frameResourceSlot, const UniformBuffer& data) {
    using Clock = std::chrono::high_resolution_clock;
    const uint32_t vertexCount = static_cast<uint32_t>(hostVertices.size());
    auto& attrs = frameVertexAttributesHost[frameResourceSlot];
    if (attrs.size() != vertexCount) {
        attrs.resize(vertexCount);
    }
    std::vector<uint32_t> overlaps(vertexCount, 0u);
    std::vector<uint32_t> prefix(vertexCount, 0u);
    std::vector<uint32_t> keysLow;
    std::vector<uint32_t> keysHigh;
    std::vector<uint32_t> payloads;
    keysLow.reserve(vertexCount);
    keysHigh.reserve(vertexCount);
    payloads.reserve(vertexCount);

    const auto [width, height] = swapchain->swapchainExtent;
    const uint32_t tileX = (width + 16 - 1) / 16;
    const uint32_t tileY = (height + 16 - 1) / 16;
    const uint32_t numTiles = tileX * tileY;
    const auto tPreStart = Clock::now();

    constexpr float SH_C0 = 0.28209479177387814f;
    constexpr float SH_C1 = 0.4886025119029199f;
    constexpr float SH_C2[5] = {
        1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f
    };
    constexpr float SH_C3[7] = {
        -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
        -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f
    };
    constexpr uint32_t kMagic = 0x4d415449u;

    auto floatBits = [](float v) -> uint32_t {
        uint32_t u = 0;
        std::memcpy(&u, &v, sizeof(uint32_t));
        return u;
    };
    auto ndc2Pix = [](float v, int s) -> float {
        return ((v + 1.0f) * static_cast<float>(s) - 1.0f) * 0.5f;
    };
    auto computeEllipseIntersection = [](const glm::vec4& con_o, float disc, float t, const glm::vec2& p, bool isY,
                                         float coord) -> glm::vec2 {
        const float p_u = isY ? p.y : p.x;
        const float p_v = isY ? p.x : p.y;
        const float coeff = isY ? con_o.x : con_o.z;
        const float h = coord - p_u;
        const float sqrtTerm = std::sqrt(std::max(0.0f, disc * h * h + t * coeff));
        return glm::vec2(
            (-con_o.y * h - sqrtTerm) / coeff + p_v,
            (-con_o.y * h + sqrtTerm) / coeff + p_v
        );
    };

    auto processTilesCount = [&](const glm::vec4& con_o, float disc, float t, const glm::vec2& p,
                                 glm::vec2 bboxMin, glm::vec2 bboxMax, glm::vec2 bboxArgMin, glm::vec2 bboxArgMax,
                                 glm::ivec2 rectMin, glm::ivec2 rectMax, bool isY) -> uint32_t {
        const float blockU = isY ? 16.0f : 16.0f;
        const float blockV = isY ? 16.0f : 16.0f;
        if (isY) {
            rectMin = glm::ivec2(rectMin.y, rectMin.x);
            rectMax = glm::ivec2(rectMax.y, rectMax.x);
            bboxMin = glm::vec2(bboxMin.y, bboxMin.x);
            bboxMax = glm::vec2(bboxMax.y, bboxMax.x);
            bboxArgMin = glm::vec2(bboxArgMin.y, bboxArgMin.x);
            bboxArgMax = glm::vec2(bboxArgMax.y, bboxArgMax.x);
        }
        uint32_t count = 0;
        glm::vec2 intersectMaxLine(bboxMax.y, bboxMin.y);
        glm::vec2 intersectMinLine = (bboxMin.x <= static_cast<float>(rectMin.x) * blockU)
                                         ? computeEllipseIntersection(con_o, disc, t, p, isY, static_cast<float>(rectMin.x) * blockU)
                                         : intersectMaxLine;
        float minLine = static_cast<float>(rectMin.x) * blockU;
        for (int u = rectMin.x; u < rectMax.x; ++u) {
            const float maxLine = minLine + blockU;
            if (maxLine <= bboxMax.x) {
                intersectMaxLine = computeEllipseIntersection(con_o, disc, t, p, isY, maxLine);
            }
            const float ellipseMin = (minLine <= bboxArgMin.y && bboxArgMin.y < maxLine)
                                         ? bboxMin.y
                                         : std::min(intersectMinLine.x, intersectMaxLine.x);
            const float ellipseMax = (minLine <= bboxArgMax.y && bboxArgMax.y < maxLine)
                                         ? bboxMax.y
                                         : std::max(intersectMinLine.y, intersectMaxLine.y);
            const int minTileV = std::max(rectMin.y, std::min(rectMax.y, static_cast<int>(ellipseMin / blockV)));
            const int maxTileV = std::min(rectMax.y, std::max(rectMin.y, static_cast<int>(ellipseMax / blockV + 1.0f)));
            count += static_cast<uint32_t>(std::max(0, maxTileV - minTileV));
            intersectMinLine = intersectMaxLine;
            minLine = maxLine;
        }
        return count;
    };

    auto duplicateToTilesTouched = [&](const glm::vec2& p, const glm::vec4& con_o, glm::uvec4& rect) -> uint32_t {
        rect = glm::uvec4(0u);
        const float disc = con_o.y * con_o.y - con_o.x * con_o.z;
        if (con_o.x <= 0.0f || con_o.z <= 0.0f || disc >= 0.0f || con_o.w <= (1.0f / 255.0f)) {
            return 0;
        }
        const float t = 2.0f * std::log(std::max(con_o.w * 255.0f, 1e-6f));
        float xTerm = std::sqrt(std::max(0.0f, -(con_o.y * con_o.y * t) / (disc * con_o.x)));
        xTerm = (con_o.y < 0.0f) ? xTerm : -xTerm;
        float yTerm = std::sqrt(std::max(0.0f, -(con_o.y * con_o.y * t) / (disc * con_o.z)));
        yTerm = (con_o.y < 0.0f) ? yTerm : -yTerm;
        const glm::vec2 bboxArgMin(p.y - yTerm, p.x - xTerm);
        const glm::vec2 bboxArgMax(p.y + yTerm, p.x + xTerm);
        const glm::vec2 bboxMin(
            computeEllipseIntersection(con_o, disc, t, p, true, bboxArgMin.x).x,
            computeEllipseIntersection(con_o, disc, t, p, false, bboxArgMin.y).x
        );
        const glm::vec2 bboxMax(
            computeEllipseIntersection(con_o, disc, t, p, true, bboxArgMax.x).y,
            computeEllipseIntersection(con_o, disc, t, p, false, bboxArgMax.y).y
        );

        glm::ivec2 rectMin(
            std::max(0, std::min(static_cast<int>(tileX), static_cast<int>(bboxMin.x / 16.0f))),
            std::max(0, std::min(static_cast<int>(tileY), static_cast<int>(bboxMin.y / 16.0f)))
        );
        glm::ivec2 rectMax(
            std::max(0, std::min(static_cast<int>(tileX), static_cast<int>(bboxMax.x / 16.0f + 1.0f))),
            std::max(0, std::min(static_cast<int>(tileY), static_cast<int>(bboxMax.y / 16.0f + 1.0f)))
        );
        if ((rectMax.x - rectMin.x) * (rectMax.y - rectMin.y) == 0) {
            return 0;
        }
        rect = glm::uvec4(rectMin.x, rectMin.y, rectMax.x, rectMax.y);
        const bool isY = (rectMax.y - rectMin.y) < (rectMax.x - rectMin.x);
        return processTilesCount(con_o, disc, t, p, bboxMin, bboxMax, bboxArgMin, bboxArgMax, rectMin, rectMax, isY);
    };

    auto computeSh = [&](uint32_t index) -> glm::vec3 {
        const auto& v = hostVertices[index];
        glm::vec3 rayDirection = glm::vec3(v.position) - glm::vec3(data.camera_position);
        const float norm = glm::length(rayDirection);
        if (norm <= 1e-8f) {
            rayDirection = glm::vec3(0.0f, 0.0f, 1.0f);
        } else {
            rayDirection /= norm;
        }
        const float x = rayDirection.x;
        const float y = rayDirection.y;
        const float z = rayDirection.z;
        auto shv = [&](uint32_t ind) -> glm::vec3 {
            return glm::vec3(v.shs[ind * 3], v.shs[ind * 3 + 1], v.shs[ind * 3 + 2]);
        };

        glm::vec3 c = SH_C0 * shv(0);
        c -= SH_C1 * shv(1) * y;
        c += SH_C1 * shv(2) * z;
        c -= SH_C1 * shv(3) * x;
        c += SH_C2[0] * shv(4) * x * y;
        c += SH_C2[1] * shv(5) * y * z;
        c += SH_C2[2] * shv(6) * (2.0f * z * z - x * x - y * y);
        c += SH_C2[3] * shv(7) * z * x;
        c += SH_C2[4] * shv(8) * (x * x - y * y);
        c += SH_C3[0] * shv(9) * (3.0f * x * x - y * y) * y;
        c += SH_C3[1] * shv(10) * x * y * z;
        c += SH_C3[2] * shv(11) * (4.0f * z * z - x * x - y * y) * y;
        c += SH_C3[3] * shv(12) * z * (2.0f * z * z - 3.0f * x * x - 3.0f * y * y);
        c += SH_C3[4] * shv(13) * x * (4.0f * z * z - x * x - y * y);
        c += SH_C3[5] * shv(14) * (x * x - y * y) * z;
        c += SH_C3[6] * shv(15) * x * (x * x - 3.0f * y * y);
        c += 0.5f;
        c = glm::max(c, glm::vec3(0.0f));
        return c;
    };

    auto computeCov2d = [&](uint32_t index, const glm::vec3& cam) -> glm::mat2 {
        glm::vec3 t = cam;
        const float limx = 1.3f * data.tan_fovx;
        const float limy = 1.3f * data.tan_fovy;
        const float txtz = t.x / t.z;
        const float tytz = t.y / t.z;
        t.x = std::min(limx, std::max(-limx, txtz)) * t.z;
        t.y = std::min(limy, std::max(-limy, tytz)) * t.z;
        const float focalX = static_cast<float>(data.width) / (2.0f * data.tan_fovx);
        const float focalY = static_cast<float>(data.height) / (2.0f * data.tan_fovy);

        glm::mat3 J(0.0f);
        J[0][0] = focalX / t.z;
        J[1][1] = focalY / t.z;
        J[2][0] = -(focalX * t.x) / (t.z * t.z);
        J[2][1] = -(focalY * t.y) / (t.z * t.z);

        glm::mat3 W = glm::transpose(glm::mat3(data.view_mat));
        const size_t base = static_cast<size_t>(index) * 6u;
        glm::mat3 sigma(1.0f);
        sigma[0][0] = hostCov3D[base + 0];
        sigma[0][1] = hostCov3D[base + 1];
        sigma[0][2] = hostCov3D[base + 2];
        sigma[1][0] = hostCov3D[base + 1];
        sigma[1][1] = hostCov3D[base + 3];
        sigma[1][2] = hostCov3D[base + 4];
        sigma[2][0] = hostCov3D[base + 2];
        sigma[2][1] = hostCov3D[base + 4];
        sigma[2][2] = hostCov3D[base + 5];

        const glm::mat3 T = W * J;
        glm::mat3 cov = glm::transpose(T) * sigma * T;
        cov[0][0] += 0.3f;
        cov[1][1] += 0.3f;
        return glm::mat2(cov);
    };

    for (uint32_t i = 0; i < vertexCount; ++i) {
        attrs[i] = {};
        overlaps[i] = 0u;
        prefix[i] = 0u;
    }
    for (uint32_t i = 0; i < vertexCount; ++i) {
        const glm::vec4 pHom = data.proj_mat * hostVertices[i].position;
        if (std::abs(pHom.w) <= 1e-8f) {
            continue;
        }
        const glm::vec3 ndc = glm::vec3(pHom) / pHom.w;
        const glm::vec4 pView4 = data.view_mat * hostVertices[i].position;
        if (pView4.z <= 0.2f) {
            continue;
        }

        const glm::mat2 cov2d = computeCov2d(i, glm::vec3(pView4));
        const float det = glm::determinant(cov2d);
        if (det <= 0.0f) {
            continue;
        }
        const glm::mat2 conic = glm::inverse(cov2d);
        attrs[i].conic_opacity = glm::vec4(conic[0][0], conic[0][1], conic[1][1], hostVertices[i].scale_opacity.w);

        const glm::vec2 uv(ndc2Pix(ndc.x, static_cast<int>(data.width)), ndc2Pix(ndc.y, static_cast<int>(data.height)));
        const float mid = 0.5f * (cov2d[0][0] + cov2d[1][1]);
        const float root = std::sqrt(std::max(0.1f, mid * mid - det));
        const float lambda = std::max(mid + root, mid - root);
        const float radii = std::ceil(3.0f * std::sqrt(std::max(0.0f, lambda)));

        glm::uvec4 boundingBox(0u);
        uint32_t overlap = 0u;
        overlap = duplicateToTilesTouched(uv, attrs[i].conic_opacity, boundingBox);
        if (overlap == 0u) {
            continue;
        }
        attrs[i].aabb = boundingBox;
        attrs[i].depth = pView4.z;
        attrs[i].color_radii = glm::vec4(computeSh(i), radii);
        attrs[i].uv = uv;
        attrs[i].__padding[0] = kMagic;
        overlaps[i] = overlap;
    }
    const auto tPreEnd = Clock::now();

    const auto tPrefixStart = Clock::now();
    uint32_t running = 0u;
    for (uint32_t i = 0; i < vertexCount; ++i) {
        running += overlaps[i];
        prefix[i] = running;
    }
    const auto tPrefixEnd = Clock::now();

    const uint32_t numInstances = running;
    frameNumInstances[frameResourceSlot] = numInstances;
    keysLow.resize(numInstances);
    keysHigh.resize(numInstances);
    payloads.resize(numInstances);

    auto emitTiles = [&](uint32_t index, uint32_t& off) {
        const auto& a = attrs[index];
        if (a.color_radii.w == 0.0f) {
            return;
        }
        const uint32_t depthBits = floatBits(a.depth);
        const glm::vec4 con_o = a.conic_opacity;
        const float disc = con_o.y * con_o.y - con_o.x * con_o.z;
        if (con_o.x <= 0.0f || con_o.z <= 0.0f || disc >= 0.0f || con_o.w <= (1.0f / 255.0f)) {
            return;
        }
        const float t = 2.0f * std::log(std::max(con_o.w * 255.0f, 1e-6f));
        const glm::vec2 p = a.uv;
        float xTerm = std::sqrt(std::max(0.0f, -(con_o.y * con_o.y * t) / (disc * con_o.x)));
        xTerm = (con_o.y < 0.0f) ? xTerm : -xTerm;
        float yTerm = std::sqrt(std::max(0.0f, -(con_o.y * con_o.y * t) / (disc * con_o.z)));
        yTerm = (con_o.y < 0.0f) ? yTerm : -yTerm;
        const glm::vec2 bboxArgMin(p.y - yTerm, p.x - xTerm);
        const glm::vec2 bboxArgMax(p.y + yTerm, p.x + xTerm);
        const glm::vec2 bboxMin(
            computeEllipseIntersection(con_o, disc, t, p, true, bboxArgMin.x).x,
            computeEllipseIntersection(con_o, disc, t, p, false, bboxArgMin.y).x
        );
        const glm::vec2 bboxMax(
            computeEllipseIntersection(con_o, disc, t, p, true, bboxArgMax.x).y,
            computeEllipseIntersection(con_o, disc, t, p, false, bboxArgMax.y).y
        );
        glm::ivec2 rectMin(a.aabb.x, a.aabb.y);
        glm::ivec2 rectMax(a.aabb.z, a.aabb.w);
        if ((rectMax.x - rectMin.x) * (rectMax.y - rectMin.y) == 0) {
            return;
        }
        bool isY = (rectMax.y - rectMin.y) < (rectMax.x - rectMin.x);
        const float blockU = 16.0f;
        const float blockV = 16.0f;
        if (isY) {
            rectMin = glm::ivec2(rectMin.y, rectMin.x);
            rectMax = glm::ivec2(rectMax.y, rectMax.x);
        }
        glm::vec2 bmin = bboxMin;
        glm::vec2 bmax = bboxMax;
        glm::vec2 bargmin = bboxArgMin;
        glm::vec2 bargmax = bboxArgMax;
        if (isY) {
            bmin = glm::vec2(bmin.y, bmin.x);
            bmax = glm::vec2(bmax.y, bmax.x);
            bargmin = glm::vec2(bargmin.y, bargmin.x);
            bargmax = glm::vec2(bargmax.y, bargmax.x);
        }
        glm::vec2 intersectMaxLine(bmax.y, bmin.y);
        glm::vec2 intersectMinLine = (bmin.x <= static_cast<float>(rectMin.x) * blockU)
                                         ? computeEllipseIntersection(con_o, disc, t, p, isY, static_cast<float>(rectMin.x) * blockU)
                                         : intersectMaxLine;
        float minLine = static_cast<float>(rectMin.x) * blockU;
        for (int u = rectMin.x; u < rectMax.x; ++u) {
            const float maxLine = minLine + blockU;
            if (maxLine <= bmax.x) {
                intersectMaxLine = computeEllipseIntersection(con_o, disc, t, p, isY, maxLine);
            }
            const float ellipseMin = (minLine <= bargmin.y && bargmin.y < maxLine)
                                         ? bmin.y
                                         : std::min(intersectMinLine.x, intersectMaxLine.x);
            const float ellipseMax = (minLine <= bargmax.y && bargmax.y < maxLine)
                                         ? bmax.y
                                         : std::max(intersectMinLine.y, intersectMaxLine.y);
            const int minTileV = std::max(rectMin.y, std::min(rectMax.y, static_cast<int>(ellipseMin / blockV)));
            const int maxTileV = std::min(rectMax.y, std::max(rectMin.y, static_cast<int>(ellipseMax / blockV + 1.0f)));
            for (int v = minTileV; v < maxTileV; ++v) {
                const uint32_t tileIndex = isY ? (static_cast<uint32_t>(u) * tileX + static_cast<uint32_t>(v))
                                               : (static_cast<uint32_t>(v) * tileX + static_cast<uint32_t>(u));
                keysHigh[off] = tileIndex;
                keysLow[off] = depthBits;
                payloads[off] = index;
                ++off;
            }
            intersectMinLine = intersectMaxLine;
            minLine = maxLine;
        }
    };

    const auto tPreSortStart = Clock::now();
    for (uint32_t i = 0; i < vertexCount; ++i) {
        if (overlaps[i] == 0u) {
            continue;
        }
        uint32_t off = (i == 0u) ? 0u : prefix[i - 1];
        emitTiles(i, off);
    }
    const auto tPreSortEnd = Clock::now();

    const auto tSortStart = Clock::now();
    std::vector<uint32_t> order(numInstances);
    for (uint32_t i = 0; i < numInstances; ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (keysHigh[a] != keysHigh[b]) return keysHigh[a] < keysHigh[b];
        if (keysLow[a] != keysLow[b]) return keysLow[a] < keysLow[b];
        return payloads[a] < payloads[b];
    });

    auto& sortedVertices = frameSortedVerticesHost[frameResourceSlot];
    sortedVertices.resize(numInstances);
    std::vector<uint32_t> sortedHigh(numInstances);
    for (uint32_t i = 0; i < numInstances; ++i) {
        sortedVertices[i] = payloads[order[i]];
        sortedHigh[i] = keysHigh[order[i]];
    }
    const auto tSortEnd = Clock::now();

    const auto tTileStart = Clock::now();
    auto& boundaries = frameTileBoundariesHost[frameResourceSlot];
    std::fill(boundaries.begin(), boundaries.end(), 0u);
    if (numInstances > 0) {
        for (uint32_t i = 0; i < numInstances; ++i) {
            const uint32_t key = sortedHigh[i];
            const bool keyValid = key < numTiles;
            if (i == 0) {
                if (keyValid) {
                    boundaries[key * 2] = i;
                }
            } else {
                const uint32_t prevKey = sortedHigh[i - 1];
                const bool prevValid = prevKey < numTiles;
                if ((key != prevKey) || (!keyValid) || (!prevValid)) {
                    if (keyValid) {
                        boundaries[key * 2] = i;
                    }
                    if (prevValid) {
                        boundaries[prevKey * 2 + 1] = i;
                    }
                }
            }
            if (i == numInstances - 1 && keyValid) {
                boundaries[key * 2 + 1] = numInstances;
            }
        }
    }
    const auto tTileEnd = Clock::now();

    const uint32_t neededSortCapacity = std::max<uint32_t>(1u, numInstances);
    if (neededSortCapacity * sizeof(uint32_t) > frameSortVBuffers[frameResourceSlot]->size) {
        frameSortVBuffers[frameResourceSlot]->realloc(neededSortCapacity * sizeof(uint32_t));
    }
    frameVertexAttributeBuffers[frameResourceSlot]->upload(attrs.data(), vertexCount * sizeof(VertexAttributeBuffer), 0);
    if (numInstances > 0) {
        frameSortVBuffers[frameResourceSlot]->upload(sortedVertices.data(), numInstances * sizeof(uint32_t), 0);
    }
    frameTileBoundaryBuffers[frameResourceSlot]->upload(boundaries.data(),
                                                        static_cast<uint32_t>(boundaries.size() * sizeof(uint32_t)), 0);

    cpuStageTiming.preprocessMs = std::chrono::duration<double, std::milli>(tPreEnd - tPreStart).count();
    cpuStageTiming.prefixSumMs = std::chrono::duration<double, std::milli>(tPrefixEnd - tPrefixStart).count();
    cpuStageTiming.preprocessSortMs = std::chrono::duration<double, std::milli>(tPreSortEnd - tPreSortStart).count();
    cpuStageTiming.sortMs = std::chrono::duration<double, std::milli>(tSortEnd - tSortStart).count();
    cpuStageTiming.tileBoundaryMs = std::chrono::duration<double, std::milli>(tTileEnd - tTileStart).count();
}


bool Renderer::recordRenderCommandBuffer(uint32_t currentFrame, uint32_t frameResourceSlot) {
    if (renderCommandBuffers.empty()) {
        vk::CommandBufferAllocateInfo allocateInfo = {commandPool.get(), vk::CommandBufferLevel::ePrimary, FRAMES_IN_FLIGHT};
        renderCommandBuffers = context->device->allocateCommandBuffersUnique(allocateInfo);
    }
    auto& renderCommandBuffer = renderCommandBuffers[frameResourceSlot];
    const uint32_t numInstances = frameNumInstances[frameResourceSlot];
    if (configuration.enableGui) {
        guiManager.pushTextMetric("instances", numInstances);
    }
    renderCommandBuffer->reset({});
    renderCommandBuffer->begin(vk::CommandBufferBeginInfo{});
    renderCommandBuffer->resetQueryPool(frameRenderQueryPools[frameResourceSlot].get(), 0, 2);

    renderPipeline->bind(renderCommandBuffer, currentFrame, std::vector<uint32_t>{frameResourceSlot, currentImageIndex});
    renderCommandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, frameRenderQueryPools[frameResourceSlot].get(), 0);
    auto [width, height] = swapchain->swapchainExtent;
    uint32_t constants[2] = {width, height};
    renderCommandBuffer->pushConstants(renderPipeline->pipelineLayout.get(),
                                       vk::ShaderStageFlagBits::eCompute, 0,
                                       sizeof(uint32_t) * 2, constants);

    vk::ImageMemoryBarrier imageMemoryBarrier{};
    imageMemoryBarrier.oldLayout = vk::ImageLayout::eUndefined;
    imageMemoryBarrier.newLayout = vk::ImageLayout::eGeneral;
    imageMemoryBarrier.image = swapchain->swapchainImages[currentImageIndex]->image;
    imageMemoryBarrier.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eNoneKHR;
    imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderWrite;
    imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    renderCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                         vk::PipelineStageFlagBits::eComputeShader,
                                         vk::DependencyFlags(),
                                         nullptr, nullptr, imageMemoryBarrier);

    renderCommandBuffer->dispatch((width + 15) / 16, (height + 15) / 16, 1);

    imageMemoryBarrier.oldLayout = vk::ImageLayout::eGeneral;
    imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
    if (configuration.enableGui) {
        imageMemoryBarrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
        imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        renderCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                             vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                             vk::DependencyFlags(),
                                             nullptr, nullptr, imageMemoryBarrier);
    } else {
        imageMemoryBarrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
        imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
        renderCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                             vk::PipelineStageFlagBits::eBottomOfPipe,
                                             vk::DependencyFlags(),
                                             nullptr, nullptr, imageMemoryBarrier);
    }
    renderCommandBuffer->writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, frameRenderQueryPools[frameResourceSlot].get(), 1);

    if (configuration.enableGui) {
        imguiManager->draw(renderCommandBuffer.get(), currentImageIndex, std::bind(&GUIManager::buildGui, &guiManager));
        imageMemoryBarrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
        imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
        imageMemoryBarrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
        imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
        renderCommandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                             vk::PipelineStageFlagBits::eComputeShader,
                                             vk::DependencyFlags(),
                                             nullptr, nullptr, imageMemoryBarrier);
    }
    renderCommandBuffer->end();
    return true;
}

Renderer::UniformBuffer Renderer::buildUniformData() const {
    UniformBuffer data{};
    auto [width, height] = swapchain->swapchainExtent;
    data.width = width;
    data.height = height;
    float tan_fovx = 0.0f;
    float tan_fovy = 0.0f;

    const auto* cameraOverride = activeCameraOverride.has_value()
                                     ? &activeCameraOverride.value()
                                     : (configuration.cameraOverride.has_value() ? &configuration.cameraOverride.value() : nullptr);
    if (cameraOverride != nullptr) {
        const auto& cameraJson = *cameraOverride;
        const glm::vec3 cameraPosition(
            static_cast<float>(cameraJson.position[0]),
            static_cast<float>(cameraJson.position[1]),
            static_cast<float>(cameraJson.position[2]));
        data.camera_position = glm::vec4(cameraPosition, 1.0f);

        glm::mat3 rotationC2W(1.0f);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                rotationC2W[col][row] = static_cast<float>(cameraJson.rotation[row][col]);
            }
        }
        // Python camera_to_JSON stores camera->world rotation. Build view from world->camera.
        const glm::mat3 rotationW2C = glm::transpose(rotationC2W);
        glm::mat4 view(1.0f);
        view[0][0] = rotationW2C[0][0];
        view[1][0] = rotationW2C[1][0];
        view[2][0] = rotationW2C[2][0];
        view[0][1] = rotationW2C[0][1];
        view[1][1] = rotationW2C[1][1];
        view[2][1] = rotationW2C[2][1];
        view[0][2] = rotationW2C[0][2];
        view[1][2] = rotationW2C[1][2];
        view[2][2] = rotationW2C[2][2];

        const glm::vec3 translation = -rotationW2C * cameraPosition;
        view[3][0] = translation.x;
        view[3][1] = translation.y;
        view[3][2] = translation.z;

        // Match Python cameras.json semantics:
        // fx/fy are tied to the camera entry's native width/height, not the current swapchain size.
        const float camWidth = static_cast<float>(std::max<uint32_t>(cameraJson.width, 1u));
        const float camHeight = static_cast<float>(std::max<uint32_t>(cameraJson.height, 1u));
        tan_fovx = camWidth / (2.0f * std::max(cameraJson.fx, 1e-6f));
        tan_fovy = camHeight / (2.0f * std::max(cameraJson.fy, 1e-6f));
        data.view_mat = view;
        data.proj_mat = glm::perspective(std::atan(tan_fovy) * 2.0f,
                                         static_cast<float>(width) / static_cast<float>(height),
                                         0.01f,
                                         100.0f) * view;
    } else {
        data.camera_position = glm::vec4(camera.position, 1.0f);
        auto rotation = glm::mat4_cast(camera.rotation);
        auto translation = glm::translate(glm::mat4(1.0f), camera.position);
        auto view = glm::inverse(translation * rotation);

        tan_fovx = std::tan(glm::radians(camera.fov) / 2.0);
        tan_fovy = tan_fovx * static_cast<float>(height) / static_cast<float>(width);
        data.view_mat = view;
        data.proj_mat = glm::perspective(std::atan(tan_fovy) * 2.0f,
                                         static_cast<float>(width) / static_cast<float>(height),
                                         camera.nearPlane,
                                         camera.farPlane) * view;
    }

    data.view_mat[0][1] *= -1.0f;
    data.view_mat[1][1] *= -1.0f;
    data.view_mat[2][1] *= -1.0f;
    data.view_mat[3][1] *= -1.0f;
    data.view_mat[0][2] *= -1.0f;
    data.view_mat[1][2] *= -1.0f;
    data.view_mat[2][2] *= -1.0f;
    data.view_mat[3][2] *= -1.0f;

    data.proj_mat[0][1] *= -1.0f;
    data.proj_mat[1][1] *= -1.0f;
    data.proj_mat[2][1] *= -1.0f;
    data.proj_mat[3][1] *= -1.0f;
    data.tan_fovx = tan_fovx;
    data.tan_fovy = tan_fovy;
    return data;
}

void Renderer::updateUniforms(const UniformBuffer& data, uint32_t frameResourceSlot) {
    (void)frameResourceSlot;
    // Kept for API symmetry; preprocessing is CPU-side and uploads direct render inputs.
}

Renderer::~Renderer() {
    stopWritebackThread();
}
