#include <iostream>
#include <utility>
#include <vulkan/vulkan.h>
#include "Buffer.h"
#include <vk_mem_alloc.h>

#include "Utils.h"
#include "spdlog/spdlog.h"

void Buffer::alloc() {
    auto bufferInfo = vk::BufferCreateInfo()
            .setSize(size)
            .setUsage(usage);
    
    // Check if graphics and compute families are the same to avoid sharing mode concurrent errors
    auto graphicsFamily = context->queues[VulkanContext::Queue::Type::GRAPHICS].queueFamily;
    auto computeFamily = context->queues[VulkanContext::Queue::Type::COMPUTE].queueFamily;
    
    if (shared && graphicsFamily != computeFamily) {
        uint32_t queueFamilyIndices[] = {graphicsFamily, computeFamily};
        bufferInfo.setSharingMode(vk::SharingMode::eConcurrent)
                  .setQueueFamilyIndexCount(2)
                  .setPQueueFamilyIndices(queueFamilyIndices);
    } else {
        bufferInfo.setSharingMode(vk::SharingMode::eExclusive);
    }

    auto vkBufferInfo = static_cast<VkBufferCreateInfo>(bufferInfo);

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = vmaUsage;
    allocInfo.flags = flags;
    
    // 树莓派上的 Broadcom V3D 驱动可能对某些内存标志较为敏感。
    // 如果是 GPU_ONLY 模式，尝试更通用的标志以减少内存分页冲突导致的 -8 错误（VK_ERROR_FEATURE_NOT_PRESENT 或某些资源不足）
    if (vmaUsage == VMA_MEMORY_USAGE_GPU_ONLY) {
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.priority = 1.0f;
    }

    VkBuffer vkBuffer = VK_NULL_HANDLE;

    VkResult res;
    if (alignment != 0) {
        res = vmaCreateBufferWithAlignment(context->allocator, &vkBufferInfo, &allocInfo, alignment, &vkBuffer,
                                           &allocation, &allocation_info);
    } else {
        res = vmaCreateBuffer(context->allocator, &vkBufferInfo, &allocInfo, &vkBuffer, &allocation, &allocation_info);
    }
    if (res != VK_SUCCESS) {
        spdlog::error("vmaCreateBuffer failed with error code: {}", (int)res);
        throw std::runtime_error("Failed to create buffer");
    }
    buffer = vk::Buffer(vkBuffer);

    if (context->validationLayersEnabled) {
        context->device->setDebugUtilsObjectNameEXT(
                vk::DebugUtilsObjectNameInfoEXT {vk::ObjectType::eBuffer, reinterpret_cast<uint64_t>(static_cast<VkBuffer>(buffer)), debugName.c_str()});
    }
}

Buffer::Buffer(const std::shared_ptr<VulkanContext>& _context, uint32_t size, vk::BufferUsageFlags usage,
               VmaMemoryUsage vmaUsage, VmaAllocationCreateFlags flags, bool shared, vk::DeviceSize alignment, std::string debugName)
    : context(_context),
      size(size),
      alignment(alignment),
      shared(shared),
      usage(usage),
      vmaUsage(vmaUsage),
      flags(flags),
      allocation(nullptr),
      debugName(std::move(debugName)) {
    alloc();
}

Buffer Buffer::createStagingBuffer(uint32_t size) {
    return Buffer(context, size, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                  VMA_MEMORY_USAGE_AUTO,
                  VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, false);
}

void Buffer::upload(const void* data, uint32_t size, uint32_t offset) {
    if (size + offset > this->size) {
        throw std::runtime_error("Buffer overflow");
    }

    if (!(flags & VMA_ALLOCATION_CREATE_MAPPED_BIT)) {
        auto stagingBuffer = createStagingBuffer(size);
        memcpy(stagingBuffer.allocation_info.pMappedData, ((char *) data) + offset, size);
        vmaFlushAllocation(context->allocator, stagingBuffer.allocation, 0, size);
        auto commandBuffer = context->beginOneTimeCommandBuffer();
        vk::BufferCopy copyRegion = {};
        copyRegion.setSize(size);
        commandBuffer->copyBuffer(stagingBuffer.buffer, buffer, 1, &copyRegion);
        context->endOneTimeCommandBuffer(std::move(commandBuffer), VulkanContext::Queue::COMPUTE);
    } else {
        memcpy(allocation_info.pMappedData, ((char *) data) + offset, size);
        vmaFlushAllocation(context->allocator, allocation, offset, size);
    }
}

void Buffer::uploadFrom(std::shared_ptr<Buffer> buffer) {
    if (buffer->size > size) {
        throw std::runtime_error("Buffer overflow");
    }

    if (!(flags & VMA_ALLOCATION_CREATE_MAPPED_BIT)) {
        auto commandBuffer = context->beginOneTimeCommandBuffer();
        vk::BufferCopy copyRegion = {};
        copyRegion.setSize(buffer->size);
        commandBuffer->copyBuffer(buffer->buffer, this->buffer, 1, &copyRegion);
        context->endOneTimeCommandBuffer(std::move(commandBuffer), VulkanContext::Queue::COMPUTE);
    } else {
        vmaInvalidateAllocation(context->allocator, buffer->allocation, 0, buffer->size);
        memcpy(allocation_info.pMappedData, buffer->allocation_info.pMappedData, buffer->size);
        vmaFlushAllocation(context->allocator, allocation, 0, buffer->size);
    }
}

void Buffer::downloadTo(std::shared_ptr<Buffer> buffer, vk::DeviceSize srcOffset, vk::DeviceSize dstOffset) {
    if (!(flags & VMA_ALLOCATION_CREATE_MAPPED_BIT)) {
        auto commandBuffer = context->beginOneTimeCommandBuffer();
        vk::BufferCopy copyRegion = {};
        copyRegion.setSize(buffer->size);
        commandBuffer->copyBuffer(this->buffer, buffer->buffer, 1, &copyRegion);
        context->endOneTimeCommandBuffer(std::move(commandBuffer), VulkanContext::Queue::COMPUTE);
    } else {
        vmaInvalidateAllocation(context->allocator, allocation, srcOffset, buffer->size);
        memcpy(buffer->allocation_info.pMappedData, allocation_info.pMappedData, buffer->size);
        vmaFlushAllocation(context->allocator, buffer->allocation, dstOffset, buffer->size);
    }
}

Buffer::~Buffer() {
    vmaDestroyBuffer(context->allocator, static_cast<VkBuffer>(buffer), allocation);
    spdlog::debug("Buffer destroyed");
}

void Buffer::realloc(uint64_t newSize) {
    if (newSize <= size) {
        return;
    }

    vmaDestroyBuffer(context->allocator, static_cast<VkBuffer>(buffer), allocation);

    size = newSize;
    alloc();

    // 关键修复：VkDescriptorBufferInfo 的 offset 是相对于 VkBuffer 句柄的偏移。
    // 在这里由于我们要绑定整个重分配后的 Buffer，offset 必须为 0。
    // 旧代码中传递了 allocation_info.offset，那是相对于父 memory 块的偏移，会导致越界报错。
    vk::DescriptorBufferInfo bufferInfo(buffer, 0, size);

    std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
    for (auto& tuple: boundDescriptorSets) {
        auto descriptorSet = std::get<0>(tuple);
        auto shared = descriptorSet.lock();
        if (shared) {
            writeDescriptorSets.emplace_back(shared->descriptorSets[std::get<1>(tuple)].get(),
                                            std::get<2>(tuple), 0, 1,
                                            std::get<3>(tuple), nullptr,
                                            &bufferInfo);
        }
    }
    if (!writeDescriptorSets.empty()) {
        context->device->updateDescriptorSets(writeDescriptorSets, nullptr);
    }
}

void Buffer::boundToDescriptorSet(std::weak_ptr<DescriptorSet> descriptorSet, uint32_t set, uint32_t binding,
    vk::DescriptorType type) {
    boundDescriptorSets.push_back({descriptorSet, set, binding, type});
}

std::shared_ptr<Buffer> Buffer::uniform(std::shared_ptr<VulkanContext> context, uint32_t size, bool shared) {
    return std::make_shared<Buffer>(context, size, vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,
                  VMA_MEMORY_USAGE_AUTO,
                  VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, shared);
}

std::shared_ptr<Buffer> Buffer::staging(std::shared_ptr<VulkanContext> context, unsigned long size) {
    return std::make_shared<Buffer>(context, size, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                  VMA_MEMORY_USAGE_AUTO,
                  VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, false);
}

std::shared_ptr<Buffer> Buffer::storage(std::shared_ptr<VulkanContext> context, uint64_t size, bool shared, vk::DeviceSize alignment, std::string debugName) {
    return std::make_shared<Buffer>(context, size, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
                  VMA_MEMORY_USAGE_AUTO,
                  0, shared, alignment, std::move(debugName));
}

void Buffer::assertEquals(char* data, size_t length) {
    if (length > size) {
        throw std::runtime_error("Buffer overflow");
    }

    if (!(flags & VMA_ALLOCATION_CREATE_MAPPED_BIT)) {
        auto stagingBuffer = Buffer::staging(context, length);
        downloadTo(stagingBuffer);
        if (memcmp(data, stagingBuffer->allocation_info.pMappedData, length) != 0) {
            throw std::runtime_error("Buffer content does not match");
        }
    } else {
        if (memcmp(data, allocation_info.pMappedData, length) != 0) {
            throw std::runtime_error("Buffer content does not match");
        }
    }
}

void Buffer::computeWriteReadBarrier(vk::CommandBuffer commandBuffer) {
    Utils::BarrierBuilder().queueFamilyIndex(context->queues[VulkanContext::Queue::COMPUTE].queueFamily)
            .addBufferBarrier(shared_from_this(), vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead)
            .build(commandBuffer, vk::PipelineStageFlagBits::eComputeShader,
                   vk::PipelineStageFlagBits::eComputeShader);
}

void Buffer::computeReadWriteBarrier(vk::CommandBuffer commandBuffer) {
    Utils::BarrierBuilder().queueFamilyIndex(context->queues[VulkanContext::Queue::COMPUTE].queueFamily)
            .addBufferBarrier(shared_from_this(), vk::AccessFlagBits::eShaderRead,
                              vk::AccessFlagBits::eShaderWrite)
            .build(commandBuffer, vk::PipelineStageFlagBits::eComputeShader,
                   vk::PipelineStageFlagBits::eComputeShader);
}

void Buffer::computeWriteWriteBarrier(vk::CommandBuffer commandBuffer) {
    Utils::BarrierBuilder().queueFamilyIndex(context->queues[VulkanContext::Queue::COMPUTE].queueFamily)
            .addBufferBarrier(shared_from_this(), vk::AccessFlagBits::eShaderWrite,
                              vk::AccessFlagBits::eShaderWrite)
            .build(commandBuffer, vk::PipelineStageFlagBits::eComputeShader,
                   vk::PipelineStageFlagBits::eComputeShader);
}

std::vector<char> Buffer::download() {
    auto stagingBuffer = Buffer::staging(context, size);
    downloadTo(stagingBuffer);
    return {
        (char *) stagingBuffer->allocation_info.pMappedData,
        ((char *) stagingBuffer->allocation_info.pMappedData) + size
    };
}
