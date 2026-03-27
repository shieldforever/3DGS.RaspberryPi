#ifndef RENDERER_H
#define RENDERER_H

#define GLM_SWIZZLE

#include <atomic>
#include <array>
#include "3dgs.h"

#include "vulkan/Window.h"
#include "GSScene.h"
#include "vulkan/pipelines/ComputePipeline.h"
#include "vulkan/Swapchain.h"
#include <glm/gtc/quaternion.hpp>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>

#include "GUIManager.h"
#include "vulkan/ImguiManager.h"
#include "vulkan/QueryManager.h"

class Renderer {
public:
    struct alignas(16) UniformBuffer {
        glm::vec4 camera_position;
        glm::mat4 proj_mat;
        glm::mat4 view_mat;
        uint32_t width;
        uint32_t height;
        float tan_fovx;
        float tan_fovy;
    };

    struct VertexAttributeBuffer {
        glm::vec4 conic_opacity;
        glm::vec4 color_radii;
        glm::uvec4 aabb;
        glm::vec2 uv;
        float depth;
        uint32_t __padding[1];
    };

    struct Camera {
        glm::vec3 position;
        glm::quat rotation;
        float fov;
        float nearPlane;
        float farPlane;

        void translate(glm::vec3 translation) {
            position += rotation * translation;
        }
    };

    struct RadixSortPushConstants {
        uint32_t g_num_elements; // == NUM_ELEMENTS
        uint32_t g_shift; // (*)
        uint32_t g_num_workgroups; // == NUMBER_OF_WORKGROUPS as defined in the section above
        uint32_t g_num_blocks_per_workgroup; // == NUM_BLOCKS_PER_WORKGROUP
    };

    explicit Renderer(VulkanSplatting::RendererConfiguration configuration);

    void createGui();

    void initialize();

    void handleInput();

    std::array<double, 6> retrieveTimestamps(bool logTiming = true);

    void recreateSwapchain();

    void draw();

    void run();

    void stop();

    ~Renderer();

    Camera camera {
        .position = glm::vec3(0.0f, 0.0f, 5.0f),   
        .rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        .fov = 45.0f,
        .nearPlane = 0.1f,
        .farPlane = 1000.0f
    };

private:
    VulkanSplatting::RendererConfiguration configuration;
    std::shared_ptr<Window> window;
    std::shared_ptr<VulkanContext> context;
    std::shared_ptr<ImguiManager> imguiManager;
    std::shared_ptr<GSScene> scene;
    std::shared_ptr<QueryManager> queryManager = std::make_shared<QueryManager>();
    GUIManager guiManager {};

    std::shared_ptr<ComputePipeline> preprocessPipeline;
    std::shared_ptr<ComputePipeline> renderPipeline;
    std::shared_ptr<ComputePipeline> prefixSumPipeline;
    std::shared_ptr<ComputePipeline> preprocessSortPipeline;
    std::shared_ptr<ComputePipeline> sortHistPipeline;
    std::shared_ptr<ComputePipeline> sortScanPipeline;
    std::shared_ptr<ComputePipeline> sortScatterPipeline;
    std::shared_ptr<ComputePipeline> clearTileBoundaryPipeline;
    std::shared_ptr<ComputePipeline> tileBoundaryPipeline;

    std::shared_ptr<Buffer> uniformBuffer;
    std::shared_ptr<Buffer> gaussianMaskBuffer;
    std::shared_ptr<Buffer> vertexAttributeBuffer;
    std::shared_ptr<Buffer> tileOverlapBuffer;
    std::shared_ptr<Buffer> prefixSumPingBuffer;
    std::shared_ptr<Buffer> prefixSumPongBuffer;
    std::shared_ptr<Buffer> totalSumBufferHost;

    std::shared_ptr<Buffer> sortKBufferEvenLow;
    std::shared_ptr<Buffer> sortKBufferEvenHigh;
    std::shared_ptr<Buffer> sortKBufferOddLow;
    std::shared_ptr<Buffer> sortKBufferOddHigh;
    std::shared_ptr<Buffer> sortKBufferEven;
    std::shared_ptr<Buffer> sortKBufferOdd;
    std::shared_ptr<Buffer> sortVBufferEven;
    std::shared_ptr<Buffer> sortVBufferOdd;
    std::shared_ptr<Buffer> sortHistBuffer;
    std::shared_ptr<Buffer> sortOffsetBuffer;
    std::shared_ptr<Buffer> tileBoundaryBuffer;

    // CPU-preprocess + per-frame render input resources.
    std::vector<GSScene::Vertex> hostVertices;
    std::vector<float> hostCov3D;
    std::vector<std::vector<VertexAttributeBuffer>> frameVertexAttributesHost;
    std::vector<std::vector<uint32_t>> frameSortedVerticesHost;
    std::vector<std::vector<uint32_t>> frameTileBoundariesHost;
    std::vector<uint32_t> frameNumInstances;
    std::vector<std::shared_ptr<Buffer>> frameVertexAttributeBuffers;
    std::vector<std::shared_ptr<Buffer>> frameSortVBuffers;
    std::vector<std::shared_ptr<Buffer>> frameTileBoundaryBuffers;
    uint32_t frameSlot = 0;
    uint64_t frameSequenceId = 0;
    std::optional<VulkanSplatting::RendererConfiguration::CameraOverride> activeCameraOverride = std::nullopt;
    std::string activeOutputRawPath {};

    std::shared_ptr<DescriptorSet> inputSet;

    std::atomic<bool> running = true;

    std::vector<vk::UniqueFence> inflightFences;

    std::shared_ptr<Swapchain> swapchain;

    vk::UniqueCommandPool commandPool;

    vk::UniqueCommandBuffer preprocessCommandBuffer;
    std::vector<vk::UniqueCommandBuffer> renderCommandBuffers;

    uint32_t currentImageIndex;

    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;

#ifdef __APPLE__
    uint32_t numRadixSortBlocksPerWorkgroup = 256;
#else
    uint32_t numRadixSortBlocksPerWorkgroup = 32;
#endif

    int fpsCounter = 0;
    std::chrono::high_resolution_clock::time_point lastFpsTime = std::chrono::high_resolution_clock::now();

    unsigned int sortBufferSizeMultiplier = 1;
    uint32_t radixBits = 8;
    uint32_t radixBins = 256;
    uint32_t radixPasses = 8;
    bool debugMaskStats = false;
    bool debugMaskStatsLogged = false;
    struct CpuStageTiming {
        double preprocessMs = 0.0;
        double prefixSumMs = 0.0;
        double preprocessSortMs = 0.0;
        double sortMs = 0.0;
        double tileBoundaryMs = 0.0;
    } cpuStageTiming;
    struct FrameTimingSnapshot {
        CpuStageTiming cpu {};
        bool valid = false;
    };
    struct PendingReadback {
        uint64_t frameId = 0;
        uint32_t imageIndex = 0;
        std::string outputPath {};
        bool valid = false;
    };
    struct InflightReadback {
        std::shared_ptr<Buffer> stagingBuffer;
        vk::UniqueFence fence;
        vk::UniqueCommandBuffer commandBuffer;
        VkDeviceSize imageSize = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        std::string outputPath {};
    };
    struct WritebackTask {
        std::shared_ptr<Buffer> stagingBuffer;
        VkDeviceSize imageSize = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        std::string outputPath {};
    };
    uint64_t timingSampleCount = 0;
    double timingCpuTotalMsAccum = 0.0;
    double timingRenderMsAccum = 0.0;
    std::vector<double> frameE2EMsSamples;
    std::vector<vk::UniqueQueryPool> frameRenderQueryPools;
    std::vector<FrameTimingSnapshot> framePendingTiming;
    std::vector<PendingReadback> framePendingReadback;
    std::deque<InflightReadback> inflightReadbacks;
    std::deque<WritebackTask> writebackQueue;
    std::mutex writebackMutex;
    std::condition_variable writebackCv;
    std::thread writebackThread;
    std::atomic<bool> writebackThreadRunning = false;
    std::atomic<uint64_t> pendingWritebackCount = 0;

    void initializeVulkan();

    void loadSceneToGPU();

    void createPreprocessPipeline();

    void createPrefixSumPipeline();

    void createRadixSortPipeline();

    void createPreprocessSortPipeline();

    void createTileBoundaryPipeline();
    void createClearTileBoundaryPipeline();

    void createRenderPipeline();

    void recordPreprocessCommandBuffer();

    bool recordRenderCommandBuffer(uint32_t currentFrame, uint32_t frameResourceSlot);

    void createCommandPool();

    UniformBuffer buildUniformData() const;
    void updateUniforms(const UniformBuffer& data, uint32_t frameResourceSlot);
    void initializeCpuPreprocessResources();
    void runCpuPreprocess(uint32_t frameResourceSlot, const UniformBuffer& data);
    void createPerFrameRenderQueryPools();
    void flushCompletedTimingForSlot(uint32_t slot, bool logTiming);
    void flushCompletedReadbackForSlot(uint32_t slot);
    void enqueueAsyncReadback(uint32_t imageIndex, const std::string& outputPath);
    void pollAsyncReadbacks();
    void startWritebackThread();
    void stopWritebackThread();
    void writebackThreadMain();
    static std::string addFrameSuffixToPath(const std::string& basePath, uint64_t frameId);
    void saveFrameRawByImageIndex(uint32_t imageIndex, const std::string& outputPath);
    void saveCurrentFrameRaw(const std::string& outputPath);
};


#endif //RENDERER_H
