#ifndef VULKANSPLATTING_H
#define VULKANSPLATTING_H

#include <optional>
#include <string>
#include <memory>
#include <array>
#include <cstdint>
#include <vector>

class Window;
class Renderer;

class VulkanSplatting {
public:
    struct RendererConfiguration {
        bool enableVulkanValidationLayers = false;
        std::optional<uint8_t> physicalDeviceId = std::nullopt;
        bool immediateSwapchain = false;
        std::string scene;
        std::string outputRawPath = "output/output.raw";

        float fov = 45.0f;
        float near = 0.2f;
        float far = 1000.0f;
        bool enableGui = false;

        struct CameraOverride {
            std::array<double, 3> position {};
            std::array<std::array<double, 3>, 3> rotation {};
            float fx = 0.0f;
            float fy = 0.0f;
            uint32_t width = 0;
            uint32_t height = 0;
        };
        struct RenderJob {
            CameraOverride camera;
            std::string outputRawPath;
        };
        std::optional<CameraOverride> cameraOverride = std::nullopt;
        std::vector<RenderJob> renderJobs {};

        std::shared_ptr<Window> window;
    };

    explicit VulkanSplatting(RendererConfiguration configuration) : configuration(configuration) {}

#ifdef VKGS_ENABLE_GLFW
    static std::shared_ptr<Window> createGlfwWindow(std::string name, int width, int height);
#endif

#ifdef VKGS_ENABLE_METAL
    static std::shared_ptr<Window> createMetalWindow(void *caMetalLayer, int width, int height);
#endif

    void start();

    void initialize();

    void draw();

    void logTranslation(float x, float y);

    void logMovement(float x, float y, float z);

    void stop();
private:
    RendererConfiguration configuration;
    std::shared_ptr<Renderer> renderer;
};

#endif //VULKANSPLATTING_H
