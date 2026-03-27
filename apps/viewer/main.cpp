#include <filesystem>
#include <iostream>
#include <fstream>
#include <optional>
#include <cstdlib>
#include <cctype>
#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <libenvpp/env.hpp>

#include "3dgs.h"
#include "args.hxx"
#include "spdlog/spdlog.h"

namespace {
struct CameraJsonRecord {
    std::string imgName;
    uint32_t width = 0;
    uint32_t height = 0;
    float fx = 0.0f;
    float fy = 0.0f;
    std::array<double, 3> position {};
    std::array<std::array<double, 3>, 3> rotation {};
};

template <typename T>
bool readBinary(std::ifstream& ifs, T& value) {
    ifs.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(ifs);
}

uint32_t colmapCameraNumParams(int32_t modelId) {
    switch (modelId) {
        case 0: return 3;  // SIMPLE_PINHOLE
        case 1: return 4;  // PINHOLE
        case 2: return 4;  // SIMPLE_RADIAL
        case 3: return 5;  // RADIAL
        case 4: return 8;  // OPENCV
        case 5: return 8;  // OPENCV_FISHEYE
        case 6: return 12; // FULL_OPENCV
        case 7: return 5;  // FOV
        case 8: return 4;  // SIMPLE_RADIAL_FISHEYE
        case 9: return 5;  // RADIAL_FISHEYE
        case 10: return 12; // THIN_PRISM_FISHEYE
        default: return 0;
    }
}

struct ColmapIntrinsics {
    uint32_t width = 0;
    uint32_t height = 0;
    float fx = 0.0f;
    float fy = 0.0f;
};

std::optional<std::unordered_map<uint32_t, ColmapIntrinsics>>
loadColmapCamerasBin(const std::string& camerasBinPath, std::string& err) {
    std::ifstream ifs(camerasBinPath, std::ios::binary);
    if (!ifs.is_open()) {
        err = "Cannot open cameras.bin: " + camerasBinPath;
        return std::nullopt;
    }

    uint64_t numCameras = 0;
    if (!readBinary(ifs, numCameras)) {
        err = "Failed to read cameras.bin header";
        return std::nullopt;
    }

    std::unordered_map<uint32_t, ColmapIntrinsics> cameras;
    cameras.reserve(static_cast<size_t>(numCameras));
    for (uint64_t i = 0; i < numCameras; ++i) {
        uint32_t cameraId = 0;
        int32_t modelId = 0;
        uint64_t width = 0;
        uint64_t height = 0;
        if (!readBinary(ifs, cameraId) ||
            !readBinary(ifs, modelId) ||
            !readBinary(ifs, width) ||
            !readBinary(ifs, height)) {
            err = "Failed to read camera entry in cameras.bin";
            return std::nullopt;
        }

        const uint32_t numParams = colmapCameraNumParams(modelId);
        if (numParams < 3) {
            err = "Unsupported COLMAP camera model id: " + std::to_string(modelId);
            return std::nullopt;
        }
        std::vector<double> params(numParams, 0.0);
        ifs.read(reinterpret_cast<char*>(params.data()), static_cast<std::streamsize>(numParams * sizeof(double)));
        if (!ifs) {
            err = "Failed to read camera params in cameras.bin";
            return std::nullopt;
        }

        ColmapIntrinsics intr {};
        intr.width = static_cast<uint32_t>(width);
        intr.height = static_cast<uint32_t>(height);
        if (modelId == 0 || modelId == 2 || modelId == 3 || modelId == 8 || modelId == 9) {
            intr.fx = static_cast<float>(params[0]);
            intr.fy = static_cast<float>(params[0]);
        } else {
            intr.fx = static_cast<float>(params[0]);
            intr.fy = static_cast<float>(params[1]);
        }
        cameras[cameraId] = intr;
    }
    return cameras;
}

std::optional<std::vector<CameraJsonRecord>>
loadAllCamerasFromColmapBin(const std::string& camerasBinPath,
                            const std::string& imagesBinPath,
                            std::string& err) {
    auto intrMapOpt = loadColmapCamerasBin(camerasBinPath, err);
    if (!intrMapOpt.has_value()) {
        return std::nullopt;
    }
    const auto& intrMap = intrMapOpt.value();

    std::ifstream ifs(imagesBinPath, std::ios::binary);
    if (!ifs.is_open()) {
        err = "Cannot open images.bin: " + imagesBinPath;
        return std::nullopt;
    }

    uint64_t numImages = 0;
    if (!readBinary(ifs, numImages)) {
        err = "Failed to read images.bin header";
        return std::nullopt;
    }
    const uint64_t fileSize = std::filesystem::file_size(imagesBinPath);

    std::vector<CameraJsonRecord> records;
    records.reserve(static_cast<size_t>(numImages));
    for (uint64_t i = 0; i < numImages; ++i) {
        if (static_cast<uint64_t>(ifs.tellg()) + 64 > fileSize) {
            spdlog::warn("images.bin appears truncated at image index {} (parsed {}/{} entries)", i, records.size(),
                         numImages);
            break;
        }
        uint32_t imageId = 0;
        double qvec[4] = {1.0, 0.0, 0.0, 0.0}; // qw qx qy qz
        double tvec[3] = {0.0, 0.0, 0.0};
        uint32_t cameraId = 0;
        if (!readBinary(ifs, imageId) ||
            !readBinary(ifs, qvec[0]) ||
            !readBinary(ifs, qvec[1]) ||
            !readBinary(ifs, qvec[2]) ||
            !readBinary(ifs, qvec[3]) ||
            !readBinary(ifs, tvec[0]) ||
            !readBinary(ifs, tvec[1]) ||
            !readBinary(ifs, tvec[2]) ||
            !readBinary(ifs, cameraId)) {
            if (!records.empty()) {
                spdlog::warn("Stopped parsing images.bin at index {} due to incomplete image header", i);
                break;
            }
            err = "Failed to read image entry in images.bin";
            return std::nullopt;
        }

        std::string imageName;
        while (true) {
            char ch = '\0';
            if (!ifs.get(ch)) {
                err = "Failed to read image name in images.bin";
                return std::nullopt;
            }
            if (ch == '\0') break;
            imageName.push_back(ch);
        }

        uint64_t numPoints2D = 0;
        if (!readBinary(ifs, numPoints2D)) {
            if (!records.empty()) {
                spdlog::warn("Stopped parsing images.bin at index {} due to incomplete points2D count", i);
                break;
            }
            err = "Failed to read num_points2D in images.bin";
            return std::nullopt;
        }
        const std::streamoff pointsBytes = static_cast<std::streamoff>(numPoints2D) *
                                           static_cast<std::streamoff>(sizeof(double) * 2 + sizeof(int64_t));
        if (static_cast<uint64_t>(ifs.tellg()) + static_cast<uint64_t>(pointsBytes) > fileSize) {
            spdlog::warn("Stopped parsing images.bin at index {} due to truncated points2D block (parsed {}/{} entries)",
                         i, records.size(), numImages);
            break;
        }
        ifs.seekg(pointsBytes, std::ios::cur);
        if (!ifs) {
            if (!records.empty()) {
                spdlog::warn("Stopped parsing images.bin at index {} while skipping points2D", i);
                break;
            }
            err = "Failed to skip points2D in images.bin";
            return std::nullopt;
        }

        auto intrIt = intrMap.find(cameraId);
        if (intrIt == intrMap.end()) {
            err = "images.bin references unknown camera_id=" + std::to_string(cameraId);
            return std::nullopt;
        }
        const auto& intr = intrIt->second;

        // qvec/tvec in images.bin are world->camera: Xc = R*Xw + t
        // JSON expects camera center (world) and camera->world rotation.
        const double qw = qvec[0], qx = qvec[1], qy = qvec[2], qz = qvec[3];
        std::array<std::array<double, 3>, 3> Rw2c {{
            {{1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz),       2.0 * (qx * qz + qw * qy)}},
            {{2.0 * (qx * qy + qw * qz),       1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)}},
            {{2.0 * (qx * qz - qw * qy),       2.0 * (qy * qz + qw * qx),       1.0 - 2.0 * (qx * qx + qy * qy)}}
        }};
        std::array<std::array<double, 3>, 3> Rc2w {{
            {{Rw2c[0][0], Rw2c[1][0], Rw2c[2][0]}},
            {{Rw2c[0][1], Rw2c[1][1], Rw2c[2][1]}},
            {{Rw2c[0][2], Rw2c[1][2], Rw2c[2][2]}}
        }};
        const double cx = -(Rc2w[0][0] * tvec[0] + Rc2w[0][1] * tvec[1] + Rc2w[0][2] * tvec[2]);
        const double cy = -(Rc2w[1][0] * tvec[0] + Rc2w[1][1] * tvec[1] + Rc2w[1][2] * tvec[2]);
        const double cz = -(Rc2w[2][0] * tvec[0] + Rc2w[2][1] * tvec[1] + Rc2w[2][2] * tvec[2]);

        CameraJsonRecord rec {};
        rec.imgName = imageName;
        rec.width = intr.width;
        rec.height = intr.height;
        rec.fx = intr.fx;
        rec.fy = intr.fy;
        rec.position = {cx, cy, cz};
        rec.rotation = Rc2w;
        records.push_back(rec);
    }

    if (records.empty()) {
        err = "No camera/image entries found in COLMAP bin files";
        return std::nullopt;
    }
    return records;
}

bool parseJsonNumber(const std::string& text, size_t start, double& out, size_t* endPos = nullptr) {
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) {
        ++start;
    }
    const char* begin = text.data() + start;
    char* end = nullptr;
    out = std::strtod(begin, &end);
    if (end == begin) {
        return false;
    }
    if (endPos != nullptr) {
        *endPos = static_cast<size_t>(end - text.data());
    }
    return true;
}

size_t findKeyPosition(const std::string& objectText, const std::string& key) {
    return objectText.find("\"" + key + "\"");
}

bool parseScalarField(const std::string& objectText, const std::string& key, double& value) {
    const size_t keyPos = findKeyPosition(objectText, key);
    if (keyPos == std::string::npos) {
        return false;
    }
    const size_t colonPos = objectText.find(':', keyPos);
    if (colonPos == std::string::npos) {
        return false;
    }
    return parseJsonNumber(objectText, colonPos + 1, value);
}

bool parseStringField(const std::string& objectText, const std::string& key, std::string& value) {
    const size_t keyPos = findKeyPosition(objectText, key);
    if (keyPos == std::string::npos) {
        return false;
    }
    const size_t colonPos = objectText.find(':', keyPos);
    if (colonPos == std::string::npos) {
        return false;
    }
    const size_t quoteStart = objectText.find('"', colonPos + 1);
    if (quoteStart == std::string::npos) {
        return false;
    }
    const size_t quoteEnd = objectText.find('"', quoteStart + 1);
    if (quoteEnd == std::string::npos) {
        return false;
    }
    value = objectText.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
    return true;
}

bool parseVector3Field(const std::string& objectText, const std::string& key, std::array<double, 3>& value) {
    const size_t keyPos = findKeyPosition(objectText, key);
    if (keyPos == std::string::npos) {
        return false;
    }
    size_t cursor = objectText.find('[', keyPos);
    if (cursor == std::string::npos) {
        return false;
    }
    ++cursor;
    for (size_t i = 0; i < 3; ++i) {
        if (!parseJsonNumber(objectText, cursor, value[i], &cursor)) {
            return false;
        }
        if (i < 2) {
            cursor = objectText.find(',', cursor);
            if (cursor == std::string::npos) {
                return false;
            }
            ++cursor;
        }
    }
    return true;
}

bool parseMatrix3x3Field(const std::string& objectText, const std::string& key,
                         std::array<std::array<double, 3>, 3>& value) {
    const size_t keyPos = findKeyPosition(objectText, key);
    if (keyPos == std::string::npos) {
        return false;
    }
    size_t cursor = objectText.find('[', keyPos);
    if (cursor == std::string::npos) {
        return false;
    }
    ++cursor;
    for (size_t row = 0; row < 3; ++row) {
        cursor = objectText.find('[', cursor);
        if (cursor == std::string::npos) {
            return false;
        }
        ++cursor;
        for (size_t col = 0; col < 3; ++col) {
            if (!parseJsonNumber(objectText, cursor, value[row][col], &cursor)) {
                return false;
            }
            if (col < 2) {
                cursor = objectText.find(',', cursor);
                if (cursor == std::string::npos) {
                    return false;
                }
                ++cursor;
            }
        }
        cursor = objectText.find(']', cursor);
        if (cursor == std::string::npos) {
            return false;
        }
        ++cursor;
    }
    return true;
}

bool extractObjectByIndex(const std::string& arrayText, uint32_t objectIndex, std::string& outObject) {
    bool inString = false;
    bool escaping = false;
    int depth = 0;
    uint32_t seenObjects = 0;
    size_t objectStart = std::string::npos;

    for (size_t i = 0; i < arrayText.size(); ++i) {
        const char c = arrayText[i];
        if (inString) {
            if (escaping) {
                escaping = false;
            } else if (c == '\\') {
                escaping = true;
            } else if (c == '"') {
                inString = false;
            }
            continue;
        }

        if (c == '"') {
            inString = true;
            continue;
        }
        if (c == '{') {
            ++depth;
            if (depth == 1) {
                objectStart = i;
            }
            continue;
        }
        if (c == '}') {
            --depth;
            if (depth == 0 && objectStart != std::string::npos) {
                if (seenObjects == objectIndex) {
                    outObject = arrayText.substr(objectStart, i - objectStart + 1);
                    return true;
                }
                ++seenObjects;
                objectStart = std::string::npos;
            }
        }
    }
    return false;
}

std::optional<std::vector<CameraJsonRecord>> loadAllCamerasFromJson(const std::string& jsonPath, std::string& err) {
    std::ifstream ifs(jsonPath, std::ios::binary);
    if (!ifs.is_open()) {
        err = "Cannot open camera json: " + jsonPath;
        return std::nullopt;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    std::vector<CameraJsonRecord> cameras;
    uint32_t idx = 0;
    while (true) {
        std::string objectText;
        if (!extractObjectByIndex(content, idx, objectText)) {
            break;
        }
        CameraJsonRecord record;
        double width = 0.0, height = 0.0, fx = 0.0, fy = 0.0;
        if (!parseStringField(objectText, "img_name", record.imgName) ||
            !parseScalarField(objectText, "width", width) ||
            !parseScalarField(objectText, "height", height) ||
            !parseScalarField(objectText, "fx", fx) ||
            !parseScalarField(objectText, "fy", fy) ||
            !parseVector3Field(objectText, "position", record.position) ||
            !parseMatrix3x3Field(objectText, "rotation", record.rotation)) {
            err = "Failed to parse camera entry at index " + std::to_string(idx);
            return std::nullopt;
        }
        record.width = static_cast<uint32_t>(width);
        record.height = static_cast<uint32_t>(height);
        record.fx = static_cast<float>(fx);
        record.fy = static_cast<float>(fy);
        cameras.push_back(record);
        ++idx;
    }

    if (cameras.empty()) {
        err = "No camera entries found in " + jsonPath;
        return std::nullopt;
    }
    return cameras;
}
}

int main(int argc, char** argv) {
    spdlog::set_pattern("[%H:%M:%S] [%^%L%$] %v");

    args::ArgumentParser parser("Vulkan Splatting");
    args::HelpFlag helpFlag{parser, "help", "Display this help menu", {'h', "help"}};
    args::Flag validationLayersFlag{
        parser, "validation-layers", "Enable Vulkan validation layers", {"validation"}
    };
    args::Flag verboseFlag{parser, "verbose", "Enable verbose logging", {'v', "verbose"}};
    args::ValueFlag<uint32_t> physicalDeviceIdFlag{
        parser, "physical-device", "Select physical device by index", {'d', "device"}
    };
    args::Flag immediateSwapchainFlag{
        parser, "immediate-swapchain", "Set swapchain mode to immediate (VK_PRESENT_MODE_IMMEDIATE_KHR)",
        {'i', "immediate-swapchain"}
    };
    args::ValueFlag<uint32_t> widthFlag{parser, "width", "Set window width", {'w', "width"}};
    args::ValueFlag<uint32_t> heightFlag{parser, "height", "Set window height", {'h', "height"}};
    args::ValueFlag<std::string> cameraJsonFlag{
        parser, "camera-json", "Path to cameras.json", {"camera-json"}
    };
    args::ValueFlag<uint32_t> cameraIndexFlag{
        parser, "camera-index", "Camera index inside cameras.json", {"camera-index"}
    };
    args::ValueFlag<uint32_t> maxViewsFlag{
        parser, "max-views", "Render first m camera views from cameras.json", {'m', "max-views"}
    };
    args::ValueFlag<std::string> outputRawFlag{
        parser, "output-raw", "Output raw file path", {"output-raw"}
    };
    args::Positional<std::string> scenePath{parser, "scene", "Path to scene file"};

    try {
        parser.ParseCLI(argc, argv);
    } catch (const args::Completion& e) {
        std::cout << e.what();
        return 0;
    }
    catch (const args::Help&) {
        std::cout << parser;
        return 0;
    }
    catch (const args::ParseError& e) {
        std::cout << e.what() << std::endl;
        std::cout << parser;
        return 1;
    }

    auto pre = env::prefix("VKGS");
    auto validationLayers = pre.register_variable<bool>("VALIDATION_LAYERS");
    auto physicalDeviceId = pre.register_variable<uint8_t>("PHYSICAL_DEVICE");
    auto immediateSwapchain = pre.register_variable<bool>("IMMEDIATE_SWAPCHAIN");
    auto envVars = pre.parse_and_validate();

    if (args::get(verboseFlag)) {
        spdlog::set_level(spdlog::level::debug);
    }

    if (!scenePath) {
        spdlog::critical("Missing required model path: scene");
        std::cout << parser;
        return 1;
    }

    VulkanSplatting::RendererConfiguration config{
        envVars.get_or(validationLayers, false),
        envVars.get(physicalDeviceId).has_value()
            ? std::make_optional(envVars.get(physicalDeviceId).value())
            : std::nullopt,
        envVars.get_or(immediateSwapchain, false),
        args::get(scenePath)
    };
    config.outputRawPath = outputRawFlag ? args::get(outputRawFlag) : "output/output.raw";

    // check that the scene file exists
    if (!std::filesystem::exists(config.scene)) {
        spdlog::critical("File does not exist: {}", config.scene);
        return 0;
    }

    if (validationLayersFlag) {
        config.enableVulkanValidationLayers = args::get(validationLayersFlag);
    }

    if (physicalDeviceIdFlag) {
        config.physicalDeviceId = std::make_optional<uint8_t>(static_cast<uint8_t>(args::get(physicalDeviceIdFlag)));
    }

    if (immediateSwapchainFlag) {
        config.immediateSwapchain = args::get(immediateSwapchainFlag);
    }

    if (cameraJsonFlag) {
        std::vector<CameraJsonRecord> allCameras;
        std::string cameraSourceName;
        const std::string cameraJsonPath = args::get(cameraJsonFlag);
        if (!std::filesystem::exists(cameraJsonPath)) {
            spdlog::critical("Camera json not found: {}", cameraJsonPath);
            return 1;
        }
        std::string parseErr;
        auto allCamerasOpt = loadAllCamerasFromJson(cameraJsonPath, parseErr);
        if (!allCamerasOpt.has_value()) {
            spdlog::critical("{}", parseErr);
            return 1;
        }
        allCameras = std::move(allCamerasOpt.value());
        cameraSourceName = cameraJsonPath;

        if (allCameras.empty()) {
            spdlog::critical("No cameras found from source {}", cameraSourceName);
            return 1;
        }

        std::vector<uint32_t> renderIndices;
        if (cameraIndexFlag) {
            const uint32_t idx = args::get(cameraIndexFlag);
            if (idx >= allCameras.size()) {
                spdlog::critical("camera-index {} is out of range [0, {}]", idx, allCameras.size() - 1);
                return 1;
            }
            renderIndices.push_back(idx);
        } else {
            const uint32_t requestedViews = maxViewsFlag ? args::get(maxViewsFlag) : static_cast<uint32_t>(allCameras.size());
            const uint32_t numViews = std::min<uint32_t>(requestedViews, static_cast<uint32_t>(allCameras.size()));
            if (numViews == 0) {
                spdlog::critical("max-views must be > 0");
                return 1;
            }
            renderIndices.reserve(numViews);
            for (uint32_t i = 0; i < numViews; ++i) {
                renderIndices.push_back(i);
            }
        }

        const std::filesystem::path outputArgPath = outputRawFlag ? args::get(outputRawFlag) : "output";
        std::filesystem::path outputDir = outputArgPath;
        if (outputArgPath.has_extension()) {
            outputDir = outputArgPath.parent_path().empty() ? std::filesystem::path("output") : outputArgPath.parent_path();
        }
        std::error_code ec;
        std::filesystem::create_directories(outputDir, ec);
        if (ec) {
            spdlog::critical("Failed to create output directory {}: {}", outputDir.string(), ec.message());
            return 1;
        }

        config.enableGui = false;
        config.renderJobs.clear();
        config.renderJobs.reserve(renderIndices.size());

        for (uint32_t iter = 0; iter < renderIndices.size(); ++iter) {
            const uint32_t cameraIndex = renderIndices[iter];
            const auto& camera = allCameras[cameraIndex];

            VulkanSplatting::RendererConfiguration::RenderJob job {};
            job.camera.position = camera.position;
            job.camera.rotation = camera.rotation;
            job.camera.fx = camera.fx;
            job.camera.fy = camera.fy;
            job.camera.width = camera.width;
            job.camera.height = camera.height;

            if (cameraIndexFlag && outputRawFlag) {
                job.outputRawPath = outputArgPath.string();
            } else if (cameraIndexFlag && !outputRawFlag) {
                job.outputRawPath = "output/output.raw";
            } else {
                std::string stem = camera.imgName.empty()
                                       ? ("camera_" + std::to_string(cameraIndex))
                                       : std::filesystem::path(camera.imgName).stem().string();
                job.outputRawPath = (outputDir / (stem + ".raw")).string();
            }
            config.renderJobs.push_back(std::move(job));
        }

        const auto& firstJob = config.renderJobs.front();
        const uint32_t width = widthFlag ? args::get(widthFlag) : firstJob.camera.width;
        const uint32_t height = heightFlag ? args::get(heightFlag) : firstJob.camera.height;
        config.cameraOverride = firstJob.camera;
        config.outputRawPath = firstJob.outputRawPath;
        config.window = VulkanSplatting::createGlfwWindow("Vulkan Splatting", width, height);
        spdlog::info("Start rendering {} camera views from {} (single model load)", renderIndices.size(), cameraSourceName);
    } else {
        if (std::find_if(argv, argv + argc, [](const char* s) { return std::string(s) == "--cameras-bin" || std::string(s) == "--images-bin"; }) != argv + argc) {
            spdlog::critical("COLMAP bin loading is disabled. Please use --camera-json <path/to/cameras.json>.");
            return 1;
        }
        const uint32_t width = widthFlag ? args::get(widthFlag) : 1280u;
        const uint32_t height = heightFlag ? args::get(heightFlag) : 720u;
        config.enableGui = true;
        config.cameraOverride = std::nullopt;
        config.renderJobs.clear();
        config.window = VulkanSplatting::createGlfwWindow("Vulkan Splatting", width, height);
        spdlog::info("No --camera-json provided, using default initial camera pose.");
    }
#ifndef DEBUG
    try {
#endif
        auto renderer = VulkanSplatting(config);
        renderer.start();
#ifndef DEBUG
    } catch (const std::exception& e) {
        spdlog::critical(e.what());
        std::cout << e.what() << std::endl;
        return 1;
    }
#endif
    spdlog::info("All camera views rendered successfully.");
    return 0;
}
