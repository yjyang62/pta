#include "torch_npu/csrc/core/npu/PtaOomInjector.h"

#include <atomic>
#include <cstdlib>
#include <string>

#include <c10/util/Exception.h>

#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"

namespace c10_npu {
namespace pta_oom {
namespace {

constexpr int64_t kDefaultTriggerCount = 30004;

std::atomic<int64_t> g_candidate_count{0};
std::atomic<bool> g_oom_triggered{false};

std::string formatAllocSize(uint64_t size)
{
    if (size <= 1024) {
        return std::to_string(size) + " bytes";
    }
    if (size <= 1048576) {
        return std::to_string(size / 1024.0) + " KiB";
    }
    if (size <= 1073741824ULL) {
        return std::to_string(size / 1048576.0) + " MiB";
    }
    return std::to_string(size / 1073741824.0) + " GiB";
}

int64_t getTriggerCount()
{
    const static int64_t trigger_count = []() -> int64_t {
        char *env_val = c10_npu::option::get_and_log_env("PTA_OOM_TRIGGER_COUNT");
        return (env_val != nullptr) ? strtol(env_val, nullptr, 10) : kDefaultTriggerCount;
    }();
    return trigger_count;
}

int64_t getOpTriggerCount()
{
    const static int64_t trigger_count = []() -> int64_t {
        char *env_val = c10_npu::option::get_and_log_env("PTA_OOM_OP_TRIGGER_COUNT");
        if (env_val != nullptr) {
            return strtol(env_val, nullptr, 10);
        }
        return getTriggerCount();
    }();
    return trigger_count;
}

bool shouldIgnoreCapture()
{
    const static bool ignore_capture = []() -> bool {
        char *env_val = c10_npu::option::get_and_log_env("PTA_OOM_DURING_CAPTURE");
        return (env_val != nullptr) && (strtol(env_val, nullptr, 10) != 0);
    }();
    return ignore_capture;
}

bool isCaptureActive()
{
    if (shouldIgnoreCapture()) {
        return false;
    }
    return c10_npu::currentStreamCaptureStatus() != c10_npu::CaptureStatus::None;
}

bool shouldTrigger(int64_t trigger_count, int64_t current_count, const std::string &message)
{
    if (current_count > trigger_count && current_count < trigger_count + 2) {
        g_oom_triggered.store(true);
        TORCH_CHECK_WITH(OutOfMemoryError, false, message.c_str());
        return true;
    }
    return false;
}

} // namespace

void maybeThrowAllocOom(int device, size_t size)
{
    if (size == 0 || g_oom_triggered.load() || isCaptureActive()) {
        return;
    }

    const int64_t trigger_count = getTriggerCount();
    if (trigger_count <= 0) {
        return;
    }

    const int64_t current_count = ++g_candidate_count;
    shouldTrigger(
        trigger_count,
        current_count,
        std::string("NPU out of memory. Injected PTA alloc OOM after ") +
        std::to_string(current_count) + " PTA events. Tried to allocate " +
        formatAllocSize(size) + " on NPU " + std::to_string(device) + ".");
}

void maybeThrowOpOom(const std::string &op_name)
{
    if (g_oom_triggered.load() || isCaptureActive()) {
        return;
    }

    const int64_t trigger_count = getOpTriggerCount();
    if (trigger_count <= 0) {
        return;
    }

    const int64_t current_count = ++g_candidate_count;
    shouldTrigger(
        trigger_count,
        current_count,
        std::string("NPU out of memory. Injected PTA op OOM after ") +
        std::to_string(current_count) + " PTA events, last op is " + op_name + ".");
}

} // namespace pta_oom
} // namespace c10_npu
