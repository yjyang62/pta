#pragma once

#include <cstddef>
#include <string>

namespace c10_npu {
namespace pta_oom {

// Trigger injected PTA OOM after PTA_OOM_TRIGGER_COUNT tensor allocations.
// Useful for prefill (P) and other allocation-heavy phases.
void maybeThrowAllocOom(int device, size_t size);

// Trigger injected PTA OOM after PTA_OOM_OP_TRIGGER_COUNT NPU op executions.
// Useful for decode (D) and graph-replay inference where allocations are reused.
void maybeThrowOpOom(const std::string &op_name);

} // namespace pta_oom
} // namespace c10_npu
