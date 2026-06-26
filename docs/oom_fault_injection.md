# Synthetic OOM fault injection

This repository provides synthetic OOM injection hooks for recovery validation.
The op and allocator hooks are disabled by default; the HCCL hook keeps its
existing count-based default. All hooks throw `OutOfMemoryError` without
consuming device memory, so a serving layer can exercise its fast recovery path
without leaving real HBM pressure behind.

## Recommended PD inference flow

Use op-level automatic triggering for both prefill and decode nodes. Add these
values to the MindIE launch environment:

```text
HCCL_OOM_COUNT=1
NPU_ALLOCATOR_OOM_TRIGGER_COUNT=0
HCCL_OOM_TRIGGER_COUNT=0
```

Start the PD inference service normally with this environment. No extra command
is required during inference. The hook counts NPU operations inside each
process; once the configured count is reached, each rank keeps throwing a
synthetic OOM on every subsequent operation for that device.

This is the preferred mode for Qwen-235B PD separation tests because it does not
depend on hitting a specific HCCL operation and does not require real HBM
pressure. Increase `HCCL_OOM_COUNT` if the fault must be delayed until
after warmup requests.

## Environment variables

| Variable | Scope | Default | Meaning |
| --- | --- | --- | --- |
| `NPU_OOM_TRIGGER_FILE` | op, allocator, HCCL fallback | unset | Shared trigger file. If it exists, inject OOM. |
| `NPU_OP_OOM_TRIGGER_FILE` | op | unset | Op-only trigger file. Overrides the shared file for op injection. |
| `NPU_ALLOCATOR_OOM_TRIGGER_FILE` | allocator | unset | Allocator-only trigger file. Overrides the shared file for allocator injection. |
| `HCCL_OOM_TRIGGER_FILE` | HCCL | unset | HCCL-only trigger file. Overrides the shared file for HCCL injection. |
| `NPU_OOM_TRIGGER_COUNT` | op, allocator, HCCL fallback | `0` for op/allocator, HCCL keeps its legacy default if unset | Shared automatic trigger count. |
| `HCCL_OOM_COUNT` | op | `0` | Inject when per-device op count reaches this value. `0` disables count injection. After the count is reached, injection continues on every subsequent op. |
| `NPU_ALLOCATOR_OOM_TRIGGER_COUNT` | allocator | `0` | Inject when per-device allocation count reaches this value. `0` disables count injection. After the count is reached, injection continues on every subsequent allocation. |
| `HCCL_OOM_TRIGGER_COUNT` | HCCL | `6000` | Inject when HCCL call count reaches this value. `0` disables count injection. After the count is reached, injection continues on every subsequent HCCL call. |
| `NPU_OOM_TRIGGER_MODE` | op, allocator, HCCL fallback | `once` | Shared one-shot or repeated mode for file-based triggers only. |
| `NPU_OP_OOM_TRIGGER_MODE` | op | `once` | Set to `always`, `repeat`, or `1` to inject repeatedly from file-based triggers. |
| `NPU_ALLOCATOR_OOM_TRIGGER_MODE` | allocator | `once` | Set to `always`, `repeat`, or `1` to inject repeatedly from file-based triggers. |
| `HCCL_OOM_TRIGGER_MODE` | HCCL | `once` | Set to `always`, `repeat`, or `1` to inject repeatedly from file-based triggers. |

Count-based triggers always inject OOM on every subsequent operation once the
configured count is reached. The `*_TRIGGER_MODE` variables only affect
file-based triggers. Trigger files remain available for manual debugging, but
automatic count-based triggering is the recommended PD inference path.
