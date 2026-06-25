# Synthetic OOM fault injection

This repository provides synthetic OOM injection hooks for recovery validation.
The op and allocator hooks are disabled by default; the HCCL hook keeps its
existing count-based default. All hooks throw `OutOfMemoryError` without
consuming device memory, so a serving layer can exercise its fast recovery path
without leaving real HBM pressure behind.

## Recommended PD inference flow

Use a shared trigger file for both prefill and decode nodes:

```bash
export NPU_OOM_TRIGGER_FILE=/tmp/mindie_oom.trigger
rm -f "${NPU_OOM_TRIGGER_FILE}"
```

Start the PD inference service normally. After the service enters the inference
phase, create the trigger file on each node:

```bash
touch "${NPU_OOM_TRIGGER_FILE}"
```

The next NPU op, allocator request, or HCCL operation in each rank throws a
synthetic OOM once per local device by default. Remove the trigger after the
fault is observed:

```bash
rm -f "${NPU_OOM_TRIGGER_FILE}"
```

This is the preferred mode for Qwen-235B PD separation tests because it does not
depend on hitting a specific HCCL operation count and can be armed only after
model loading and warmup.

## Environment variables

| Variable | Scope | Default | Meaning |
| --- | --- | --- | --- |
| `NPU_OOM_TRIGGER_FILE` | op, allocator, HCCL fallback | unset | Shared trigger file. If it exists, inject OOM. |
| `NPU_OP_OOM_TRIGGER_FILE` | op | unset | Op-only trigger file. Overrides the shared file for op injection. |
| `NPU_ALLOCATOR_OOM_TRIGGER_FILE` | allocator | unset | Allocator-only trigger file. Overrides the shared file for allocator injection. |
| `HCCL_OOM_TRIGGER_FILE` | HCCL | unset | HCCL-only trigger file. Overrides the shared file for HCCL injection. |
| `NPU_OP_OOM_TRIGGER_COUNT` | op | `0` | Inject when per-device op count reaches this value. `0` disables count injection. |
| `NPU_ALLOCATOR_OOM_TRIGGER_COUNT` | allocator | `0` | Inject when per-device allocation count reaches this value. `0` disables count injection. |
| `HCCL_OOM_TRIGGER_COUNT` | HCCL | `6000` | Inject when HCCL call count reaches this value. `0` disables count injection. |
| `NPU_OP_OOM_TRIGGER_MODE` | op | `once` | Set to `always`, `repeat`, or `1` to inject repeatedly. |
| `NPU_ALLOCATOR_OOM_TRIGGER_MODE` | allocator | `once` | Set to `always`, `repeat`, or `1` to inject repeatedly. |
| `HCCL_OOM_TRIGGER_MODE` | HCCL | `once` | Set to `always`, `repeat`, or `1` to inject repeatedly. |

For fast recovery validation, keep the default one-shot mode. Repeated mode is
mainly useful for stress tests where the serving process is expected to restart
or the trigger file is removed immediately.
