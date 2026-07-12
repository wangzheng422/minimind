#!/usr/bin/env python
"""Step-by-step Google Colab runner for the MiniMind learning path.

Run this from the repository root or pass --root explicitly.  The runner uses
the repository's existing trainer and model code; it does not fork training
logic into a second implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path


PROFILES = {
    "micro": {
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "batch_size": 8,
        "accumulation_steps": 1,
        "max_seq_len": 128,
        "epochs": 1,
        "data_suffix": "micro",
    },
    "zero": {
        "hidden_size": 768,
        "num_hidden_layers": 8,
        "batch_size": 32,
        "accumulation_steps": 8,
        "max_seq_len": 768,
        "epochs": 1,
        "data_suffix": "mini",
    },
}
DATASET_REPOSITORY = "jingyaogong/minimind_dataset"
DATASET_FILES = ("pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl")


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def repository_root(value: str | None) -> Path:
    root = Path(value).expanduser().resolve() if value else Path(__file__).resolve().parents[1]
    required = (root / "model" / "model_minimind.py", root / "trainer" / "train_pretrain.py")
    if not all(path.exists() for path in required):
        fail(f"{root} is not a MiniMind repository root")
    return root


def require_file(path: Path, action: str) -> None:
    if not path.exists():
        fail(f"Missing {path}. Run {action} first.")


def profile(name: str) -> dict[str, int | str]:
    return PROFILES[name]


def profile_weight(name: str, stage: str) -> str:
    return f"learn_{name}_{stage}"


def profile_data(root: Path, name: str, stage: str) -> Path:
    suffix = profile(name)["data_suffix"]
    return root / "dataset" / f"{stage}_t2t_{suffix}.jsonl"


def command_setup(args: argparse.Namespace, root: Path) -> None:
    venv = Path(args.venv).expanduser().resolve() if args.venv else root / ".colab-venv"
    venv_python = venv / "bin" / "python"
    if not venv_python.exists():
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(venv)],
            check=True,
        )
    host_pip = shutil.which("pip")
    pip_command = [host_pip] if host_pip else [sys.executable, "-m", "pip"]
    subprocess.run(
        [*pip_command, "--python", str(venv_python), "install", "--upgrade", "pip", "-r", str(root / "requirements.txt")],
        check=True,
    )
    print(json.dumps({"venv_python": str(venv_python), "next": f"{venv_python} colab/minimind_colab.py preflight"}, ensure_ascii=False, indent=2))


def command_preflight(args: argparse.Namespace, root: Path) -> None:
    import psutil
    import torch

    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "memory_gib": round(properties.total_memory / 1024 ** 3, 2),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "cuda_runtime": torch.version.cuda,
        }
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "repo": str(root),
        "torch": torch.__version__,
        "gpu": gpu,
        "ram_gib": round(psutil.virtual_memory().total / 1024 ** 3, 2),
        "disk_free_gib": round(shutil.disk_usage(root).free / 1024 ** 3, 2),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_a100 and (gpu is None or "A100" not in gpu["name"].upper()):
        fail("An NVIDIA A100 was required but is not the active accelerator.")
    if gpu is None:
        fail("No CUDA GPU is available. In Colab select a GPU runtime before training.")
    if not gpu["bf16_supported"]:
        fail("The active GPU does not report bfloat16 support; use a BF16-capable CUDA GPU such as an L4 or A100.")


def command_download(args: argparse.Namespace, root: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        fail(f"huggingface_hub is unavailable ({error}); run setup first.")
    target = root / "dataset"
    target.mkdir(exist_ok=True)
    files = DATASET_FILES if args.all else (f"{args.stage}_t2t_mini.jsonl",)
    for filename in files:
        path = hf_hub_download(
            repo_id=DATASET_REPOSITORY,
            repo_type="dataset",
            filename=filename,
            local_dir=target,
        )
        print(f"downloaded {filename}: {path}")


def command_make_micro(args: argparse.Namespace, root: Path) -> None:
    for stage in ("pretrain", "sft"):
        source = root / "dataset" / f"{stage}_t2t_mini.jsonl"
        destination = profile_data(root, "micro", stage)
        require_file(source, "download --all")
        copied = 0
        with source.open("r", encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
            for line in reader:
                if line.strip():
                    writer.write(line)
                    copied += 1
                if copied >= args.rows:
                    break
        if copied == 0:
            fail(f"{source} did not contain JSONL records")
        print(f"wrote {copied} records to {destination}")


def load_minimind_modules(root: Path):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    from transformers import AutoTokenizer
    from dataset.lm_dataset import PretrainDataset, SFTDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    return torch, AutoTokenizer, PretrainDataset, SFTDataset, MiniMindConfig, MiniMindForCausalLM


def make_model(root: Path, name: str, stage: str, device: str, load_weight: bool):
    torch, AutoTokenizer, _, _, MiniMindConfig, MiniMindForCausalLM = load_minimind_modules(root)
    settings = profile(name)
    config = MiniMindConfig(hidden_size=settings["hidden_size"], num_hidden_layers=settings["num_hidden_layers"])
    model = MiniMindForCausalLM(config)
    if load_weight:
        weight = root / "out" / f"{profile_weight(name, stage)}_{settings['hidden_size']}.pth"
        require_file(weight, f"train --profile {name} --stage {stage}")
        model.load_state_dict(torch.load(weight, map_location="cpu"), strict=False)
    tokenizer = AutoTokenizer.from_pretrained(root / "model")
    return torch, tokenizer, model.to(device), config


def command_lesson(args: argparse.Namespace, root: Path) -> None:
    torch, AutoTokenizer, PretrainDataset, SFTDataset, MiniMindConfig, MiniMindForCausalLM = load_minimind_modules(root)
    settings = profile(args.profile)
    tokenizer = AutoTokenizer.from_pretrained(root / "model")
    if args.topic == "tokenizer":
        text = args.text
        encoded = tokenizer(text, add_special_tokens=False).input_ids
        print(json.dumps({
            "text": text,
            "input_ids": encoded,
            "raw_tokens": tokenizer.convert_ids_to_tokens(encoded),
            "decoded": tokenizer.decode(encoded),
            "special_ids": {"bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id, "pad": tokenizer.pad_token_id},
            "chat_template": tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True),
        }, ensure_ascii=False, indent=2))
        return
    if args.topic == "model":
        config = MiniMindConfig(hidden_size=settings["hidden_size"], num_hidden_layers=settings["num_hidden_layers"])
        model = MiniMindForCausalLM(config).eval()
        input_ids = torch.tensor([[tokenizer.bos_token_id, *tokenizer(args.text, add_special_tokens=False).input_ids[:7]]])
        with torch.inference_mode():
            output = model(input_ids)
        print(json.dumps({
            "profile": args.profile,
            "input_ids_shape": list(input_ids.shape),
            "embedding_shape": list(model.model.embed_tokens(input_ids).shape),
            "q_projection_weight_shape": list(model.model.layers[0].self_attn.q_proj.weight.shape),
            "k_projection_weight_shape": list(model.model.layers[0].self_attn.k_proj.weight.shape),
            "v_projection_weight_shape": list(model.model.layers[0].self_attn.v_proj.weight.shape),
            "logits_shape": list(output.logits.shape),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "embedding_and_lm_head_share_storage": model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr(),
        }, ensure_ascii=False, indent=2))
        return
    stage = "pretrain" if args.topic == "pretrain-labels" else "sft"
    data_path = profile_data(root, args.profile, stage)
    require_file(data_path, "download --all" if args.profile == "zero" else f"make-micro --rows {args.rows}")
    dataset = PretrainDataset(data_path, tokenizer, max_length=settings["max_seq_len"]) if stage == "pretrain" else SFTDataset(data_path, tokenizer, max_length=settings["max_seq_len"])
    input_ids, labels = dataset[args.index]
    rows = []
    for position in range(min(args.rows, input_ids.numel() - 1)):
        target = int(labels[position + 1])
        rows.append({
            "position": position,
            "input": tokenizer.decode([int(input_ids[position])]),
            "next_token_target": None if target == -100 else tokenizer.decode([target]),
            "loss": target != -100,
        })
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def command_one_step(args: argparse.Namespace, root: Path) -> None:
    torch, AutoTokenizer, PretrainDataset, SFTDataset, _, _ = load_minimind_modules(root)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    settings = profile(args.profile)
    data_path = profile_data(root, args.profile, args.stage)
    require_file(data_path, "download --all" if args.profile == "zero" else f"make-micro --rows {args.rows}")
    tokenizer = AutoTokenizer.from_pretrained(root / "model")
    dataset_class = PretrainDataset if args.stage == "pretrain" else SFTDataset
    dataset = dataset_class(data_path, tokenizer, max_length=settings["max_seq_len"])
    batch = [dataset[index] for index in range(min(settings["batch_size"], len(dataset)))]
    input_ids = torch.stack([item[0] for item in batch]).to(device)
    labels = torch.stack([item[1] for item in batch]).to(device)
    previous_stage = "pretrain" if args.stage == "sft" else args.stage
    torch, _, model, _ = make_model(root, args.profile, previous_stage, device, load_weight=args.stage == "sft")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
    model.train()
    with autocast:
        output = model(input_ids, labels=labels)
        loss_before = output.loss + output.aux_loss
    loss_before.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode(), autocast:
        loss_after = model(input_ids, labels=labels).loss
    result = {
        "stage": args.stage,
        "profile": args.profile,
        "device": device,
        "batch_shape": list(input_ids.shape),
        "logits_shape": list(output.logits.shape),
        "loss_before": float(loss_before),
        "grad_norm_before_clip": grad_norm,
        "loss_after_same_batch": float(loss_after),
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 3) if device.startswith("cuda") else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def trainer_command(root: Path, name: str, stage: str, args: argparse.Namespace) -> list[str]:
    settings = profile(name)
    data_path = profile_data(root, name, stage)
    require_file(data_path, "download --all" if name == "zero" else f"make-micro --rows {args.rows}")
    script = root / "trainer" / ("train_pretrain.py" if stage == "pretrain" else "train_full_sft.py")
    command = [
        sys.executable, str(script),
        "--save_dir", str(root / "out"),
        "--save_weight", profile_weight(name, stage),
        "--epochs", str(args.epochs or settings["epochs"]),
        "--batch_size", str(args.batch_size or settings["batch_size"]),
        "--accumulation_steps", str(settings["accumulation_steps"]),
        "--max_seq_len", str(args.max_seq_len or settings["max_seq_len"]),
        "--data_path", str(data_path),
        "--hidden_size", str(settings["hidden_size"]),
        "--num_hidden_layers", str(settings["num_hidden_layers"]),
        "--dtype", "bfloat16",
        "--num_workers", str(args.num_workers),
        "--from_resume", "1" if args.resume else "0",
        "--use_compile", "1" if args.compile else "0",
    ]
    command.extend(["--from_weight", "none" if stage == "pretrain" else profile_weight(name, "pretrain")])
    return command


def command_train(args: argparse.Namespace, root: Path) -> None:
    stages = ("pretrain", "sft") if args.stage == "all" else (args.stage,)
    for stage in stages:
        command = trainer_command(root, args.profile, stage, args)
        print("RUN:", " ".join(command), flush=True)
        log_path = root / "logs" / f"{profile_weight(args.profile, stage)}.log"
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"RUN: {' '.join(command)}\n")
            process = subprocess.Popen(command, cwd=root / "trainer", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            return_code = process.wait()
            log.write(f"EXIT_CODE: {return_code}\n")
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)


def command_tokenizer_experiment(args: argparse.Namespace, root: Path) -> None:
    require_file(root / "dataset" / "sft_t2t_mini.jsonl", "download --stage sft")
    command = [sys.executable, str(root / "trainer" / "train_tokenizer.py")]
    log_path = root / "logs" / "tokenizer_experiment.log"
    log_path.parent.mkdir(exist_ok=True)
    print("RUN:", " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=root / "trainer", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
        log.write(f"EXIT_CODE: {return_code}\n")
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    print("Learning-only tokenizer output: model_learn_tokenizer/. Do not use it with MiniMind weights.")


def command_infer(args: argparse.Namespace, root: Path) -> None:
    torch, tokenizer, model, _ = make_model(root, args.profile, args.stage, args.device or ("cuda" if torch.cuda.is_available() else "cpu"), load_weight=True)
    device = next(model.parameters()).device
    model.eval()
    if args.stage == "pretrain":
        prompt = tokenizer.bos_token + args.prompt
    else:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True, open_thinking=False)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            inputs=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    print(completion)


def copy_changed(source: Path, destination: Path) -> int:
    copied = 0
    if not source.exists():
        return copied
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or path.stat().st_mtime_ns > target.stat().st_mtime_ns or path.stat().st_size != target.stat().st_size:
            shutil.copy2(path, target)
            copied += 1
    return copied


def command_backup(args: argparse.Namespace, root: Path) -> None:
    target = Path(args.drive_dir).expanduser().resolve() / "minimind-colab"
    copied = sum(copy_changed(root / name, target / name) for name in ("out", "checkpoints", "logs"))
    if args.include_data:
        copied += copy_changed(root / "dataset", target / "dataset")
    print(json.dumps({"backup": str(target), "files_copied": copied}, ensure_ascii=False))


def command_restore(args: argparse.Namespace, root: Path) -> None:
    source = Path(args.drive_dir).expanduser().resolve() / "minimind-colab"
    if not source.exists():
        if args.allow_missing:
            print(json.dumps({"restore": str(source), "files_copied": 0, "status": "no saved state"}, ensure_ascii=False))
            return
        fail(f"No saved Colab state at {source}")
    copied = sum(copy_changed(source / name, root / name) for name in ("out", "checkpoints", "logs"))
    if args.include_data:
        copied += copy_changed(source / "dataset", root / "dataset")
    print(json.dumps({"restore": str(source), "files_copied": copied}, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="MiniMind repository root; defaults to this file's parent repository")
    parser.add_argument("--venv", help="Virtual environment path used by setup")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="Create a Colab venv that reuses the installed CUDA PyTorch, then install project dependencies")
    preflight = commands.add_parser("preflight", help="Print GPU, BF16, RAM, and disk checks")
    preflight.add_argument("--require-a100", action="store_true")
    download = commands.add_parser("download", help="Download the recommended mini datasets from Hugging Face")
    download_group = download.add_mutually_exclusive_group(required=True)
    download_group.add_argument("--all", action="store_true")
    download_group.add_argument("--stage", choices=("pretrain", "sft"))
    micro = commands.add_parser("make-micro", help="Stream a small learning dataset from the mini JSONL files")
    micro.add_argument("--rows", type=int, default=2048)
    commands.add_parser("tokenizer-experiment", help="Run the repository's learning-only BPE tokenizer script; do not use its output with MiniMind weights")
    lesson = commands.add_parser("lesson", help="Inspect tokenizer, model tensors, and loss masks")
    lesson.add_argument("topic", choices=("tokenizer", "model", "pretrain-labels", "sft-labels"))
    lesson.add_argument("--profile", choices=PROFILES, default="micro")
    lesson.add_argument("--text", default="语言模型通过预测下一个 token 来学习文本。")
    lesson.add_argument("--index", type=int, default=0)
    lesson.add_argument("--rows", type=int, default=32)
    one_step = commands.add_parser("one-step", help="Run one real forward/backward/AdamW update and report the metrics")
    one_step.add_argument("--profile", choices=PROFILES, default="micro")
    one_step.add_argument("--stage", choices=("pretrain", "sft"), required=True)
    one_step.add_argument("--device")
    one_step.add_argument("--learning-rate", type=float, default=1e-4)
    one_step.add_argument("--rows", type=int, default=2048)
    train = commands.add_parser("train", help="Run the repository's native training scripts")
    train.add_argument("--profile", choices=PROFILES, default="micro")
    train.add_argument("--stage", choices=("pretrain", "sft", "all"), required=True)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--max-seq-len", type=int)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--compile", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--rows", type=int, default=2048)
    infer = commands.add_parser("infer", help="Run a single non-interactive generation from a trained local weight")
    infer.add_argument("--profile", choices=PROFILES, default="micro")
    infer.add_argument("--stage", choices=("pretrain", "sft"), required=True)
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--max-new-tokens", type=int, default=128)
    infer.add_argument("--temperature", type=float, default=0.7)
    infer.add_argument("--top-p", type=float, default=0.9)
    infer.add_argument("--device")
    backup = commands.add_parser("backup", help="Incrementally copy checkpoints and outputs to a mounted Google Drive directory")
    backup.add_argument("--drive-dir", required=True)
    backup.add_argument("--include-data", action="store_true")
    restore = commands.add_parser("restore", help="Restore a previously backed-up Colab state into this repository before native --resume training")
    restore.add_argument("--drive-dir", required=True)
    restore.add_argument("--include-data", action="store_true")
    restore.add_argument("--allow-missing", action="store_true", help="Treat an empty Google Drive backup location as a new run")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = repository_root(args.root)
    handlers = {
        "setup": command_setup,
        "preflight": command_preflight,
        "download": command_download,
        "make-micro": command_make_micro,
        "tokenizer-experiment": command_tokenizer_experiment,
        "lesson": command_lesson,
        "one-step": command_one_step,
        "train": command_train,
        "infer": command_infer,
        "backup": command_backup,
        "restore": command_restore,
    }
    handlers[args.command](args, root)


if __name__ == "__main__":
    main()
