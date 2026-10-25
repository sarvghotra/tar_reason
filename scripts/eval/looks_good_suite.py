"""Prepare (and optionally submit) a looks_good evaluation suite in the uv environment.

  python scripts/eval/looks_good_suite.py --suite NAME --config sft=PATH:sft --config ck300=PATH \\
      --decoding sample --max_refinements 4 [--benchmarks geneval2 tiif genai] [--submit]

Per configuration and benchmark the stages are: generation shards -> finalize ->
replay of images 1..max_refinements (image max_refinements+1 is the final cap) ->
replay summary -> instruction-judge scores (TIIF, GenAI) -> score summary, then a
report. The commands match the earlier derived suites (run_shard.py, seed 421,
gen_batch_size 16, SDPA); what changes is the placement:

  gen-large  one GPU of TAR_GPU_LARGE per task   GenEval2 generation (~47 GB), GenAI judge (~52 GB)
  gen        one GPU of TAR_GPU_GEN per task     TIIF/GenAI generation (~28 GB), GenEval2 replay
                                                 (~22 GB), TIIF judge (~17 GB)
  light      LIGHT_PACK tasks share one GPU      TIIF/GenAI replay (~4.5 GB each)
  tiny       one TAR_GPU_TINY GPU                finalize (needs CUDA, uses no GPU memory)
  cpu        TAR_CPU_ACCOUNT, no GPU             summaries and the report

instead of whole 4-GPU nodes. Shards are seeded by batch start index, so placement
does not change samples; the GPU model does change numerics, so every GPU stage
of a suite uses one model (pass --gpu_type to choose it). Code is frozen into the
suite directory; nothing is submitted without --submit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
SHARDS = dict(gen=dict(geneval2=8, tiif=8, genai=12), replay=4, score=2)
GENEVAL2 = 'benchmarks/geneval2/a6e82d2289e8d418f27f0adee77908b07060eea3/geneval2_data.jsonl'
INSTRUCTION = 'benchmarks/instruction_20260916/{}.json'
JUDGES = dict(geneval2='models/judges/Qwen3-VL-8B-Instruct', tiif='models/judges/Qwen2.5-VL-7B-Instruct',
              genai='models/judges/Qwen3.5-27B')
TIME = dict(gen='02:30:00', finalize='00:45:00', replay='02:30:00', score='01:15:00', cpu='00:30:00')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def stage_list(benchmarks):
    out = []
    for bench in benchmarks:
        out += [f'gen-{bench}', f'gen-{bench}-finalize', f'replay-{bench}', f'replay-{bench}-summary']
        if bench != 'geneval2':
            out += [f'score-{bench}', f'score-{bench}-summary']
    return out + ['report']


def dependencies(name, benchmarks):
    if name == 'report':
        return [f'gen-geneval2-finalize' if b == 'geneval2' else f'score-{b}-summary' for b in benchmarks] + \
               (['replay-geneval2-summary'] if 'geneval2' in benchmarks else [])
    if name.endswith(('-finalize', '-summary')):
        return [name.rsplit('-', 1)[0]]
    stage, bench = name.split('-')[:2]
    if stage == 'gen':
        return []
    if stage == 'replay':
        return [f'gen-{bench}-finalize']
    return [f'gen-{bench}-finalize', f'replay-{bench}-summary']


def profile_of(name):
    stage, bench = name.split('-')[:2] if name != 'report' else ('report', None)
    if name.endswith('-finalize'):
        return 'tiny'
    if name.endswith('-summary') or name == 'report':
        return 'cpu'
    if stage == 'gen':
        return 'gen-large' if bench == 'geneval2' else 'gen'
    if stage == 'replay':
        return 'gen' if bench == 'geneval2' else 'light'
    return 'gen-large' if bench == 'genai' else 'gen'


class Suite:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.results)/'evaluations'/args.suite
        self.code = self.root/'code'
        self.logs = Path(args.results)/'logs'/args.suite
        self.shared = Path(args.shared)
        self.python = str(Path(args.python))
        self.images = args.max_refinements + 1

    def module_args(self, config, bench):
        checkpoint, kind = config['checkpoint'], config['type']
        common = ['--checkpoint', checkpoint, '--checkpoint_type', kind, '--protocol', 'looks_good',
                  '--max_refinements', str(self.args.max_refinements), '--reset_context_per_turn',
                  '--decoding', self.args.decoding, '--verdict_rule', 'correction',
                  '--benchmark', str(self.shared/(GENEVAL2 if bench == 'geneval2' else INSTRUCTION.format(bench))),
                  '--output_dir', str(self.root/config['name']/f'{bench}_final'),
                  '--code_manifest', str(self.code/'manifest.json'),
                  '--ar_path', str(self.shared/'models/tar/ar_dtok_lp_512px.pth'),
                  '--encoder_path', str(self.shared/'models/tar/ta_tok.pth'),
                  '--decoder_path', str(self.shared/'models/tar/vq_ds16_t2i.pt'),
                  '--seed', '421', '--gen_batch_size', '16', '--reflect_tokens', '128', '--max_seq_len', '4096',
                  '--cfg_scale', '4', '--attn_implementation', 'sdpa', '--report_to', 'wandb']
        if bench == 'geneval2':
            common += ['--reward_model', str(self.shared/JUDGES['geneval2']), '--reward_python', self.python]
        return ('llava.train.rl.eval_geneval2' if bench == 'geneval2'
                else 'llava.train.rl.generate_instruction_benchmarks'), common

    def tasks(self, config, name):
        """(environment, argv) per task of a stage; argv runs from the frozen code."""
        c, py = config['name'], self.python
        parts = name.split('-')
        stage, bench = (parts[0], parts[1]) if name != 'report' else ('report', None)
        out_dir = lambda tag: str(self.root/c/f'{bench}_{tag}')
        run_name = f'lg_{self.args.suite_tag}_{c}_{bench}'
        if stage == 'gen':
            module, margs = self.module_args(config, bench)
            shards = SHARDS['gen'][bench]
            shard = [str(self.code/'scripts/eval/run_shard.py'), module] + margs
            if name.endswith('-finalize'):
                return [(dict(SHARD_MODE='finalize', RANK='0', WORLD_SIZE=str(shards), LOCAL_RANK='0',
                              RUN_NAME=run_name, WANDB_RUN_ID=run_name), [py] + shard)]
            return [(dict(SHARD_MODE='work', RANK=str(k), WORLD_SIZE=str(shards), LOCAL_RANK='0',
                          RUN_NAME=run_name, WANDB_RUN_ID=run_name), [py] + shard) for k in range(shards)]
        ordinals = range(1, self.args.max_refinements + 1)  # the cap image is the final directory
        if stage == 'replay':
            if name.endswith('-summary'):
                return [({}, [py, '-m', 'llava.train.rl.intermediate_eval', '--output_dir', out_dir(f'image{k}'),
                              '--summary_only']) for k in ordinals]
            return [(dict(RANK=str(s), WORLD_SIZE=str(SHARDS['replay']), LOCAL_RANK='0'),
                     [py, '-m', 'llava.train.rl.intermediate_eval', '--source_dir', out_dir('final'),
                      '--output_dir', out_dir(f'image{k}'), '--image_ordinal', str(k), '--reward_python', py])
                    for k in ordinals for s in range(SHARDS['replay'])]
        if stage == 'score':
            dirs = [out_dir('final')] + [out_dir(f'image{k}') for k in ordinals]
            base = lambda d: [py, '-m', 'llava.train.rl.score_instruction_benchmarks', '--output_dir', d,
                              '--model', str(self.shared/JUDGES[bench]), '--seed', '421', '--report_to', 'wandb']
            if name.endswith('-summary'):
                return [({}, base(d) + ['--summary_only']) for d in dirs]
            return [(dict(RANK=str(s), WORLD_SIZE=str(SHARDS['score']), LOCAL_RANK='0'), base(d))
                    for d in dirs for s in range(SHARDS['score'])]
        return [({}, [py, '-m', 'scripts.rl_ft.report_intermediate_eval', '--config',
                      str(self.root/c/'comparison.json'), '--report_to', 'wandb'])]

    def resources(self, profile):
        a = self.args
        gpu = {'gen-large': a.gpu_large, 'gen': a.gpu_gen, 'light': a.gpu_gen, 'tiny': a.gpu_tiny}.get(profile)
        if profile == 'cpu':
            lines = [f'--account={a.cpu_account}', '--ntasks=1', '--cpus-per-task=2', '--mem=32G',
                     f'--time={TIME["cpu"]}']
        else:
            lines = [f'--account={a.gpu_account}', '--ntasks=1', f'--gpus-per-task={gpu}:1',
                     f'--cpus-per-task={8 if profile == "light" else 6}',
                     f'--mem-per-gpu={"96G" if profile == "light" else "64G"}']
        return lines

    def array_size(self, profile, n_tasks):
        pack = self.args.light_pack if profile == 'light' else 1
        return (n_tasks + pack - 1) // pack, pack

    def script(self, config, name):
        profile = profile_of(name)
        tasks = self.tasks(config, name)
        elements, pack = self.array_size(profile, len(tasks))
        stage_time = (TIME['finalize'] if name.endswith('-finalize') else TIME['cpu'] if profile == 'cpu'
                      else TIME[name.split('-')[0]])
        c = config['name']
        log = self.logs/c
        header = [f'--job-name=lg_{self.args.suite_tag}_{c}-{name}'[:64], *self.resources(profile),
                  f'--time={stage_time}' if profile != 'cpu' else None,
                  f'--output={log}/{name}-%A_%a.out', f'--error={log}/{name}-%A_%a.err']
        is_array = profile not in ('cpu', 'tiny') and len(tasks) > 1
        if is_array:
            header.append(f'--array=0-{elements - 1}')
        lines = ['#!/bin/bash'] + [f'#SBATCH {h}' for h in header if h] + [
            'set -euo pipefail',
            f'source {shlex.quote(str(self.root/"env.sh"))}',
            f'export WANDB_DIR={shlex.quote(str(self.root/c/"wandb"))} WANDB_DATA_DIR={shlex.quote(str(self.root/c/"wandb/artifacts"))}',
            'mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR"',
            'run_task() {', '    case "$1" in']
        for index, (env, argv) in enumerate(tasks):
            assignments = ' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())
            lines.append(f'    {index}) {("env " + assignments + " ") if assignments else ""}{shlex.join(argv)} ;;')
        lines += ['    *) echo "bad task $1" >&2; return 1 ;;', '    esac', '}']
        if not is_array:
            lines += [f'for task in $(seq 0 {len(tasks) - 1}); do run_task "$task"; done']
        elif pack == 1:
            lines += ['run_task "$SLURM_ARRAY_TASK_ID"']
        else:
            # Light stages: PACK independent single-GPU tasks share this element's GPU.
            lines += [f'PACK={pack} TASKS={len(tasks)} PIDS=() STATUS=0',
                      'for slot in $(seq 0 $((PACK - 1))); do',
                      '    task=$((SLURM_ARRAY_TASK_ID * PACK + slot))',
                      '    (( task < TASKS )) || continue',
                      f'    run_task "$task" > {shlex.quote(str(log))}/{name}-${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}-slot$slot.out 2>&1 &',
                      '    PIDS+=("$!")',
                      'done',
                      'for pid in "${PIDS[@]}"; do wait "$pid" || STATUS=1; done',
                      'exit $STATUS']
        return '\n'.join(lines) + '\n', len(tasks), elements, profile

    def comparison(self, config):
        c = config['name']
        ordinals = {f'image{k}': k for k in range(1, self.images + 1)}
        benchmarks = {b: {'final': str(self.root/c/f'{b}_final'),
                          **{f'image{k}': str(self.root/c/(f'{b}_image{k}' if k < self.images else f'{b}_final'))
                             for k in range(1, self.images + 1)}} for b in self.args.benchmarks}
        refinements = self.args.max_refinements
        return dict(model=f'{c}: looks_good, reset context, <= {refinements} refinements, {self.args.decoding} decoding',
                    image_ordinals=ordinals, benchmarks=benchmarks,
                    final_note=('Final = the looks_good episode result: stop on a "looks good" Correction verdict, '
                                f'otherwise a forced refinement, at most {refinements} refinements '
                                f'(image {self.images} is the cap).'),
                    checkpoint_step=config.get('step'))

    def prepare(self):
        if self.root.exists():
            raise SystemExit(f'{self.root} exists; a suite is prepared once')
        self.code.mkdir(parents=True)
        files = subprocess.run(['git', '-C', str(REPO), 'ls-files', '-z', '-co', '--exclude-standard'],
                               check=True, capture_output=True).stdout.decode().split('\0')
        manifest = {}
        for rel in filter(None, files):
            source = REPO/rel
            if source.is_file() and not source.is_symlink():
                (self.code/rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, self.code/rel)
                manifest[rel] = sha256(source)
        git = lambda *a: subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, text=True).stdout
        (self.code/'manifest.json').write_text(json.dumps(dict(
            git_head=git('rev-parse', 'HEAD').strip(), git_status=git('status', '--porcelain=v1'),
            files=manifest), indent=1, sort_keys=True))
        (self.root/'env.sh').write_text('\n'.join([
            '# Shared by every stage of this suite (uv environment, frozen code).',
            f'export TAR_ROOT={shlex.quote(self.args.tar_root)} TAR_SHARED={shlex.quote(str(self.shared))}',
            f'export VIRTUAL_ENV={shlex.quote(str(Path(self.python).parent.parent))}',
            'export PATH="$VIRTUAL_ENV/bin:$PATH"',
            f'export PYTHONPATH={shlex.quote(str(self.code))}',
            'export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=6',
            f'export HF_HUB_CACHE={shlex.quote(str(self.shared/"hf_hub"))} HF_HUB_OFFLINE=1',
            f'export TAR_VISION_MODEL={shlex.quote(str(self.shared/"models/tar/ta_tok.pth"))}',
            f'export WANDB_MODE={shlex.quote(self.args.wandb_mode)} WANDB_PROJECT=tar_reasoning WANDB__SERVICE_WAIT=300',
            f'export WANDB_CONFIG_DIR={shlex.quote(str(Path(self.args.results)/"config/wandb"))}',
            f'export WANDB_CACHE_DIR={shlex.quote(self.args.tar_root + "/cache/wandb")}',
            f'export TRITON_CACHE_DIR={shlex.quote(self.args.tar_root + "/cache/triton")}',
            f'export TORCHINDUCTOR_CACHE_DIR={shlex.quote(self.args.tar_root + "/cache/torchinductor")}',
            'export TMPDIR="${SLURM_TMPDIR:-/tmp}"',
            f'cd {shlex.quote(str(self.code))}',
            'echo "host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-none} start $(date -Is)"', '']))
        plan = {}
        for config in self.args.configs:
            (self.root/config['name']/'jobs').mkdir(parents=True)
            (self.logs/config['name']).mkdir(parents=True, exist_ok=True)
            (self.root/config['name']/'comparison.json').write_text(json.dumps(self.comparison(config), indent=2))
            for name in stage_list(self.args.benchmarks):
                text, n_tasks, elements, profile = self.script(config, name)
                path = self.root/config['name']/'jobs'/f'{name}.sbatch'
                path.write_text(text)
                plan[f'{config["name"]}/{name}'] = dict(script=str(path), sha256=sha256(path), tasks=n_tasks,
                                                        array_elements=elements, profile=profile,
                                                        depends_on=dependencies(name, self.args.benchmarks))
        launch = dict(suite=self.args.suite, configs=self.args.configs, decoding=self.args.decoding,
                      max_refinements=self.args.max_refinements, benchmarks=self.args.benchmarks,
                      gpu=dict(large=self.args.gpu_large, gen=self.args.gpu_gen, tiny=self.args.gpu_tiny,
                               light_pack=self.args.light_pack), python=self.python, stages=plan)
        (self.root/'launch.json').write_text(json.dumps(launch, indent=1))
        return launch

    def submit(self, launch):
        for config in self.args.configs:
            ids = {}
            for name in stage_list(self.args.benchmarks):
                entry = launch['stages'][f'{config["name"]}/{name}']
                if sha256(entry['script']) != entry['sha256']:
                    raise SystemExit(f'{entry["script"]} changed after preparation')
                depends = [ids[d] for d in entry['depends_on']]
                command = ['sbatch', '--parsable'] + (
                    [f'--dependency=afterok:{":".join(depends)}'] if depends else []) + [entry['script']]
                env = {k: v for k, v in os.environ.items() if k not in ('TMPDIR', 'SLURM_TMPDIR')}
                job = subprocess.run(command, check=True, capture_output=True, text=True, env=env).stdout.strip()
                ids[name] = job.split(';')[0]
                entry['job_id'] = ids[name]
                print(f'{config["name"]:>20} {name:<24} {ids[name]}  ({entry["profile"]}, {entry["tasks"]} tasks)')
        (self.root/'launch.json').write_text(json.dumps(launch, indent=1))


PROFILE_KEYS = ('SLURM_ACCOUNT', 'GPU_TYPE', 'TAR_GPU_LARGE', 'TAR_GPU_GEN', 'TAR_GPU_TINY', 'TAR_CPU_ACCOUNT',
                'TAR_CLUSTER')


def profile_settings():
    """Cluster profile values (scripts/clusters/<cluster>/profile.sh); the environment overrides them."""
    script = ('source scripts/clusters/profile.sh >/dev/null && for k in ' + ' '.join(PROFILE_KEYS) +
              '; do printf "%s=%s\\0" "$k" "${!k:-}"; done')
    out = subprocess.run(['bash', '-c', f'REPO_ROOT={shlex.quote(str(REPO))}; {script}'], cwd=REPO,
                         capture_output=True, text=True)
    values = dict(item.split('=', 1) for item in out.stdout.split('\0') if '=' in item) if out.returncode == 0 else {}
    return {k: os.environ.get(k) or values.get(k, '') for k in PROFILE_KEYS}


def parse_config(text):
    name, _, rest = text.partition('=')
    path, _, kind = rest.partition(':')
    if not name or not path or kind not in ('', 'sft', 'rl_adapter'):
        raise argparse.ArgumentTypeError('expected NAME=CHECKPOINT[:sft|:rl_adapter]')
    step = None
    state = Path(path)/'state.json'
    if state.is_file():
        step = json.loads(state.read_text()).get('step')
    return dict(name=name, checkpoint=str(Path(path).resolve()), type=kind or 'rl_adapter', step=step)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--suite', required=True)
    p.add_argument('--config', dest='configs', action='append', type=parse_config, required=True)
    p.add_argument('--decoding', choices=['sample', 'greedy'], default='sample')
    p.add_argument('--max_refinements', type=int, default=4)
    p.add_argument('--benchmarks', nargs='+', default=['geneval2', 'tiif', 'genai'],
                   choices=['geneval2', 'tiif', 'genai'])
    root = os.environ.get('TAR_ROOT', '')
    p.add_argument('--tar_root', default=root)
    p.add_argument('--shared', default=os.environ.get('TAR_SHARED', root and f'{root}/shared'))
    p.add_argument('--results', default=os.environ.get('TAR_RESULTS', root and f'{root}/results'))
    # The resolved venv (lock-hash directory), not the interpreter it links to.
    p.add_argument('--python', default=os.environ.get('TAR_EVAL_PYTHON', str(Path(sys.prefix).resolve()/'bin/python')))
    profile = profile_settings()
    gpu = profile['GPU_TYPE'] or 'h100'
    p.add_argument('--gpu_type', default=None, help='one GPU model for every GPU stage (default: profile settings)')
    p.add_argument('--gpu_large', default=profile['TAR_GPU_LARGE'] or gpu)
    p.add_argument('--gpu_gen', default=profile['TAR_GPU_GEN'] or gpu)
    p.add_argument('--gpu_tiny', default=profile['TAR_GPU_TINY'] or gpu)
    p.add_argument('--light_pack', type=int, default=int(os.environ.get('LIGHT_PACK', 4)))
    p.add_argument('--gpu_account', default=profile['SLURM_ACCOUNT'])
    p.add_argument('--cpu_account', default=profile['TAR_CPU_ACCOUNT'] or profile['SLURM_ACCOUNT'])
    p.add_argument('--wandb_mode', default=os.environ.get('WANDB_MODE', 'online'))
    p.add_argument('--submit', action='store_true')
    args = p.parse_args(argv)
    if not (args.tar_root and args.shared and args.results):
        p.error('Set TAR_ROOT (or --tar_root/--shared/--results): suites run in the uv layout')
    if args.gpu_type:
        args.gpu_large = args.gpu_gen = args.gpu_type
    if not args.gpu_account:
        p.error('Set SLURM_ACCOUNT or --gpu_account')
    if len({c['name'] for c in args.configs}) != len(args.configs):
        p.error('Configuration names must be unique')
    args.suite_tag = args.suite[:24]
    suite = Suite(args)
    if args.submit and (suite.root/'launch.json').is_file():
        launch = json.loads((suite.root/'launch.json').read_text())   # submit what was prepared
        if any('job_id' in e for e in launch['stages'].values()):
            raise SystemExit(f'{suite.root} was already submitted')
        if [c['name'] for c in launch['configs']] != [c['name'] for c in args.configs]:
            raise SystemExit('Configurations differ from the prepared suite')
        suite.submit(launch)
        return
    launch = suite.prepare()
    for key, entry in launch['stages'].items():
        print(f'{key:<44} {entry["profile"]:<9} tasks={entry["tasks"]:<3} elements={entry["array_elements"]}')
    if args.submit:
        suite.submit(launch)
    else:
        print(f'Prepared {suite.root}; rerun the same command with --submit (scripts are sha-checked).')


if __name__ == '__main__':
    main()
