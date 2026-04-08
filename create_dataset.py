import json
import os
import random
import numpy as np
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from pathlib import Path
from PIL import Image
from tqdm import tqdm

SPRITE_ROOT = Path('./Universal-LPC-spritesheet')
OUTPUT_DIR  = Path('./data_output')
FRAMES_DIR  = OUTPUT_DIR / 'frames'
ROW_HEIGHT  = 64
SEED        = 42
NUM_WORKERS = max(1, cpu_count() - 1)

random.seed(SEED)

LAYER_ORDER = ['body', 'behind_body', 'legs', 'feet', 'torso', 'belt',
               'hands', 'head', 'hair', 'accessories', 'facial', 'weapons']

_body_candidates: dict[str, list] = {}

def get_gender(path: Path) -> str:
    lower_parts = [p.lower() for p in path.parts]
    has_female  = 'female' in lower_parts
    has_male    = 'male'   in lower_parts
    if has_male and not has_female:
        return 'male'
    if has_male and has_female and lower_parts.index('male') > lower_parts.index('female'):
        return 'female'
    return 'female' if has_female else 'unisex'


def scan(root: Path) -> dict:
    catalog: dict = {}
    for png in root.rglob('*.png'):
        try:
            cat = png.relative_to(root).parts[0]
        except ValueError:
            continue
        g = get_gender(png)
        catalog.setdefault(cat, {'male': [], 'female': [], 'unisex': []})[g].append(png)
    return catalog


def pick(catalog: dict, category: str, gender: str) -> Path | None:
    if category not in catalog:
        return None
    pool = catalog[category].get(gender) or catalog[category].get('unisex') or []
    return random.choice(pool) if pool else None


def pick_body(root: Path, gender: str) -> Path | None:
    global _body_candidates
    if gender not in _body_candidates:
        _body_candidates[gender] = list((root / 'body' / gender).glob('*.png'))
    pool = _body_candidates[gender]
    return random.choice(pool) if pool else None


@lru_cache(maxsize=256)
def _open_layer_cached(p: Path) -> Image.Image:
    return Image.open(p).convert('RGBA')


def composite(paths: list, size: tuple) -> Image.Image:
    canvas = Image.new('RGBA', size, (0, 0, 0, 0))
    for i, p in enumerate(paths):
        try:
            layer = _open_layer_cached(p)
            if layer.size != size:
                layer = layer.resize(size, Image.NEAREST)
            canvas = Image.alpha_composite(canvas, layer)
        except Exception as e:
            if i == 0:
                raise RuntimeError(f'body layer failed: {p}: {e}') from e
    return canvas


def extract_rows(img: Image.Image, entry_id: int, gender: str) -> None:
    arr    = np.asarray(img)
    n_rows = arr.shape[0] // ROW_HEIGHT
    for row in range(n_rows):
        strip = arr[row * ROW_HEIGHT:(row + 1) * ROW_HEIGHT]
        Image.fromarray(strip).save(
            FRAMES_DIR / f'entry_{entry_id:03d}_{gender}_row{row}.png',
            compress_level=1,
        )


def generate_entry(entry_id: int, gender: str, catalog: dict, categories: list) -> bool:
    body = pick_body(SPRITE_ROOT, gender)
    if body is None:
        return False
    try:
        base = _open_layer_cached(body)
    except Exception:
        return False

    layers = [body]
    meta   = {'gender': gender, 'layers': {'body': str(body)}, 'skipped_categories': []}
    for cat in categories:
        if cat == 'body':
            continue
        chosen = pick(catalog, cat, gender)
        if chosen is None:
            meta['skipped_categories'].append(cat)
        else:
            layers.append(chosen)
            meta['layers'][cat] = str(chosen)

    try:
        sheet = composite(layers, base.size)
    except Exception:
        return False

    sheet_path = OUTPUT_DIR / f'entry_{entry_id:03d}_{gender}.png'
    try:
        sheet.save(sheet_path, compress_level=1)
    except Exception:
        sheet_path.unlink(missing_ok=True)
        return False

    with open(OUTPUT_DIR / f'entry_{entry_id:03d}_{gender}_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    try:
        extract_rows(sheet, entry_id, gender)
    except Exception as e:
        print(f'  warning: row extraction failed entry {entry_id}: {e}')

    return True



def _worker_init(seed_base: int) -> None:
    """Initialise per-worker RNG and pre-build the body-candidates cache."""
    worker_seed = seed_base + os.getpid()
    random.seed(worker_seed)
    for g in ('male', 'female'):
        pick_body(SPRITE_ROOT, g)


def _task(args: tuple) -> tuple[int, str, bool]:
    """Top-level picklable function executed by each Pool worker."""
    entry_id, gender, catalog, categories = args
    ok = generate_entry(entry_id, gender, catalog, categories)
    return entry_id, gender, ok

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Scanning sprite root: {SPRITE_ROOT}')
    catalog    = scan(SPRITE_ROOT)
    categories = [c for c in LAYER_ORDER if c in catalog]
    print(f'Found {sum(len(v) for cats in catalog.values() for v in cats.values())} sprites '
          f'across {len(catalog)} categories')

    tasks = [('male',   i)       for i in range(500)] + \
            [('female', i)       for i in range(500, 1000)]
    worker_args = [(eid, g, catalog, categories) for g, eid in tasks]

    generated = skipped = 0

    print(f'Generating {len(tasks)} entries using {NUM_WORKERS} worker(s) …')
    with Pool(
        processes=NUM_WORKERS,
        initializer=_worker_init,
        initargs=(SEED,),
    ) as pool:
        with tqdm(total=len(tasks), unit='entry', dynamic_ncols=True) as bar:
            for entry_id, gender, ok in pool.imap_unordered(_task, worker_args, chunksize=10):
                if ok:
                    generated += 1
                else:
                    skipped += 1
                    tqdm.write(f'  skipped entry_{entry_id:03d}_{gender}')
                bar.set_postfix(gen=generated, skip=skipped)
                bar.update()

    print(f'\ndone. generated={generated}  skipped={skipped}')


if __name__ == '__main__':
    main()
