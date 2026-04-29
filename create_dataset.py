import json
import os
import random
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


SPRITE_ROOT = Path('./Universal-LPC-spritesheet')
OUTPUT_DIR = Path('./data_output')
FRAMES_DIR = OUTPUT_DIR / 'frames'
ROW_HEIGHT = 64
SEED = 42
NUM_WORKERS = max(1, cpu_count() - 1)

# Layers are composited from body outward.
LAYER_ORDER = [
    'body',
    'behind_body',
    'legs',
    'feet',
    'torso',
    'belt',
    'hands',
    'head',
    'hair',
    'accessories',
    'facial',
    'weapons',
]

_body_candidates: dict[str, list[Path]] = {}


def get_gender(path: Path) -> str:
    # Gender is inferred from folder names in the LPC tree.
    lower_parts = [part.lower() for part in path.parts]
    has_female = 'female' in lower_parts
    has_male = 'male' in lower_parts
    if has_male and not has_female:
        return 'male'
    if has_male and has_female and lower_parts.index('male') > lower_parts.index('female'):
        return 'female'
    return 'female' if has_female else 'unisex'


def scan(root: Path) -> dict[str, dict[str, list[Path]]]:
    # Catalog sprites by top-level category and gender.
    catalog: dict[str, dict[str, list[Path]]] = {}
    for png in root.rglob('*.png'):
        try:
            category = png.relative_to(root).parts[0]
        except ValueError:
            continue
        gender = get_gender(png)
        catalog.setdefault(category, {'male': [], 'female': [], 'unisex': []})[gender].append(png)
    return catalog


def pick(catalog: dict, category: str, gender: str) -> Path | None:
    # Unisex sprites fill gaps when gender-specific assets are missing.
    if category not in catalog:
        return None
    pool = catalog[category].get(gender) or catalog[category].get('unisex') or []
    return random.choice(pool) if pool else None


def pick_body(root: Path, gender: str) -> Path | None:
    # Body candidates are cached because every entry needs one.
    if gender not in _body_candidates:
        _body_candidates[gender] = list((root / 'body' / gender).glob('*.png'))
    pool = _body_candidates[gender]
    return random.choice(pool) if pool else None


@lru_cache(maxsize=256)
def open_layer_cached(path: Path) -> Image.Image:
    # Layer images are reused heavily across generated characters.
    return Image.open(path).convert('RGBA')


def composite(paths: list[Path], size: tuple[int, int]) -> Image.Image:
    # Alpha compositing preserves transparent sprite-sheet layers.
    canvas = Image.new('RGBA', size, (0, 0, 0, 0))
    for index, path in enumerate(paths):
        try:
            layer = open_layer_cached(path)
            if layer.size != size:
                layer = layer.resize(size, Image.NEAREST)
            canvas = Image.alpha_composite(canvas, layer)
        except Exception as exc:
            if index == 0:
                raise RuntimeError(f'body layer failed: {path}: {exc}') from exc
    return canvas


def extract_rows(img: Image.Image, entry_id: int, gender: str) -> None:
    # Each 64px row becomes one supervised animation strip.
    pixels = np.asarray(img)
    n_rows = pixels.shape[0] // ROW_HEIGHT
    for row in range(n_rows):
        strip = pixels[row * ROW_HEIGHT:(row + 1) * ROW_HEIGHT]
        out_path = FRAMES_DIR / f'entry_{entry_id:03d}_{gender}_row{row}.png'
        Image.fromarray(strip).save(out_path, compress_level=1)


def generate_entry(entry_id: int, gender: str, catalog: dict, categories: list[str]) -> bool:
    # One entry is a randomized layered LPC character sheet.
    body = pick_body(SPRITE_ROOT, gender)
    if body is None:
        return False
    try:
        base = open_layer_cached(body)
    except Exception:
        return False

    layers = [body]
    meta = {'gender': gender, 'layers': {'body': str(body)}, 'skipped_categories': []}
    for category in categories:
        if category == 'body':
            continue
        chosen = pick(catalog, category, gender)
        if chosen is None:
            # Missing optional categories are recorded, not fatal.
            meta['skipped_categories'].append(category)
        else:
            layers.append(chosen)
            meta['layers'][category] = str(chosen)

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

    metadata_path = OUTPUT_DIR / f'entry_{entry_id:03d}_{gender}_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    try:
        extract_rows(sheet, entry_id, gender)
    except Exception as exc:
        print(f'  warning: row extraction failed entry {entry_id}: {exc}')

    return True


def worker_init(seed_base: int) -> None:
    # PID-based seeds keep worker sampling independent.
    random.seed(seed_base + os.getpid())
    for gender in ('male', 'female'):
        pick_body(SPRITE_ROOT, gender)


def task(args: tuple[int, str, dict, list[str]]) -> tuple[int, str, bool]:
    # Pool workers need a top-level picklable function.
    entry_id, gender, catalog, categories = args
    return entry_id, gender, generate_entry(entry_id, gender, catalog, categories)


def main() -> None:
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Scanning sprite root: {SPRITE_ROOT}')
    catalog = scan(SPRITE_ROOT)
    categories = [category for category in LAYER_ORDER if category in catalog]
    sprite_count = sum(len(paths) for cats in catalog.values() for paths in cats.values())
    print(f'Found {sprite_count} sprites across {len(catalog)} categories')

    # Entry IDs are split by gender for predictable filenames.
    tasks = [('male', i) for i in range(500)] + [('female', i) for i in range(500, 1000)]
    worker_args = [(entry_id, gender, catalog, categories) for gender, entry_id in tasks]
    generated = 0
    skipped = 0

    print(f'Generating {len(tasks)} entries using {NUM_WORKERS} worker(s) ...')
    with Pool(processes=NUM_WORKERS, initializer=worker_init, initargs=(SEED,)) as pool:
        with tqdm(total=len(tasks), unit='entry', dynamic_ncols=True) as bar:
            for entry_id, gender, ok in pool.imap_unordered(task, worker_args, chunksize=10):
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
