import json
import random
from pathlib import Path
from PIL import Image

SPRITE_ROOT = Path('./Universal-LPC-spritesheet')
OUTPUT_DIR = Path('./data_output')
FRAMES_DIR = OUTPUT_DIR / 'frames'
ROW_HEIGHT = 64
SEED = 42

random.seed(SEED)

LAYER_ORDER = ['body', 'behind_body', 'legs', 'feet', 'torso', 'belt',
               'hands', 'head', 'hair', 'accessories', 'facial', 'weapons']


def get_gender(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if 'male' in parts and 'female' not in parts[:parts.index('male')]:
        return 'male'
    return 'female' if 'female' in parts else 'unisex'


def scan(root: Path) -> dict:
    catalog = {}
    for png in root.rglob('*.png'):
        try:
            cat = png.relative_to(root).parts[0]
        except ValueError:
            continue
        g = get_gender(png)
        catalog.setdefault(cat, {'male': [], 'female': [], 'unisex': []})[g].append(png)
    return catalog


def pick(catalog, category, gender):
    if category not in catalog:
        return None
    pool = catalog[category].get(gender) or catalog[category].get('unisex') or []
    return random.choice(pool) if pool else None


def pick_body(root, gender):
    candidates = list((root / 'body' / gender).glob('*.png'))
    return random.choice(candidates) if candidates else None


def composite(paths, size):
    canvas = Image.new('RGBA', size, (0, 0, 0, 0))
    for i, p in enumerate(paths):
        try:
            layer = Image.open(p).convert('RGBA')
            if layer.size != size:
                layer = layer.resize(size, Image.NEAREST)
            canvas = Image.alpha_composite(canvas, layer)
        except Exception as e:
            if i == 0:
                raise RuntimeError(f'body layer failed: {p}: {e}') from e
    return canvas


def extract_rows(img, entry_id, gender):
    w, h = img.size
    for row in range(h // ROW_HEIGHT):
        top = row * ROW_HEIGHT
        strip = img.crop((0, top, w, top + ROW_HEIGHT))
        strip.save(FRAMES_DIR / f'entry_{entry_id:03d}_{gender}_row{row}.png')


def generate_entry(entry_id, gender, catalog, categories):
    body = pick_body(SPRITE_ROOT, gender)
    if body is None:
        return False
    try:
        base = Image.open(body).convert('RGBA')
    except Exception:
        return False

    layers = [body]
    meta = {'gender': gender, 'layers': {'body': str(body)}, 'skipped_categories': []}
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
        sheet.save(sheet_path)
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    catalog = scan(SPRITE_ROOT)
    categories = [c for c in LAYER_ORDER if c in catalog]

    generated = skipped = 0
    tasks = [('male', i) for i in range(50)] + [('female', i) for i in range(50, 100)]
    for gender, entry_id in tasks:
        if generate_entry(entry_id, gender, catalog, categories):
            generated += 1
        else:
            skipped += 1
            print(f'skipped {entry_id} ({gender})')

    print(f'done. generated={generated} skipped={skipped}')


if __name__ == '__main__':
    main()
