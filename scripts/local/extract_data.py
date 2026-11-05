"""Extrai os .tif da Track 2 dos zips na raiz do projeto para data/dfc2019/track2.

Extrai todos os arquivos dos zips de RGB (LEFT_RGB, RIGHT_RGB, LEFT_AGL) e, do zip
de Reference (Truth), só as disparidades (LEFT_DSP) dos pares que têm RGB.
Pode rodar de novo quando baixar outro zip: o que já foi extraído é pulado.

Uso (na raiz do projeto):  .venv\\Scripts\\python scripts\\local\\extract_data.py
"""
import argparse
import zipfile
from pathlib import Path

from tqdm import tqdm


def extract(zf, infos, out):
    done = 0
    for info in tqdm(infos, unit="arq"):
        dest = out / Path(info.filename).name
        if dest.exists() and dest.stat().st_size == info.file_size:
            continue  # já extraído por inteiro
        with zf.open(info) as src, open(dest, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        done += 1
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", default=".", help="pasta onde estão os .zip")
    ap.add_argument("--out", default="data/dfc2019/track2")
    ap.add_argument("--all-truth", action="store_true", help="extrai também CLS e DSP sem RGB correspondente")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rgb_zips = sorted(Path(args.zips).glob("Train-Track2-RGB-*.zip"))
    truth_zip = Path(args.zips) / "Train-Track2-Truth.zip"
    if not rgb_zips:
        raise SystemExit("Nenhum Train-Track2-RGB-*.zip encontrado.")

    bases = set()
    for z in rgb_zips:
        with zipfile.ZipFile(z) as zf:
            infos = [i for i in zf.infolist() if i.filename.endswith(".tif")]
            bases |= {Path(i.filename).name[: -len("LEFT_RGB.tif")] for i in infos if i.filename.endswith("LEFT_RGB.tif")}
            print(f"{z.name}: {len(infos)} arquivos")
            n = extract(zf, infos, out)
            print(f"  extraídos agora: {n}")

    if truth_zip.exists():
        with zipfile.ZipFile(truth_zip) as zf:
            infos = [i for i in zf.infolist() if i.filename.endswith(".tif")]
            if not args.all_truth:
                infos = [i for i in infos if i.filename.endswith("LEFT_DSP.tif")
                         and Path(i.filename).name[: -len("LEFT_DSP.tif")] in bases]
            print(f"{truth_zip.name}: {len(infos)} disparidades dos pares com RGB")
            print(f"  extraídos agora: {extract(zf, infos, out)}")
    else:
        print("Aviso: Train-Track2-Truth.zip não encontrado; sem disparidades.")
    print(f"Pronto: {len(bases)} pares com RGB em {out}")


if __name__ == "__main__":
    main()
