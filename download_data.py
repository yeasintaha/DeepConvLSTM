import argparse
import os
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/raw", help="folder to unzip into")
    p.add_argument("--url", default=URL)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    zip_path = os.path.join(args.out, "opportunity.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {args.url}\n -> {zip_path} (about 300 MB, this can take a while)")
        urllib.request.urlretrieve(args.url, zip_path)
    else:
        print(f"Found existing {zip_path}, skipping download")

    print("Unzipping ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.out)
    # The UCI zip sometimes contains a nested zip with the actual dataset.
    for root, _, files in os.walk(args.out):
        for f in files:
            if f.lower().endswith(".zip") and os.path.join(root, f) != zip_path:
                inner = os.path.join(root, f)
                print(f"Unzipping nested archive {inner}")
                with zipfile.ZipFile(inner) as z:
                    z.extractall(root)
    print("Done. Next: python preprocess.py --raw", args.out)


if __name__ == "__main__":
    main()
