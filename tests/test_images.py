import io
import json
import random

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw, ImageEnhance, ImageOps  # noqa: E402

from dotlet import cli  # noqa: E402
from dotlet.image import mia as imia  # noqa: E402
from dotlet.image import overlap as iov  # noqa: E402
from dotlet.image import probe as iprobe  # noqa: E402
from dotlet.image.features import HashFeatures, find_images  # noqa: E402

FAST = {"n_permutations": 2000, "n_bootstrap": 300}


def synth(seed, size=128):
    r = random.Random(seed)
    rng = np.random.default_rng(seed)
    im = Image.fromarray((rng.random((6, 6, 3)) * 255).astype("uint8")).resize((size, size), Image.BICUBIC)
    d = ImageDraw.Draw(im)
    for _ in range(6):
        x0, y0 = r.randrange(size), r.randrange(size)
        box = [x0, y0, x0 + r.randrange(10, 60), y0 + r.randrange(10, 60)]
        (d.ellipse if r.random() < 0.5 else d.rectangle)(box, fill=tuple(r.randrange(256) for _ in range(3)))
    return im


def distort(im, mirror=False):
    """What a regurgitated copy looks like: resized, brighter, JPEG-compressed, maybe mirrored."""
    im = ImageEnhance.Brightness(im.resize((160, 160))).enhance(1.1)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=55)
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return ImageOps.mirror(im) if mirror else im


def write_set(dirpath, seeds, caption=True):
    dirpath.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in seeds:
        p = dirpath / f"img{s}.png"
        synth(s).save(p)
        if caption:
            p.with_suffix(".txt").write_text(f"an abstract painting with shapes, number {s}")
        paths.append(p)
    return paths


@pytest.fixture
def sets(tmp_path):
    mine = write_set(tmp_path / "mine", range(0, 30))
    ctrl = write_set(tmp_path / "ctrl", range(1000, 1030))
    return tmp_path, mine, ctrl


def test_hash_similarity_separates_copies_from_unrelated():
    f = HashFeatures()
    origs = [synth(i) for i in range(40)]
    S = f.similarity(f(origs), f([distort(o) for o in origs]))
    same = np.diag(S)
    off = S[~np.eye(40, dtype=bool)]
    assert same.min() > 0.85 > off.max()


def test_find_images_skips_captions(sets):
    tmp, mine, _ = sets
    assert find_images([tmp / "mine"]) == sorted(mine)


def test_image_overlap_detects_regurgitation(sets):
    tmp, mine, ctrl = sets
    out = tmp / "out"
    out.mkdir()
    for i, p in enumerate(mine[:12]):  # 12 of 30 of my images leak, some mirrored
        distort(Image.open(p).convert("RGB"), mirror=i % 2 == 0).save(out / f"leak{i}.jpg")
    for s in range(5000, 5030):
        synth(s).save(out / f"novel{s}.png")
    rep = iov.analyze(mine, ctrl, find_images([out]), HashFeatures(), **FAST)
    assert rep.result.p_permutation < 0.01 and rep.result.auc > 0.6  # max ~0.70 when 12/30 leak
    assert len(rep.near_duplicates) == 12
    assert rep.max_control_similarity < rep.threshold
    assert rep.matches[0].group == "candidate"


def test_image_overlap_null(sets):
    tmp, mine, ctrl = sets
    out = tmp / "out"
    out.mkdir()
    for s in range(5000, 5040):
        synth(s).save(out / f"novel{s}.png")
    rep = iov.analyze(mine, ctrl, find_images([out]), HashFeatures(), **FAST)
    assert rep.result.verdict.startswith("NO") and not rep.near_duplicates


def test_captions_from_csv_and_missing(tmp_path):
    paths = write_set(tmp_path / "a", [1, 2], caption=False)
    csvf = tmp_path / "caps.csv"
    csvf.write_text("image,prompt\nimg1.png,a red thing\n" + f"{paths[1]},a blue thing\n")
    caps = iprobe.load_captions(paths, csvf)
    assert caps[str(paths[0])] == "a red thing" and caps[str(paths[1])] == "a blue thing"
    with pytest.raises(SystemExit):
        iprobe.load_captions(paths)


def test_image_probe_end_to_end(sets, capsys):
    tmp, mine, ctrl = sets
    probes_path = tmp / "probes.jsonl"
    cli.main(["image", "probe", "make", "--materials", str(tmp / "mine"), "--controls", str(tmp / "ctrl"),
              "--out", str(probes_path)])
    probes = [json.loads(l) for l in probes_path.read_text().splitlines()]
    assert len(probes) == 60 and probes[0]["prompt"].startswith("an abstract painting")
    gen = tmp / "gen"
    gen.mkdir()
    rows = []
    for i, p in enumerate(probes):  # a "memorizing" model: recreates my images, invents new ones for controls
        files = []
        for k in range(2):
            f = gen / f"{i}_{k}.png"
            (distort(Image.open(p["image"]).convert("RGB")) if p["group"] == "candidate" and k == 1
             else synth(9000 + 2 * i + k)).save(f)
            files.append(str(f))
        rows.append({"id": p["id"], "files": files})
    (tmp / "resp.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cli.main(["image", "probe", "score", "--probes", str(probes_path), "--responses", str(tmp / "resp.jsonl"),
              "--permutations", "1000"])
    out = capsys.readouterr().out
    assert "STRONG" in out


def test_image_mia_with_fake_loss(sets):
    _, mine, ctrl = sets
    members = {str(p) for p in mine}
    rng = random.Random(0)
    fake = lambda paths, prompts, model: [rng.gauss(0.10 if str(p) in members else 0.12, 0.01) for p in paths]
    res = imia.analyze(mine, ctrl, "fake-sd", loss_fn=fake, **FAST)
    assert res.p_permutation < 0.01 and res.auc > 0.8


def test_cli_image_overlap(sets, capsys):
    tmp, mine, ctrl = sets
    out = tmp / "out"
    out.mkdir()
    distort(Image.open(mine[3]).convert("RGB")).save(out / "copy.jpg")
    cli.main(["image", "overlap", "--materials", str(tmp / "mine"), "--controls", str(tmp / "ctrl"),
              "--outputs", str(out), "--permutations", "500", "--json", str(tmp / "r.json")])
    text = capsys.readouterr().out
    assert "FLAG" in text and "img3.png" in text
    assert json.loads((tmp / "r.json").read_text())["matches"][0]["group"] == "candidate"
