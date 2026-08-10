from sigverify.data.datasets import (
    StaticSignatureTripletDataset,
    kfold_writers,
    load_manifest,
    split_writers,
)
from sigverify.data.synthetic import build_demo_dataset


def test_split_writers_is_disjoint_and_covers_all_writers(tmp_path):
    manifests = build_demo_dataset(tmp_path, num_writers=10, genuine_per_writer=2, forged_per_writer=1)
    static_manifest = manifests["static_manifest"]
    train_writers, val_writers = split_writers(static_manifest, val_fraction=0.2, seed=1)
    assert train_writers.isdisjoint(val_writers)
    all_writers = {r["writer_id"] for r in load_manifest(static_manifest)}
    assert train_writers | val_writers == all_writers
    assert len(val_writers) == 2  # 20% of 10


def test_kfold_writers_every_writer_held_out_exactly_once(tmp_path):
    manifests = build_demo_dataset(tmp_path, num_writers=10, genuine_per_writer=2, forged_per_writer=1)
    static_manifest = manifests["static_manifest"]
    splits = kfold_writers(static_manifest, k=5, seed=1)
    assert len(splits) == 5

    all_val_writers = []
    for train_writers, val_writers in splits:
        assert train_writers.isdisjoint(val_writers)
        all_val_writers.extend(val_writers)

    # every writer appears in exactly one fold's held-out set
    assert sorted(all_val_writers) == sorted({r["writer_id"] for r in load_manifest(static_manifest)})
    assert len(all_val_writers) == len(set(all_val_writers))


def test_kfold_writers_respects_max_writers_cap(tmp_path):
    manifests = build_demo_dataset(tmp_path, num_writers=10, genuine_per_writer=2, forged_per_writer=1)
    static_manifest = manifests["static_manifest"]
    splits = kfold_writers(static_manifest, k=4, seed=1, max_writers=8)
    covered = set()
    for train_writers, val_writers in splits:
        covered |= train_writers | val_writers
    assert len(covered) == 8


def test_kfold_writers_rejects_more_folds_than_writers(tmp_path):
    manifests = build_demo_dataset(tmp_path, num_writers=3, genuine_per_writer=2, forged_per_writer=1)
    static_manifest = manifests["static_manifest"]
    try:
        kfold_writers(static_manifest, k=5, seed=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_static_dataset_shared_denoise_cache_is_reused_across_instances(tmp_path):
    manifests = build_demo_dataset(tmp_path, num_writers=2, genuine_per_writer=3, forged_per_writer=1)
    static_manifest = manifests["static_manifest"]
    shared_cache = {}
    ds_a = StaticSignatureTripletDataset(static_manifest, target_size=(32, 32), cache_in_memory=True, shared_denoise_cache=shared_cache)
    ds_a[0]
    assert len(shared_cache) > 0

    ds_b = StaticSignatureTripletDataset(static_manifest, target_size=(32, 32), cache_in_memory=True, shared_denoise_cache=shared_cache)
    assert ds_b._denoised_cache is shared_cache
