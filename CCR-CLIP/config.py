config = {
    'epoch': 40,
    'train_dataset': '/mnt/storage/kellen/lmdb/adult_data/train',
    'test_dataset': '/mnt/storage/kellen/lmdb/adult_data/val',
    'batch': 128,
    'imageW': 128,
    'imageH': 128,
    'alphabet_path': './data/radical_alphabet_27533_benchmark.txt',
    'decompose_path': './data/decompose_27533_benchmark.txt',
    'max_len': 30,
    'lr': 1e-4,
    'exp_name': 'adult_with_notation',
}