import argparse
import gzip
import urllib.request
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh


DOCWORD_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz'
VOCAB_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/vocab.nytimes.txt'


def download_file(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def load_docword_gz(path):
    with gzip.open(path, 'rt') as f:
        n_docs = int(f.readline())
        n_words = int(f.readline())
        nnz = int(f.readline())
        rows = np.empty(nnz, dtype=np.int32)
        cols = np.empty(nnz, dtype=np.int32)
        vals = np.empty(nnz, dtype=np.int64)
        for idx, line in enumerate(f):
            doc_id, word_id, count = line.split()
            rows[idx] = int(doc_id) - 1
            cols[idx] = int(word_id) - 1
            vals[idx] = int(count)
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n_docs, n_words)).tocsr()


def normalized_matrix(A):
    row_sums = np.asarray(A.sum(axis=1)).ravel()
    col_sums = np.asarray(A.sum(axis=0)).ravel()
    tau = row_sums.mean()
    left_scale = 1.0 / np.sqrt(row_sums + tau)
    right_scale = 1.0 / np.sqrt(col_sums + col_sums.mean())
    L = sparse.diags(left_scale) @ A @ sparse.diags(right_scale)
    return L.tocsr(), row_sums, col_sums


def centered_products(L):
    n_rows, n_cols = L.shape
    row_sums = np.asarray(L.sum(axis=1)).ravel()
    col_sums = np.asarray(L.sum(axis=0)).ravel()
    grand_mean = L.sum() / (n_rows * n_cols)

    def ax(x):
        return (
            L @ x
            - ((col_sums @ x) / n_rows - x.sum() * grand_mean)
            - row_sums * x.mean()
        )

    def atx(x):
        return (
            L.T @ x
            - ((row_sums @ x) / n_cols - x.sum() * grand_mean)
            - col_sums * x.mean()
        )

    return ax, atx, row_sums, col_sums


def compute_rohe_pca(A, k=50, random_state=0):
    L, row_sums_a, col_sums_a = normalized_matrix(A)
    ax, atx, row_sums_l, col_sums_l = centered_products(L)
    n_rows = L.shape[0]

    operator = LinearOperator(
        shape=(n_rows, n_rows),
        matvec=lambda x: ax(atx(x)),
        dtype=np.float64,
    )
    values, vectors = eigsh(operator, k=k, which='LA', tol=1e-6, v0=np.random.default_rng(random_state).normal(size=n_rows))
    order = np.argsort(values)[::-1]
    values = values[order]
    U = vectors[:, order]
    scree = np.sqrt(np.maximum(values, 0.0))

    projected = (U.T @ L).T
    V, _, _ = np.linalg.svd(projected, full_matrices=False)

    return {
        'U': U,
        'V': V,
        'scree': scree,
        'sym': np.array(False),
        'centered': np.array(True),
        'normalized': np.array(True),
        'loc_U_stat': np.sum(U ** 4, axis=0),
        'loc_V_stat': np.sum(V ** 4, axis=0),
        'rsA': row_sums_a,
        'csA': col_sums_a,
        'rsL': row_sums_l,
        'csL': col_sums_l,
    }


def prepare_nytimes_data(data_dir=Path('data'), k=50, force_download=False):
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / 'raw'
    docword_path = raw_dir / 'docword.nytimes.txt.gz'
    vocab_path = raw_dir / 'vocab.nytimes.txt'

    if force_download or not docword_path.exists():
        print(f'Downloading {DOCWORD_URL}')
        download_file(DOCWORD_URL, docword_path)
    if force_download or not vocab_path.exists():
        print(f'Downloading {VOCAB_URL}')
        download_file(VOCAB_URL, vocab_path)

    print('Building sparse NYTimes matrix')
    A = load_docword_gz(docword_path)
    sparse.save_npz(data_dir / 'nytA.npz', A, compressed=True)

    print(f'Computing Rohe-style normalized, centered PCA factors with k={k}')
    pca = compute_rohe_pca(A, k=k)
    np.savez_compressed(data_dir / 'nytimes_pca.npz', **pca)
    print('Saved data/nytA.npz and data/nytimes_pca.npz')


def parse_args():
    parser = argparse.ArgumentParser(description='Download and preprocess NYTimes data for the pairplot figures.')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--force-download', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_nytimes_data(args.data_dir, args.k, args.force_download)


if __name__ == '__main__':
    main()
