#  Copyright 2020 Maruan Al-Shedivat. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  =============================================================================
"""Data utils."""

from concurrent import futures
from functools import partial

import numpy as np
import scipy as sp


def get_tokenizer(name="bert-base-uncased", max_workers=16):

    import transformers

    def tokenize(inputs, max_length=None):
        # Build tokenizer.
        if name.startswith("bert"):
            tokenizer = transformers.BertTokenizer.from_pretrained(name)
        else:
            raise ValueError(f"Unknown tokenizer name: {name}.")
        encode = partial(
            tokenizer.encode,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
        )
        # Tokenize inputs.
        chunksize = len(inputs) // max_workers
        with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            tokenized = list(executor.map(encode, inputs, chunksize=chunksize))
        return np.asarray(tokenized, dtype=np.int32)

    return tokenize


def get_zca_whitening_mat(X, eps=1e-6):
    flat_X = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
    Sigma = np.dot(flat_X.T, flat_X) / flat_X.shape[0]
    U, S, _ = sp.linalg.svd(Sigma)
    M = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(S + eps))), U.T)
    return M


def zca_whiten(X, W):
    shape = X.shape
    flat_X = np.reshape(X, (shape[0], np.prod(shape[1:])))
    white_X = np.dot(flat_X, W)
    return np.reshape(white_X, shape)
