# pipeline step 4 — model training
# svd matrix factorization baseline using the surprise library
# learns latent vectors for users and items that capture preference patterns
# the trained latent vectors are later used to measure item-item similarity
# reference koren bell volinsky 2009, matrix factorization techniques for recommender systems
# https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf

# SVD baseline model (Surprise library).
# key design choices:
# - surprise uses its own internal integer ids, translate back/forth
#   so callers always work with our dense userId/movieId integers
# - item similarity is computed on-demand (dot product of latent vectors)
#   to stay memory-safe for 10M/20M datasets
# - an optional precompute_similarity_matrix() method exists for ML-1M
#   where the full 3700x3700 matrix fits comfortably in ram

import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader


class SVDBaseline:
    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.01,
        reg_all: float = 0.05,
        biased: bool = True,
        random_state: int = 42,
        rating_scale: tuple = (1, 5),  # (0, 1) for Arm D binary targets
    ):
        # svd learns: pu (user latent matrix), qi (item latent matrix), bu, bi, global mean
        # prediction: r_hat(u,i) = mu + bu + bi + pu[u] dot qi[i]
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.biased = biased
        self.random_state = random_state
        self.rating_scale = rating_scale

        self._model: Optional[SVD] = None
        self._trainset = None
        # movieId (our int)  Surprise inner id
        self._raw_to_inner: Dict[int, int] = {}
        # Surprise inner id  movieId (our int)
        self._inner_to_raw: Dict[int, int] = {}
        self._sim_matrix: Optional[np.ndarray] = None
        # Dense numpy lookup arrays built by _ensure_fast_lookup() after fit.
        # None = not built yet (or old pickled model without these attrs).
        self._item_inner_arr: Optional[np.ndarray] = None
        self._user_inner_arr: Optional[np.ndarray] = None

    def get_params(self) -> dict:
        # returns hyperparameters as a dict  used when saving config to disk
        return {
            "n_factors": self.n_factors,
            "n_epochs": self.n_epochs,
            "lr_all": self.lr_all,
            "reg_all": self.reg_all,
            "biased": self.biased,
            "random_state": self.random_state,
            "rating_scale": list(self.rating_scale),
        }

    def fit(self, train_df: pd.DataFrame) -> "SVDBaseline":
        # modell training
        # Train SVD on train_df (columns= userId, movieId, rating)
        # Surprise treats IDs as strings internally; we store mappings

        # surprise requires string ids, convert our integers to strings first
        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(
            train_df[["userId", "movieId", "rating"]].astype(
                {"userId": str, "movieId": str}
            ),
            reader,
        )
        self._trainset = data.build_full_trainset()

        # sgd training: for each rating compute error, update pu, qi, bu, bi
        # update rule: param += lr * (error * gradient - reg * param)
        self._model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            biased=self.biased,
            random_state=self.random_state,
        )
        self._model.fit(self._trainset)

        # Build movieId <  > Surprise inner-id mappings
        # surprise assigns its own internal ids in arbitrary order during training
        #  movieId 1193 might become inner id 742
        # without this mapping get_similarity() would access wrong latent vectors
        self._raw_to_inner = {}
        for inner_id in self._trainset.all_items():
            raw_str = self._trainset.to_raw_iid(inner_id)
            self._raw_to_inner[int(raw_str)] = inner_id
        self._inner_to_raw = {v: k for k, v in self._raw_to_inner.items()}

        self._ensure_fast_lookup()
        return self

    def _ensure_fast_lookup(self) -> None:
        """Build dense numpy arrays for O(1) ID translation without Python loops.
        Safe to call on old pickled models rebuilds from _raw_to_inner if missing.
        """
        if (getattr(self, '_item_inner_arr', None) is not None and
                getattr(self, '_user_inner_arr', None) is not None):
            return

        # Dense item array: raw item ID → Surprise inner ID  (-1 = unknown)
        if self._raw_to_inner:
            max_item = max(self._raw_to_inner.keys())
            self._item_inner_arr = np.full(max_item + 1, -1, dtype=np.int32)
            for raw, inner in self._raw_to_inner.items():
                self._item_inner_arr[raw] = inner
        else:
            self._item_inner_arr = np.array([], dtype=np.int32)

        # Dense user array: raw user ID → Surprise inner ID  (-1 = unknown)
        user_raw_to_inner: Dict[int, int] = {}
        for inner_uid in self._trainset.all_users():
            user_raw_to_inner[int(self._trainset.to_raw_uid(inner_uid))] = inner_uid
        if user_raw_to_inner:
            max_user = max(user_raw_to_inner.keys())
            self._user_inner_arr = np.full(max_user + 1, -1, dtype=np.int32)
            for raw, inner in user_raw_to_inner.items():
                self._user_inner_arr[raw] = inner
        else:
            self._user_inner_arr = np.array([], dtype=np.int32)

    def predict(self, user_id: int, item_id: int) -> float:
        # returns the estimated rating for (user, item) pair
        # surprise expects string ids convert here
        return self._model.predict(str(user_id), str(item_id)).est

    def get_similarity(self, item_a: int, item_b: int) -> float:
        # Cosine similarity between two items using their latent vectors
        # cos(theta) = (a dot b) / (norm_a * norm_b)
        # if two vestors are simmilar cos(0) 1 very similar ontop of eachother
        # neutral cos(90) 0 no similarity
        # cos(180) -1 opposite negative
        # Returns 0.0 if either item is unknown (not in training data)

        # why cosine instead of euclidean distance?
        # cosine normalizes by vector length so popular items (longer vectors) dont dominate
        # measures angle between vectors — direction of preference, not magnitude

        # Fast path: pre-computed matrix
        if self._sim_matrix is not None:
            ia = self._raw_to_inner.get(item_a)
            ib = self._raw_to_inner.get(item_b)
            if ia is None or ib is None:
                return 0.0
            return float(self._sim_matrix[ia, ib])

        ia = self._raw_to_inner.get(item_a)
        ib = self._raw_to_inner.get(item_b)
        if ia is None or ib is None:
            return 0.0

        # qi is the item latent matrix shape (n_items, n_factors)
        vec_a = self._model.qi[ia]
        vec_b = self._model.qi[ib]
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def precompute_similarity_matrix(self) -> None:
        # Pre-compute the full item-item cosine similarity matrix
        # Only call this for ML-1M (~3700 items, ~55 MB as float32) Too large for 10M/20M....
        # float32 saves memory vs float64 — similarity values dont need full double precision
        from sklearn.metrics.pairwise import cosine_similarity

        qi = self._model.qi  # shape (n_surprise_items, n_factors)
        self._sim_matrix = cosine_similarity(qi).astype(np.float32)

    def predict_batch(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        # Vectorized scoring: mu + bu[u] + bi[items] + qi[items] @ pu[u].
        # Dense numpy arrays built at fit() time no Python loops, no dict lookups.
        self._ensure_fast_lookup()
        global_mean = self._trainset.global_mean

        # User lookup: O(1) array index, no try/except, no dict
        uid = int(user_id)
        if 0 <= uid < len(self._user_inner_arr) and self._user_inner_arr[uid] >= 0:
            u_inner  = int(self._user_inner_arr[uid])
            user_vec = self._model.pu[u_inner]            # shape (n_factors,)
            user_bias = float(self._model.bu[u_inner]) if self.biased else 0.0
        else:
            user_bias = 0.0
            user_vec  = np.zeros(self.n_factors)

        # Item lookup: fully vectorized, no Python loop
        item_arr  = np.asarray(item_ids, dtype=np.int64)
        in_bounds = (item_arr >= 0) & (item_arr < len(self._item_inner_arr))
        clipped   = np.where(in_bounds, item_arr, 0)
        i_inner   = self._item_inner_arr[clipped]
        unknown   = ~in_bounds | (i_inner < 0)
        i_inner_s = np.where(unknown, 0, i_inner)

        item_vecs = self._model.qi[i_inner_s]             # shape (n_items, n_factors)
        if self.biased:
            item_biases = self._model.bi[i_inner_s]
        else:
            item_biases = np.zeros(len(i_inner_s), dtype=np.float64)
        preds = global_mean + user_bias + item_biases + item_vecs @ user_vec
        preds[unknown] = global_mean
        return preds

    def similarity_matrix_batch(
        self, items_a: List[int], items_b: List[int]
    ) -> np.ndarray:
        # Cosine similarity between every pair in items_a x items_b: one matmul.
        # Returns shape (len(items_a), len(items_b)).
        self._ensure_fast_lookup()

        def _get_vecs(item_ids: List[int]) -> np.ndarray:
            arr       = np.asarray(item_ids, dtype=np.int64)
            in_bounds = (arr >= 0) & (arr < len(self._item_inner_arr))
            clipped   = np.where(in_bounds, arr, 0)
            i_inner   = self._item_inner_arr[clipped]
            unknown   = ~in_bounds | (i_inner < 0)
            i_inner_s = np.where(unknown, 0, i_inner)
            vecs = self._model.qi[i_inner_s].astype(np.float32)
            vecs[unknown] = 0.0
            return vecs

        vecs_a = _get_vecs(items_a)  # shape (n_a, n_factors)
        vecs_b = _get_vecs(items_b)  # shape (n_b, n_factors)

        norms_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
        norms_a[norms_a == 0] = 1.0
        norms_b[norms_b == 0] = 1.0
        vecs_a /= norms_a
        vecs_b /= norms_b

        return (vecs_a @ vecs_b.T).astype(np.float32)

    def rank_items_for_user(
        self, user_id: int, candidate_items: List[int]
    ) -> List[Tuple[int, float]]:
        # scores each candidate with svd prediction, returns sorted list descending
        # uses predict_batch() instead of 500 individual predict() calls
        preds = self.predict_batch(user_id, candidate_items)
        scores = list(zip(candidate_items, preds.tolist()))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def save(self, path: str) -> None:
        # serializes the whole object including trained weights and id mappings
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "SVDBaseline":
        # loads a previously saved model so training doesnt need to repeat
        with open(path, "rb") as f:
            return pickle.load(f)