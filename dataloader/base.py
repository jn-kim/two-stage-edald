from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from utils.query_utils import decode_queries

class BaseDataset(Dataset):
    def __init__(self, transform=None):
        super(BaseDataset, self).__init__()
        self.transform = transform
        self.list_labelled_queries: Optional[List[np.ndarray]] = None
        self.global_queries: Optional[List[np.ndarray]] = None
        self.local_candidates: Optional[List[np.ndarray]] = None

    def label_queries(
        self,
        dict_queries: Dict[str, dict],
        dict_global_queries: Dict[str, dict]
    ):
        queries = decode_queries(dict_queries, self.list_inputs)
        assert len(queries) == len(self.queries), "Decoded query length does not match existing query length."

        self.queries = [np.logical_or(prev, new) for prev, new in zip(self.queries, queries)]

        global_qs = decode_queries(dict_global_queries, self.list_inputs)
        assert len(global_qs) == len(self.queries), "global_queries should match the number of images."

        self.global_queries = [
            np.logical_or(prev_gq, new_gq)
            for prev_gq, new_gq in zip(self.global_queries, global_qs)
        ]

        assert len(self.queries) == len(self.list_labels)
        assert len(self.global_queries) == len(self.list_labels)

    def __getitem__(self, ind):
        p_img: str = self.list_inputs[ind]
        x = Image.open(p_img).convert("RGB")
        y = np.load(self.list_labels[ind])
        w, h = x.size

        queries = torch.tensor(self.queries[ind].astype(bool), dtype=torch.bool) \
            if self.queries is not None else torch.ones((h, w), dtype=torch.bool)

        labelled_queries = torch.tensor(self.list_labelled_queries[ind]) \
            if self.list_labelled_queries is not None else torch.zeros((h, w), dtype=torch.uint8)

        return {
            'x': self.transform(x) if self.transform else TF.to_tensor(x) * 2 - 1,
            'y': torch.tensor(np.asarray(y, dtype=np.int64), dtype=torch.long),
            "queries": queries,
            "labelled_queries": labelled_queries,
            "p_img": p_img
        }