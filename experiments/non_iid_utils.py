import numpy as np
from torch.utils.data import Subset

def create_non_iid_clients(
train_dataset,
num_clients=10,
shards_per_client=2,
seed=42,
):
    rng = np.random.default_rng(seed)

    # Get labels for the examples in the supplied dataset.
    if isinstance(train_dataset, Subset):
        base_dataset = train_dataset.dataset
        base_indices = np.asarray(train_dataset.indices)
        targets = np.asarray(base_dataset.targets)[base_indices]
    else:
        targets = np.asarray(train_dataset.targets)

    # Sort examples by their digit labels.
    sorted_indices = np.argsort(targets)

    # Divide the sorted examples into shards.
    num_shards = num_clients * shards_per_client
    shards = np.array_split(sorted_indices, num_shards)

    # Randomly assign shards to clients.
    shard_order = rng.permutation(num_shards)

    client_datasets = []

    for client_id in range(num_clients):
        start = client_id * shards_per_client
        end = start + shards_per_client

        chosen_shards = shard_order[start:end]

        client_indices = np.concatenate(
            [shards[index] for index in chosen_shards]
        ).tolist()

        # These indices refer to positions in train_dataset.
        client_datasets.append(
            Subset(train_dataset, client_indices)
        )

    return client_datasets
