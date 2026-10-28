import copy

import torch
import torch.nn as nn

def fedsgd_round(
    global_model,
    client_datasets,
    learning_rate=0.01,
    batch_size=1024,
    device="cpu",
):


    """
    Perform one communication round of FedSGD.

    Each client computes gradients over its complete local dataset.
    The server aggregates gradients weighted by local dataset size
    and performs one global model update.
    """

    global_model.to(device)
    global_model.train()

    loss_function = nn.CrossEntropyLoss(reduction="mean")


    # Initialize accumulated gradients for every model parameter.
    aggregated_gradients = [
        torch.zeros_like(parameter)
        for parameter in global_model.parameters()
    ]

    total_examples = sum(len(dataset) for dataset in client_datasets)

    if total_examples == 0:
        raise ValueError("The total number of client examples is zero.")

    # Each client computes gradients independently.
    for dataset in client_datasets:

        client_model = copy.deepcopy(global_model)
        client_model.to(device)
        client_model.train()

        client_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        client_size = len(dataset)

        # Compute the gradient of the mean loss over the client's
        # complete local dataset.
        for parameter in client_model.parameters():
            parameter.grad = None

        total_loss = 0.0

        for images, labels in client_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = client_model(images)
            loss = loss_function(outputs, labels)

            # Scale each batch's mean loss by its number of examples.
            current_batch_size = images.size(0)

            weighted_loss = (
                loss * current_batch_size / client_size
            )

            weighted_loss.backward()

            total_loss += loss.item() * current_batch_size


        # Aggregate client gradients, weighted by client dataset size.
        with torch.no_grad():
            for aggregated, parameter in zip(
                aggregated_gradients,
                client_model.parameters(),
            ):
                if parameter.grad is not None:
                    aggregated.add_(
                        parameter.grad,
                        alpha=client_size / total_examples,
                    )

        print(
            f"Client processed {client_size} examples; "
            f"local loss: {total_loss / client_size:.4f}"
        )

        del client_model
    # Clip the global aggregated gradient.
    torch.nn.utils.clip_grad_norm_(
        aggregated_gradients,
        max_norm=1.0,
    )

    # Apply the clipped gradient to the global model.
    with torch.no_grad():
        for parameter, gradient in zip(
            global_model.parameters(),
            aggregated_gradients,
        ):
            parameter.add_(gradient, alpha=-learning_rate)
