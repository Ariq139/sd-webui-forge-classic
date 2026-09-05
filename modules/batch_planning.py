from contextlib import contextmanager


@contextmanager
def restore_grouped_batch_settings(processing):
    """Do not leak temporary grouping state after success or an exception."""
    processing._extra_network_batch_plan = None
    try:
        yield
    finally:
        if processing._extra_network_batch_plan is not None:
            processing.batch_size = processing._extra_network_original_batch_size
            processing.n_iter = processing._extra_network_original_n_iter
            processing._current_batch_indices = None
            processing._current_output_index = None
            processing._extra_network_batch_plan = None


def group_extra_network_batches(signatures, batch_size):
    """Group equal signatures without exceeding the requested batch size."""
    requested_batches = [
        list(range(start, min(start + batch_size, len(signatures))))
        for start in range(0, len(signatures), batch_size)
    ]
    if not any(len({signatures[index] for index in batch}) > 1 for batch in requested_batches):
        return None

    grouped_indices = {}
    for index, signature in enumerate(signatures):
        grouped_indices.setdefault(signature, []).append(index)

    return [
        indices[start : start + batch_size]
        for indices in grouped_indices.values()
        for start in range(0, len(indices), batch_size)
    ]


def restore_grouped_output_order(output_images, output_records, infotext_records):
    ordered_images = []
    for _, _, start, end in sorted(output_records):
        ordered_images.extend(output_images[start:end])
    ordered_infotexts = [text for _, _, text in sorted(infotext_records)]
    return ordered_images, ordered_infotexts


def select_batch_values(values, iteration, batch_size, batch_indices=None):
    if batch_indices is not None:
        return [values[index] for index in batch_indices]

    start = iteration * batch_size
    return values[start : start + batch_size]


def effective_prompt_index(processing, index, prompts):
    batch_indices = getattr(processing, "_current_batch_indices", None)
    if batch_indices is not None and prompts is not processing.all_prompts:
        return batch_indices[index]
    return index


def effective_output_batch_size(processing):
    if getattr(processing, "_current_output_index", None) is not None:
        return processing._extra_network_original_batch_size
    return processing.batch_size


def effective_output_batch_index(processing):
    output_index = getattr(processing, "_current_output_index", None)
    if output_index is not None:
        return output_index % effective_output_batch_size(processing)
    return processing.batch_index


def effective_output_generation_index(processing):
    output_index = getattr(processing, "_current_output_index", None)
    if output_index is not None:
        return output_index
    return processing.iteration * processing.batch_size + processing.batch_index
