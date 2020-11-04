import resource


def peak_memory() -> float:
    """
    Peak memory in GB
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
