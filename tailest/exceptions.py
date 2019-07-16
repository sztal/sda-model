"""Tail estimation exception classes."""


class HillEstimatorWarning(Warning):
    pass

class MomentsEstimatorWarning(Warning):
    pass

class KernelTypeEstimatorWarning(Warning):
    pass

class KernelTypeEstimatorError(Exception):
    pass
