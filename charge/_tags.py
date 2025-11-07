def verifier(func):
    func.__verifier_tag = True
    return func


def is_verifier(func):
    return hasattr(func, "__verifier_tag")


def hypothesis(func):
    func.__hypothesis_tag = True
    return func


def is_hypothesis(func):
    return hasattr(func, "__hypothesis_tag")
