"""The pipeline's own log lines, and keeping them visible for the whole run.

Something imported part-way through a run can silence them. On the CGL full
run every INFO and WARNING line of ours stopped after training - checkpoint
selection's progress ("validation generation: n/N row(s)") and its choice
("selected ... from N candidates") never reached the log in seven hours - while
vLLM, which logs through its own handler, kept printing. The likely cause is
ms-swift, imported into the pipeline process for the pre-launch masking check
(training.train._masking_encoder): its logger setup raises the root console
handlers. :func:`reassert` puts our lines back, and is called before every
stage and after anything known to touch logging.
"""

from __future__ import annotations

import logging

FORMAT = "%(levelname)s %(name)s %(message)s"

#: This repository's top-level packages: their loggers are never left disabled.
OWN_PACKAGES = frozenset({
    "artifact_registry", "calibration", "common", "data_pipeline", "evaluation", "inference_core",
    "orchestration", "pilot", "postprocessing", "registry_utils", "scripts", "serving", "testing",
    "training", "azure_service",
})

_level = logging.INFO


class ConsoleHandler(logging.StreamHandler):
    """The run's console handler - a subclass on purpose.

    ms-swift's and modelscope's ``get_logger`` silence the root logger's console
    by raising every handler whose type *is* ``logging.StreamHandler`` (what
    ``basicConfig`` installs) to ERROR. A subclass is not that type, so ours is
    left alone at the source; :func:`reassert` stays as the backstop for any
    other library.
    """


def setup(level: int = logging.INFO) -> None:
    """The run's logging: ``level`` and up, to the console, in :data:`FORMAT`."""
    global _level
    _level = level
    root = logging.getLogger()
    if not any(isinstance(handler, ConsoleHandler) for handler in root.handlers):
        root.addHandler(_console())
    root.setLevel(level)
    reassert()


def _console() -> ConsoleHandler:
    handler = ConsoleHandler()
    handler.setFormatter(logging.Formatter(FORMAT))
    return handler


def reassert() -> None:
    """Undo what a library did to our lines since :func:`setup`.

    Lowers the root logger and its handlers back to the run's level where they
    were raised, restores a console handler if they were removed, and re-enables
    this repository's loggers if a ``dictConfig`` disabled them. It never raises
    a level: a library that logs less than this keeps logging less.
    """
    root = logging.getLogger()
    if root.level > _level:
        root.setLevel(_level)
    if not root.handlers:
        root.addHandler(_console())
    for handler in root.handlers:
        if handler.level > _level:
            handler.setLevel(_level)
    for name, logger in list(logging.root.manager.loggerDict.items()):
        if isinstance(logger, logging.Logger) and name.split(".", 1)[0] in OWN_PACKAGES:
            logger.disabled = False
            if logger.level > _level:
                logger.setLevel(logging.NOTSET)
