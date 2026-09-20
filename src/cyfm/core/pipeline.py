"""The shape every entry point has, stated once.

``train.py`` and ``evaluate.py`` were each a single Hydra ``main``: 535 lines and
226 lines respectively, with no seam between deciding *what* to run and running
it. Neither had a unit test, and neither could: the only way to reach any of that
logic was to stand up Hydra and execute the whole thing.

A pipeline separates the three phases that were interleaved there. ``setup``
builds what the run needs and validates the configuration, so an unusable
combination fails before a dataset is indexed or a model is built. ``execute``
does the work and returns it as data. ``teardown`` releases what ``setup``
acquired, and runs even when ``execute`` raises -- which the old ``train.py`` did
not do for the process group or the W&B run, so an exception mid-epoch left both
dangling.

Deliberately not a framework. There is no hook registry, no event bus and no
plugin lookup; a pipeline is three methods and the order they run in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ResultT = TypeVar("ResultT")

__all__ = ["BasePipeline"]


class BasePipeline(ABC, Generic[ResultT]):
    """Template method: ``setup``, then ``execute``, then ``teardown``.

    Subclasses override the phases, never :meth:`run`. The ordering and the
    guarantee that ``teardown`` happens are the only things this class provides,
    and they are the two things every entry point was getting subtly wrong.
    """

    def run(self) -> ResultT:
        """Execute the pipeline and return its result.

        Returns:
            Whatever :meth:`execute` produced.
        """
        self.setup()
        try:
            return self.execute()
        finally:
            self.teardown()

    def setup(self) -> None:
        """Acquire resources and validate the configuration.

        Validation belongs here rather than at first use: the alternative is
        discovering an unusable combination after a cohort has been indexed and a
        model built. Does nothing by default.
        """

    @abstractmethod
    def execute(self) -> ResultT:
        """Do the work and return it as data.

        Returning rather than printing is what makes a pipeline testable, and
        what lets the reproduction layer render the same numbers as text, JSON or
        LaTeX without re-running anything.
        """

    def teardown(self) -> None:
        """Release what :meth:`setup` acquired.

        Runs even when :meth:`execute` raises. Does nothing by default.
        """
