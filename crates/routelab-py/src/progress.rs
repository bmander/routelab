//! Progress reporting for preprocessing that has released the GIL.

use pyo3::prelude::*;

use routelab_core::util::progress::Progress as CoreProgress;

/// How far along a piece of preprocessing is.
///
/// Hand one to a preprocessing call and read it from another thread while that
/// call runs. Reading takes no lock and does not need the GIL back, which is
/// the point: preprocessing releases the GIL precisely so that something else
/// can be doing something, and a progress report that took it back would be
/// asking the work to stop in order to say it had not.
#[pyclass(name = "Progress", module = "routelab._routelab", frozen)]
#[derive(Default)]
pub struct PyProgress {
    pub(crate) inner: CoreProgress,
}

/// The counter a preprocessing call should write into: the one that was
/// handed in, or a fresh one nobody reads.
///
/// The same shape as `footpaths_or_none`, and for the same reason: every
/// binding that takes an optional argument unwraps it the same way, so the
/// unwrapping is written once.
pub(crate) fn counter(progress: Option<&PyProgress>) -> CoreProgress {
    progress.map_or_else(CoreProgress::new, |held| held.inner.clone())
}

#[pymethods]
impl PyProgress {
    #[new]
    fn new() -> Self {
        PyProgress::default()
    }

    /// `(phase, done, total)`. `total` is 0 when the work cannot say — which is
    /// different from not having started, and worth being able to tell apart.
    fn read(&self) -> (&'static str, u64, u64) {
        let (done, total) = self.inner.read();
        (self.inner.phase(), done, total)
    }

    /// What is happening now — `"contracting"`, `"assembling"`, `"measuring"`.
    ///
    /// A fraction is always within a phase, so this is half the answer: a build
    /// with two unequal halves reports the second from zero rather than letting
    /// the first sit at 100% while it runs.
    #[getter]
    fn phase(&self) -> &'static str {
        self.inner.phase()
    }

    /// How far along, 0 to 1, or `None` from work with no honest measure.
    #[getter]
    fn fraction(&self) -> Option<f64> {
        self.inner.fraction()
    }

    fn __repr__(&self) -> String {
        match self.inner.fraction() {
            Some(fraction) => {
                format!("Progress({} {:.0}%)", self.inner.phase(), fraction * 100.0)
            }
            None => "Progress(unknown)".to_string(),
        }
    }
}
