pub mod engine;
pub mod error;
pub mod graph;
pub mod storage;
pub mod stores;
pub mod types;
pub mod vector;
pub mod wal;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// PyO3 module entry point. The function name must match `module-name` in
/// cogdb_engine/pyproject.toml so Python can import it as `cogdb_engine`.
#[cfg(feature = "python")]
#[pymodule]
fn cogdb_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
