use thiserror::Error;

#[derive(Debug, Error)]
pub enum CogError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("Serialization error: {0}")]
    Serialize(String),

    #[error("WAL error: {0}")]
    Wal(String),

    #[error("Store error: {0}")]
    Store(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid argument: {0}")]
    InvalidArg(String),
}

impl From<bincode::error::EncodeError> for CogError {
    fn from(e: bincode::error::EncodeError) -> Self {
        CogError::Serialize(e.to_string())
    }
}

impl From<bincode::error::DecodeError> for CogError {
    fn from(e: bincode::error::DecodeError) -> Self {
        CogError::Serialize(e.to_string())
    }
}

impl From<serde_json::Error> for CogError {
    fn from(e: serde_json::Error) -> Self {
        CogError::Serialize(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, CogError>;
