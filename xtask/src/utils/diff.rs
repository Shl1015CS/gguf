use super::{OperateError, OutputConfig, file_info::FileInfo};
use std::path::PathBuf;

pub fn diff(_a: PathBuf, _b: PathBuf, _out: OutputConfig) -> Result<Vec<FileInfo>, OperateError> {
    todo!()
}
