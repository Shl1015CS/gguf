use crate::{
    LogArgs,
    utils::{OutputArgs, diff, show_file_info},
};
use std::path::PathBuf;

#[derive(Args, Default)]
pub struct DiffArgs {
    /// The file as reference
    a: PathBuf,
    /// The file to diff
    b: PathBuf,

    #[clap(flatten)]
    output: OutputArgs,
    #[clap(flatten)]
    log: LogArgs,
}

impl DiffArgs {
    pub fn diff(self) {
        let Self { a, b, output, log } = self;
        log.init();

        let files = diff(a, b, output.into()).unwrap();
        show_file_info(&files);
    }
}
