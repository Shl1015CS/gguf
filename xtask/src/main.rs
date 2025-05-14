// #![deny(warnings)]

mod convert;
mod diff;
mod merge;
mod set_meta;
mod show;
mod split;
mod utils;

#[cfg(test)]
mod tests;

#[macro_use]
extern crate clap;
use clap::Parser;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        Show(args) => args.show(),
        Split(args) => args.split(),
        Merge(args) => args.merge(),
        Convert(args) => args.convert(),
        Diff(args) => args.diff(),
        SetMeta(args) => args.set_meta(),
    }
}

/// gguf-utils is a command-line tool for working with gguf files.
#[derive(Parser)]
#[clap(version)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show the contents of gguf files
    Show(show::ShowArgs),
    /// Split gguf files into shards
    Split(split::SplitArgs),
    /// Merge shards into a single gguf file
    Merge(merge::MergeArgs),
    /// Convert gguf files to different format
    Convert(convert::ConvertArgs),
    /// Diff two gguf files
    Diff(diff::DiffArgs),
    /// Set metadata of gguf files
    SetMeta(set_meta::SetMetaArgs),
}

#[derive(Args, Default)]
struct LogArgs {
    /// Log level, may be "off", "trace", "debug", "info" or "error".
    #[clap(long)]
    log: Option<String>,
}

impl LogArgs {
    fn init(self) {
        use log::LevelFilter;
        use simple_logger::SimpleLogger;
        use time::UtcOffset;

        let level = self
            .log
            .and_then(|level| match level.to_lowercase().as_str() {
                "off" | "none" => Some(LevelFilter::Off),
                "all" | "trace" => Some(LevelFilter::Trace),
                "debug" => Some(LevelFilter::Debug),
                "info" => Some(LevelFilter::Info),
                "error" => Some(LevelFilter::Error),
                _ => None,
            })
            .unwrap_or(LevelFilter::Warn);

        const EAST8: UtcOffset = match UtcOffset::from_hms(8, 0, 0) {
            Ok(it) => it,
            Err(_) => unreachable!(),
        };
        SimpleLogger::new()
            .with_level(level)
            .with_utc_offset(UtcOffset::current_local_offset().unwrap_or(EAST8))
            .init()
            .unwrap();
    }
}
