#[cfg(test)]
mod tests {
    use crate::*;
    use clap::Parser;

    #[test]
    fn test_cli_show_command() {
        let args = vec!["gguf-utils", "show", "test.gguf"];
        let cli = Cli::parse_from(args);
        match cli.command {
            Commands::Show(_) => assert!(true),
            _ => assert!(false, "Expected Show command"),
        }
    }

    #[test]
    fn test_cli_split_command() {
        let args = vec!["gguf-utils", "split", "test.gguf", "-t", "2"];
        let cli = Cli::parse_from(args);
        match cli.command {
            Commands::Split(_) => assert!(true),
            _ => assert!(false, "Expected Split command"),
        }
    }

    #[test]
    fn test_cli_merge_command() {
        let args = vec!["gguf-utils", "merge", "shard.00.gguf"];
        let cli = Cli::parse_from(args);
        match cli.command {
            Commands::Merge(_) => assert!(true),
            _ => assert!(false, "Expected Merge command"),
        }
    }

    #[test]
    fn test_log_args() {
        let log_args = LogArgs {
            log: Some("debug".to_string()),
        };
        log_args.init();
        // 由于日志初始化是全局的，我们只能验证它不会panic
        assert!(true);
    }
} 